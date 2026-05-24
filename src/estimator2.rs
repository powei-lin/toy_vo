use anyhow::Result;
use camera_intrinsic_model::generic_model::GenericModel;
use image::GrayImage;
use nalgebra as na;
use patch_tracker::StereoPatchTracker;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tiny_solver::manifold::se3::SE3;
use tiny_solver::manifold::so3::SO3;

use crate::frame::Frame;
use crate::keyframe_sliding_window::KeyframeSlidingWindow;
use crate::optimization::bundle_adjustment;

const TRANSLATION_THRESHOLD: f64 = 0.4; // meters
const ROTATION_THRESHOLD: f64 = 0.25; // radians
const EPIPOLAR_ERROR_THRESHOLD: f32 = 0.005; // epipolar error threshold for filtering matches
const MAX_FRAMES_BETWEEN_KEYFRAMES: u64 = 10; // Maximum number of frames between keyframes

pub fn triangulate_points(
    undist_pt0: &na::Vector3<f32>,
    undist_pt1: &na::Vector3<f32>,
    t_1_0: na::SMatrixView<f32, 3, 4>,
) -> na::Point3<f32> {
    // println!("{}", t_1_0);
    let r0 = undist_pt1[0] * t_1_0.row(2) - t_1_0.row(0);
    let r1 = undist_pt1[1] * t_1_0.row(2) - t_1_0.row(1);
    let design_matrix = unsafe {
        na::Matrix4::new(
            -1.0,
            0.0,
            undist_pt0[0],
            0.0,
            0.0,
            -1.0,
            undist_pt0[1],
            0.0,
            *r0.get_unchecked(0),
            *r0.get_unchecked(1),
            *r0.get_unchecked(2),
            *r0.get_unchecked(3),
            *r1.get_unchecked(0),
            *r1.get_unchecked(1),
            *r1.get_unchecked(2),
            *r1.get_unchecked(3),
        )
    };
    let svd = design_matrix.svd(false, true);
    let vt: na::Matrix4<f32> = svd.v_t.unwrap();
    let p3d = vt.row(3) / vt[(3, 3)];
    na::Point3::from(p3d.transpose().fixed_rows::<3>(0).into_owned())
}

fn try_add_point() {}

/// Placeholder estimator implementation.
/// Currently mimics the control flow and logging structure of the C++ Estimator::process_frame,
/// but uses dummy values for tracking, optimization, and mapping.
pub struct Estimator2 {
    frame_id_counter: u64,
    frames_since_last_keyframe: u64,
    pub tracker: StereoPatchTracker,
    cam0: GenericModel<f32>,
    cam1: GenericModel<f32>,
    t_cam1_cam0: na::Isometry3<f32>,
    t_hat_rmat: na::Matrix3<f32>,
    t_1_0_matrix: na::SMatrix<f32, 3, 4>,
    // Full trajectory of keyframes
    trajectory: Vec<na::Isometry3<f32>>,
    // Latest computed pose (T_W_Cl: left camera in world)
    pub current_t_w_cam0: na::Isometry3<f32>,
    pub keyframe_window: KeyframeSlidingWindow,
    pub landmarks: HashMap<usize, na::Point3<f32>>, // Map from feature ID to 3D position in world
}

impl Estimator2 {
    /// Create a new estimator configured with camera intrinsics and distortion
    /// loaded from the YAML configuration.
    ///
    /// The `viewer` reference must outlive the estimator.
    pub fn new(
        cam0: GenericModel<f32>,
        cam1: GenericModel<f32>,
        t_cam1_cam0: na::Isometry3<f32>,
        tracker_optical_flow_levels: u32,
        tracker_grid_size: u32,
        keyframe_window_size: usize,
    ) -> Self {
        let t_hat = SO3::hat(t_cam1_cam0.translation.vector.normalize().as_view());
        let t_hat_rmat = t_hat * t_cam1_cam0.rotation.to_rotation_matrix().matrix();
        let t_1_0_matrix = {
            let mut mat = na::SMatrix::<f32, 3, 4>::zeros();
            mat.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&t_cam1_cam0.rotation.to_rotation_matrix().matrix());
            mat.fixed_view_mut::<3, 1>(0, 3)
                .copy_from(&t_cam1_cam0.translation.vector);
            mat
        };
        Self {
            frame_id_counter: 0,
            frames_since_last_keyframe: 0,
            tracker: StereoPatchTracker::new(tracker_optical_flow_levels, tracker_grid_size),
            cam0,
            cam1,
            t_cam1_cam0,
            t_hat_rmat,
            trajectory: Vec::new(),
            current_t_w_cam0: na::Isometry3::identity(),
            keyframe_window: KeyframeSlidingWindow::new(keyframe_window_size),
            landmarks: HashMap::new(),
            t_1_0_matrix,
        }
    }

    /// Process a single stereo frame.
    pub fn process_frame(
        &mut self,
        left_gray_image: &GrayImage,
        right_gray_image: &GrayImage,
    ) -> Result<()> {
        self.tracker
            .process_frame(&left_gray_image, &right_gray_image);
        let tracked_points = self.tracker.get_track_points();

        let mut bad_points = Vec::new();
        let mut cam0_observations = HashMap::new();
        let mut cam1_observations = HashMap::new();
        let mut new_point_ids = Vec::new();

        // initialize landmarks if this is the first frame
        if self.landmarks.is_empty() {
            // Initialize landmarks from the first frame's tracked points
            for (id, &(kp0_x, kp0_y)) in &tracked_points[0] {
                if let Some(&(kp1_x, kp1_y)) = tracked_points[1].get(id) {
                    println!("Initializing landmark {} from first frame", id);
                    let mut undistorted_pt_cam0 =
                        self.cam0.unproject_one(&na::Vector2::new(kp0_x, kp0_y));
                    let mut undistorted_pt_cam1 =
                        self.cam1.unproject_one(&na::Vector2::new(kp1_x, kp1_y));
                    if undistorted_pt_cam0.z <= 0.0 || undistorted_pt_cam1.z <= 0.0 {
                        println!(
                            "Warning: Landmark {} has non-positive depth after unprojection, skipping",
                            id
                        );
                        bad_points.push(*id);
                        continue; // Skip points that cannot be unprojected
                    }
                    undistorted_pt_cam0 /= undistorted_pt_cam0.z;
                    undistorted_pt_cam1 /= undistorted_pt_cam1.z;
                    let epipolar_error =
                        undistorted_pt_cam1.transpose() * self.t_hat_rmat * undistorted_pt_cam0;
                    if epipolar_error[(0, 0)].abs() > EPIPOLAR_ERROR_THRESHOLD {
                        bad_points.push(*id);
                        println!(
                            "Warning: Landmark {} has high epipolar error ({:.6}), skipping",
                            id,
                            epipolar_error[(0, 0)].abs()
                        );
                        continue; // Skip points that violate the epipolar constraint
                    }
                    // Triangulate the 3D position in the left camera frame (assuming a fixed depth of 1.0 meter)
                    let landmark_pos_cam0 = triangulate_points(
                        &undistorted_pt_cam0,
                        &undistorted_pt_cam1,
                        self.t_1_0_matrix.as_view(),
                    );
                    if landmark_pos_cam0.z <= 0.0 {
                        println!(
                            "Warning: Landmark {} has non-positive depth after triangulation, skipping",
                            id
                        );
                        bad_points.push(*id);
                        continue; // Skip points that cannot be triangulated
                    }
                    cam0_observations.insert(*id, (undistorted_pt_cam0.x, undistorted_pt_cam0.y));
                    cam1_observations.insert(*id, (undistorted_pt_cam1.x, undistorted_pt_cam1.y));
                    self.landmarks.insert(*id, landmark_pos_cam0);
                    new_point_ids.push(*id);
                }
            }
            self.tracker.remove_id(&bad_points);
            let frame = Frame::new(
                na::Isometry::identity(),
                cam0_observations,
                cam1_observations,
                new_point_ids,
            );
            self.keyframe_window.add_keyframe(frame);
        } else {
            let mut known_landmark_pnp = Vec::new();
            let mut observations_pnp = Vec::new();
            // let mut local_landmark = Vec::new();

            // pnp to find pose first
            for (id, &(kp0_x, kp0_y)) in &tracked_points[0] {
                if let Some(p3d) = self.landmarks.get(id) {
                    let mut undistorted_pt_cam0 =
                        self.cam0.unproject_one(&na::Vector2::new(kp0_x, kp0_y));
                    if undistorted_pt_cam0.z <= 0.0 {
                        println!(
                            "Warning: Landmark {} has non-positive depth after unprojection, skipping",
                            id
                        );
                        bad_points.push(*id);
                        continue; // Skip points that cannot be unprojected
                    }
                    undistorted_pt_cam0 /= undistorted_pt_cam0.z;
                    let p3d = p3d.cast::<f64>();
                    known_landmark_pnp.push((p3d.x, p3d.y, p3d.z));
                    observations_pnp
                        .push((undistorted_pt_cam0.x as f64, undistorted_pt_cam0.y as f64));
                    continue; // Skip already initialized landmarks
                }
            }
            println!(
                "Solving PnP with {} known landmarks",
                known_landmark_pnp.len()
            );

            let (rvec, tvec) = sqpnp_simple::sqpnp_solve(&known_landmark_pnp, &observations_pnp)
                .expect("solve pnp failed");
            let t_cam0_w_pnp = na::Isometry3::new(
                na::Vector3::new(tvec.0, tvec.1, tvec.2),
                na::Vector3::new(rvec.0, rvec.1, rvec.2),
            );
            let t_cam0_w_pnp_f32 = t_cam0_w_pnp.cast::<f32>();
            self.current_t_w_cam0 = t_cam0_w_pnp_f32.inverse();

            let t_last_key_cam0_current_cam0 =
                self.keyframe_window.last_keyframe_t_cam0_w().cast() * t_cam0_w_pnp.inverse();
            let need_keyframe = t_last_key_cam0_current_cam0
                .translation
                .vector
                .norm_squared()
                > TRANSLATION_THRESHOLD.powi(2)
                || t_last_key_cam0_current_cam0
                    .rotation
                    .scaled_axis()
                    .norm_squared()
                    > ROTATION_THRESHOLD.powi(2)
                || self.frames_since_last_keyframe > MAX_FRAMES_BETWEEN_KEYFRAMES; // Rough rotation threshold based on trace

            if need_keyframe {
                // add observations for existing landmarks
                for (id, &(kp0_x, kp0_y)) in &tracked_points[0] {
                    if self.landmarks.contains_key(id) {
                        let mut undistorted_pt_cam0 =
                            self.cam0.unproject_one(&na::Vector2::new(kp0_x, kp0_y));
                        if undistorted_pt_cam0.z <= 0.0 {
                            println!(
                                "Warning: Landmark {} has non-positive depth after unprojection, skipping",
                                id
                            );
                            bad_points.push(*id);
                            continue; // Skip points that cannot be unprojected
                        }
                        undistorted_pt_cam0 /= undistorted_pt_cam0.z;
                        cam0_observations
                            .insert(*id, (undistorted_pt_cam0.x, undistorted_pt_cam0.y));
                    }
                }
                for (id, &(kp1_x, kp1_y)) in &tracked_points[1] {
                    if self.landmarks.contains_key(id) {
                        let mut undistorted_pt_cam1 =
                            self.cam1.unproject_one(&na::Vector2::new(kp1_x, kp1_y));
                        if undistorted_pt_cam1.z <= 0.0 {
                            println!(
                                "Warning: Landmark {} has non-positive depth after unprojection, skipping",
                                id
                            );
                            bad_points.push(*id);
                            continue; // Skip points that cannot be unprojected
                        }
                        undistorted_pt_cam1 /= undistorted_pt_cam1.z;
                        cam1_observations
                            .insert(*id, (undistorted_pt_cam1.x, undistorted_pt_cam1.y));
                    }
                }

                // add new points from current frame to the map
                for (id, &(kp0_x, kp0_y)) in &tracked_points[0] {
                    if self.landmarks.contains_key(id) {
                        continue; // Skip already initialized landmarks
                    }
                    if let Some(&(kp1_x, kp1_y)) = tracked_points[1].get(id) {
                        // println!("Triangulating new landmark {} from current frame", id);
                        let mut undistorted_pt_cam0 =
                            self.cam0.unproject_one(&na::Vector2::new(kp0_x, kp0_y));
                        let mut undistorted_pt_cam1 =
                            self.cam1.unproject_one(&na::Vector2::new(kp1_x, kp1_y));
                        if undistorted_pt_cam0.z <= 0.0 || undistorted_pt_cam1.z <= 0.0 {
                            println!(
                                "Warning: Landmark {} has non-positive depth after unprojection, skipping",
                                id
                            );
                            bad_points.push(*id);
                            continue; // Skip points that cannot be unprojected
                        }
                        undistorted_pt_cam0 /= undistorted_pt_cam0.z;
                        undistorted_pt_cam1 /= undistorted_pt_cam1.z;
                        let epipolar_error =
                            undistorted_pt_cam1.transpose() * self.t_hat_rmat * undistorted_pt_cam0;
                        if epipolar_error[(0, 0)].abs() > EPIPOLAR_ERROR_THRESHOLD {
                            bad_points.push(*id);
                            println!(
                                "Warning: Landmark {} has high epipolar error ({:.6}), skipping",
                                id,
                                epipolar_error[(0, 0)].abs()
                            );
                            continue; // Skip points that violate the epipolar constraint
                        }
                        // Triangulate the 3D position in the left camera frame
                        let landmark_pos_cam0 = triangulate_points(
                            &undistorted_pt_cam0,
                            &undistorted_pt_cam1,
                            self.t_1_0_matrix.as_view(),
                        );
                        if landmark_pos_cam0.z <= 0.0 {
                            println!(
                                "Warning: Landmark {} has non-positive depth after triangulation, skipping",
                                id
                            );
                            bad_points.push(*id);
                            continue; // Skip points that cannot be triangulated
                        }
                        cam0_observations
                            .insert(*id, (undistorted_pt_cam0.x, undistorted_pt_cam0.y));
                        cam1_observations
                            .insert(*id, (undistorted_pt_cam1.x, undistorted_pt_cam1.y));
                        let landmark_pos_world = self.current_t_w_cam0 * landmark_pos_cam0;
                        self.landmarks.insert(*id, landmark_pos_world);
                        new_point_ids.push(*id);
                    } else {
                        println!(
                            "Warning: Landmark {} does not have a match in the right image, skipping",
                            id
                        );
                        bad_points.push(*id);
                    }
                }

                println!("Adding new keyframe with pose: {:?}", self.current_t_w_cam0);
                let frame = Frame::new(
                    t_cam0_w_pnp_f32,
                    cam0_observations,
                    cam1_observations,
                    new_point_ids,
                );
                let marg_keyframe_pose_and_ids = self.keyframe_window.add_keyframe(frame);
                self.frames_since_last_keyframe = 0;

                if let Some((marg_keyframe_pose, marg_keyframe_ids)) = marg_keyframe_pose_and_ids {
                    println!(
                        "Marginalizing out keyframe with pose: {:?}",
                        marg_keyframe_pose
                    );
                    for id in &marg_keyframe_ids {
                        self.landmarks.remove(id);
                        bad_points.push(*id);
                    }
                }
                let bad_landmarks = bundle_adjustment(
                    &mut self.keyframe_window,
                    &mut self.landmarks,
                    &self.t_cam1_cam0,
                );
                for id in bad_landmarks {
                    self.landmarks.remove(&id);
                    bad_points.push(id);
                }
            } else {
                for id in tracked_points[0].keys() {
                    if !self.landmarks.contains_key(id) {
                        bad_points.push(*id);
                    }
                }
                println!("Not adding keyframe, pose change is too small");
            }

            // For subsequent frames, we can attempt to triangulate new landmarks from newly tracked points
            self.tracker.remove_id(&bad_points);
        }

        self.frame_id_counter += 1;
        self.frames_since_last_keyframe += 1;

        Ok(())
    }
}
