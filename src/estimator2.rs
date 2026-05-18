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

const TRANSLATION_THRESHOLD: f64 = 0.4; // meters
const ROTATION_THRESHOLD: f64 = 0.25; // radians
const EPIPOLAR_ERROR_THRESHOLD: f32 = 0.005; // epipolar error threshold for filtering matches

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
    pub current_pose: na::Isometry3<f32>,
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
            current_pose: na::Isometry3::identity(),
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
        let total_start_time = Instant::now();

        // // New frame: update counters
        // self.frame_id_counter += 1;
        // self.frames_since_last_keyframe += 1;

        // // Timing placeholders
        // let mut frame_creation_time_ms = 0.0f64;
        // let mut patch_tracking_time_ms = 0.0f64;
        // let mut motion_tracking_time_ms = 0.0f64;
        // let mut optimization_time_ms = 0.0f64;

        // // Create frame (images are not stored, only features will be added)
        // let mut current_frame = Frame2::from_stereo_images(
        //     self.frame_id_counter as i32,
        //     self.cam0.clone(),
        //     self.cam1.clone(),
        //     self.t_cam1_cam0,
        // );

        // // frame_creation_time_ms = frame_creation_start.elapsed().as_secs_f64() * 1000.0;

        // // Patch tracking
        // let tracking_start = Instant::now();
        self.tracker
            .process_frame(&left_gray_image, &right_gray_image);
        let tracked_points = self.tracker.get_track_points();
        let mut bad_points = Vec::new();
        let mut cam0_observations = HashMap::new();
        let mut cam1_observations = HashMap::new();

        // initialize landmarks if this is the first frame
        if self.landmarks.is_empty() {
            // Initialize landmarks from the first frame's tracked points
            for (id, &(kp0_x, kp0_y)) in &tracked_points[0] {
                // For simplicity, we initialize all landmarks at a fixed depth of 1.0 meter in front of the camera
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
                }
            }
            self.tracker.remove_id(&bad_points);
            let frame = Frame::new(
                na::Isometry::identity(),
                cam0_observations,
                cam1_observations,
            );
            self.keyframe_window.add_keyframe(frame);
        } else {
            let mut known_landmark_pnp = Vec::new();
            let mut observations_pnp = Vec::new();
            let mut local_landmark = Vec::new();
            // For subsequent frames, we can attempt to triangulate new landmarks from newly tracked points
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
                if let Some(&(kp1_x, kp1_y)) = tracked_points[1].get(id) {
                    println!("Triangulating new landmark {} from current frame", id);
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
                    cam0_observations.insert(*id, (undistorted_pt_cam0.x, undistorted_pt_cam0.y));
                    cam1_observations.insert(*id, (undistorted_pt_cam1.x, undistorted_pt_cam1.y));
                    local_landmark.push((*id, landmark_pos_cam0));
                }
            }
            self.tracker.remove_id(&bad_points);

            let (rvec, tvec) = sqpnp_simple::sqpnp_solve(&known_landmark_pnp, &observations_pnp)
                .expect("solve pnp failed");
            let t_w_cam0_pnp = na::Isometry3::new(
                na::Vector3::new(tvec.0, tvec.1, tvec.2),
                na::Vector3::new(rvec.0, rvec.1, rvec.2),
            );
            let t_w_cam0_pnp_f32 = t_w_cam0_pnp.cast::<f32>();
            local_landmark.iter().for_each(|(id, pos)| {
                let pos_cam0 = t_w_cam0_pnp_f32 * pos;
                self.landmarks.insert(*id, pos_cam0);
            });
            self.current_pose = t_w_cam0_pnp_f32;
            let t_diff = self.keyframe_window.last_keyframe_pose().inverse().cast() * t_w_cam0_pnp;
            let need_keyframe = t_diff.translation.vector.norm_squared()
                > TRANSLATION_THRESHOLD.powi(2)
                || t_diff.rotation.scaled_axis().norm_squared() > ROTATION_THRESHOLD.powi(2); // Rough rotation threshold based on trace

            if need_keyframe {
                println!("Adding new keyframe with pose: {:?}", t_w_cam0_pnp);
                let frame = Frame::new(t_w_cam0_pnp.cast(), cam0_observations, cam1_observations);
                self.keyframe_window.add_keyframe(frame);
            } else {
                println!("Not adding keyframe, pose change is too small");
            }
        }

        // // filter epipolar outliers and add features to current frame
        // let t_hat = hat(&self.t_cam1_cam0.translation.vector.normalize());
        // let t_hat_rmat = t_hat * self.t_cam1_cam0.rotation.to_rotation_matrix().matrix();
        // let mut max_err = 0.0_f32;

        // let good_idx: HashSet<usize> =
        // tracked_points[0]
        //     .iter()
        //     .filter_map(|(left_idx, pt_l)| {
        //         if let Some(pt_r) = tracked_points[1].get(left_idx) {
        //             // Check epipolar constraint
        //             let undistorted_pt0 = self.cam0.unproject_one(&na::Vector2::new(pt_l.0, pt_l.1));
        //             let undistorted_pt1 = self.cam1.unproject_one(&na::Vector2::new(pt_r.0, pt_r.1));

        //             // epipolar error
        //             let epipolar_error = undistorted_pt0.transpose() * t_hat_rmat * undistorted_pt1;
        //             // println!("epipolar error for match ({}, {}): {}", left_idx, right_idx, epipolar_error);
        //             max_err = max_err.max(epipolar_error[(0, 0)].abs());
        //             if epipolar_error[(0, 0)].abs() < 0.005 as f32 {
        //                 Some(*left_idx)
        //             } else {
        //                 None
        //             }
        //         } else {
        //             None
        //         }
        //     })
        //     .collect();
        // println!("good matches: {}, max epipolar error: {}", good_idx.len(), max_err);

        // for (id, &(kp_x, kp_y)) in &tracked_points[0]{
        //     if !good_idx.contains(id) {
        //         continue;
        //     }
        //     current_frame.add_left_feature(Feature::new(*id, [kp_x, kp_y]));
        // }
        // for (id, &(kp_x, kp_y)) in &tracked_points[1]{
        //     if !good_idx.contains(id) {
        //         continue;
        //     }
        //     current_frame.add_right_feature(Feature::new(*id, [kp_x, kp_y]));
        // }
        // self.tracked_points = tracked_points;
        // patch_tracking_time_ms = tracking_start.elapsed().as_secs_f64() * 1000.0;

        // // Motion tracking - only if the sliding window is full (has initialized keyframes)
        // if self.sliding_window.is_full() {
        //     // DEBUG
        //     let motion_tracking_start = Instant::now();
        //     let motion_tracking_result = self.sliding_window.track_motion(&current_frame);
        //     match motion_tracking_result {
        //         Ok(Some(T_W_Cl)) => {
        //             // Apply the optimized pose to the current frame
        //             current_frame.state.T_W_Cl = T_W_Cl;
        //             self.current_pose = T_W_Cl;

        //             // Check if translation and rotation since last keyframe is large enough to trigger a keyframe
        //             let T_W_Cl_last_kf = self
        //                 .sliding_window
        //                 .get_keyframe_poses()
        //                 .last()
        //                 .unwrap()
        //                 .clone();
        //             let T_rel = T_W_Cl * T_W_Cl_last_kf.try_inverse().unwrap();
        //             let t_rel = T_rel.fixed_view::<3, 1>(0, 3).into_owned();
        //             let R_rel = T_rel.fixed_view::<3, 3>(0, 0).into_owned();
        //             let e_rel = Vector3::from([
        //                 UnitQuaternion::from_matrix(&R_rel).euler_angles().0,
        //                 UnitQuaternion::from_matrix(&R_rel).euler_angles().1,
        //                 UnitQuaternion::from_matrix(&R_rel).euler_angles().2,
        //             ]);
        //             log::debug!(
        //                 "[Estimator] Translation since last keyframe: {:.2?}, Euler angles since last keyframe: {:.2?}",
        //                 t_rel,
        //                 e_rel
        //             );

        //             if t_rel.norm() > TRANSLATION_THRESHOLD || e_rel.norm() > ROTATION_THRESHOLD {
        //                 log::debug!(
        //                     "[Estimator] Translation and rotation since last keyframe are large enough to trigger a keyframe"
        //                 );
        //                 current_frame.is_keyframe = true;
        //             } else {
        //                 current_frame.is_keyframe = false;
        //             }
        //         }
        //         Ok(None) => {
        //             log::warn!(
        //                 "[Estimator] Motion tracking failed (optimization did not converge)"
        //             );
        //         }
        //         Err(e) => {
        //             log::error!("[Estimator] Motion tracking error: {:?}", e);
        //         }
        //     }
        //     motion_tracking_time_ms = motion_tracking_start.elapsed().as_secs_f64() * 1000.0;
        // } else {
        //     log::debug!("[Estimator] Sliding window is not full, skipping motion tracking");
        // }
        // // current_frame.is_keyframe = true; // For now, treat every frame as keyframe to test the pipeline

        // // View map points and keyframe poses
        // // Bundle adjustment
        // if current_frame.is_keyframe {
        //     let optimization_start = Instant::now();
        //     self.sliding_window.add_frame(current_frame);
        //     if let Err(e) = self.sliding_window.optimize() {
        //         log::warn!("[Estimator] Optimization skipped: {}", e);
        //     }
        //     optimization_time_ms = optimization_start.elapsed().as_secs_f64() * 1000.0;
        // }

        // // Final timing summary
        // let total_duration_ms = total_start_time.elapsed().as_secs_f64() * 1000.0;
        // log::debug!(
        //     "[Timing] frame_creation={:.3} ms, patch_tracking={:.3} ms, motion_tracking={:.3} ms, optimization={:.3} ms, total={:.3} ms",
        //     frame_creation_time_ms,
        //     patch_tracking_time_ms,
        //     motion_tracking_time_ms,
        //     optimization_time_ms,
        //     total_duration_ms
        // );

        Ok(())
    }
}
