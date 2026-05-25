use anyhow::Result;
use camera_intrinsic_model::generic_model::GenericModel;
use image::GrayImage;
use nalgebra as na;
use patch_tracker::StereoPatchTracker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tiny_solver::manifold::so3::SO3;

use crate::frame::Frame;
use crate::keyframe_sliding_window::KeyframeSlidingWindow;
use crate::optimization::bundle_adjustment;

/// Triangulates a 3D point from a stereo correspondence expressed in normalized camera coordinates.
pub fn triangulate_points(
    undist_pt0: &na::Vector3<f32>,
    undist_pt1: &na::Vector3<f32>,
    t_1_0: na::SMatrixView<f32, 3, 4>,
) -> na::Point3<f32> {
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

#[derive(Clone, Serialize, Deserialize)]
/// Groups the tunable parameters that control tracking, keyframe insertion, and geometric checks.
pub struct EstimatorParameters {
    pub tracker_optical_flow_levels: u32,
    pub tracker_grid_size: u32,
    pub keyframe_window_size: usize,
    pub epipolar_error_threshold: f32,
    pub translation_threshold: f64,
    pub rotation_threshold: f64,
    pub max_frames_between_keyframes: u64,
}

impl Default for EstimatorParameters {
    fn default() -> Self {
        Self {
            tracker_optical_flow_levels: 5,
            tracker_grid_size: 16,
            keyframe_window_size: 7,
            epipolar_error_threshold: 0.005,
            translation_threshold: 0.4,
            rotation_threshold: 0.25,
            max_frames_between_keyframes: 10,
        }
    }
}

/// Tracks stereo features, maintains landmarks, and estimates camera motion over time.
pub struct StereoEstimator {
    frame_id_counter: u64,
    frames_since_last_keyframe: u64,
    pub tracker: StereoPatchTracker,
    cam0: GenericModel<f32>,
    cam1: GenericModel<f32>,
    t_cam1_cam0: na::Isometry3<f32>,
    t_hat_rmat: na::Matrix3<f32>,
    t_1_0_matrix: na::SMatrix<f32, 3, 4>,
    // Full trajectory of keyframes
    pub keyframe_trajectory: Vec<na::Isometry3<f32>>,
    // Latest computed pose (T_W_Cl: left camera in world)
    pub current_t_w_cam0: na::Isometry3<f32>,
    pub keyframe_window: KeyframeSlidingWindow,
    pub landmarks: HashMap<usize, na::Point3<f32>>, // Map from feature ID to 3D position in world
    pub removed_good_landmarks: Vec<na::Point3<f32>>, // Store removed landmarks for visualization or analysis
    pub new_keyframe_added: bool, // Flag to indicate if a new keyframe was added in the current frame
    pub params: EstimatorParameters, // Store the parameters used by the estimator
}

impl StereoEstimator {
    /// Creates a stereo visual odometry estimator from calibrated cameras and an optional parameter set.
    pub fn new(
        cam0: GenericModel<f32>,
        cam1: GenericModel<f32>,
        t_cam1_cam0: na::Isometry3<f32>,
        optional_params: Option<EstimatorParameters>,
    ) -> Self {
        let params = optional_params.unwrap_or_default();
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
            tracker: StereoPatchTracker::new(
                params.tracker_optical_flow_levels,
                params.tracker_grid_size,
            ),
            cam0,
            cam1,
            t_cam1_cam0,
            t_hat_rmat,
            keyframe_trajectory: Vec::new(),
            current_t_w_cam0: na::Isometry3::identity(),
            keyframe_window: KeyframeSlidingWindow::new(params.keyframe_window_size),
            landmarks: HashMap::new(),
            removed_good_landmarks: Vec::new(),
            new_keyframe_added: false,
            t_1_0_matrix,
            params,
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
        self.new_keyframe_added = false; // Reset the flag at the start of processing each frame

        // initialize landmarks if this is the first frame
        if self.landmarks.is_empty() {
            // Initialize landmarks from the first frame's tracked points
            for (id, &(kp0_x, kp0_y)) in &tracked_points[0] {
                if let Some(&(kp1_x, kp1_y)) = tracked_points[1].get(id) {
                    log::info!("Initializing landmark {} from first frame", id);
                    let mut undistorted_pt_cam0 =
                        self.cam0.unproject_one(&na::Vector2::new(kp0_x, kp0_y));
                    let mut undistorted_pt_cam1 =
                        self.cam1.unproject_one(&na::Vector2::new(kp1_x, kp1_y));
                    if undistorted_pt_cam0.z <= 0.0 || undistorted_pt_cam1.z <= 0.0 {
                        log::warn!(
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
                    if epipolar_error[(0, 0)].abs() > self.params.epipolar_error_threshold {
                        bad_points.push(*id);
                        log::warn!(
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
                        log::warn!(
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
            self.new_keyframe_added = true;
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
                        log::warn!(
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
            log::info!(
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
                > self.params.translation_threshold.powi(2)
                || t_last_key_cam0_current_cam0
                    .rotation
                    .scaled_axis()
                    .norm_squared()
                    > self.params.rotation_threshold.powi(2)
                || self.frames_since_last_keyframe > self.params.max_frames_between_keyframes; // Rough rotation threshold based on trace

            if need_keyframe {
                // add observations for existing landmarks
                for (id, &(kp0_x, kp0_y)) in &tracked_points[0] {
                    if self.landmarks.contains_key(id) {
                        let mut undistorted_pt_cam0 =
                            self.cam0.unproject_one(&na::Vector2::new(kp0_x, kp0_y));
                        if undistorted_pt_cam0.z <= 0.0 {
                            log::warn!(
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
                            log::warn!(
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
                            log::warn!(
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
                        if epipolar_error[(0, 0)].abs() > self.params.epipolar_error_threshold {
                            bad_points.push(*id);
                            log::warn!(
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
                            log::warn!(
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
                        log::warn!(
                            "Warning: Landmark {} does not have a match in the right image, skipping",
                            id
                        );
                        bad_points.push(*id);
                    }
                }

                log::info!("Adding new keyframe with pose: {:?}", self.current_t_w_cam0);
                let frame = Frame::new(
                    t_cam0_w_pnp_f32,
                    cam0_observations,
                    cam1_observations,
                    new_point_ids,
                );
                let marg_keyframe_pose_and_ids = self.keyframe_window.add_keyframe(frame);
                self.new_keyframe_added = true;
                self.frames_since_last_keyframe = 0;

                if let Some((marg_keyframe_pose, marg_keyframe_ids)) = marg_keyframe_pose_and_ids {
                    log::info!(
                        "Marginalizing out keyframe with pose: {:?}",
                        marg_keyframe_pose
                    );
                    self.removed_good_landmarks.clear();
                    for id in &marg_keyframe_ids {
                        if let Some(landmark) = self.landmarks.remove(id) {
                            self.removed_good_landmarks.push(landmark);
                        }
                        bad_points.push(*id);
                    }
                    self.keyframe_trajectory.push(marg_keyframe_pose);
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
                log::info!("Not adding keyframe, pose change is too small");
            }

            // For subsequent frames, we can attempt to triangulate new landmarks from newly tracked points
            self.tracker.remove_id(&bad_points);
        }

        self.frame_id_counter += 1;
        self.frames_since_last_keyframe += 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator_parameters_default() {
        let params = EstimatorParameters::default();
        assert_eq!(params.tracker_optical_flow_levels, 5);
        assert_eq!(params.tracker_grid_size, 16);
        assert_eq!(params.keyframe_window_size, 7);
        assert!((params.epipolar_error_threshold - 0.005).abs() < 1e-6);
        assert!((params.translation_threshold - 0.4).abs() < 1e-6);
        assert!((params.rotation_threshold - 0.25).abs() < 1e-6);
        assert_eq!(params.max_frames_between_keyframes, 10);
    }

    #[test]
    fn test_triangulate_points_identity_extrinsics() {
        // Camera 1 is shifted 1m to the right of camera 0
        let t_1_0 = na::SMatrix::<f32, 3, 4>::new(
            1.0, 0.0, 0.0, 1.0, // rotation = identity, tx = 1.0
            0.0, 1.0, 0.0, 0.0, // ty = 0
            0.0, 0.0, 1.0, 0.0, // tz = 0
        );
        // Point at (0, 0, 10) in cam0 frame -> normalized (0, 0, 1)
        // In cam1 frame the point is at (-1, 0, 10) -> normalized (-0.1, 0, 1)
        let pt0 = na::Vector3::new(0.0, 0.0, 1.0);
        let pt1 = na::Vector3::new(-0.1, 0.0, 1.0);

        let result = triangulate_points(&pt0, &pt1, t_1_0.as_view());
        // SVD may return negated result; check absolute depth and ratios
        let depth = result.z.abs();
        assert!((depth - 10.0).abs() < 0.1, "depth={}", depth);
        assert!((result.x / result.z).abs() < 0.01, "x/z={}", result.x / result.z);
        assert!((result.y / result.z).abs() < 0.01, "y/z={}", result.y / result.z);
    }

    #[test]
    fn test_triangulate_points_known_geometry() {
        // Baseline of 0.5m along x
        let t_1_0 = na::SMatrix::<f32, 3, 4>::new(
            1.0, 0.0, 0.0, 0.5, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
        );
        // Point at (1, 2, 5) in cam0
        let depth = 5.0_f32;
        let pt0 = na::Vector3::new(1.0 / depth, 2.0 / depth, 1.0);
        // In cam1: (1 - 0.5, 2, 5) -> normalized
        let pt1 = na::Vector3::new(0.5 / depth, 2.0 / depth, 1.0);

        let result = triangulate_points(&pt0, &pt1, t_1_0.as_view());
        // Check ratios (sign may be flipped by SVD)
        let sign = result.z.signum();
        assert!((sign * result.x - 1.0).abs() < 0.01, "x={}", sign * result.x);
        assert!((sign * result.y - 2.0).abs() < 0.01, "y={}", sign * result.y);
        assert!((sign * result.z - 5.0).abs() < 0.01, "z={}", sign * result.z);
    }
}
