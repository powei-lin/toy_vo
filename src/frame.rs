use std::collections::HashMap;

use nalgebra as na;

pub struct Frame {
    pub t_cam0_w: na::Isometry3<f32>,
    /// undistorted pixel coordinates of observed features, keyed by feature ID
    pub cam0_observations: HashMap<usize, (f32, f32)>,
    /// undistorted pixel coordinates of observed features in the right camera, keyed by feature ID
    pub cam1_observations: HashMap<usize, (f32, f32)>,
}

impl Frame {
    pub fn new(
        t_cam0_w: na::Isometry3<f32>,
        cam0_observations: HashMap<usize, (f32, f32)>,
        cam1_observations: HashMap<usize, (f32, f32)>,
    ) -> Self {
        Self {
            t_cam0_w,
            cam0_observations,
            cam1_observations,
        }
    }
}
