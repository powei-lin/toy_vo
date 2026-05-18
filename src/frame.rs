use std::collections::HashMap;

use nalgebra as na;

pub struct Frame {
    pub t_w_cam0: na::Isometry3<f32>,
    pub cam0_observations: HashMap<usize, (f32, f32)>,
    pub cam1_observations: HashMap<usize, (f32, f32)>,
}

impl Frame {
    pub fn new(
        t_w_cam0: na::Isometry3<f32>,
        cam0_observations: HashMap<usize, (f32, f32)>,
        cam1_observations: HashMap<usize, (f32, f32)>,
    ) -> Self {
        Self {
            t_w_cam0,
            cam0_observations,
            cam1_observations,
        }
    }
}
