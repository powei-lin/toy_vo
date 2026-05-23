use crate::frame::Frame;
use nalgebra as na;
use std::collections::VecDeque;

pub struct KeyframeSlidingWindow {
    pub max_size: usize,
    pub keyframes: VecDeque<Frame>,
}

impl KeyframeSlidingWindow {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            keyframes: VecDeque::with_capacity(max_size),
        }
    }

    pub fn add_keyframe(&mut self, frame: Frame) {
        if self.keyframes.len() == self.max_size {
            log::debug!("Sliding window is full, removing the oldest keyframe");
            println!("!!!!!!!!!!!!!!!!!! Sliding window is full, removing the oldest keyframe");
            self.keyframes.pop_front();
        }
        self.keyframes.push_back(frame);
    }

    pub fn is_full(&self) -> bool {
        self.keyframes.len() == self.max_size
    }

    pub fn last_keyframe_t_cam0_w(&self) -> na::Isometry3<f32> {
        if self.keyframes.is_empty() {
            na::Isometry3::identity()
        } else {
            self.keyframes.back().unwrap().t_cam0_w
        }
    }
}
