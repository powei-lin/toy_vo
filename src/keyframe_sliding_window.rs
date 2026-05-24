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

    pub fn add_keyframe(&mut self, frame: Frame) -> Option<Frame> {
        let marg_keyframe = if self.keyframes.len() == self.max_size {
            log::debug!("Sliding window is full, removing the oldest keyframe");
            self.keyframes.pop_front()
        } else {
            None
        };
        self.keyframes.push_back(frame);
        if let Some(ref marg_kf) = marg_keyframe {
            log::debug!("Marginalizing keyframe with pose: {:?}", marg_kf.t_cam0_w);
            for id in &marg_kf.new_point_ids {
                self.keyframes.iter_mut().for_each(|kf| {
                    kf.cam0_observations.remove(id);
                    kf.cam1_observations.remove(id);
                });
            }
        }
        marg_keyframe
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
