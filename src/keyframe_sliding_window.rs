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

    pub fn add_keyframe(&mut self, frame: Frame) -> Option<(na::Isometry3<f32>, Vec<usize>)> {
        let marg_keyframe = if self.keyframes.len() == self.max_size {
            log::debug!("Sliding window is full, removing the oldest keyframe");
            self.keyframes.pop_front()
        } else {
            None
        };
        self.keyframes.push_back(frame);
        if let Some(marg_kf) = marg_keyframe {
            log::debug!("Marginalizing keyframe with pose: {:?}", marg_kf.t_cam0_w);
            let (mut kept, removed): (Vec<_>, Vec<_>) = marg_kf.new_point_ids.iter().cloned().partition(|id|{
                if self.keyframes[0].cam0_observations.contains_key(id) && self.keyframes[0].cam1_observations.contains_key(id) {
                    log::debug!("Landmark {} is still observed in the oldest keyframe, keeping it in the sliding window", id);
                    true
                } else {
                    log::debug!("Landmark {} is not observed in the oldest keyframe, removing it from the sliding window", id);
                    false
                }
            });
            for id in &removed {
                self.keyframes.iter_mut().for_each(|kf| {
                    kf.cam0_observations.remove(id);
                    kf.cam1_observations.remove(id);
                });
            }
            self.keyframes[0].new_point_ids.append(&mut kept);
            Some((marg_kf.t_cam0_w, removed))
        } else {
            None
        }
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
