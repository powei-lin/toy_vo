use crate::frame::Frame;
use nalgebra as na;
use std::collections::VecDeque;

/// Maintains a fixed-size queue of keyframes for local bundle adjustment.
pub struct KeyframeSlidingWindow {
    pub max_size: usize,
    pub keyframes: VecDeque<Frame>,
}

impl KeyframeSlidingWindow {
    /// Creates an empty sliding window with the requested maximum number of keyframes.
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            keyframes: VecDeque::with_capacity(max_size),
        }
    }

    /// Adds a keyframe and, when the window is full, returns the marginalized pose and removed landmark IDs.
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

    /// Returns whether the sliding window has reached its configured capacity.
    pub fn is_full(&self) -> bool {
        self.keyframes.len() == self.max_size
    }

    /// Returns the pose of the newest keyframe, or identity if the window is empty.
    pub fn last_keyframe_t_cam0_w(&self) -> na::Isometry3<f32> {
        if self.keyframes.is_empty() {
            na::Isometry3::identity()
        } else {
            self.keyframes.back().unwrap().t_cam0_w
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_frame(new_point_ids: Vec<usize>) -> Frame {
        let mut cam0_obs = HashMap::new();
        let mut cam1_obs = HashMap::new();
        for &id in &new_point_ids {
            cam0_obs.insert(id, (id as f32 * 0.1, id as f32 * 0.2));
            cam1_obs.insert(id, (id as f32 * 0.1 - 0.01, id as f32 * 0.2));
        }
        Frame::new(na::Isometry3::identity(), cam0_obs, cam1_obs, new_point_ids)
    }

    #[test]
    fn test_new_sliding_window() {
        let sw = KeyframeSlidingWindow::new(5);
        assert_eq!(sw.max_size, 5);
        assert!(sw.keyframes.is_empty());
        assert!(!sw.is_full());
    }

    #[test]
    fn test_add_keyframe_below_capacity() {
        let mut sw = KeyframeSlidingWindow::new(3);
        let result = sw.add_keyframe(make_frame(vec![1, 2, 3]));
        assert!(result.is_none());
        assert_eq!(sw.keyframes.len(), 1);
        assert!(!sw.is_full());
    }

    #[test]
    fn test_add_keyframe_at_capacity_marginalizes() {
        let mut sw = KeyframeSlidingWindow::new(2);
        // First keyframe with landmarks 1,2,3
        sw.add_keyframe(make_frame(vec![1, 2, 3]));
        // Second keyframe also observes landmark 1,2,3
        sw.add_keyframe(make_frame(vec![4, 5]));
        assert!(sw.is_full());
        // Third keyframe should marginalize the first
        // Only landmarks also seen in the new oldest frame are kept
        let result = sw.add_keyframe(make_frame(vec![6]));
        assert!(result.is_some());
        let (_pose, removed_ids) = result.unwrap();
        // Landmarks 1,2,3 were owned by the first frame; check which were removed
        assert!(!removed_ids.is_empty());
    }

    #[test]
    fn test_last_keyframe_t_cam0_w_empty() {
        let sw = KeyframeSlidingWindow::new(3);
        let pose = sw.last_keyframe_t_cam0_w();
        assert_eq!(pose, na::Isometry3::identity());
    }

    #[test]
    fn test_last_keyframe_t_cam0_w_with_frames() {
        let mut sw = KeyframeSlidingWindow::new(3);
        let mut frame = make_frame(vec![1]);
        let expected_pose = na::Isometry3::new(
            na::Vector3::new(1.0, 2.0, 3.0),
            na::Vector3::new(0.1, 0.2, 0.3),
        );
        frame.t_cam0_w = expected_pose;
        sw.add_keyframe(frame);
        let pose = sw.last_keyframe_t_cam0_w();
        assert!((pose.translation.vector - expected_pose.translation.vector).norm() < 1e-6);
    }
}
