use crate::keyframe_sliding_window::KeyframeSlidingWindow;
use nalgebra as na;
use std::collections::HashMap;
use std::sync::Arc;
use tiny_solver::factors::Factor;
use tiny_solver::loss_functions::HuberLoss;
use tiny_solver::manifold::se3::{SE3, SE3Manifold};
use tiny_solver::sparse::LinearSolverType;
use tiny_solver::{Optimizer, OptimizerOptions, Problem};

const HUBER_LOSS_DELTA: f64 = 0.001;

/// Trait for converting vector types to nalgebra DVector.
pub trait Vec3DVec<T: Clone> {
    fn to_dvec(&self) -> na::DVector<T>;
}
/// Trait for converting DVector to vector types.
pub trait DVecVec3<T: Clone> {
    fn to_vec3(&self) -> nalgebra::Vector3<T>;
}
impl<T: Clone> DVecVec3<T> for na::DVector<T> {
    fn to_vec3(&self) -> nalgebra::Vector3<T> {
        na::Vector3::new(self[0].clone(), self[1].clone(), self[2].clone())
    }
}
impl<T: Clone> Vec3DVec<T> for na::Vector3<T> {
    fn to_dvec(&self) -> na::DVector<T> {
        na::dvector![self[0].clone(), self[1].clone(), self[2].clone()]
    }
}
#[derive(Clone)]
pub struct ReprojectionFactorCam0 {
    pub observed_normalized_pt: (f32, f32),
}

impl<T: na::RealField> Factor<T> for ReprojectionFactorCam0 {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        let t_cam0_w = na::Isometry3::new(params[0].to_vec3(), params[1].to_vec3());
        let point3d = na::Point3::new(
            params[2][0].clone(),
            params[2][1].clone(),
            params[2][2].clone(),
        );
        let landmark_in_cam0 = t_cam0_w * point3d;
        na::dvector![
            landmark_in_cam0.x.clone() / landmark_in_cam0.z.clone()
                - T::from_f32(self.observed_normalized_pt.0).unwrap(),
            landmark_in_cam0.y.clone() / landmark_in_cam0.z.clone()
                - T::from_f32(self.observed_normalized_pt.1).unwrap()
        ]
    }
}

#[derive(Clone)]
pub struct ReprojectionFactorCam1 {
    pub observed_normalized_pt: (f32, f32),
    pub t_cam1_cam0: na::Isometry3<f32>,
}

impl<T: na::RealField> Factor<T> for ReprojectionFactorCam1 {
    fn residual_func(&self, params: &[nalgebra::DVector<T>]) -> nalgebra::DVector<T> {
        let t_cam0_w = na::Isometry3::new(params[0].to_vec3(), params[1].to_vec3());
        let point3d = na::Point3::new(
            params[2][0].clone(),
            params[2][1].clone(),
            params[2][2].clone(),
        );
        let landmark_in_cam0 = t_cam0_w * point3d;
        let landmark_in_cam1 = self.t_cam1_cam0.cast() * landmark_in_cam0;
        na::dvector![
            landmark_in_cam1.x.clone() / landmark_in_cam1.z.clone()
                - T::from_f32(self.observed_normalized_pt.0).unwrap(),
            landmark_in_cam1.y.clone() / landmark_in_cam1.z.clone()
                - T::from_f32(self.observed_normalized_pt.1).unwrap()
        ]
    }
}

pub fn bundle_adjustment(
    sliding_window: &mut KeyframeSlidingWindow,
    landmarks: &mut HashMap<usize, na::Point3<f32>>,
    t_cam1_cam0: &na::Isometry3<f32>,
) -> Vec<usize> {
    let mut bad_landmarks = Vec::new();
    println!(
        "Running bundle adjustment on {} keyframes and {} landmarks",
        sliding_window.keyframes.len(),
        landmarks.len()
    );
    let mut count_observations: HashMap<usize, u16> = HashMap::new();
    for frame in sliding_window.keyframes.iter() {
        for id in frame.cam0_observations.keys() {
            *count_observations.entry(*id).or_insert(0) += 1;
        }
    }
    // println!("Observation counts for landmarks: {:?}", count_observations);
    // Placeholder for bundle adjustment implementation
    let mut problem = tiny_solver::Problem::new();
    let mut initial_values = HashMap::<String, na::DVector<f64>>::new();

    let mut count_added_variables = HashMap::<String, usize>::new();

    for (i, frame) in sliding_window.keyframes.iter().enumerate() {
        let tvec_name = format!("tvec_cam0_w{}", i);
        let rvec_name = format!("rvec_cam0_w{}", i);
        println!(
            "Processing keyframe {} with {} observations",
            i,
            frame.cam0_observations.len() + frame.cam1_observations.len()
        );
        let tvec_cam0_w = frame.t_cam0_w.translation.vector.to_dvec();
        let rvec_cam0_w = frame.t_cam0_w.rotation.scaled_axis().to_dvec();
        frame
            .cam0_observations
            .iter()
            .for_each(|(l_id, observed_pt)| {
                if count_observations.get(l_id).unwrap() > &1 {
                    if let Some(landmark_pos) = landmarks.get(l_id) {
                        // Create a reprojection factor for this observation
                        let factor = ReprojectionFactorCam0 {
                            observed_normalized_pt: *observed_pt,
                        };
                        let landmark_name = format!("l{}", l_id);
                        // Add the factor to the problem, with parameters for the camera pose and landmark position
                        problem.add_residual_block(
                            2, // 2 residuals for x and y reprojection error
                            &[
                                tvec_name.as_str(),
                                rvec_name.as_str(),
                                landmark_name.as_str(),
                            ], // parameter blocks for camera pose and landmark position
                            Box::new(factor),
                            Some(Box::new(HuberLoss::new(HUBER_LOSS_DELTA))), // no loss function for now
                        );
                        count_added_variables.insert(
                            landmark_name.clone(),
                            count_added_variables.get(&landmark_name).unwrap_or(&0) + 1,
                        );
                        initial_values.insert(landmark_name, landmark_pos.coords.cast().to_dvec());
                    }
                }
            });
        frame
            .cam1_observations
            .iter()
            .for_each(|(l_id, observed_pt)| {
                if count_observations.get(l_id).unwrap() > &1 {
                    if let Some(landmark_pos) = landmarks.get(l_id) {
                        // Create a reprojection factor for this observation
                        let factor = ReprojectionFactorCam1 {
                            observed_normalized_pt: *observed_pt,
                            t_cam1_cam0: *t_cam1_cam0,
                        };
                        let landmark_name = format!("l{}", l_id);
                        // Add the factor to the problem, with parameters for the camera pose and landmark position
                        problem.add_residual_block(
                            2, // 2 residuals for x and y reprojection error
                            &[
                                tvec_name.as_str(),
                                rvec_name.as_str(),
                                landmark_name.as_str(),
                            ], // parameter blocks for camera pose and landmark position
                            Box::new(factor),
                            Some(Box::new(HuberLoss::new(HUBER_LOSS_DELTA))), // no loss function for now
                        );
                        initial_values.insert(landmark_name, landmark_pos.coords.cast().to_dvec());
                    }
                }
            });
        // problem.set_variable_manifold(qtvec_name.as_str(), Arc::new(SE3Manifold));
        initial_values.insert(tvec_name, tvec_cam0_w.cast::<f64>());
        initial_values.insert(rvec_name, rvec_cam0_w.cast::<f64>());
    }
    // for (name, count) in count_added_variables.iter() {
    //     println!("Variable {} is involved in {} residuals", name, count);
    // }
    // panic!("stop here for now");

    // println!("initial values: {:?}", initial_values);
    for i in 0..3 {
        problem.fix_variable("tvec_cam0_w0", i);
        problem.fix_variable("rvec_cam0_w0", i);
    }
    // let res = problem.compute_residuals(&problem.initialize_parameter_blocks(&initial_values), false);
    // println!("Initial residuals: {:?}", res);
    // panic!("stop here for now");

    // initialize optimizer
    let optimizer = tiny_solver::LevenbergMarquardtOptimizer::new(1e-6, 1e-2, 1e2);
    // let optimizer = tiny_solver::GaussNewtonOptimizer::new();
    // println!(
    //     "starting pose tvec_cam0_w1: {:?}",
    //     initial_values["tvec_cam0_w1"]
    // );
    // // optimize
    let result = optimizer
        .optimize(
            &problem,
            &initial_values,
            Some(OptimizerOptions {
                linear_solver_type: LinearSolverType::SparseCholesky,
                verbosity_level: 10,
                max_iteration: 50,
                ..Default::default()
            }), // use default optimizer options
        )
        .unwrap();

    // println!("optimized pose tvec_cam0_w1: {:?}", result["tvec_cam0_w1"]);
    sliding_window
        .keyframes
        .iter_mut()
        .enumerate()
        .for_each(|(i, frame)| {
            if i == 0 {
                return; // skip the first keyframe since it's fixed
            }
            let tvec_name = format!("tvec_cam0_w{}", i);
            let rvec_name = format!("rvec_cam0_w{}", i);
            if let (Some(tvec_opt), Some(rvec_opt)) =
                (result.get(&tvec_name), result.get(&rvec_name))
            {
                let tvec = tvec_opt.to_vec3();
                let rvec = rvec_opt.to_vec3();
                frame.t_cam0_w = na::Isometry3::new(tvec, rvec).cast();
            }
        });
    landmarks.iter_mut().for_each(|(l_id, landmark_pos)| {
        let landmark_name = format!("l{}", l_id);
        if let Some(landmark_opt) = result.get(&landmark_name) {
            let landmark_vec = landmark_opt.to_vec3().cast::<f32>();
            *landmark_pos = na::Point3::new(landmark_vec.x, landmark_vec.y, landmark_vec.z);
        }
    });
    let mut error_w_id: Vec<(f32, usize)> = count_added_variables
        .iter()
        .filter_map(|(landmark_name, count)| {
            // reprojection error
            if let Some(landmark_opt) = result.get(landmark_name) {
                let landmark_vec = landmark_opt.to_vec3().cast::<f32>();
                let landmark_pos = na::Point3::new(landmark_vec.x, landmark_vec.y, landmark_vec.z);
                let l_id: usize = landmark_name[1..].parse().unwrap();
                let mut total_error = 0.0;
                sliding_window.keyframes.iter().for_each(|frame| {
                    if let Some(observed_pt) = frame.cam0_observations.get(&l_id) {
                        let projected_pt = frame.t_cam0_w * landmark_pos;
                        let reprojection_error =
                            ((projected_pt.x / projected_pt.z - observed_pt.0).powi(2)
                                + (projected_pt.y / projected_pt.z - observed_pt.1).powi(2))
                            .sqrt();
                        total_error += reprojection_error;
                    }
                    if let Some(observed_pt) = frame.cam1_observations.get(&l_id) {
                        let projected_pt_cam1 = *t_cam1_cam0 * (frame.t_cam0_w * landmark_pos);
                        let reprojection_error_cam1 = ((projected_pt_cam1.x / projected_pt_cam1.z
                            - observed_pt.0)
                            .powi(2)
                            + (projected_pt_cam1.y / projected_pt_cam1.z - observed_pt.1).powi(2))
                        .sqrt();
                        total_error += reprojection_error_cam1;
                    }
                });
                let avg_error = total_error / (*count as f32);
                Some((avg_error, l_id))
            } else {
                None
            }
        })
        .collect();
    error_w_id.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (error, id) in error_w_id.iter() {
        println!("Landmark {} has average reprojection error {}", id, error);
        if *error > 0.01 {
            bad_landmarks.push(*id);
        }
    }
    // println!("Landmark reprojection errors (sorted): {:?}", error_w_id);

    bad_landmarks
}
