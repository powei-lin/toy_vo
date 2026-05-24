use camera_intrinsic_model::generic_model::GenericModel;
use camera_intrinsic_model::{EUCM, FovCamera, KannalaBrandt4, OpenCVModel5, UCM};
use image::GrayImage;
use nalgebra as na;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::estimator::{EstimatorParameters, StereoEstimator};

fn build_generic_model(
    camera_matrix: &na::Matrix3<f32>,
    distortion_params: &[f32],
    model_name: &str,
    width: u32,
    height: u32,
) -> PyResult<GenericModel<f32>> {
    let fx = camera_matrix[(0, 0)];
    let fy = camera_matrix[(1, 1)];
    let cx = camera_matrix[(0, 2)];
    let cy = camera_matrix[(1, 2)];

    let model = match model_name {
        "EUCM" | "eucm" => {
            if distortion_params.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "EUCM requires 2 distortion parameters: [alpha, beta]",
                ));
            }
            GenericModel::EUCM(EUCM {
                fx,
                fy,
                cx,
                cy,
                alpha: distortion_params[0],
                beta: distortion_params[1],
                width,
                height,
            })
        }
        "UCM" | "ucm" => {
            if distortion_params.len() != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "UCM requires 1 distortion parameter: [alpha]",
                ));
            }
            GenericModel::UCM(UCM {
                fx,
                fy,
                cx,
                cy,
                alpha: distortion_params[0],
                width,
                height,
            })
        }
        "OpenCVModel5" | "opencv5" | "OPENCV5" => {
            if distortion_params.len() != 5 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "OpenCVModel5 requires 5 distortion parameters: [k1, k2, p1, p2, k3]",
                ));
            }
            GenericModel::OpenCVModel5(OpenCVModel5 {
                fx,
                fy,
                cx,
                cy,
                k1: distortion_params[0],
                k2: distortion_params[1],
                p1: distortion_params[2],
                p2: distortion_params[3],
                k3: distortion_params[4],
                width,
                height,
            })
        }
        "KannalaBrandt4" | "kb4" | "KB4" => {
            if distortion_params.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "KannalaBrandt4 requires 4 distortion parameters: [k1, k2, k3, k4]",
                ));
            }
            GenericModel::KannalaBrandt4(KannalaBrandt4 {
                fx,
                fy,
                cx,
                cy,
                k1: distortion_params[0],
                k2: distortion_params[1],
                k3: distortion_params[2],
                k4: distortion_params[3],
                width,
                height,
            })
        }
        "FovCamera" | "fov_camera" | "FOV_CAMERA" => {
            if distortion_params.len() != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "FovCamera requires 1 distortion parameter: [w]",
                ));
            }
            GenericModel::FovCamera(FovCamera {
                fx,
                fy,
                cx,
                cy,
                w: distortion_params[0],
                width,
                height,
            })
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model name: '{}'. Supported: EUCM, UCM, OpenCVModel5, KannalaBrandt4, FovCamera",
                model_name
            )));
        }
    };
    Ok(model)
}

#[pyclass(name = "StereoEstimator")]
pub struct PyStereoEstimator {
    inner: StereoEstimator,
}

#[pymethods]
impl PyStereoEstimator {
    #[new]
    #[pyo3(signature = (
        cam0_camera_matrix,
        cam0_distortion_params,
        cam0_model_name,
        cam0_width,
        cam0_height,
        cam1_camera_matrix,
        cam1_distortion_params,
        cam1_model_name,
        cam1_width,
        cam1_height,
        t_cam1_cam0_rvec,
        t_cam1_cam0_tvec,
        params=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cam0_camera_matrix: PyReadonlyArray2<f64>,
        cam0_distortion_params: PyReadonlyArray1<f64>,
        cam0_model_name: &str,
        cam0_width: u32,
        cam0_height: u32,
        cam1_camera_matrix: PyReadonlyArray2<f64>,
        cam1_distortion_params: PyReadonlyArray1<f64>,
        cam1_model_name: &str,
        cam1_width: u32,
        cam1_height: u32,
        t_cam1_cam0_rvec: PyReadonlyArray1<f64>,
        t_cam1_cam0_tvec: PyReadonlyArray1<f64>,
        params: Option<PyEstimatorParameters>,
    ) -> PyResult<Self> {
        // Parse cam0
        let cam0_mat_arr = cam0_camera_matrix.as_array();
        let cam0_mat = na::Matrix3::new(
            cam0_mat_arr[[0, 0]] as f32,
            cam0_mat_arr[[0, 1]] as f32,
            cam0_mat_arr[[0, 2]] as f32,
            cam0_mat_arr[[1, 0]] as f32,
            cam0_mat_arr[[1, 1]] as f32,
            cam0_mat_arr[[1, 2]] as f32,
            cam0_mat_arr[[2, 0]] as f32,
            cam0_mat_arr[[2, 1]] as f32,
            cam0_mat_arr[[2, 2]] as f32,
        );
        let cam0_dist: Vec<f32> = cam0_distortion_params
            .as_array()
            .iter()
            .map(|&v| v as f32)
            .collect();

        // Parse cam1
        let cam1_mat_arr = cam1_camera_matrix.as_array();
        let cam1_mat = na::Matrix3::new(
            cam1_mat_arr[[0, 0]] as f32,
            cam1_mat_arr[[0, 1]] as f32,
            cam1_mat_arr[[0, 2]] as f32,
            cam1_mat_arr[[1, 0]] as f32,
            cam1_mat_arr[[1, 1]] as f32,
            cam1_mat_arr[[1, 2]] as f32,
            cam1_mat_arr[[2, 0]] as f32,
            cam1_mat_arr[[2, 1]] as f32,
            cam1_mat_arr[[2, 2]] as f32,
        );
        let cam1_dist: Vec<f32> = cam1_distortion_params
            .as_array()
            .iter()
            .map(|&v| v as f32)
            .collect();

        let cam0_model = build_generic_model(&cam0_mat, &cam0_dist, cam0_model_name, cam0_width, cam0_height)?;
        let cam1_model = build_generic_model(&cam1_mat, &cam1_dist, cam1_model_name, cam1_width, cam1_height)?;

        // Parse extrinsics
        let rvec_arr = t_cam1_cam0_rvec.as_array();
        let tvec_arr = t_cam1_cam0_tvec.as_array();
        let t_cam1_cam0 = na::Isometry3::new(
            na::Vector3::new(tvec_arr[0] as f32, tvec_arr[1] as f32, tvec_arr[2] as f32),
            na::Vector3::new(rvec_arr[0] as f32, rvec_arr[1] as f32, rvec_arr[2] as f32),
        );

        let estimator_params = params.map(|p| p.into());

        let estimator = StereoEstimator::new(cam0_model, cam1_model, t_cam1_cam0, estimator_params);

        Ok(Self { inner: estimator })
    }

    fn process_frame<'py>(
        &mut self,
        py: Python<'py>,
        left_image: PyReadonlyArray2<u8>,
        right_image: PyReadonlyArray2<u8>,
    ) -> PyResult<Py<pyo3::types::PyDict>> {
        let left_arr = left_image.as_array();
        let right_arr = right_image.as_array();

        let (h_left, w_left) = (left_arr.shape()[0] as u32, left_arr.shape()[1] as u32);
        let (h_right, w_right) = (right_arr.shape()[0] as u32, right_arr.shape()[1] as u32);

        let left_gray = GrayImage::from_raw(w_left, h_left, left_arr.iter().copied().collect())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Failed to create left GrayImage")
            })?;
        let right_gray = GrayImage::from_raw(w_right, h_right, right_arr.iter().copied().collect())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Failed to create right GrayImage")
            })?;

        self.inner
            .process_frame(&left_gray, &right_gray)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Return a dict with current pose and status
        let dict = pyo3::types::PyDict::new(py);

        // Current pose as 4x4 matrix (row-major)
        let t_w_cam0 = self.inner.current_t_w_cam0;
        let pose_mat = t_w_cam0.to_homogeneous();
        let pose_vec: Vec<Vec<f64>> = (0..4)
            .map(|r| (0..4).map(|c| pose_mat[(r, c)] as f64).collect())
            .collect();
        let pose_array = PyArray2::from_vec2(py, &pose_vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("T_w_cam0", &pose_array)?;

        dict.set_item("new_keyframe_added", self.inner.new_keyframe_added)?;
        dict.set_item("num_landmarks", self.inner.landmarks.len())?;

        Ok(dict.unbind())
    }

    #[getter]
    fn current_pose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let t_w_cam0 = self.inner.current_t_w_cam0;
        let pose_mat = t_w_cam0.to_homogeneous();
        let pose_vec: Vec<Vec<f64>> = (0..4)
            .map(|r| (0..4).map(|c| pose_mat[(r, c)] as f64).collect())
            .collect();
        PyArray2::from_vec2(py, &pose_vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn landmarks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data: Vec<Vec<f64>> = self
            .inner
            .landmarks
            .values()
            .map(|pt| vec![pt.x as f64, pt.y as f64, pt.z as f64])
            .collect();
        if data.is_empty() {
            PyArray2::from_vec2(py, &[vec![] as Vec<f64>; 0])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        } else {
            PyArray2::from_vec2(py, &data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    #[getter]
    fn keyframe_trajectory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let traj = &self.inner.keyframe_trajectory;
        let data: Vec<Vec<f64>> = traj
            .iter()
            .map(|iso| {
                let mat = iso.to_homogeneous();
                (0..4)
                    .flat_map(|r| (0..4).map(move |c| mat[(r, c)] as f64))
                    .collect()
            })
            .collect();
        if data.is_empty() {
            PyArray2::from_vec2(py, &[vec![] as Vec<f64>; 0])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        } else {
            PyArray2::from_vec2(py, &data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    #[getter]
    fn new_keyframe_added(&self) -> bool {
        self.inner.new_keyframe_added
    }

    #[getter]
    fn num_landmarks(&self) -> usize {
        self.inner.landmarks.len()
    }

    #[getter]
    fn removed_good_landmarks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data: Vec<Vec<f64>> = self
            .inner
            .removed_good_landmarks
            .iter()
            .map(|pt| vec![pt.x as f64, pt.y as f64, pt.z as f64])
            .collect();
        if data.is_empty() {
            PyArray2::from_vec2(py, &[vec![] as Vec<f64>; 0])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        } else {
            PyArray2::from_vec2(py, &data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    #[getter]
    fn keyframe_poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data: Vec<Vec<f64>> = self
            .inner
            .keyframe_window
            .keyframes
            .iter()
            .map(|kf| {
                let mat = kf.t_cam0_w.inverse().to_homogeneous();
                (0..4)
                    .flat_map(|r| (0..4).map(move |c| mat[(r, c)] as f64))
                    .collect()
            })
            .collect();
        if data.is_empty() {
            PyArray2::from_vec2(py, &[vec![] as Vec<f64>; 0])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        } else {
            PyArray2::from_vec2(py, &data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    #[getter]
    fn landmarks_with_ids<'py>(&self, py: Python<'py>) -> PyResult<Py<pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        for (&id, pt) in &self.inner.landmarks {
            let coords = (pt.x as f64, pt.y as f64, pt.z as f64);
            dict.set_item(id, coords)?;
        }
        Ok(dict.unbind())
    }

    #[getter]
    fn track_points<'py>(&self, py: Python<'py>) -> PyResult<Py<pyo3::types::PyDict>> {
        let points = self.inner.tracker.get_track_points();
        let dict = pyo3::types::PyDict::new(py);

        for (cam_idx, cam_points) in points.iter().enumerate() {
            let cam_dict = pyo3::types::PyDict::new(py);
            for (&id, &(x, y)) in cam_points {
                let pt = (x as f64, y as f64);
                cam_dict.set_item(id, pt)?;
            }
            dict.set_item(format!("cam{}", cam_idx), cam_dict)?;
        }

        Ok(dict.unbind())
    }
}

#[derive(Clone, FromPyObject)]
pub struct PyEstimatorParameters {
    pub tracker_optical_flow_levels: u32,
    pub tracker_grid_size: u32,
    pub keyframe_window_size: usize,
    pub epipolar_error_threshold: f32,
    pub translation_threshold: f64,
    pub rotation_threshold: f64,
    pub max_frames_between_keyframes: u64,
}

impl From<PyEstimatorParameters> for EstimatorParameters {
    fn from(p: PyEstimatorParameters) -> Self {
        EstimatorParameters {
            tracker_optical_flow_levels: p.tracker_optical_flow_levels,
            tracker_grid_size: p.tracker_grid_size,
            keyframe_window_size: p.keyframe_window_size,
            epipolar_error_threshold: p.epipolar_error_threshold,
            translation_threshold: p.translation_threshold,
            rotation_threshold: p.rotation_threshold,
            max_frames_between_keyframes: p.max_frames_between_keyframes,
        }
    }
}

#[pymodule]
pub fn toy_vo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStereoEstimator>()?;
    Ok(())
}
