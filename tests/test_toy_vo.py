import numpy as np
import pytest

from toy_vo import EstimatorParameters, StereoEstimator


class TestEstimatorParameters:
    def test_default_construction(self):
        params = EstimatorParameters()
        assert params.tracker_optical_flow_levels == 5
        assert params.tracker_grid_size == 16
        assert params.keyframe_window_size == 7
        assert abs(params.epipolar_error_threshold - 0.005) < 1e-6
        assert abs(params.translation_threshold - 0.4) < 1e-6
        assert abs(params.rotation_threshold - 0.25) < 1e-6
        assert params.max_frames_between_keyframes == 10

    def test_custom_construction(self):
        params = EstimatorParameters(
            tracker_optical_flow_levels=3,
            tracker_grid_size=8,
            keyframe_window_size=5,
            epipolar_error_threshold=0.01,
            translation_threshold=0.5,
            rotation_threshold=0.3,
            max_frames_between_keyframes=20,
        )
        assert params.tracker_optical_flow_levels == 3
        assert params.tracker_grid_size == 8
        assert params.keyframe_window_size == 5
        assert abs(params.epipolar_error_threshold - 0.01) < 1e-6
        assert abs(params.translation_threshold - 0.5) < 1e-6
        assert abs(params.rotation_threshold - 0.3) < 1e-6
        assert params.max_frames_between_keyframes == 20

    def test_partial_kwargs(self):
        params = EstimatorParameters(tracker_grid_size=32)
        assert params.tracker_grid_size == 32
        # Other fields should be default
        assert params.tracker_optical_flow_levels == 5

    def test_setattr(self):
        params = EstimatorParameters()
        params.tracker_grid_size = 64
        assert params.tracker_grid_size == 64

    def test_repr(self):
        params = EstimatorParameters()
        r = repr(params)
        assert "EstimatorParameters" in r
        assert "tracker_grid_size=16" in r


class TestStereoEstimator:
    @pytest.fixture
    def stereo_estimator(self):
        """Create a StereoEstimator with pinhole cameras (no distortion)."""
        cam_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        dist_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        # Identity rotation, 0.1m baseline along x
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.1, 0.0, 0.0])
        return StereoEstimator(
            cam0_camera_matrix=cam_matrix,
            cam0_distortion_params=dist_params,
            cam0_model_name="opencv5",
            cam0_width=640,
            cam0_height=480,
            cam1_camera_matrix=cam_matrix,
            cam1_distortion_params=dist_params,
            cam1_model_name="opencv5",
            cam1_width=640,
            cam1_height=480,
            t_cam1_cam0_rvec=rvec,
            t_cam1_cam0_tvec=tvec,
        )

    def test_construction(self, stereo_estimator):
        assert stereo_estimator is not None

    def test_construction_with_params(self):
        cam_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        dist_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.1, 0.0, 0.0])
        params = EstimatorParameters(tracker_grid_size=32)
        est = StereoEstimator(
            cam0_camera_matrix=cam_matrix,
            cam0_distortion_params=dist_params,
            cam0_model_name="opencv5",
            cam0_width=640,
            cam0_height=480,
            cam1_camera_matrix=cam_matrix,
            cam1_distortion_params=dist_params,
            cam1_model_name="opencv5",
            cam1_width=640,
            cam1_height=480,
            t_cam1_cam0_rvec=rvec,
            t_cam1_cam0_tvec=tvec,
            params=params,
        )
        assert est is not None

    def test_initial_pose_is_identity(self, stereo_estimator):
        pose = stereo_estimator.current_pose
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(pose, np.eye(4), atol=1e-6)

    def test_initial_landmarks_empty(self, stereo_estimator):
        landmarks = stereo_estimator.landmarks
        assert landmarks.shape[0] == 0

    def test_initial_num_landmarks(self, stereo_estimator):
        assert stereo_estimator.num_landmarks == 0

    def test_process_frame_returns_dict(self, stereo_estimator):
        left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        result = stereo_estimator.process_frame(left, right)
        assert "T_w_cam0" in result
        assert "new_keyframe_added" in result
        assert "num_landmarks" in result
        assert result["T_w_cam0"].shape == (4, 4)

    def test_invalid_model_name_raises(self):
        cam_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        dist_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.1, 0.0, 0.0])
        with pytest.raises(ValueError, match="Unknown model name"):
            StereoEstimator(
                cam0_camera_matrix=cam_matrix,
                cam0_distortion_params=dist_params,
                cam0_model_name="invalid_model",
                cam0_width=640,
                cam0_height=480,
                cam1_camera_matrix=cam_matrix,
                cam1_distortion_params=dist_params,
                cam1_model_name="opencv5",
                cam1_width=640,
                cam1_height=480,
                t_cam1_cam0_rvec=rvec,
                t_cam1_cam0_tvec=tvec,
            )

    def test_wrong_distortion_params_raises(self):
        cam_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        # EUCM requires exactly 2 params
        dist_params = np.array([0.1, 0.2, 0.3])
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.1, 0.0, 0.0])
        with pytest.raises(ValueError, match="EUCM requires 2"):
            StereoEstimator(
                cam0_camera_matrix=cam_matrix,
                cam0_distortion_params=dist_params,
                cam0_model_name="EUCM",
                cam0_width=640,
                cam0_height=480,
                cam1_camera_matrix=cam_matrix,
                cam1_distortion_params=np.array([0.5, 1.0]),
                cam1_model_name="EUCM",
                cam1_width=640,
                cam1_height=480,
                t_cam1_cam0_rvec=rvec,
                t_cam1_cam0_tvec=tvec,
            )
