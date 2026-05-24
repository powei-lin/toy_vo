from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray


class EstimatorParameters(TypedDict):
    tracker_optical_flow_levels: int
    tracker_grid_size: int
    keyframe_window_size: int
    epipolar_error_threshold: float
    translation_threshold: float
    rotation_threshold: float
    max_frames_between_keyframes: int


class ProcessFrameResult(TypedDict):
    T_w_cam0: NDArray[np.float64]
    new_keyframe_added: bool
    num_landmarks: int


class StereoEstimator:
    def __init__(
        self,
        cam0_camera_matrix: NDArray[np.float64],
        cam0_distortion_params: NDArray[np.float64],
        cam0_model_name: str,
        cam0_width: int,
        cam0_height: int,
        cam1_camera_matrix: NDArray[np.float64],
        cam1_distortion_params: NDArray[np.float64],
        cam1_model_name: str,
        cam1_width: int,
        cam1_height: int,
        t_cam1_cam0_rvec: NDArray[np.float64],
        t_cam1_cam0_tvec: NDArray[np.float64],
        params: Optional[EstimatorParameters] = None,
    ) -> None:
        """Create a stereo visual odometry estimator.

        Args:
            cam0_camera_matrix: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            cam0_distortion_params: Distortion parameters (model-dependent)
            cam0_model_name: Camera model name. Supported: "EUCM", "UCM", "OpenCVModel5", "KannalaBrandt4", "FovCamera"
            cam0_width: Image width for camera 0
            cam0_height: Image height for camera 0
            cam1_camera_matrix: 3x3 intrinsic matrix for camera 1
            cam1_distortion_params: Distortion parameters for camera 1
            cam1_model_name: Camera model name for camera 1
            cam1_width: Image width for camera 1
            cam1_height: Image height for camera 1
            t_cam1_cam0_rvec: Rotation vector (3,) of T_cam1_cam0 extrinsics (axis-angle)
            t_cam1_cam0_tvec: Translation vector (3,) of T_cam1_cam0 extrinsics
            params: Optional estimator parameters
        """
        ...

    def process_frame(
        self,
        left_image: NDArray[np.uint8],
        right_image: NDArray[np.uint8],
    ) -> ProcessFrameResult:
        """Process a stereo image pair.

        Args:
            left_image: Grayscale left image as HxW uint8 numpy array
            right_image: Grayscale right image as HxW uint8 numpy array

        Returns:
            Dict with keys: "T_w_cam0" (4x4 pose), "new_keyframe_added" (bool), "num_landmarks" (int)
        """
        ...

    @property
    def current_pose(self) -> NDArray[np.float64]:
        """Current camera pose T_w_cam0 as a 4x4 homogeneous matrix."""
        ...

    @property
    def landmarks(self) -> NDArray[np.float64]:
        """Active 3D landmarks as Nx3 array."""
        ...

    @property
    def landmarks_with_ids(self) -> dict[int, tuple[float, float, float]]:
        """Active 3D landmarks as dict mapping feature_id -> (x, y, z)."""
        ...

    @property
    def keyframe_trajectory(self) -> NDArray[np.float64]:
        """Keyframe trajectory as Nx16 array (each row is a flattened 4x4 pose matrix, row-major)."""
        ...

    @property
    def keyframe_poses(self) -> NDArray[np.float64]:
        """Current keyframe window poses (T_w_cam0) as Nx16 array (each row is a flattened 4x4 matrix, row-major)."""
        ...

    @property
    def removed_good_landmarks(self) -> NDArray[np.float64]:
        """Recently removed good landmarks as Nx3 array."""
        ...

    @property
    def new_keyframe_added(self) -> bool:
        """Whether a new keyframe was added in the last process_frame call."""
        ...

    @property
    def num_landmarks(self) -> int:
        """Number of active landmarks."""
        ...

    @property
    def track_points(self) -> dict[str, dict[int, tuple[float, float]]]:
        """Tracked 2D points per camera. Returns {"cam0": {id: (x, y), ...}, "cam1": {id: (x, y), ...}}."""
        ...
