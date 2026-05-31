"""
Stereo Visual Odometry example using toy_vo Python bindings.
Equivalent to examples/run_stereo.rs with full rerun visualization.
"""

import json
import glob
import time
from pathlib import Path
import argparse

import numpy as np
from numpy.typing import NDArray
import cv2
import rerun as rr
from toy_vo import StereoEstimator, EstimatorParameters


def id_to_color(feature_id: int) -> list[int]:
    """Generate a deterministic color from a feature ID."""
    import hashlib

    h = hashlib.md5(feature_id.to_bytes(8, "little")).digest()
    return [h[0], h[1], h[2]]


def load_camera_config(config_path: str):
    """Load camera model from JSON config file.

    Returns camera_matrix (3x3), distortion_params, model_name, width, height.
    """
    with open(config_path, "r") as f:
        data = json.load(f)

    model_name = list(data.keys())[0]
    params = data[model_name]

    fx = params["fx"]
    fy = params["fy"]
    cx = params["cx"]
    cy = params["cy"]
    width = params["width"]
    height = params["height"]

    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    if model_name in ("EUCM", "eucm"):
        distortion_params = np.array([params["alpha"], params["beta"]])
    elif model_name in ("UCM", "ucm"):
        distortion_params = np.array([params["alpha"]])
    elif model_name in ("OpenCVModel5", "opencv5", "OPENCV5"):
        distortion_params = np.array(
            [params["k1"], params["k2"], params["p1"], params["p2"], params["k3"]]
        )
    elif model_name in ("KannalaBrandt4", "kb4", "KB4"):
        distortion_params = np.array(
            [params["k1"], params["k2"], params["k3"], params["k4"]]
        )
    elif model_name in ("FovCamera", "fov_camera", "FOV_CAMERA"):
        distortion_params = np.array([params["w"]])
    else:
        raise ValueError(f"Unknown camera model: {model_name}")

    return camera_matrix, distortion_params, model_name, width, height


def load_extrinsics(extrinsics_path: str):
    """Load stereo extrinsics (T_cam1_cam0) from JSON.

    Returns rvec (3,) and tvec (3,) numpy arrays.
    """
    with open(extrinsics_path, "r") as f:
        data = json.load(f)

    rtvec = data["rtvecs"][1]
    rvec = np.array(rtvec["rvec"])
    tvec = np.array(rtvec["tvec"])
    return rvec, tvec


def load_camera_images(root: str, cam_idx: int) -> list[str]:
    """Load sorted list of image paths for a given camera."""
    pattern = str(Path(root) / f"mav0/cam{cam_idx}/data/*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pattern = str(Path(root) / f"mav0/cam{cam_idx}/data/*.jpg")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise ValueError(f"No images found for camera {cam_idx} in {root}")
    return paths


def rvec_tvec_to_4x4(rvec: NDArray, tvec: NDArray) -> NDArray[np.float64]:
    """Convert axis-angle rotation vector and translation to 4x4 homogeneous matrix."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_rotvec(rvec)
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = tvec
    return mat


def pose_to_rerun_transform(pose_4x4: NDArray) -> "rr.Transform3D":
    """Convert a 4x4 pose matrix to a rerun Transform3D."""
    from scipy.spatial.transform import Rotation

    rot_mat = pose_4x4[:3, :3]
    quat = Rotation.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
    translation = pose_4x4[:3, 3]

    return rr.Transform3D.from_translation_rotation(
        translation=translation.tolist(),
        rotation=rr.Quaternion(xyzw=quat.tolist()),
    )


def log_rerun_image(image: NDArray, entity_path: str):
    """Log a grayscale image to rerun as encoded JPEG."""
    _, buf = cv2.imencode(
        ".jpg", image
    )  # Ensure OpenCV is used to encode (for consistency)
    rr.log(
        entity_path, rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg")
    )


def log_image_keypoints(
    curr_keypoints: dict[int, tuple[float, float]],
    prev_keypoints: dict[int, tuple[float, float]],
    image_entity_path: str,
):
    """Log 2D keypoints and optical flow lines on an image."""

    # Points with per-ID colors
    points = []
    colors = []
    for fid, (x, y) in curr_keypoints.items():
        points.append([x + 0.5, y + 0.5])
        colors.append(id_to_color(fid))

    if points:
        rr.log(
            f"{image_entity_path}/points",
            rr.Points2D(points, colors=colors),
        )

    # Lines from previous to current keypoints
    line_strips = []
    line_colors = []
    for fid, (curr_x, curr_y) in curr_keypoints.items():
        if fid in prev_keypoints:
            prev_x, prev_y = prev_keypoints[fid]
            line_strips.append(
                [[prev_x + 0.5, prev_y + 0.5], [curr_x + 0.5, curr_y + 0.5]]
            )
            line_colors.append(id_to_color(fid))

    if line_strips:
        rr.log(
            f"{image_entity_path}/lines",
            rr.LineStrips2D(line_strips, colors=line_colors),
        )


def log_pose(
    pose_4x4: NDArray,
    entity_path: str,
    size: float,
    camera_frustum=None,
):
    """Log a 3D pose (transform + axes or camera frustum)."""

    rr.log(
        entity_path,
        rr.Transform3D(translation=pose_4x4[:3, 3], mat3x3=pose_4x4[:3, :3]),
    )
    if camera_frustum is not None:
        rr.log(entity_path, camera_frustum)
    else:
        rr.log(entity_path, rr.TransformAxes3D(axis_length=size))


def log_active_landmarks(
    landmarks_with_ids: dict[int, tuple[float, float, float]],
    current_pose_4x4: NDArray,
    entity_path: str,
    range_threshold: float,
):
    """Log active 3D landmarks, filtered by distance from current camera."""

    cam_position = current_pose_4x4[:3, 3]
    range_threshold2 = range_threshold * range_threshold

    points = []
    colors = []
    for fid, (x, y, z) in landmarks_with_ids.items():
        pt = np.array([x, y, z])
        dist2 = float(np.sum((pt - cam_position) ** 2))
        if dist2 < range_threshold2:
            points.append([x, y, z])
            colors.append(id_to_color(fid))

    if points:
        rr.log(entity_path, rr.Points3D(points, colors=colors))


def log_old_landmarks(
    landmarks: NDArray,
    entity_path: str,
):
    """Log removed landmarks as 3D points."""

    if landmarks.shape[0] == 0:
        return
    rr.log(entity_path, rr.Points3D(landmarks.tolist()))


def log_trajectory(
    trajectory: NDArray,
    entity_path: str,
):
    """Log the keyframe trajectory as a 3D line strip.

    Trajectory rows are flattened 4x4 T_cam0_w matrices.
    We invert each to get world position.
    """

    if trajectory.shape[0] == 0:
        return

    points = []
    for i in range(trajectory.shape[0]):
        t_cam0_w = trajectory[i].reshape(4, 4)
        # Invert to get T_w_cam0, then extract translation
        t_w_cam0 = np.linalg.inv(t_cam0_w)
        points.append(t_w_cam0[:3, 3].tolist())

    if points:
        rr.log(entity_path, rr.LineStrips3D([points]))


def make_camera_frustum(camera_matrix: NDArray, width: int, height: int):
    """Create a rerun Pinhole from camera intrinsics."""
    import rerun as rr

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    return rr.Pinhole(
        focal_length=[fx, fy],
        resolution=[width, height],
        principal_point=[cx, cy],
        image_plane_distance=0.1,
    )


def main():

    parser = argparse.ArgumentParser(description="Stereo Visual Odometry")
    parser.add_argument(
        "-d", "--dataset-folder", required=True, help="Path to dataset folder"
    )
    parser.add_argument(
        "-c", "--config-folder", required=True, help="Path to config folder"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Enable rerun visualization"
    )
    args = parser.parse_args()

    # Load camera configurations
    cam0_matrix, cam0_dist, cam0_model, cam0_w, cam0_h = load_camera_config(
        f"{args.config_folder}/cam0.json"
    )
    cam1_matrix, cam1_dist, cam1_model, cam1_w, cam1_h = load_camera_config(
        f"{args.config_folder}/cam1.json"
    )

    # Load extrinsics
    rvec, tvec = load_extrinsics(f"{args.config_folder}/extrinsics.json")
    t_cam0_cam1_4x4 = np.linalg.inv(rvec_tvec_to_4x4(rvec, tvec))

    # Create estimator
    estimator = StereoEstimator(
        cam0_camera_matrix=cam0_matrix,
        cam0_distortion_params=cam0_dist,
        cam0_model_name=cam0_model,
        cam0_width=cam0_w,
        cam0_height=cam0_h,
        cam1_camera_matrix=cam1_matrix,
        cam1_distortion_params=cam1_dist,
        cam1_model_name=cam1_model,
        cam1_width=cam1_w,
        cam1_height=cam1_h,
        t_cam1_cam0_rvec=rvec,
        t_cam1_cam0_tvec=tvec,
        params=EstimatorParameters(
            tracker_optical_flow_levels=8,
            tracker_grid_size=30,
            translation_threshold=2.0,
        ),
    )

    # Setup rerun
    camera_frustum = None
    if args.rerun:
        import rerun as rr

        rr.init("vo")
        rr.spawn(port=9875)
        camera_frustum = make_camera_frustum(cam0_matrix, cam0_w, cam0_h)
        rr.log("/", rr.ViewCoordinates.RDF, static=True)

    # Load images
    left_images = load_camera_images(args.dataset_folder, 0)
    right_images = load_camera_images(args.dataset_folder, 1)
    assert len(left_images) == len(right_images) and len(left_images) > 0, (
        "Mismatched or empty camera images"
    )
    print(f"Found {len(left_images)} stereo image pairs")

    prev_points0: dict[int, tuple[float, float]] = {}
    prev_points1: dict[int, tuple[float, float]] = {}

    start_time = time.time()
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        estimator.process_frame(left_img, right_img)

        curr_points = estimator.track_points
        curr_points0 = curr_points["cam0"]
        curr_points1 = curr_points["cam1"]

        if args.rerun:
            # Set timeline
            timestamp_ns = 0
            try:
                stem = Path(left_path).stem
                timestamp_ns = int(stem)
            except ValueError:
                pass
            if (
                timestamp_ns < 1262304000000000000
            ):  # If timestamp is before 2010, assume it's a frame index instead
                rr.set_time("frame", sequence=i)
            else:
                rr.set_time("epoch_time", timestamp=timestamp_ns / 1e9)

            # Log images
            log_rerun_image(left_img, "camera/left")
            log_rerun_image(right_img, "camera/right")

            # Log keypoints and optical flow
            log_image_keypoints(curr_points0, prev_points0, "camera/left")
            log_image_keypoints(curr_points1, prev_points1, "camera/right")

            # Log active landmarks
            log_active_landmarks(
                estimator.landmarks_with_ids,
                estimator.current_pose,
                "/active_landmarks",
                50.0,
            )

            # Log current pose (left)
            log_pose(
                estimator.current_pose,
                "/current_pose/left",
                0.4,
                camera_frustum,
            )

            # Log current pose (right) = T_w_cam0 @ T_cam1_cam0
            t_w_cam1 = estimator.current_pose @ t_cam0_cam1_4x4
            log_pose(
                t_w_cam1,
                "/current_pose/right",
                0.4,
                camera_frustum,
            )

            # On new keyframe: log keyframe poses, old landmarks, trajectory
            if estimator.new_keyframe_added:
                keyframe_poses = estimator.keyframe_poses
                for ki in range(keyframe_poses.shape[0]):
                    kf_pose = keyframe_poses[ki].reshape(4, 4)
                    log_pose(
                        kf_pose,
                        f"/keyframe_poses/keyframe_{ki}",
                        0.3,
                        None,
                    )

                log_old_landmarks(estimator.removed_good_landmarks, "/old_landmarks")
                log_trajectory(estimator.keyframe_trajectory, "/trajectory")

        prev_points0 = curr_points0
        prev_points1 = curr_points1

    elapsed = time.time() - start_time
    avg_ms = (elapsed * 1000) / len(left_images)
    print(f"Processing time: {avg_ms:.2f}ms average per frame")


if __name__ == "__main__":
    main()
