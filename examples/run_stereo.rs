use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Cursor, path::PathBuf};

use anyhow::Ok;
use camera_intrinsic_model::io::model_from_json;
use clap::Parser;
use glob::glob;
use nalgebra as na;
use toy_stereo_vo::estimator::{Estimator, EstimatorParameters};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct StereoVoArgs {
    #[arg(short, long)]
    dataset_folder: String,

    #[arg(short, long)]
    config_folder: String,

    #[arg(long, action)]
    rerun: bool,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
struct RvecTvec {
    pub rvec: [f32; 3],
    pub tvec: [f32; 3],
}
impl RvecTvec {
    pub fn to_isometry3(&self) -> na::Isometry3<f32> {
        na::Isometry3::new(
            na::Vector3::new(self.tvec[0], self.tvec[1], self.tvec[2]),
            na::Vector3::new(self.rvec[0], self.rvec[1], self.rvec[2]),
        )
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct RvecTvecs {
    pub rtvecs: Vec<RvecTvec>,
}

fn rtvec_from_json(file_path: &str) -> RvecTvec {
    let contents =
        std::fs::read_to_string(file_path).expect("Should have been able to read the file");
    let rtvecs: RvecTvecs = serde_json::from_str(&contents).unwrap();
    rtvecs.rtvecs[1]
}

fn id_to_color(id: u64) -> [u8; 3] {
    const M: u32 = 2u32.pow(24);
    fastrand::seed(id);
    let color_num = fastrand::u32(0..M);
    [
        ((color_num >> 16) % 256) as u8,
        ((color_num >> 8) % 256) as u8,
        (color_num % 256) as u8,
    ]
}

fn load_camera_images(root: &str, cam_idx: usize) -> anyhow::Result<Vec<PathBuf>> {
    let pattern = PathBuf::from(root).join(format!("mav0/cam{}/data/*.png", cam_idx));

    glob(pattern.to_str().unwrap())?
        .map(|p| p.map_err(anyhow::Error::from))
        .collect()
}

fn log_rerun_image(rec: &rerun::RecordingStream, image: &image::GrayImage, entity_path: &str) {
    let mut bytes: Vec<u8> = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Jpeg)
        .expect("fail to write jpeg");
    rec.log(entity_path, &rerun::EncodedImage::from_file_contents(bytes))
        .unwrap();
}

fn log_image_keypoints(
    rec: &rerun::RecordingStream,
    curr_keypoints: &HashMap<usize, (f32, f32)>,
    prev_keypoints: &HashMap<usize, (f32, f32)>,
    image_entity_path: &str,
) {
    // Cam0 points and lines
    let (colors0, points0): (Vec<_>, Vec<(f32, f32)>) = curr_keypoints
        .iter()
        .map(|(&id, &(x, y))| {
            let color = id_to_color(id as u64);
            (color, (x + 0.5, y + 0.5))
        })
        .unzip();
    rec.log(
        format!("{}/points", image_entity_path),
        &rerun::Points2D::new(points0).with_colors(colors0),
    )
    .unwrap();

    let mut line_strips0 = Vec::new();
    let mut line_colors0 = Vec::new();
    for (&id, &(curr_x, curr_y)) in curr_keypoints {
        if let Some(&(prev_x, prev_y)) = prev_keypoints.get(&id) {
            line_strips0.push([(prev_x + 0.5, prev_y + 0.5), (curr_x + 0.5, curr_y + 0.5)]);
            line_colors0.push(id_to_color(id as u64));
        }
    }
    if !line_strips0.is_empty() {
        rec.log(
            format!("{}/lines", image_entity_path),
            &rerun::LineStrips2D::new(line_strips0).with_colors(line_colors0),
        )
        .unwrap();
    }
}

fn to_rerun_transform(transform: &na::Isometry3<f32>) -> rerun::Transform3D {
    let qxyzw =
        na::UnitQuaternion::from_rotation_matrix(&transform.rotation.to_rotation_matrix()).coords;
    let rotation = rerun::Quaternion::from_xyzw([qxyzw[0], qxyzw[1], qxyzw[2], qxyzw[3]]);
    rerun::Transform3D::from_translation_rotation(
        [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ],
        rotation,
    )
}

fn log_pose(
    rec: &rerun::RecordingStream,
    transform: &na::Isometry3<f32>,
    entity_path: &str,
    size: f32,
    option_camera_frustum: Option<rerun::Pinhole>,
) {
    rec.log(entity_path, &to_rerun_transform(transform))
        .unwrap();
    if let Some(camera_frustum) = option_camera_frustum {
        rec.log(entity_path, &camera_frustum).unwrap();
    } else {
        rec.log(entity_path, &rerun::TransformAxes3D::new(size))
            .unwrap();
    }
}

fn log_active_landmarks(
    rec: &rerun::RecordingStream,
    landmarks: &HashMap<usize, na::Point3<f32>>,
    current_t_w_cam0: &na::Isometry3<f32>,
    entity_path: &str,
    range_threshold: f32,
) {
    let range_threshold2 = range_threshold * range_threshold;
    let cam0_position = na::Point3::from(current_t_w_cam0.translation.vector);
    let (colors, points): (Vec<[u8; 3]>, Vec<[f32; 3]>) = landmarks
        .iter()
        .filter_map(|(&feature_id, &point)| {
            let distance2 = (point - cam0_position).norm_squared();
            if distance2 < range_threshold2 {
                Some((id_to_color(feature_id as u64), [point.x, point.y, point.z]))
            } else {
                None
            }
        })
        .unzip();
    rec.log(
        entity_path,
        &rerun::Points3D::new(points).with_colors(colors),
    )
    .unwrap();
}

fn log_old_landmarks(
    rec: &rerun::RecordingStream,
    landmarks: &[na::Point3<f32>],
    entity_path: &str,
) {
    if landmarks.is_empty() {
        return;
    }
    let points: Vec<[f32; 3]> = landmarks.iter().map(|pt| [pt.x, pt.y, pt.z]).collect();
    rec.log(entity_path, &rerun::Points3D::new(points)).unwrap();
}

fn log_trajectory(
    rec: &rerun::RecordingStream,
    trajectory: &[na::Isometry3<f32>],
    entity_path: &str,
) {
    let line_strips: Vec<[f32; 3]> = trajectory
        .iter()
        .map(|t_cam0_w| {
            let tvec_w = t_cam0_w.inverse().translation.vector;
            [tvec_w.x, tvec_w.y, tvec_w.z]
        })
        .collect();
    rec.log(entity_path, &rerun::LineStrips3D::new([line_strips]))
        .unwrap();
}

fn camera_model_to_rerun_pinhole(
    model: &camera_intrinsic_model::GenericModel<f64>,
) -> rerun::Pinhole {
    let projection_matrix = model
        .estimate_new_camera_matrix_for_undistort(0.0, None)
        .cast();
    rerun::Pinhole::from_focal_length_and_resolution(
        [projection_matrix[(0, 0)], projection_matrix[(1, 1)]],
        [model.width() as f32, model.height() as f32],
    )
    .with_principal_point([projection_matrix[(0, 2)], projection_matrix[(1, 2)]])
    .with_image_plane_distance(0.1)
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = StereoVoArgs::parse();

    let root_folder = cli.dataset_folder;

    let camera_images: Vec<Vec<PathBuf>> = (0..2)
        .map(|i| load_camera_images(&root_folder, i))
        .collect::<Result<_, _>>()?;

    anyhow::ensure!(
        camera_images[0].len() == camera_images[1].len() && !camera_images[0].is_empty(),
        "Mismatched or empty camera images"
    );
    println!("Found {} stereo image pairs", camera_images[0].len());

    // load camera models and extrinsics
    let config_folder = cli.config_folder;
    let left_cam_model = model_from_json(&format!("{}/cam0.json", config_folder));
    let right_cam_model = model_from_json(&format!("{}/cam1.json", config_folder));
    let t_cam1_cam0 = rtvec_from_json(&format!("{}/extrinsics.json", config_folder)).to_isometry3();

    let mut estimator = Estimator::new(
        left_cam_model.cast(),
        right_cam_model.cast(),
        t_cam1_cam0,
        Some(EstimatorParameters::default()),
    );
    let (rec, camera_frustum) = if cli.rerun {
        (
            Some(
                rerun::RecordingStreamBuilder::new("vo")
                    .spawn_opts(&rerun::SpawnOptions {
                        port: 9875,
                        ..Default::default()
                    })
                    .unwrap(),
            ),
            Some(camera_model_to_rerun_pinhole(&left_cam_model.cast())),
        )
    } else {
        (None, None)
    };

    if let Some(rec) = &rec {
        rec.log_static("/", &rerun::ViewCoordinates::RDF()).unwrap();
    }

    let mut prev_points0: HashMap<usize, (f32, f32)> = HashMap::new();
    let mut prev_points1: HashMap<usize, (f32, f32)> = HashMap::new();

    let start_time = std::time::Instant::now();
    for (i, (left_img_path, right_img_path)) in
        camera_images[0].iter().zip(&camera_images[1]).enumerate()
    {
        let left_image = image::open(left_img_path)?.to_luma8();
        let right_image = image::open(right_img_path)?.to_luma8();
        estimator.process_frame(&left_image, &right_image)?;
        let [curr_points0, curr_points1] = &estimator.tracker.get_track_points();

        if let Some(rec) = &rec {
            let timestamp_ns = left_img_path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);
            if timestamp_ns == 0 {
                log::warn!(
                    "Failed to parse timestamp from filename '{}'",
                    left_img_path.display()
                );
                rec.set_time_sequence("frame", i as i64);
            } else {
                rec.set_time(
                    "epoch_time",
                    rerun::TimeCell::from_timestamp_nanos_since_epoch(timestamp_ns),
                );
            }
            log_rerun_image(&rec, &left_image, "camera/left");
            log_rerun_image(&rec, &right_image, "camera/right");
            log_image_keypoints(&rec, curr_points0, &prev_points0, "camera/left");
            log_image_keypoints(&rec, curr_points1, &prev_points1, "camera/right");

            log_active_landmarks(
                &rec,
                &estimator.landmarks,
                &estimator.current_t_w_cam0,
                "/active_landmarks",
                50.0,
            );

            log_pose(
                &rec,
                &estimator.current_t_w_cam0,
                "/current_pose/left",
                0.4,
                camera_frustum.clone(),
            );
            log_pose(
                &rec,
                &(estimator.current_t_w_cam0 * t_cam1_cam0),
                "/current_pose/right",
                0.4,
                camera_frustum.clone(),
            );
            if estimator.new_keyframe_added {
                for (i, kf) in estimator.keyframe_window.keyframes.iter().enumerate() {
                    log_pose(
                        &rec,
                        &kf.t_cam0_w.inverse(),
                        &format!("/keyframe_poses/keyframe_{}", i),
                        0.3,
                        None,
                    );
                }
                log_old_landmarks(&rec, &estimator.removed_good_landmarks, "/old_landmarks");
                log_trajectory(&rec, &estimator.keyframe_trajectory, "/trajectory");
            }
        }

        // for visualization purposes
        prev_points0 = curr_points0.clone();
        prev_points1 = curr_points1.clone();
    }

    let elapsed = start_time.elapsed();
    println!(
        "Processing time: {:.2?}ms average per frame",
        elapsed.as_millis() as f64 / camera_images[0].len() as f64
    );

    Ok(())
}
