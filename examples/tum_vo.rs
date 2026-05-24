use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Cursor, path::PathBuf};

use anyhow::Ok;
use camera_intrinsic_model::io::model_from_json;
use clap::Parser;
use glob::glob;
use nalgebra as na;
use patch_tracker::StereoPatchTracker;
use toy_vo::estimator2::Estimator2;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct StereoVoArgs {
    #[arg(short, long)]
    folder: String,

    #[arg(short, long)]
    left_camera_info: String,

    #[arg(short, long)]
    right_camera_info: String,

    #[arg(long)]
    rtvec: String,

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

fn log_keypoints(
    rec: &rerun::RecordingStream,
    keypoints: &[[f32; 2]],
    ids: &[usize],
    entity_path: &str,
) {
    let (points, colors) = keypoints
        .iter()
        .zip(ids.iter())
        .filter_map(|(&kp, &id)| {
            if id == usize::MAX {
                return None; // Skip keypoints without valid IDs
            } else {
                let color = id_to_color(id as u64);
                Some((kp, color))
            }
        })
        .unzip::<[f32; 2], [u8; 3], Vec<_>, Vec<_>>();
    let keypoint_ids: Vec<u16> = ids
        .iter()
        .filter_map(|&id| {
            if id == usize::MAX {
                None
            } else {
                Some(id as u16)
            }
        })
        .collect();
    rec.log(
        entity_path,
        &rerun::Points2D::new(points)
            .with_colors(colors)
            .with_keypoint_ids(keypoint_ids),
    )
    .unwrap();
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

fn log_pose(rec: &rerun::RecordingStream, transform: &na::Isometry3<f32>, entity_path: &str) {
    rec.log(entity_path, &to_rerun_transform(transform))
        .unwrap();
    rec.log(entity_path, &rerun::TransformAxes3D::new(1.0))
        .unwrap();
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = StereoVoArgs::parse();

    let root_folder = cli.folder;

    let camera_images: Vec<Vec<PathBuf>> = (0..2)
        .map(|i| load_camera_images(&root_folder, i))
        .collect::<Result<_, _>>()?;

    anyhow::ensure!(
        camera_images[0].len() == camera_images[1].len() && !camera_images[0].is_empty(),
        "Mismatched or empty camera images"
    );
    println!("Found {} stereo image pairs", camera_images[0].len());
    let left_cam_model = model_from_json(&cli.left_camera_info);
    let right_cam_model = model_from_json(&cli.right_camera_info);
    let t_cam1_cam0 = rtvec_from_json(&cli.rtvec).to_isometry3();

    let mut estimator = Estimator2::new(
        left_cam_model.cast(),
        right_cam_model.cast(),
        t_cam1_cam0,
        5,  // Tracker optical flow levels
        16, // Tracker grid size
        10, // Keyframe window size
    );

    let rec = rerun::RecordingStreamBuilder::new("vo")
        .spawn_opts(&rerun::SpawnOptions {
            port: 9875,
            ..Default::default()
        })
        .unwrap();
    rec.log_static("/", &rerun::ViewCoordinates::RDF()).unwrap();

    let mut trajectory: Vec<[f32; 3]> = Vec::new();
    let mut prev_points0: HashMap<usize, (f32, f32)> = HashMap::new();
    let mut prev_points1: HashMap<usize, (f32, f32)> = HashMap::new();

    for (i, (left_img_path, right_img_path)) in
        camera_images[0].iter().zip(&camera_images[1]).enumerate()
    {
        // if i > 305 {
        //     break;
        // }
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

        // println!(
        //     "Pair {}: Left: {}, Right: {}",
        //     i,
        //     left_img_path.display(),
        //     right_img_path.display()
        // );

        let left_image = image::open(left_img_path)?.to_luma8();
        let right_image = image::open(right_img_path)?.to_luma8();
        log_rerun_image(&rec, &left_image, "camera/left");
        log_rerun_image(&rec, &right_image, "camera/right");
        estimator.process_frame(&left_image, &right_image)?;

        let [curr_points0, curr_points1] = &estimator.tracker.get_track_points();

        // Cam0 points and lines
        let (colors0, points0): (Vec<_>, Vec<(f32, f32)>) = curr_points0
            .iter()
            .map(|(&id, &(x, y))| {
                let color = id_to_color(id as u64);
                (color, (x + 0.5, y + 0.5))
            })
            .unzip();
        rec.log(
            "camera/left/points",
            &rerun::Points2D::new(points0).with_colors(colors0),
        )
        .unwrap();

        let mut line_strips0 = Vec::new();
        let mut line_colors0 = Vec::new();
        for (&id, &(curr_x, curr_y)) in curr_points0 {
            if let Some(&(prev_x, prev_y)) = prev_points0.get(&id) {
                line_strips0.push([(prev_x + 0.5, prev_y + 0.5), (curr_x + 0.5, curr_y + 0.5)]);
                line_colors0.push(id_to_color(id as u64));
            }
        }
        if !line_strips0.is_empty() {
            rec.log(
                "camera/left/lines",
                &rerun::LineStrips2D::new(line_strips0).with_colors(line_colors0),
            )
            .unwrap();
        }

        // Cam1 points and lines
        let (colors1, points1): (Vec<_>, Vec<(f32, f32)>) = curr_points1
            .iter()
            .map(|(&id, &(x, y))| {
                let color = id_to_color(id as u64);
                (color, (x + 0.5, y + 0.5))
            })
            .unzip();
        rec.log(
            "camera/right/points",
            &rerun::Points2D::new(points1).with_colors(colors1),
        )
        .unwrap();

        let mut line_strips1 = Vec::new();
        let mut line_colors1 = Vec::new();
        for (&id, &(curr_x, curr_y)) in curr_points1 {
            if let Some(&(prev_x, prev_y)) = prev_points1.get(&id) {
                line_strips1.push([(prev_x + 0.5, prev_y + 0.5), (curr_x + 0.5, curr_y + 0.5)]);
                line_colors1.push(id_to_color(id as u64));
            }
        }
        if !line_strips1.is_empty() {
            rec.log(
                "camera/right/lines",
                &rerun::LineStrips2D::new(line_strips1).with_colors(line_colors1),
            )
            .unwrap();
        }

        prev_points0 = curr_points0.clone();
        prev_points1 = curr_points1.clone();

        let (colors, points): (Vec<[u8; 3]>, Vec<[f32; 3]>) = estimator
            .landmarks
            .iter()
            .filter_map(|(&feature_id, &point)| {
                if point[0] * point[0] + point[1] * point[1] + point[2] * point[2] < 10000.0 {
                    Some((id_to_color(feature_id as u64), [point.x, point.y, point.z]))
                } else {
                    None
                }
            })
            .unzip();
        rec.log(
            "/map/points",
            &rerun::Points3D::new(points).with_colors(colors),
        )
        .unwrap();
        log_pose(&rec, &estimator.current_t_w_cam0, "/current_pose");

        //     // Keyframe poses — get_keyframe_poses() now returns T_W_Cl for each frame
        //     let system_poses = estimator.sliding_window.get_keyframe_poses();
        //     for (pose_id, T_W_Cl) in system_poses.iter().enumerate() {
        //         let pose_path = format!("pose_{}", pose_id);
        //         rec.log(pose_path.clone(), &to_rerun_transform(T_W_Cl))
        //             .unwrap();
        //         rec.log(pose_path, &rerun::TransformAxes3D::new(1.0))
        //             .unwrap();
        //     }

        //     // History of keyframe poses
        //     let mat = estimator
        //         .sliding_window
        //         .get_keyframe_poses()
        //         .first()
        //         .unwrap()
        //         .cast();
        //     trajectory.push([mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]]);
        //     rec.log(
        //         "/trajectory",
        //         &rerun::LineStrips3D::new([trajectory.clone()]),
        //     )
        //     .unwrap();
    }

    Ok(())
}
