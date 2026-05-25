use diol::prelude::*;
use eyre::Result;
use nalgebra as na;
use toy_vo::estimator::triangulate_points;
use toy_vo::optimization::{ReprojectionFactorCam0, ReprojectionFactorCam1};
use tiny_solver::factors::Factor;

fn main() -> Result<()> {
    let bench = Bench::new(Config::from_args()?);
    bench.register_many(
        "triangulate_points",
        list![bench_triangulate_points],
        [1, 10, 100, 1000],
    );
    bench.register_many(
        "reprojection_factor",
        list![
            bench_reprojection_cam0,
            bench_reprojection_cam1,
        ],
        [1, 10, 100, 1000],
    );
    bench.run()?;
    Ok(())
}

fn bench_triangulate_points(bencher: Bencher, num_points: usize) {
    let t_1_0 = na::SMatrix::<f32, 3, 4>::new(
        1.0, 0.0, 0.0, 0.1, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
    );
    let pts0: Vec<na::Vector3<f32>> = (0..num_points)
        .map(|i| {
            let x = (i as f32 * 0.1) % 1.0 - 0.5;
            let y = (i as f32 * 0.07) % 1.0 - 0.5;
            na::Vector3::new(x, y, 1.0)
        })
        .collect();
    let pts1: Vec<na::Vector3<f32>> = pts0
        .iter()
        .map(|p| {
            // Simulate right camera observation with baseline shift
            let depth = 5.0 + (p.x * 10.0).abs();
            let world_pt = na::Vector3::new(p.x * depth, p.y * depth, depth);
            let cam1_pt = world_pt - na::Vector3::new(0.1, 0.0, 0.0);
            na::Vector3::new(cam1_pt.x / cam1_pt.z, cam1_pt.y / cam1_pt.z, 1.0)
        })
        .collect();

    bencher.bench(|| {
        for i in 0..num_points {
            black_box(triangulate_points(
                &pts0[i],
                &pts1[i],
                t_1_0.as_view(),
            ));
        }
    });
}

fn bench_reprojection_cam0(bencher: Bencher, num_points: usize) {
    let factors: Vec<ReprojectionFactorCam0> = (0..num_points)
        .map(|i| ReprojectionFactorCam0 {
            observed_normalized_pt: ((i as f32) * 0.01, (i as f32) * 0.02),
        })
        .collect();
    let tvec: na::DVector<f64> = na::dvector![0.1, 0.2, 0.3];
    let rvec: na::DVector<f64> = na::dvector![0.01, 0.02, 0.03];
    let landmarks: Vec<na::DVector<f64>> = (0..num_points)
        .map(|i| na::dvector![i as f64 * 0.5, i as f64 * 0.3, 5.0 + i as f64 * 0.1])
        .collect();

    bencher.bench(|| {
        for i in 0..num_points {
            let params = [tvec.clone(), rvec.clone(), landmarks[i].clone()];
            black_box(Factor::<f64>::residual_func(&factors[i], &params));
        }
    });
}

fn bench_reprojection_cam1(bencher: Bencher, num_points: usize) {
    let t_cam1_cam0 = na::Isometry3::new(
        na::Vector3::new(0.1, 0.0, 0.0),
        na::Vector3::new(0.0, 0.0, 0.0),
    );
    let factors: Vec<ReprojectionFactorCam1> = (0..num_points)
        .map(|i| ReprojectionFactorCam1 {
            observed_normalized_pt: ((i as f32) * 0.01, (i as f32) * 0.02),
            t_cam1_cam0,
        })
        .collect();
    let tvec: na::DVector<f64> = na::dvector![0.1, 0.2, 0.3];
    let rvec: na::DVector<f64> = na::dvector![0.01, 0.02, 0.03];
    let landmarks: Vec<na::DVector<f64>> = (0..num_points)
        .map(|i| na::dvector![i as f64 * 0.5, i as f64 * 0.3, 5.0 + i as f64 * 0.1])
        .collect();

    bencher.bench(|| {
        for i in 0..num_points {
            let params = [tvec.clone(), rvec.clone(), landmarks[i].clone()];
            black_box(Factor::<f64>::residual_func(&factors[i], &params));
        }
    });
}
