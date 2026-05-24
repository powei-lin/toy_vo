# toy_vo
[![crate](https://img.shields.io/crates/v/toy-vo.svg)](https://crates.io/crates/toy-vo)
[![PyPI - Version](https://img.shields.io/pypi/v/toy-vo.svg)](https://pypi.org/project/toy-vo)

A very simple stereo visual odometry library.
<img src="docs/euroc.jpg" width="600" alt="Euroc">


## Run Example

You can download datasets from

* [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/)
* [TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
* [TUM 4Seasons Dataset](https://cvg.cit.tum.de/data/datasets/4seasons-dataset/download)

#### Python
```bash
pip install toy-vo rerun-sdk==0.32 opencv-python scipy
python3 examples/run_stereo.py -d {your_path/V1_01_easy} -c configs/euroc --rerun
```
#### Rust
```bash
git clone https://github.com/powei-lin/toy_vo.git && cd toy_vo
cargo run -r --example run_stereo -- -d {your_path/dataset-corridor4_512_16} -c configs/tum_vi --rerun
```

## Calibrate your own camera and run toy-vo
Use [camera-intrinsic-calibration](https://github.com/powei-lin/camera-intrinsic-calibration-rs) and format your images like euroc dataset.

## Acknowledgements
Thanks to all the authors of these libraries.
- [basalt](https://gitlab.com/VladyslavUsenko/basalt)
- [faer](https://codeberg.org/sarah-quinones/faer)
- [nalgebra](https://github.com/dimforge/nalgebra)
- [rerun](https://github.com/rerun-io/rerun)

And thanks to [Hossam R.](https://github.com/jumpinthefire) for implementing the LM method in [tiny-solver](https://github.com/powei-lin/tiny-solver-rs).