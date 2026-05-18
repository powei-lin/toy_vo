#[derive(Debug, Clone)]
pub struct Feature {
    /// Unique identifier of this feature (within the current frame or globally).
    pub feature_id: usize,

    /// Pixel coordinate in the left image (u, v).
    pub pixel_coord: [f32; 2],

    /// Undistorted pixel coordinate (u, v). `[-1, -1]` means invalid.
    pub undistorted_coord: [f32; 2],
}

impl Feature {
    pub fn new(feature_id: usize, pixel_coord: [f32; 2]) -> Self {
        Self {
            feature_id,
            pixel_coord,
            undistorted_coord: [-1.0, -1.0],
        }
    }
}
