//! Rotation kernel used by the scan hot path (pure Rust, no PyO3).

use ndarray::Array3;

use crate::rotate::rotate_grid3d_pair_core;

/// Rotate a single template+mask pair by the inverse of `rotmat`, sampling
/// `template` trilinearly and `mask` by nearest-neighbor within `radius` of
/// the origin. Zero-fills `out_template`/`out_mask` first, then delegates to
/// `crate::rotate::rotate_grid3d_pair_core`, which writes only the in-radius
/// voxels (wrapped so the origin stays at index `[0, 0, 0]`) — voxels outside
/// `radius` are left at zero, not copied from the input.
pub fn rotate_pair_internal_into(
    template: &Array3<f32>,
    mask: &Array3<f32>,
    rotmat: &[[f32; 3]; 3],
    radius: i32,
    out_template: &mut Array3<f32>,
    out_mask: &mut Array3<f32>,
) {
    out_template.fill(0.0);
    out_mask.fill(0.0);
    rotate_grid3d_pair_core(
        template.view(),
        mask.view(),
        rotmat,
        radius,
        out_template.view_mut(),
        out_mask.view_mut(),
    );
}

/// Convert a flat (3,3) rotation matrix slice to [[f32;3];3].
pub fn rot_slice_to_mat(rot: &ndarray::ArrayView2<f32>) -> [[f32; 3]; 3] {
    [
        [rot[[0, 0]], rot[[0, 1]], rot[[0, 2]]],
        [rot[[1, 0]], rot[[1, 1]], rot[[1, 2]]],
        [rot[[2, 0]], rot[[2, 1]], rot[[2, 2]]],
    ]
}
