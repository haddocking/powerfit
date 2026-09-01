//! Rotation kernel used by the scan hot path (pure Rust, no PyO3).

use ndarray::Array3;

/// Rotate a single template+mask pair by the inverse of `rotmat`, sampling
/// `template` trilinearly and `mask` by nearest-neighbor within `radius` of
/// the origin. Zero-fills `out_template`/`out_mask` first, then writes only
/// the in-radius voxels (wrapped so the origin stays at index `[0, 0, 0]`) —
/// voxels outside `radius` are left at zero, not copied from the input.
/// Mirrors the logic of `rotate_grid3d` / `rotate_grid3d_pair` from `crate::rotate`.
pub fn rotate_pair_internal_into(
    template: &Array3<f32>,
    mask: &Array3<f32>,
    rotmat: &[[f32; 3]; 3],
    radius: i32,
    out_template: &mut Array3<f32>,
    out_mask: &mut Array3<f32>,
) {
    let gs = template.shape();
    let gs0 = gs[0] as isize;
    let gs1 = gs[1] as isize;
    let gs2 = gs[2] as isize;
    let grid_slice = gs1 * gs2;
    let grid_size = gs0 * grid_slice;

    let out_slice = gs1 * gs2;
    let out_size = gs0 * out_slice;
    let radius2 = (radius * radius) as isize;

    out_template.fill(0.0);
    out_mask.fill(0.0);

    let template_raw = template.as_ptr();
    let mask_raw = mask.as_ptr();
    let out_template_raw = out_template.as_mut_ptr();
    let out_mask_raw = out_mask.as_mut_ptr();

    let r = rotmat;

    for z in -radius..=radius {
        let dist2_z = (z * z) as isize;
        if dist2_z > radius2 {
            continue;
        }
        let zf = z as f32;
        let xcoor_z = r[2][0] * zf;
        let ycoor_z = r[2][1] * zf;
        let zcoor_z = r[2][2] * zf;

        let mut out_z = z as isize * out_slice;
        if z < 0 {
            out_z += out_size;
        }

        for y in -radius..=radius {
            let dist2_zy = dist2_z + (y * y) as isize;
            if dist2_zy > radius2 {
                continue;
            }
            let yf = y as f32;
            let xcoor_zy = xcoor_z + r[1][0] * yf;
            let ycoor_zy = ycoor_z + r[1][1] * yf;
            let zcoor_zy = zcoor_z + r[1][2] * yf;

            let mut out_zy = out_z + y as isize * gs2;
            if y < 0 {
                out_zy += out_slice;
            }

            for x in -radius..=radius {
                let dist2_zyx = dist2_zy + (x * x) as isize;
                if dist2_zyx > radius2 {
                    continue;
                }
                let xf = x as f32;
                let xcoor_zyx = xcoor_zy + r[0][0] * xf;
                let ycoor_zyx = ycoor_zy + r[0][1] * xf;
                let zcoor_zyx = zcoor_zy + r[0][2] * xf;

                let mut out_zyx = out_zy + x as isize;
                if x < 0 {
                    out_zyx += gs2;
                }

                // Trilinear for template
                let x0 = xcoor_zyx.floor() as isize;
                let y0 = ycoor_zyx.floor() as isize;
                let z0 = zcoor_zyx.floor() as isize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let z1 = z0 + 1;

                let mut grid_zyx = z0 * grid_slice + y0 * gs2 + x0;
                if x0 < 0 {
                    grid_zyx += gs2;
                }
                if y0 < 0 {
                    grid_zyx += grid_slice;
                }
                if z0 < 0 {
                    grid_zyx += grid_size;
                }

                let dx = xcoor_zyx - x0 as f32;
                let dy = ycoor_zyx - y0 as f32;
                let dz = zcoor_zyx - z0 as f32;
                let dx1 = 1.0 - dx;
                let dy1 = 1.0 - dy;
                let dz1 = 1.0 - dz;

                let off1: isize = if x1 == 0 { 1 - gs2 } else { 1 };
                let c00 = unsafe {
                    *template_raw.offset(grid_zyx) * dx1
                        + *template_raw.offset(grid_zyx + off1) * dx
                };
                let off0y: isize = if y1 == 0 { gs2 - grid_slice } else { gs2 };
                let off1y = off0y + if x1 == 0 { 1 - gs2 } else { 1 };
                let c10 = unsafe {
                    *template_raw.offset(grid_zyx + off0y) * dx1
                        + *template_raw.offset(grid_zyx + off1y) * dx
                };
                let off0z: isize = if z1 == 0 {
                    grid_slice - grid_size
                } else {
                    grid_slice
                };
                let off1z = off0z + if x1 == 0 { 1 - gs2 } else { 1 };
                let c01 = unsafe {
                    *template_raw.offset(grid_zyx + off0z) * dx1
                        + *template_raw.offset(grid_zyx + off1z) * dx
                };
                let mut off0zy = if z1 == 0 {
                    grid_slice - grid_size
                } else {
                    grid_slice
                };
                off0zy += if y1 == 0 { gs2 - grid_slice } else { gs2 };
                let off1zy = off0zy + if x1 == 0 { 1 - gs2 } else { 1 };
                let c11 = unsafe {
                    *template_raw.offset(grid_zyx + off0zy) * dx1
                        + *template_raw.offset(grid_zyx + off1zy) * dx
                };
                let c0 = c00 * dy1 + c10 * dy;
                let c1 = c01 * dy1 + c11 * dy;

                // Nearest-neighbor for mask
                let xm = xcoor_zyx.round() as isize;
                let ym = ycoor_zyx.round() as isize;
                let zm = zcoor_zyx.round() as isize;
                let mut mask_idx = zm * grid_slice + ym * gs2 + xm;
                if xm < 0 {
                    mask_idx += gs2;
                }
                if ym < 0 {
                    mask_idx += grid_slice;
                }
                if zm < 0 {
                    mask_idx += grid_size;
                }

                unsafe {
                    *out_template_raw.offset(out_zyx) = c0 * dz1 + c1 * dz;
                    *out_mask_raw.offset(out_zyx) = *mask_raw.offset(mask_idx);
                }
            }
        }
    }
}

/// Convert a flat (3,3) rotation matrix slice to [[f32;3];3].
pub fn rot_slice_to_mat(rot: &ndarray::ArrayView2<f32>) -> [[f32; 3]; 3] {
    [
        [rot[[0, 0]], rot[[0, 1]], rot[[0, 2]]],
        [rot[[1, 0]], rot[[1, 1]], rot[[1, 2]]],
        [rot[[2, 0]], rot[[2, 1]], rot[[2, 2]]],
    ]
}
