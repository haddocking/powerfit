use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Rotate a 3D float32 grid by the inverse of rotmat, sampling within a sphere of given radius.
/// nearest=false: trilinear interpolation. nearest=true: nearest-neighbor.
/// out is modified in-place.
#[pyfunction]
pub fn rotate_grid3d<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray3<'py, f32>,
    rotmat: PyReadonlyArray2<'py, f32>,
    radius: i32,
    out: &Bound<'py, PyArray3<f32>>,
    nearest: bool,
) -> PyResult<()> {
    let grid = grid.as_array();
    let rotmat = rotmat.as_array();
    let mut out = unsafe { out.as_array_mut() };

    let gs = grid.shape();
    let gs0 = gs[0] as isize;
    let gs1 = gs[1] as isize;
    let gs2 = gs[2] as isize;
    let grid_slice = gs1 * gs2;
    let grid_size = gs0 * grid_slice;

    let os = out.shape().to_owned();
    let os0 = os[0] as isize;
    let os1 = os[1] as isize;
    let os2 = os[2] as isize;
    let out_slice = os1 * os2;
    let out_size = os0 * out_slice;

    let radius2 = (radius * radius) as isize;

    let r00 = rotmat[[0, 0]];
    let r01 = rotmat[[0, 1]];
    let r02 = rotmat[[0, 2]];
    let r10 = rotmat[[1, 0]];
    let r11 = rotmat[[1, 1]];
    let r12 = rotmat[[1, 2]];
    let r20 = rotmat[[2, 0]];
    let r21 = rotmat[[2, 1]];
    let r22 = rotmat[[2, 2]];

    let grid_raw = grid.as_ptr();
    let out_raw = out.as_mut_ptr();

    if nearest {
        for z in -radius..=radius {
            let dist2_z = (z * z) as isize;
            if dist2_z > radius2 {
                continue;
            }
            let zf = z as f32;
            let xcoor_z = r20 * zf;
            let ycoor_z = r21 * zf;
            let zcoor_z = r22 * zf;

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
                let xcoor_zy = xcoor_z + r10 * yf;
                let ycoor_zy = ycoor_z + r11 * yf;
                let zcoor_zy = zcoor_z + r12 * yf;

                let mut out_zy = out_z + y as isize * os2;
                if y < 0 {
                    out_zy += out_slice;
                }

                for x in -radius..=radius {
                    let dist2_zyx = dist2_zy + (x * x) as isize;
                    if dist2_zyx > radius2 {
                        continue;
                    }
                    let xf = x as f32;
                    let xcoor_zyx = xcoor_zy + r00 * xf;
                    let ycoor_zyx = ycoor_zy + r01 * xf;
                    let zcoor_zyx = zcoor_zy + r02 * xf;

                    let mut out_zyx = out_zy + x as isize;
                    if x < 0 {
                        out_zyx += os2;
                    }

                    let x0 = xcoor_zyx.round() as isize;
                    let y0 = ycoor_zyx.round() as isize;
                    let z0 = zcoor_zyx.round() as isize;

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

                    unsafe {
                        *out_raw.offset(out_zyx) = *grid_raw.offset(grid_zyx);
                    }
                }
            }
        }
    } else {
        for z in -radius..=radius {
            let dist2_z = (z * z) as isize;
            if dist2_z > radius2 {
                continue;
            }
            let zf = z as f32;
            let xcoor_z = r20 * zf;
            let ycoor_z = r21 * zf;
            let zcoor_z = r22 * zf;

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
                let xcoor_zy = xcoor_z + r10 * yf;
                let ycoor_zy = ycoor_z + r11 * yf;
                let zcoor_zy = zcoor_z + r12 * yf;

                let mut out_zy = out_z + y as isize * os2;
                if y < 0 {
                    out_zy += out_slice;
                }

                for x in -radius..=radius {
                    let dist2_zyx = dist2_zy + (x * x) as isize;
                    if dist2_zyx > radius2 {
                        continue;
                    }
                    let xf = x as f32;
                    let xcoor_zyx = xcoor_zy + r00 * xf;
                    let ycoor_zyx = ycoor_zy + r01 * xf;
                    let zcoor_zyx = zcoor_zy + r02 * xf;

                    let mut out_zyx = out_zy + x as isize;
                    if x < 0 {
                        out_zyx += os2;
                    }

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
                        *grid_raw.offset(grid_zyx) * dx1 + *grid_raw.offset(grid_zyx + off1) * dx
                    };

                    let off0y: isize = if y1 == 0 { gs2 - grid_slice } else { gs2 };
                    let off1y: isize = off0y + if x1 == 0 { 1 - gs2 } else { 1 };
                    let c10 = unsafe {
                        *grid_raw.offset(grid_zyx + off0y) * dx1
                            + *grid_raw.offset(grid_zyx + off1y) * dx
                    };

                    let off0z: isize = if z1 == 0 { grid_slice - grid_size } else { grid_slice };
                    let off1z: isize = off0z + if x1 == 0 { 1 - gs2 } else { 1 };
                    let c01 = unsafe {
                        *grid_raw.offset(grid_zyx + off0z) * dx1
                            + *grid_raw.offset(grid_zyx + off1z) * dx
                    };

                    let mut off0zy: isize =
                        if z1 == 0 { grid_slice - grid_size } else { grid_slice };
                    off0zy += if y1 == 0 { gs2 - grid_slice } else { gs2 };
                    let off1zy: isize = off0zy + if x1 == 0 { 1 - gs2 } else { 1 };
                    let c11 = unsafe {
                        *grid_raw.offset(grid_zyx + off0zy) * dx1
                            + *grid_raw.offset(grid_zyx + off1zy) * dx
                    };

                    let c0 = c00 * dy1 + c10 * dy;
                    let c1 = c01 * dy1 + c11 * dy;
                    unsafe {
                        *out_raw.offset(out_zyx) = c0 * dz1 + c1 * dz;
                    }
                }
            }
        }
    }
    let _ = py;
    Ok(())
}

/// Rotate both template (trilinear) and mask (nearest) in one pass.
#[pyfunction]
pub fn rotate_grid3d_pair<'py>(
    py: Python<'py>,
    template: PyReadonlyArray3<'py, f32>,
    mask: PyReadonlyArray3<'py, f32>,
    rotmat: PyReadonlyArray2<'py, f32>,
    radius: i32,
    out_template: &Bound<'py, PyArray3<f32>>,
    out_mask: &Bound<'py, PyArray3<f32>>,
) -> PyResult<()> {
    let template = template.as_array();
    let mask = mask.as_array();
    let rotmat = rotmat.as_array();
    let mut out_template = unsafe { out_template.as_array_mut() };
    let mut out_mask = unsafe { out_mask.as_array_mut() };

    if template.shape() != mask.shape()
        || template.shape() != out_template.shape()
        || template.shape() != out_mask.shape()
    {
        return Err(PyValueError::new_err(
            "template, mask, out_template, and out_mask must have identical shapes",
        ));
    }

    let gs = template.shape();
    let gs0 = gs[0] as isize;
    let gs1 = gs[1] as isize;
    let gs2 = gs[2] as isize;
    let grid_slice = gs1 * gs2;
    let grid_size = gs0 * grid_slice;

    let os = out_template.shape().to_owned();
    let os0 = os[0] as isize;
    let os1 = os[1] as isize;
    let os2 = os[2] as isize;
    let out_slice = os1 * os2;
    let out_size = os0 * out_slice;

    let radius2 = (radius * radius) as isize;

    let r00 = rotmat[[0, 0]];
    let r01 = rotmat[[0, 1]];
    let r02 = rotmat[[0, 2]];
    let r10 = rotmat[[1, 0]];
    let r11 = rotmat[[1, 1]];
    let r12 = rotmat[[1, 2]];
    let r20 = rotmat[[2, 0]];
    let r21 = rotmat[[2, 1]];
    let r22 = rotmat[[2, 2]];

    let template_raw = template.as_ptr();
    let mask_raw = mask.as_ptr();
    let out_template_raw = out_template.as_mut_ptr();
    let out_mask_raw = out_mask.as_mut_ptr();

    for z in -radius..=radius {
        let dist2_z = (z * z) as isize;
        if dist2_z > radius2 {
            continue;
        }
        let zf = z as f32;
        let xcoor_z = r20 * zf;
        let ycoor_z = r21 * zf;
        let zcoor_z = r22 * zf;

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
            let xcoor_zy = xcoor_z + r10 * yf;
            let ycoor_zy = ycoor_z + r11 * yf;
            let zcoor_zy = zcoor_z + r12 * yf;

            let mut out_zy = out_z + y as isize * os2;
            if y < 0 {
                out_zy += out_slice;
            }

            for x in -radius..=radius {
                let dist2_zyx = dist2_zy + (x * x) as isize;
                if dist2_zyx > radius2 {
                    continue;
                }
                let xf = x as f32;
                let xcoor_zyx = xcoor_zy + r00 * xf;
                let ycoor_zyx = ycoor_zy + r01 * xf;
                let zcoor_zyx = zcoor_zy + r02 * xf;

                let mut out_zyx = out_zy + x as isize;
                if x < 0 {
                    out_zyx += os2;
                }

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
                let off1y: isize = off0y + if x1 == 0 { 1 - gs2 } else { 1 };
                let c10 = unsafe {
                    *template_raw.offset(grid_zyx + off0y) * dx1
                        + *template_raw.offset(grid_zyx + off1y) * dx
                };

                let off0z: isize = if z1 == 0 { grid_slice - grid_size } else { grid_slice };
                let off1z: isize = off0z + if x1 == 0 { 1 - gs2 } else { 1 };
                let c01 = unsafe {
                    *template_raw.offset(grid_zyx + off0z) * dx1
                        + *template_raw.offset(grid_zyx + off1z) * dx
                };

                let mut off0zy: isize =
                    if z1 == 0 { grid_slice - grid_size } else { grid_slice };
                off0zy += if y1 == 0 { gs2 - grid_slice } else { gs2 };
                let off1zy: isize = off0zy + if x1 == 0 { 1 - gs2 } else { 1 };
                let c11 = unsafe {
                    *template_raw.offset(grid_zyx + off0zy) * dx1
                        + *template_raw.offset(grid_zyx + off1zy) * dx
                };

                let c0 = c00 * dy1 + c10 * dy;
                let c1 = c01 * dy1 + c11 * dy;

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
    let _ = py;
    Ok(())
}
