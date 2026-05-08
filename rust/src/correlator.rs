use ndarray::{s, Array3, Zip};
use ndrustfft::{ndfft, ndfft_r2c, ndifft, ndifft_r2c, FftHandler, R2cFftHandler};
use num_complex::Complex;
use numpy::{PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Internal FFT helpers
// ---------------------------------------------------------------------------

/// Build reusable FFT handlers for a volume of shape (nz, ny, nx).
/// Returns (r2c_x, c2c_y, c2c_z).
fn make_fft_handlers(
    nz: usize,
    ny: usize,
    nx: usize,
) -> (R2cFftHandler<f32>, FftHandler<f32>, FftHandler<f32>) {
    (
        R2cFftHandler::<f32>::new(nx),
        FftHandler::<f32>::new(ny),
        FftHandler::<f32>::new(nz),
    )
}

/// 3D real-to-complex FFT: (nz, ny, nx) -> (nz, ny, nx/2+1).
/// Applies r2c along axis 2 (X), then c2c along axis 1 (Y), then axis 0 (Z).
/// Forward transform is unnormalized (matches numpy rfftn).
pub fn rfftn3(
    input: &Array3<f32>,
    h_x: &mut R2cFftHandler<f32>,
    h_y: &mut FftHandler<f32>,
    h_z: &mut FftHandler<f32>,
) -> Array3<Complex<f32>> {
    let (nz, ny, nx) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let nx_ft = nx / 2 + 1;

    let mut ft_x = Array3::<Complex<f32>>::zeros((nz, ny, nx_ft));
    ndfft_r2c(input, &mut ft_x, h_x, 2);

    let mut ft_xy = Array3::<Complex<f32>>::zeros((nz, ny, nx_ft));
    ndfft(&ft_x, &mut ft_xy, h_y, 1);

    let mut ft_xyz = Array3::<Complex<f32>>::zeros((nz, ny, nx_ft));
    ndfft(&ft_xy, &mut ft_xyz, h_z, 0);

    ft_xyz
}

/// 3D complex-to-real inverse FFT: (nz, ny, nx/2+1) -> (nz, ny, nx).
/// Normalization: each axis divides by its length, giving net 1/(nz*ny*nx).
/// Matches numpy irfftn with norm="backward" (norm=1 in project convention).
#[cfg(test)]
pub fn irfftn3(
    input: &Array3<Complex<f32>>,
    nx: usize,
    h_x: &mut R2cFftHandler<f32>,
    h_y: &mut FftHandler<f32>,
    h_z: &mut FftHandler<f32>,
) -> Array3<f32> {
    let (nz, ny, nx_ft) = (input.shape()[0], input.shape()[1], input.shape()[2]);

    let mut tmp_z = Array3::<Complex<f32>>::zeros((nz, ny, nx_ft));
    ndifft(input, &mut tmp_z, h_z, 0);

    let mut tmp_zy = Array3::<Complex<f32>>::zeros((nz, ny, nx_ft));
    ndifft(&tmp_z, &mut tmp_zy, h_y, 1);

    let mut out = Array3::<f32>::zeros((nz, ny, nx));
    ndifft_r2c(&tmp_zy, &mut out, h_x, 2);

    out
}

/// In-place 3D real-to-complex FFT using caller-provided workspace buffers.
///
/// Result is left in `ft_trans_z_out` in **zy-last** form `[ny, nx_ft, nz]`
/// (z occupies axis 2, contiguous), so that the caller can apply the spectral
/// multiply and feed directly into `irfftn3_into` without an extra transpose.
///
/// Performs 2 bulk transposes (vs. 3 in the old back-to-canonical version):
///   r2c  on axis 2 → ft_out [nz, ny, nx_ft]
///   assign ft_out →[2,0,1]→ ft_trans_y [nx_ft, nz, ny]; FFT-y on axis 2
///   assign         →[2,0,1]→ ft_trans_z [ny, nx_ft, nz]; FFT-z on axis 2 → ft_trans_z_out
fn rfftn3_into(
    input: &Array3<f32>,
    h_x: &mut R2cFftHandler<f32>,
    h_y: &mut FftHandler<f32>,
    h_z: &mut FftHandler<f32>,
    ft_out:         &mut Array3<Complex<f32>>,  // [nz, ny, nx_ft]
    ft_trans_y:     &mut Array3<Complex<f32>>,  // [nx_ft, nz, ny]
    ft_trans_y_out: &mut Array3<Complex<f32>>,  // [nx_ft, nz, ny]
    ft_trans_z:     &mut Array3<Complex<f32>>,  // [ny, nx_ft, nz]
    ft_trans_z_out: &mut Array3<Complex<f32>>,  // [ny, nx_ft, nz]
) {
    // Step 1: r2c along x (axis 2, already contiguous) → ft_out: [nz, ny, nx_ft]
    ndfft_r2c(input, ft_out, h_x, 2);

    // Step 2: y FFT — transpose to put y (length ny) at axis 2 (contiguous)
    // [nz, ny, nx_ft] -[2,0,1]-> [nx_ft, nz, ny]
    ft_trans_y.assign(&ft_out.view().permuted_axes([2, 0, 1]));
    ndfft(ft_trans_y, ft_trans_y_out, h_y, 2);

    // Step 3: z FFT — transpose to put z (length nz) at axis 2 (contiguous)
    // [nx_ft, nz, ny] -[2,0,1]-> [ny, nx_ft, nz]
    ft_trans_z.assign(&ft_trans_y_out.view().permuted_axes([2, 0, 1]));
    ndfft(ft_trans_z, ft_trans_z_out, h_z, 2);

    // Step 4: transpose back to canonical layout [nz, ny, nx_ft]
    // [ny, nx_ft, nz] -[2,0,1]-> [nz, ny, nx_ft]
    ft_out.assign(&ft_trans_z_out.view().permuted_axes([2, 0, 1]));
}

/// In-place 3D complex-to-real inverse FFT using caller-provided workspace buffers.
///
/// Mirrors rfftn3_into using inverse permutation [1,2,0]:
///   [nz, ny, nx_ft] -[1,2,0]-> [ny, nx_ft, nz]; IFFT on axis 2 (z)
///   [ny, nx_ft, nz] -[1,2,0]-> [nx_ft, nz, ny]; IFFT on axis 2 (y)
///   [nx_ft, nz, ny] -[1,2,0]-> [nz, ny, nx_ft]; c2r on axis 2 (x)
fn irfftn3_into(
    input:          &Array3<Complex<f32>>,      // [nz, ny, nx_ft]
    h_x: &mut R2cFftHandler<f32>,
    h_y: &mut FftHandler<f32>,
    h_z: &mut FftHandler<f32>,
    ft_trans_z:     &mut Array3<Complex<f32>>,  // [ny, nx_ft, nz]
    ft_trans_z_out: &mut Array3<Complex<f32>>,  // [ny, nx_ft, nz]
    ft_trans_y:     &mut Array3<Complex<f32>>,  // [nx_ft, nz, ny]
    ft_trans_y_out: &mut Array3<Complex<f32>>,  // [nx_ft, nz, ny]
    ft_back:        &mut Array3<Complex<f32>>,  // [nz, ny, nx_ft]
    out:            &mut Array3<f32>,
) {
    // Step 1: z IFFT — [nz, ny, nx_ft] -[1,2,0]-> [ny, nx_ft, nz]; IFFT on axis 2
    ft_trans_z.assign(&input.view().permuted_axes([1, 2, 0]));
    ndifft(ft_trans_z, ft_trans_z_out, h_z, 2);

    // Step 2: y IFFT — [ny, nx_ft, nz] -[1,2,0]-> [nx_ft, nz, ny]; IFFT on axis 2
    ft_trans_y.assign(&ft_trans_z_out.view().permuted_axes([1, 2, 0]));
    ndifft(ft_trans_y, ft_trans_y_out, h_y, 2);

    // Step 3: transpose back — [nx_ft, nz, ny] -[1,2,0]-> [nz, ny, nx_ft]
    ft_back.assign(&ft_trans_y_out.view().permuted_axes([1, 2, 0]));

    // Step 4: c2r along x (axis 2, contiguous)
    ndifft_r2c(ft_back, out, h_x, 2);
}

/// Reusable worker-local buffers for scan hot path.
///
/// FFT layout uses the "transpose-then-axis-2" trick to keep all ndrustfft
/// lane accesses contiguous:
///   [nz, ny, nx_ft] -permute[2,0,1]-> [nx_ft, nz, ny]  (y at axis 2)
///   [nx_ft, nz, ny] -permute[2,0,1]-> [ny, nx_ft, nz]  (z at axis 2)
///   [ny, nx_ft, nz] -permute[2,0,1]-> [nz, ny, nx_ft]  (back to canonical)
/// This replaces ~1440 small to_vec allocations per 3D FFT with a single
/// bulk ndarray::assign (strided copy), drastically reducing allocator load.
struct ScanWorkspace {
    rot_template: Array3<f32>,           // [nz, ny, nx]
    rot_mask: Array3<f32>,               // [nz, ny, nx]
    rot_mask2: Array3<f32>,              // [nz, ny, nx]

    // r2c output: [nz, ny, nx_ft]
    ft_out: Array3<Complex<f32>>,

    // Transposed buffers for y-axis FFT (y = axis 2, contiguous): [nx_ft, nz, ny]
    ft_trans_y:     Array3<Complex<f32>>,
    ft_trans_y_out: Array3<Complex<f32>>,

    // Transposed buffers for z-axis FFT (z = axis 2, contiguous): [ny, nx_ft, nz]
    ft_trans_z:     Array3<Complex<f32>>,
    ft_trans_z_out: Array3<Complex<f32>>,

    // Back-transposed intermediate before c2r step: [nz, ny, nx_ft]
    ft_back: Array3<Complex<f32>>,

    gcc: Array3<f32>,
    ave: Array3<f32>,
    ave2: Array3<f32>,
}

impl ScanWorkspace {
    fn new(shape: (usize, usize, usize)) -> Self {
        let (nz, ny, nx) = shape;
        let nx_ft = nx / 2 + 1;
        Self {
            rot_template: Array3::<f32>::zeros(shape),
            rot_mask:     Array3::<f32>::zeros(shape),
            rot_mask2:    Array3::<f32>::zeros(shape),
            ft_out:       Array3::<Complex<f32>>::zeros((nz, ny, nx_ft)),
            ft_trans_y:       Array3::<Complex<f32>>::zeros((nx_ft, nz, ny)),
            ft_trans_y_out:   Array3::<Complex<f32>>::zeros((nx_ft, nz, ny)),
            ft_trans_z:       Array3::<Complex<f32>>::zeros((ny, nx_ft, nz)),
            ft_trans_z_out:   Array3::<Complex<f32>>::zeros((ny, nx_ft, nz)),
            ft_back:      Array3::<Complex<f32>>::zeros((nz, ny, nx_ft)),
            gcc:  Array3::<f32>::zeros(shape),
            ave:  Array3::<f32>::zeros(shape),
            ave2: Array3::<f32>::zeros(shape),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal pipeline helpers
// ---------------------------------------------------------------------------

/// Normalize template: subtract mean within mask, divide by std, multiply by mask.
pub fn normalize_template(template: &Array3<f32>, mask: &Array3<f32>) -> Array3<f32> {
    let mut norm = template * mask;
    // Compute mean/std over masked voxels without building temporary vectors.
    let mut sum = 0.0_f32;
    let mut sumsq = 0.0_f32;
    let mut count = 0_usize;
    Zip::from(&norm).and(mask).for_each(|&v, &m| {
        if m != 0.0 {
            sum += v;
            sumsq += v * v;
            count += 1;
        }
    });

    if count == 0 {
        return norm;
    }

    let n = count as f32;
    let mean = sum / n;
    let variance = ((sumsq / n) - mean * mean).max(0.0);
    let std = variance.sqrt();

    Zip::from(&mut norm).and(mask).for_each(|v, &m| {
        if m != 0.0 {
            *v = (*v - mean) / std;
        } else {
            *v = 0.0;
        }
    });
    norm * mask
}

/// Compute normalization factor: count of non-zero mask voxels.
pub fn normalization_factor(mask: &Array3<f32>) -> f32 {
    mask.iter().filter(|&&v| v != 0.0).count() as f32
}

/// Compute LCC mask: voxels where target > max * 0.05.
pub fn lcc_mask(target: &Array3<f32>) -> Array3<bool> {
    let max_val = target.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_val * 0.05;
    target.mapv(|v| v > threshold)
}

/// Flattened memory-order indices for true voxels in a boolean mask.
pub fn mask_true_indices(mask: &Array3<bool>) -> Vec<usize> {
    match mask.as_slice_memory_order() {
        Some(s) => s
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect(),
        None => mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect(),
    }
}

/// Apply discrete Laplacian (finite-difference, wrap-around) to a 3D array.
pub fn laplace3d(input: &Array3<f32>) -> Array3<f32> {
    let (nz, ny, nx) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let mut out = Array3::<f32>::zeros((nz, ny, nx));
    for z in 0..nz {
        let zp = (z + 1) % nz;
        let zm = (z + nz - 1) % nz;
        for y in 0..ny {
            let yp = (y + 1) % ny;
            let ym = (y + ny - 1) % ny;
            for x in 0..nx {
                let xp = (x + 1) % nx;
                let xm = (x + nx - 1) % nx;
                let center = input[[z, y, x]];
                out[[z, y, x]] = input[[zp, y, x]]
                    + input[[zm, y, x]]
                    + input[[z, yp, x]]
                    + input[[z, ym, x]]
                    + input[[z, y, xp]]
                    + input[[z, y, xm]]
                    - 6.0 * center;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Internal rotation helper (pure Rust, no PyO3)
// ---------------------------------------------------------------------------

/// Rotate a single template+mask pair using the rotation matrix.
/// Mirrors the logic of `rotate_grid3d` / `rotate_grid3d_pair` from rotate.rs.
fn rotate_pair_internal_into(
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
                let off0z: isize = if z1 == 0 { grid_slice - grid_size } else { grid_slice };
                let off1z = off0z + if x1 == 0 { 1 - gs2 } else { 1 };
                let c01 = unsafe {
                    *template_raw.offset(grid_zyx + off0z) * dx1
                        + *template_raw.offset(grid_zyx + off1z) * dx
                };
                let mut off0zy = if z1 == 0 { grid_slice - grid_size } else { grid_slice };
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
fn rot_slice_to_mat(rot: &ndarray::ArrayView2<f32>) -> [[f32; 3]; 3] {
    [
        [rot[[0, 0]], rot[[0, 1]], rot[[0, 2]]],
        [rot[[1, 0]], rot[[1, 1]], rot[[1, 2]]],
        [rot[[2, 0]], rot[[2, 1]], rot[[2, 2]]],
    ]
}

// ---------------------------------------------------------------------------
// CpuRustCorrelator — PyO3 class
// ---------------------------------------------------------------------------

#[pyclass]
pub struct CpuRustCorrelator {
    // Shape info
    shape: (usize, usize, usize),
    laplace: bool,
    radius: i32,

    // Rotations: (n_rot, 3, 3)
    rotations: Array3<f32>,
    nproc: usize,

    // Precomputed
    template: Array3<f32>,
    mask: Array3<f32>,
    norm_factor: f32,
    target_ft: Array3<Complex<f32>>,
    target2_ft: Array3<Complex<f32>>,
    lcc_mask_arr: Array3<bool>,
    lcc_mask_indices: Vec<usize>,

    // Outputs
    lcc: Array3<f32>,
    rot: Array3<i32>,
}

#[pymethods]
impl CpuRustCorrelator {
    #[new]
    #[pyo3(signature = (target, template, rotations, mask, laplace, nproc))]
    pub fn new(
        target: PyReadonlyArray3<f32>,
        template: PyReadonlyArray3<f32>,
        rotations: PyReadonlyArray3<f32>,
        mask: PyReadonlyArray3<f32>,
        laplace: bool,
        nproc: usize,
    ) -> PyResult<Self> {
        let target = target.as_array().to_owned();
        let template = template.as_array().to_owned();
        let rotations = rotations.as_array().to_owned();
        let mask = mask.as_array().to_owned();

        let shape = (target.shape()[0], target.shape()[1], target.shape()[2]);
        let radius = shape.0.min(shape.1).min(shape.2) as i32 / 2;

        // Normalize target
        let max_val = target.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_val == 0.0 {
            return Err(PyValueError::new_err("Target is all zeros."));
        }
        let target_norm = target.mapv(|v| v / max_val);

        // Optional Laplace
        let target_filtered = if laplace {
            laplace3d(&target_norm)
        } else {
            target_norm.clone()
        };

        // LCC mask
        let lcc_mask_arr = lcc_mask(&target_norm);
        let lcc_mask_indices = mask_true_indices(&lcc_mask_arr);

        // Precompute FFTs of target
        let (mut hx, mut hy, mut hz) = make_fft_handlers(shape.0, shape.1, shape.2);
        let target_ft = rfftn3(&target_filtered, &mut hx, &mut hy, &mut hz);
        let target2 = target_filtered.mapv(|v| v * v);
        let target2_ft = rfftn3(&target2, &mut hx, &mut hy, &mut hz);

        let lcc = Array3::<f32>::zeros(shape);
        let rot = Array3::<i32>::zeros(shape);

        // set_template logic
        let norm_factor = normalization_factor(&mask);
        if norm_factor == 0.0 {
            return Err(PyValueError::new_err("Zero-filled mask is not allowed."));
        }

        let template_norm = if laplace {
            let t = laplace3d(&template);
            normalize_template(&t, &mask)
        } else {
            normalize_template(&template, &mask)
        };

        Ok(CpuRustCorrelator {
            shape,
            laplace,
            radius,
            rotations,
            nproc,
            template: template_norm,
            mask: mask.to_owned(),
            norm_factor,
            target_ft,
            target2_ft,
            lcc_mask_arr,
            lcc_mask_indices,
            lcc,
            rot,
        })
    }

    pub fn set_template(
        &mut self,
        template: PyReadonlyArray3<f32>,
        mask: PyReadonlyArray3<f32>,
    ) -> PyResult<()> {
        let template = template.as_array().to_owned();
        let mask = mask.as_array().to_owned();

        if template.shape() != [self.shape.0, self.shape.1, self.shape.2] {
            return Err(PyValueError::new_err(
                "Shape of template does not match the target.",
            ));
        }

        let norm_factor = normalization_factor(&mask);
        if norm_factor == 0.0 {
            return Err(PyValueError::new_err("Zero-filled mask is not allowed."));
        }

        self.norm_factor = norm_factor;
        self.template = if self.laplace {
            let t = laplace3d(&template);
            normalize_template(&t, &mask)
        } else {
            normalize_template(&template, &mask)
        };
        self.mask = mask;
        self.lcc.fill(0.0);
        self.rot.fill(0);
        Ok(())
    }

    pub fn scan(&mut self) -> PyResult<()> {
        let n_rot = self.rotations.shape()[0];
        let shape = self.shape;
        let radius = self.radius;

        // Reset outputs
        self.lcc.fill(0.0);
        self.rot.fill(0);

        if self.nproc <= 1 {
            // Serial path
            let (mut hx, mut hy, mut hz) = make_fft_handlers(shape.0, shape.1, shape.2);
            let mut lcc = Array3::<f32>::zeros(shape);
            let mut rot = Array3::<i32>::zeros(shape);
            let mut work = ScanWorkspace::new(shape);

            for n in 0..n_rot {
                let rotmat = rot_slice_to_mat(&self.rotations.slice(s![n, .., ..]));
                scan_one_rotation(
                    n,
                    &rotmat,
                    &self.template,
                    &self.mask,
                    &self.target_ft,
                    &self.target2_ft,
                    &self.lcc_mask_arr,
                    &self.lcc_mask_indices,
                    self.norm_factor,
                    radius,
                    &mut hx,
                    &mut hy,
                    &mut hz,
                    &mut work,
                    &mut lcc,
                    &mut rot,
                );
            }
            self.lcc = lcc;
            self.rot = rot;
        } else {
            // Parallel path: chunk rotations across nproc workers
            let nproc = self.nproc;
            let chunk_size = (n_rot + nproc - 1) / nproc;

            // Build owned copies of shared inputs for each worker
            let template = &self.template;
            let mask = &self.mask;
            let target_ft = &self.target_ft;
            let target2_ft = &self.target2_ft;
            let lcc_mask_arr = &self.lcc_mask_arr;
            let lcc_mask_indices = &self.lcc_mask_indices;
            let norm_factor = self.norm_factor;
            let rotations = &self.rotations;

            let (final_lcc, final_rot) = (0..nproc)
                .into_par_iter()
                .map(|worker| {
                    let start = worker * chunk_size;
                    if start >= n_rot {
                        return (
                            Array3::<f32>::zeros(shape),
                            Array3::<i32>::zeros(shape),
                        );
                    }
                    let end = (start + chunk_size).min(n_rot);
                    let (mut hx, mut hy, mut hz) =
                        make_fft_handlers(shape.0, shape.1, shape.2);
                    let mut lcc = Array3::<f32>::zeros(shape);
                    let mut rot = Array3::<i32>::zeros(shape);
                    let mut work = ScanWorkspace::new(shape);

                    for n in start..end {
                        let rotmat = rot_slice_to_mat(&rotations.slice(s![n, .., ..]));
                        scan_one_rotation(
                            n,
                            &rotmat,
                            template,
                            mask,
                            target_ft,
                            target2_ft,
                            lcc_mask_arr,
                            lcc_mask_indices,
                            norm_factor,
                            radius,
                            &mut hx,
                            &mut hy,
                            &mut hz,
                            &mut work,
                            &mut lcc,
                            &mut rot,
                        );
                    }
                    (lcc, rot)
                })
                .reduce(
                    || (Array3::<f32>::zeros(shape), Array3::<i32>::zeros(shape)),
                    |(mut acc_lcc, mut acc_rot), (lcc_w, rot_w)| {
                        merge_best_lcc_rot(&mut acc_lcc, &mut acc_rot, &lcc_w, &rot_w);
                        (acc_lcc, acc_rot)
                    },
                );
            self.lcc = final_lcc;
            self.rot = final_rot;
        }

        Ok(())
    }

    #[getter]
    pub fn lcc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        PyArray3::from_array(py, &self.lcc)
    }

    #[getter]
    pub fn rot<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<i32>> {
        PyArray3::from_array(py, &self.rot)
    }
}

// ---------------------------------------------------------------------------
// Per-rotation scan kernel (shared between serial and parallel paths)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn scan_one_rotation(
    n: usize,
    rotmat: &[[f32; 3]; 3],
    template: &Array3<f32>,
    mask: &Array3<f32>,
    target_ft: &Array3<Complex<f32>>,
    target2_ft: &Array3<Complex<f32>>,
    lcc_mask_arr: &Array3<bool>,
    lcc_mask_indices: &[usize],
    norm_factor: f32,
    radius: i32,
    hx: &mut R2cFftHandler<f32>,
    hy: &mut FftHandler<f32>,
    hz: &mut FftHandler<f32>,
    work: &mut ScanWorkspace,
    lcc: &mut Array3<f32>,
    rot: &mut Array3<i32>,
) {
    // Rotate template (trilinear) and mask (nearest-neighbor)
    rotate_pair_internal_into(
        template,
        mask,
        rotmat,
        radius,
        &mut work.rot_template,
        &mut work.rot_mask,
    );

    // GCC = irfftn( conj(rfftn(rot_template)) * target_ft )
    rfftn3_into(
        &work.rot_template,
        hx,
        hy,
        hz,
        &mut work.ft_out,
        &mut work.ft_trans_y,
        &mut work.ft_trans_y_out,
        &mut work.ft_trans_z,
        &mut work.ft_trans_z_out,
    );
    {
        let fo = work.ft_out.as_slice_memory_order_mut().unwrap();
        let tft = target_ft.as_slice_memory_order().unwrap();
        for i in 0..fo.len() {
            fo[i] = fo[i].conj() * tft[i];
        }
    }
    irfftn3_into(
        &work.ft_out,
        hx,
        hy,
        hz,
        &mut work.ft_trans_z,
        &mut work.ft_trans_z_out,
        &mut work.ft_trans_y,
        &mut work.ft_trans_y_out,
        &mut work.ft_back,
        &mut work.gcc,
    );

    // AVE = irfftn( conj(rfftn(rot_mask)) * target_ft )
    rfftn3_into(
        &work.rot_mask,
        hx,
        hy,
        hz,
        &mut work.ft_out,
        &mut work.ft_trans_y,
        &mut work.ft_trans_y_out,
        &mut work.ft_trans_z,
        &mut work.ft_trans_z_out,
    );
    {
        let fo = work.ft_out.as_slice_memory_order_mut().unwrap();
        let tft = target_ft.as_slice_memory_order().unwrap();
        for i in 0..fo.len() {
            fo[i] = fo[i].conj() * tft[i];
        }
    }
    irfftn3_into(
        &work.ft_out,
        hx,
        hy,
        hz,
        &mut work.ft_trans_z,
        &mut work.ft_trans_z_out,
        &mut work.ft_trans_y,
        &mut work.ft_trans_y_out,
        &mut work.ft_back,
        &mut work.ave,
    );

    // AVE2 = irfftn( conj(rfftn(rot_mask^2)) * target2_ft )
    Zip::from(&mut work.rot_mask2)
        .and(&work.rot_mask)
        .for_each(|out, &v| *out = v * v);
    rfftn3_into(
        &work.rot_mask2,
        hx,
        hy,
        hz,
        &mut work.ft_out,
        &mut work.ft_trans_y,
        &mut work.ft_trans_y_out,
        &mut work.ft_trans_z,
        &mut work.ft_trans_z_out,
    );
    {
        let fo = work.ft_out.as_slice_memory_order_mut().unwrap();
        let tft2 = target2_ft.as_slice_memory_order().unwrap();
        for i in 0..fo.len() {
            fo[i] = fo[i].conj() * tft2[i];
        }
    }
    irfftn3_into(
        &work.ft_out,
        hx,
        hy,
        hz,
        &mut work.ft_trans_z,
        &mut work.ft_trans_z_out,
        &mut work.ft_trans_y,
        &mut work.ft_trans_y_out,
        &mut work.ft_back,
        &mut work.ave2,
    );

    // LCC = gcc / sqrt(norm_factor*ave2 - ave^2), where lcc_mask != 0, else 0.
    // Fuse AVE2 normalization into this update to avoid an extra full-array pass.
    if let (
        Some(lcc_s),
        Some(rot_s),
        Some(gcc_s),
        Some(ave_s),
        Some(ave2_s),
        Some(mask_s),
    ) = (
        lcc.as_slice_memory_order_mut(),
        rot.as_slice_memory_order_mut(),
        work.gcc.as_slice_memory_order(),
        work.ave.as_slice_memory_order(),
        work.ave2.as_slice_memory_order(),
        lcc_mask_arr.as_slice_memory_order(),
    ) {
        for &i in lcc_mask_indices {
            debug_assert!(mask_s[i]);
            let ave2_v = ave2_s[i] * norm_factor;
            let var = ave2_v - ave_s[i] * ave_s[i];
            let lcc_val = if var > 0.0 {
                gcc_s[i] / var.sqrt()
            } else {
                0.0
            };
            if lcc_val > lcc_s[i] {
                lcc_s[i] = lcc_val;
                rot_s[i] = n as i32;
            }
        }
    } else {
        Zip::from(lcc)
            .and(rot)
            .and(&work.gcc)
            .and(&work.ave)
            .and(&work.ave2)
            .and(lcc_mask_arr)
            .for_each(|best_lcc, best_rot, &gcc_v, &ave_v, &ave2_v, &mask_v| {
                if mask_v {
                    let var = ave2_v * norm_factor - ave_v * ave_v;
                    let lcc_val = if var > 0.0 {
                        gcc_v / var.sqrt()
                    } else {
                        0.0
                    };
                    if lcc_val > *best_lcc {
                        *best_lcc = lcc_val;
                        *best_rot = n as i32;
                    }
                }
            });
    }
}

fn merge_best_lcc_rot(
    acc_lcc: &mut Array3<f32>,
    acc_rot: &mut Array3<i32>,
    lcc_w: &Array3<f32>,
    rot_w: &Array3<i32>,
) {
    if let (Some(acc_lcc_s), Some(acc_rot_s), Some(lcc_w_s), Some(rot_w_s)) = (
        acc_lcc.as_slice_memory_order_mut(),
        acc_rot.as_slice_memory_order_mut(),
        lcc_w.as_slice_memory_order(),
        rot_w.as_slice_memory_order(),
    ) {
        for i in 0..acc_lcc_s.len() {
            if lcc_w_s[i] > acc_lcc_s[i] {
                acc_lcc_s[i] = lcc_w_s[i];
                acc_rot_s[i] = rot_w_s[i];
            }
        }
    } else {
        Zip::from(acc_lcc)
            .and(acc_rot)
            .and(lcc_w)
            .and(rot_w)
            .for_each(|best_lcc, best_rot, &lcc_val, &rot_val| {
                if lcc_val > *best_lcc {
                    *best_lcc = lcc_val;
                    *best_rot = rot_val;
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    const ATOL: f32 = 1e-4;

    fn assert_allclose(a: &Array3<f32>, b: &Array3<f32>, atol: f32, label: &str) {
        let max_diff = Zip::from(a)
            .and(b)
            .map_collect(|&x, &y| (x - y).abs())
            .iter()
            .cloned()
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff <= atol,
            "{label}: max abs diff {max_diff} > atol {atol}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 1: rfftn3 round-trip
    // -----------------------------------------------------------------------
    #[test]
    fn test_rfftn3_roundtrip() {
        let mut input = Array3::<f32>::zeros((8, 8, 8));
        input[[2, 2, 2]] = 1.0;
        input[[3, 4, 5]] = 0.5;

        let (mut hx, mut hy, mut hz) = make_fft_handlers(8, 8, 8);
        let ft = rfftn3(&input, &mut hx, &mut hy, &mut hz);
        assert_eq!(ft.shape(), &[8, 8, 5]);

        let recovered = irfftn3(&ft, 8, &mut hx, &mut hy, &mut hz);
        assert_eq!(recovered.shape(), &[8, 8, 8]);
        assert_allclose(&input, &recovered, ATOL, "rfftn3 round-trip");
    }

    // -----------------------------------------------------------------------
    // Test 2: rfftn3 output shape matches (nz, ny, nx//2+1)
    // -----------------------------------------------------------------------
    #[test]
    fn test_rfftn3_output_shape() {
        let input = Array3::<f32>::zeros((4, 6, 8));
        let (mut hx, mut hy, mut hz) = make_fft_handlers(4, 6, 8);
        let ft = rfftn3(&input, &mut hx, &mut hy, &mut hz);
        assert_eq!(ft.shape(), &[4, 6, 5]); // 8//2+1 = 5
    }

    // -----------------------------------------------------------------------
    // Test 3: normalization factor counts non-zero mask voxels
    // -----------------------------------------------------------------------
    #[test]
    fn test_normalization_factor() {
        let mut mask = Array3::<f32>::zeros((4, 4, 4));
        mask[[1, 1, 1]] = 1.0;
        mask[[2, 2, 2]] = 1.0;
        mask[[3, 3, 3]] = 1.0;
        assert_eq!(normalization_factor(&mask), 3.0);
    }

    // -----------------------------------------------------------------------
    // Test 4: normalize_template output has zero mean and unit std within mask
    // -----------------------------------------------------------------------
    #[test]
    fn test_normalize_template() {
        let mut template = Array3::<f32>::zeros((4, 4, 4));
        let mut mask = Array3::<f32>::zeros((4, 4, 4));
        template[[1, 1, 1]] = 2.0;
        template[[1, 1, 2]] = 4.0;
        template[[1, 2, 1]] = 6.0;
        mask[[1, 1, 1]] = 1.0;
        mask[[1, 1, 2]] = 1.0;
        mask[[1, 2, 1]] = 1.0;

        let result = normalize_template(&template, &mask);

        // Collect values inside mask
        let vals: Vec<f32> = vec![result[[1, 1, 1]], result[[1, 1, 2]], result[[1, 2, 1]]];
        let mean = vals.iter().sum::<f32>() / 3.0;
        let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0).sqrt();
        assert!(mean.abs() < ATOL, "mean should be ~0, got {mean}");
        assert!((std - 1.0).abs() < ATOL, "std should be ~1, got {std}");

        // Values outside mask must be zero
        assert_eq!(result[[0, 0, 0]], 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 5: lcc_mask threshold is 5% of max
    // -----------------------------------------------------------------------
    #[test]
    fn test_lcc_mask_threshold() {
        let mut target = Array3::<f32>::zeros((4, 4, 4));
        target[[2, 2, 2]] = 1.0; // max
        target[[1, 1, 1]] = 0.06; // > 5% → inside mask
        target[[0, 0, 0]] = 0.04; // < 5% → outside mask

        let mask = lcc_mask(&target);
        assert!(mask[[2, 2, 2]]);
        assert!(mask[[1, 1, 1]]);
        assert!(!mask[[0, 0, 0]]);
    }

    // -----------------------------------------------------------------------
    // Test 6: laplace3d is approximately equivalent to scipy.ndimage.laplace
    // (wrap mode). We test that identity rotation gives laplace == 6-neighbour
    // finite difference.
    // -----------------------------------------------------------------------
    #[test]
    fn test_laplace3d_finite_difference() {
        let mut input = Array3::<f32>::zeros((4, 4, 4));
        input[[2, 2, 2]] = 1.0;
        let lap = laplace3d(&input);
        // center voxel surrounded by 6 zeros: lap = 0 + ... + 0 - 6*1 = -6
        assert!((lap[[2, 2, 2]] + 6.0).abs() < ATOL, "center: {}", lap[[2, 2, 2]]);
        // each immediate neighbour: 0 + 1 - 0 - 0 - 0 - 0 - 0 = +1
        assert!((lap[[1, 2, 2]] - 1.0).abs() < ATOL, "neighbour: {}", lap[[1, 2, 2]]);
    }

    // -----------------------------------------------------------------------
    // Test 7: scan() with identity rotation finds peak at origin
    //
    // A target with a blob centered at (0,0,0) means:
    //  - lcc_mask includes (0,0,0) (target > 5% of max)
    //  - identity rotation leaves the template unchanged
    //  - GCC peak is at shift (0,0,0) → LCC is high there
    // -----------------------------------------------------------------------
    #[test]
    fn test_scan_identity_rotation_finds_peak() {
        let mut target = Array3::<f32>::zeros((8, 8, 8));
        // blob at origin, using wrap-around neighbours for smooth falloff
        target[[0, 0, 0]] = 1.0;
        target[[0, 0, 1]] = 0.5;
        target[[0, 1, 0]] = 0.5;
        target[[1, 0, 0]] = 0.5;
        target[[0, 0, 7]] = 0.5; // x=-1 wrap
        target[[0, 7, 0]] = 0.5; // y=-1 wrap
        target[[7, 0, 0]] = 0.5; // z=-1 wrap
        let template = target.clone();
        // full-volume mask so template has non-zero std
        let mask = Array3::<f32>::ones((8, 8, 8));

        let mut rotations = Array3::<f32>::zeros((1, 3, 3));
        rotations[[0, 0, 0]] = 1.0;
        rotations[[0, 1, 1]] = 1.0;
        rotations[[0, 2, 2]] = 1.0;

        let (mut hx, mut hy, mut hz) = make_fft_handlers(8, 8, 8);
        let shape = (8usize, 8usize, 8usize);
        let radius = 4i32;

        let max_val = target.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let target_norm = target.mapv(|v| v / max_val);
        let lcc_mask_arr = lcc_mask(&target_norm);
        let target_ft = rfftn3(&target_norm, &mut hx, &mut hy, &mut hz);
        let target2 = target_norm.mapv(|v| v * v);
        let target2_ft = rfftn3(&target2, &mut hx, &mut hy, &mut hz);
        let norm_template = normalize_template(&template, &mask);
        let norm_factor = normalization_factor(&mask);

        // Confirm (0,0,0) is in lcc_mask
        assert!(lcc_mask_arr[[0, 0, 0]], "(0,0,0) must be in lcc_mask");

        let mut lcc = Array3::<f32>::zeros(shape);
        let mut rot = Array3::<i32>::zeros(shape);
        let rotmat = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let mut work = ScanWorkspace::new(shape);
        scan_one_rotation(
            0,
            &rotmat,
            &norm_template,
            &mask,
            &target_ft,
            &target2_ft,
            &lcc_mask_arr,
            &mask_true_indices(&lcc_mask_arr),
            norm_factor,
            radius,
            &mut hx,
            &mut hy,
            &mut hz,
            &mut work,
            &mut lcc,
            &mut rot,
        );

        let max_lcc = lcc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_lcc > 0.5, "Expected LCC peak > 0.5, got {max_lcc}");
    }

    // -----------------------------------------------------------------------
    // Test 8: nproc=1 and nproc=2 produce identical lcc and rot
    // -----------------------------------------------------------------------
    #[test]
    fn test_scan_nproc1_vs_nproc2_identical() {
        let mut target = Array3::<f32>::zeros((8, 8, 8));
        for z in 2..6usize {
            for y in 2..6usize {
                for x in 2..6usize {
                    target[[z, y, x]] = 1.0;
                }
        }
        }
        let template = target.clone();
        // full-volume mask so template has non-zero std
        let mask = Array3::<f32>::ones((8, 8, 8));
        let mut rotations = Array3::<f32>::zeros((2, 3, 3));
        // identity
        rotations[[0, 0, 0]] = 1.0;
        rotations[[0, 1, 1]] = 1.0;
        rotations[[0, 2, 2]] = 1.0;
        // 90° around Z
        rotations[[1, 0, 1]] = -1.0;
        rotations[[1, 1, 0]] = 1.0;
        rotations[[1, 2, 2]] = 1.0;

        let build_corr = |nproc: usize| {
            let (mut hx, mut hy, mut hz) = make_fft_handlers(8, 8, 8);
            let shape = (8usize, 8usize, 8usize);
            let radius = 4i32;
            let max_val = target.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let target_norm = target.mapv(|v| v / max_val);
            let lcc_mask_arr = lcc_mask(&target_norm);
            let target_ft = rfftn3(&target_norm, &mut hx, &mut hy, &mut hz);
            let target2 = target_norm.mapv(|v| v * v);
            let target2_ft = rfftn3(&target2, &mut hx, &mut hy, &mut hz);
            let norm_template = normalize_template(&template, &mask);
            let norm_factor = normalization_factor(&mask);
            let lcc_mask_indices = mask_true_indices(&lcc_mask_arr);

            let lcc = Array3::<f32>::zeros(shape);
            let rot = Array3::<i32>::zeros(shape);

            // Use the full scan via CpuRustCorrelator struct
            let mut corr = CpuRustCorrelator {
                shape,
                laplace: false,
                radius,
                rotations: rotations.clone(),
                nproc,
                template: norm_template,
                mask: mask.clone(),
                norm_factor,
                target_ft,
                target2_ft,
                lcc_mask_arr,
                lcc_mask_indices,
                lcc,
                rot,
            };
            corr.scan().unwrap();
            (corr.lcc.clone(), corr.rot.clone())
        };

        let (lcc1, rot1) = build_corr(1);
        let (lcc2, rot2) = build_corr(2);

        assert_allclose(&lcc1, &lcc2, ATOL, "lcc nproc=1 vs nproc=2");
        // rot should be identical
        let max_diff_rot = Zip::from(&rot1)
            .and(&rot2)
            .map_collect(|&a, &b| (a - b).abs())
            .iter()
            .cloned()
            .max()
            .unwrap_or(0);
        assert_eq!(max_diff_rot, 0, "rot arrays differ between nproc=1 and nproc=2");
    }
}
