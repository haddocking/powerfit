//! Per-rotation scan kernel, shared between the serial and rayon-parallel
//! paths in `CpuRustCorrelator::scan`.

use ndarray::{Array3, Zip};
use num_complex::Complex;

use super::fft::{FftHandlers, ForwardFftScratch, InverseFftScratch, irfftn3_into, rfftn3_into};
use super::rotation::rotate_pair_internal_into;

/// Reusable worker-local buffers for scan hot path.
///
/// FFT layout uses the "transpose-then-axis-2" trick to keep all ndrustfft
/// lane accesses contiguous:
///   [nz, ny, nx_ft] -permute[2,0,1]-> [nx_ft, nz, ny]  (y at axis 2)
///   [nx_ft, nz, ny] -permute[2,0,1]-> [ny, nx_ft, nz]  (z at axis 2)
///   [ny, nx_ft, nz] -permute[2,0,1]-> [nz, ny, nx_ft]  (back to canonical)
/// This replaces ~1440 small to_vec allocations per 3D FFT with a single
/// bulk ndarray::assign (strided copy), drastically reducing allocator load.
pub struct ScanWorkspace {
    rot_template: Array3<f32>, // [nz, ny, nx]
    rot_mask: Array3<f32>,     // [nz, ny, nx]
    rot_mask2: Array3<f32>,    // [nz, ny, nx]

    // r2c output: [nz, ny, nx_ft]
    ft_out: Array3<Complex<f32>>,

    // Transposed buffers for y-axis FFT (y = axis 2, contiguous): [nx_ft, nz, ny]
    ft_trans_y: Array3<Complex<f32>>,
    ft_trans_y_out: Array3<Complex<f32>>,

    // Transposed buffers for z-axis FFT (z = axis 2, contiguous): [ny, nx_ft, nz]
    ft_trans_z: Array3<Complex<f32>>,
    ft_trans_z_out: Array3<Complex<f32>>,

    // Back-transposed intermediate before c2r step: [nz, ny, nx_ft]
    ft_back: Array3<Complex<f32>>,

    gcc: Array3<f32>,
    ave: Array3<f32>,
    ave2: Array3<f32>,
}

impl ScanWorkspace {
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let (nz, ny, nx) = shape;
        let nx_ft = nx / 2 + 1;
        Self {
            rot_template: Array3::<f32>::zeros(shape),
            rot_mask: Array3::<f32>::zeros(shape),
            rot_mask2: Array3::<f32>::zeros(shape),
            ft_out: Array3::<Complex<f32>>::zeros((nz, ny, nx_ft)),
            ft_trans_y: Array3::<Complex<f32>>::zeros((nx_ft, nz, ny)),
            ft_trans_y_out: Array3::<Complex<f32>>::zeros((nx_ft, nz, ny)),
            ft_trans_z: Array3::<Complex<f32>>::zeros((ny, nx_ft, nz)),
            ft_trans_z_out: Array3::<Complex<f32>>::zeros((ny, nx_ft, nz)),
            ft_back: Array3::<Complex<f32>>::zeros((nz, ny, nx_ft)),
            gcc: Array3::<f32>::zeros(shape),
            ave: Array3::<f32>::zeros(shape),
            ave2: Array3::<f32>::zeros(shape),
        }
    }
}

/// Rotation to apply this step, plus the un-rotated template/mask it acts on.
pub struct RotationInput<'a> {
    pub rotmat: &'a [[f32; 3]; 3],
    pub template: &'a Array3<f32>,
    pub mask: &'a Array3<f32>,
}

/// Target-derived data that stays fixed across all rotations in a scan.
pub struct TargetContext<'a> {
    pub target_ft: &'a Array3<Complex<f32>>,
    pub target2_ft: &'a Array3<Complex<f32>>,
    pub lcc_mask_arr: &'a Array3<bool>,
    pub lcc_mask_indices: &'a [usize],
    pub norm_factor: f32,
}

/// Running best LCC score / winning rotation index, updated in place.
pub struct ScanOutput<'a> {
    pub lcc: &'a mut Array3<f32>,
    pub rot: &'a mut Array3<i32>,
}

pub fn scan_one_rotation(
    n: usize,
    rotation: RotationInput,
    target: &TargetContext,
    radius: i32,
    handlers: &mut FftHandlers,
    work: &mut ScanWorkspace,
    output: &mut ScanOutput,
) {
    // Ref doi:10.3934/biophy.2015.2.73.
    // Equation 3: global cross-correlation (GCC).
    // Rotate template (trilinear) and mask (nearest-neighbor)
    rotate_pair_internal_into(
        rotation.template,
        rotation.mask,
        rotation.rotmat,
        radius,
        &mut work.rot_template,
        &mut work.rot_mask,
    );

    // GCC = irfftn( conj(rfftn(rot_template)) * target_ft )
    rfftn3_into(
        &work.rot_template,
        handlers,
        &mut ForwardFftScratch {
            ft_out: &mut work.ft_out,
            ft_trans_y: &mut work.ft_trans_y,
            ft_trans_y_out: &mut work.ft_trans_y_out,
            ft_trans_z: &mut work.ft_trans_z,
            ft_trans_z_out: &mut work.ft_trans_z_out,
        },
    );
    {
        let fo = work.ft_out.as_slice_memory_order_mut().unwrap();
        let tft = target.target_ft.as_slice_memory_order().unwrap();
        for i in 0..fo.len() {
            fo[i] = fo[i].conj() * tft[i];
        }
    }
    irfftn3_into(
        &work.ft_out,
        handlers,
        &mut InverseFftScratch {
            ft_trans_z: &mut work.ft_trans_z,
            ft_trans_z_out: &mut work.ft_trans_z_out,
            ft_trans_y: &mut work.ft_trans_y,
            ft_trans_y_out: &mut work.ft_trans_y_out,
            ft_back: &mut work.ft_back,
        },
        &mut work.gcc,
    );

    // Ref doi:10.3934/biophy.2015.2.73.
    // Equation 4: square of average core-weighted density.
    // AVE = irfftn( conj(rfftn(rot_mask)) * target_ft )
    rfftn3_into(
        &work.rot_mask,
        handlers,
        &mut ForwardFftScratch {
            ft_out: &mut work.ft_out,
            ft_trans_y: &mut work.ft_trans_y,
            ft_trans_y_out: &mut work.ft_trans_y_out,
            ft_trans_z: &mut work.ft_trans_z,
            ft_trans_z_out: &mut work.ft_trans_z_out,
        },
    );
    {
        let fo = work.ft_out.as_slice_memory_order_mut().unwrap();
        let tft = target.target_ft.as_slice_memory_order().unwrap();
        for i in 0..fo.len() {
            fo[i] = fo[i].conj() * tft[i];
        }
    }
    irfftn3_into(
        &work.ft_out,
        handlers,
        &mut InverseFftScratch {
            ft_trans_z: &mut work.ft_trans_z,
            ft_trans_z_out: &mut work.ft_trans_z_out,
            ft_trans_y: &mut work.ft_trans_y,
            ft_trans_y_out: &mut work.ft_trans_y_out,
            ft_back: &mut work.ft_back,
        },
        &mut work.ave,
    );

    // Ref doi:10.3934/biophy.2015.2.73.
    // Equation 5: average of squared core-weighted density.
    // AVE2 = irfftn( conj(rfftn(rot_mask^2)) * target2_ft )
    Zip::from(&mut work.rot_mask2)
        .and(&work.rot_mask)
        .for_each(|out, &v| *out = v * v);
    rfftn3_into(
        &work.rot_mask2,
        handlers,
        &mut ForwardFftScratch {
            ft_out: &mut work.ft_out,
            ft_trans_y: &mut work.ft_trans_y,
            ft_trans_y_out: &mut work.ft_trans_y_out,
            ft_trans_z: &mut work.ft_trans_z,
            ft_trans_z_out: &mut work.ft_trans_z_out,
        },
    );
    {
        let fo = work.ft_out.as_slice_memory_order_mut().unwrap();
        let tft2 = target.target2_ft.as_slice_memory_order().unwrap();
        for i in 0..fo.len() {
            fo[i] = fo[i].conj() * tft2[i];
        }
    }
    irfftn3_into(
        &work.ft_out,
        handlers,
        &mut InverseFftScratch {
            ft_trans_z: &mut work.ft_trans_z,
            ft_trans_z_out: &mut work.ft_trans_z_out,
            ft_trans_y: &mut work.ft_trans_y,
            ft_trans_y_out: &mut work.ft_trans_y_out,
            ft_back: &mut work.ft_back,
        },
        &mut work.ave2,
    );

    // Ref doi:10.3934/biophy.2015.2.73.
    // Equation 6: local cross-correlation (LCC) score.
    // LCC = gcc / sqrt(norm_factor*ave2 - ave^2), where lcc_mask != 0, else 0.
    // Fuse AVE2 normalization into this update to avoid an extra full-array pass.
    let norm_factor = target.norm_factor;
    if let (Some(lcc_s), Some(rot_s), Some(gcc_s), Some(ave_s), Some(ave2_s), Some(mask_s)) = (
        output.lcc.as_slice_memory_order_mut(),
        output.rot.as_slice_memory_order_mut(),
        work.gcc.as_slice_memory_order(),
        work.ave.as_slice_memory_order(),
        work.ave2.as_slice_memory_order(),
        target.lcc_mask_arr.as_slice_memory_order(),
    ) {
        for &i in target.lcc_mask_indices {
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
        Zip::from(&mut *output.lcc)
            .and(&mut *output.rot)
            .and(&work.gcc)
            .and(&work.ave)
            .and(&work.ave2)
            .and(target.lcc_mask_arr)
            .for_each(|best_lcc, best_rot, &gcc_v, &ave_v, &ave2_v, &mask_v| {
                if mask_v {
                    let var = ave2_v * norm_factor - ave_v * ave_v;
                    let lcc_val = if var > 0.0 { gcc_v / var.sqrt() } else { 0.0 };
                    if lcc_val > *best_lcc {
                        *best_lcc = lcc_val;
                        *best_rot = n as i32;
                    }
                }
            });
    }
}

pub fn merge_best_lcc_rot(
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    use super::super::fft::make_fft_handlers;
    use super::super::fft::rfftn3;
    use super::super::pipeline::{lcc_mask, mask_true_indices, normalization_factor, normalize_template};

    // -----------------------------------------------------------------------
    // Test: scan_one_rotation with identity rotation finds peak at origin
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
        let lcc_mask_indices = mask_true_indices(&lcc_mask_arr);
        scan_one_rotation(
            0,
            RotationInput {
                rotmat: &rotmat,
                template: &norm_template,
                mask: &mask,
            },
            &TargetContext {
                target_ft: &target_ft,
                target2_ft: &target2_ft,
                lcc_mask_arr: &lcc_mask_arr,
                lcc_mask_indices: &lcc_mask_indices,
                norm_factor,
            },
            radius,
            &mut FftHandlers {
                x: &mut hx,
                y: &mut hy,
                z: &mut hz,
            },
            &mut work,
            &mut ScanOutput {
                lcc: &mut lcc,
                rot: &mut rot,
            },
        );

        let max_lcc = lcc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_lcc > 0.5, "Expected LCC peak > 0.5, got {max_lcc}");
    }
}
