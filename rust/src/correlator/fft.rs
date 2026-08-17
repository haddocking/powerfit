//! 3D real FFT primitives used by the correlator's scan hot path.
//!
//! `rfftn3`/`irfftn3` are the simple, allocating versions (used once at
//! `CpuRustCorrelator::new` time and in tests). `rfftn3_into`/`irfftn3_into`
//! are the zero-allocation versions driven by caller-provided scratch
//! buffers, used per-rotation inside `scan::scan_one_rotation`.

use ndarray::Array3;
use ndrustfft::{FftHandler, R2cFftHandler, ndfft, ndfft_r2c, ndifft, ndifft_r2c};
use num_complex::Complex;

/// Build reusable FFT handlers for a volume of shape (nz, ny, nx).
/// Returns (r2c_x, c2c_y, c2c_z).
pub fn make_fft_handlers(
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
fn irfftn3(
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

/// The three per-axis FFT handlers, grouped so functions that need all of
/// them don't have to take 3 separate `&mut` parameters.
pub struct FftHandlers<'a> {
    pub x: &'a mut R2cFftHandler<f32>,
    pub y: &'a mut FftHandler<f32>,
    pub z: &'a mut FftHandler<f32>,
}

/// Scratch buffers for `rfftn3_into`'s "transpose-then-axis-2" FFT trick
/// (see `scan::ScanWorkspace` docs). `ft_out` doubles as the function's
/// result, so it's kept separate from `InverseFftScratch` below: callers
/// read it back as `irfftn3_into`'s `input` right after, and the two borrows
/// must not overlap.
pub struct ForwardFftScratch<'a> {
    pub ft_out: &'a mut Array3<Complex<f32>>, // [nz, ny, nx_ft]
    pub ft_trans_y: &'a mut Array3<Complex<f32>>, // [nx_ft, nz, ny]
    pub ft_trans_y_out: &'a mut Array3<Complex<f32>>, // [nx_ft, nz, ny]
    pub ft_trans_z: &'a mut Array3<Complex<f32>>, // [ny, nx_ft, nz]
    pub ft_trans_z_out: &'a mut Array3<Complex<f32>>, // [ny, nx_ft, nz]
}

/// Scratch buffers for `irfftn3_into`'s inverse transpose trick. Reuses the
/// same `ft_trans_y*`/`ft_trans_z*` backing arrays as `ForwardFftScratch`
/// (never held at the same time), plus `ft_back` for the final transpose.
pub struct InverseFftScratch<'a> {
    pub ft_trans_z: &'a mut Array3<Complex<f32>>, // [ny, nx_ft, nz]
    pub ft_trans_z_out: &'a mut Array3<Complex<f32>>, // [ny, nx_ft, nz]
    pub ft_trans_y: &'a mut Array3<Complex<f32>>, // [nx_ft, nz, ny]
    pub ft_trans_y_out: &'a mut Array3<Complex<f32>>, // [nx_ft, nz, ny]
    pub ft_back: &'a mut Array3<Complex<f32>>,    // [nz, ny, nx_ft]
}

/// In-place 3D real-to-complex FFT using caller-provided workspace buffers.
///
/// Result is left in `scratch.ft_out` in **zy-last** form `[ny, nx_ft, nz]`
/// (z occupies axis 2, contiguous), so that the caller can apply the spectral
/// multiply and feed directly into `irfftn3_into` without an extra transpose.
///
/// Performs 2 bulk transposes (vs. 3 in the old back-to-canonical version):
///   r2c  on axis 2 → ft_out [nz, ny, nx_ft]
///   assign ft_out →[2,0,1]→ ft_trans_y [nx_ft, nz, ny]; FFT-y on axis 2
///   assign         →[2,0,1]→ ft_trans_z [ny, nx_ft, nz]; FFT-z on axis 2 → ft_trans_z_out
pub fn rfftn3_into(
    input: &Array3<f32>,
    handlers: &mut FftHandlers,
    scratch: &mut ForwardFftScratch,
) {
    // Step 1: r2c along x (axis 2, already contiguous) → ft_out: [nz, ny, nx_ft]
    ndfft_r2c(input, scratch.ft_out, handlers.x, 2);

    // Step 2: y FFT — transpose to put y (length ny) at axis 2 (contiguous)
    // [nz, ny, nx_ft] -[2,0,1]-> [nx_ft, nz, ny]
    scratch
        .ft_trans_y
        .assign(&scratch.ft_out.view().permuted_axes([2, 0, 1]));
    ndfft(scratch.ft_trans_y, scratch.ft_trans_y_out, handlers.y, 2);

    // Step 3: z FFT — transpose to put z (length nz) at axis 2 (contiguous)
    // [nx_ft, nz, ny] -[2,0,1]-> [ny, nx_ft, nz]
    scratch
        .ft_trans_z
        .assign(&scratch.ft_trans_y_out.view().permuted_axes([2, 0, 1]));
    ndfft(scratch.ft_trans_z, scratch.ft_trans_z_out, handlers.z, 2);

    // Step 4: transpose back to canonical layout [nz, ny, nx_ft]
    // [ny, nx_ft, nz] -[2,0,1]-> [nz, ny, nx_ft]
    scratch
        .ft_out
        .assign(&scratch.ft_trans_z_out.view().permuted_axes([2, 0, 1]));
}

/// In-place 3D complex-to-real inverse FFT using caller-provided workspace buffers.
///
/// Mirrors rfftn3_into using inverse permutation [1,2,0]:
///   [nz, ny, nx_ft] -[1,2,0]-> [ny, nx_ft, nz]; IFFT on axis 2 (z)
///   [ny, nx_ft, nz] -[1,2,0]-> [nx_ft, nz, ny]; IFFT on axis 2 (y)
///   [nx_ft, nz, ny] -[1,2,0]-> [nz, ny, nx_ft]; c2r on axis 2 (x)
pub fn irfftn3_into(
    input: &Array3<Complex<f32>>, // [nz, ny, nx_ft]
    handlers: &mut FftHandlers,
    scratch: &mut InverseFftScratch,
    out: &mut Array3<f32>,
) {
    // Step 1: z IFFT — [nz, ny, nx_ft] -[1,2,0]-> [ny, nx_ft, nz]; IFFT on axis 2
    scratch
        .ft_trans_z
        .assign(&input.view().permuted_axes([1, 2, 0]));
    ndifft(scratch.ft_trans_z, scratch.ft_trans_z_out, handlers.z, 2);

    // Step 2: y IFFT — [ny, nx_ft, nz] -[1,2,0]-> [nx_ft, nz, ny]; IFFT on axis 2
    scratch
        .ft_trans_y
        .assign(&scratch.ft_trans_z_out.view().permuted_axes([1, 2, 0]));
    ndifft(scratch.ft_trans_y, scratch.ft_trans_y_out, handlers.y, 2);

    // Step 3: transpose back — [nx_ft, nz, ny] -[1,2,0]-> [nz, ny, nx_ft]
    scratch
        .ft_back
        .assign(&scratch.ft_trans_y_out.view().permuted_axes([1, 2, 0]));

    // Step 4: c2r along x (axis 2, contiguous)
    ndifft_r2c(scratch.ft_back, out, handlers.x, 2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Zip};

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
    // Test: rfftn3 round-trip
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
    // Test: rfftn3 output shape matches (nz, ny, nx//2+1)
    // -----------------------------------------------------------------------
    #[test]
    fn test_rfftn3_output_shape() {
        let input = Array3::<f32>::zeros((4, 6, 8));
        let (mut hx, mut hy, mut hz) = make_fft_handlers(4, 6, 8);
        let ft = rfftn3(&input, &mut hx, &mut hy, &mut hz);
        assert_eq!(ft.shape(), &[4, 6, 5]); // 8//2+1 = 5
    }
}
