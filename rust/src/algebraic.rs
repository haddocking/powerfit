//! Shared algebraic (fast-math-safe, stabilized Rust 1.98) float helpers.
//! Grant the compiler permission to reorder/fuse into an FMA, without the
//! UB-on-NaN/Inf risk of the older nightly-only `f*_fast` intrinsics.
use num_complex::Complex;

/// `a + b * c`
#[inline(always)]
pub(crate) fn fma_alg(a: f32, b: f32, c: f32) -> f32 {
    a.algebraic_add(b.algebraic_mul(c))
}

/// `a * wa + b * wb`
#[inline(always)]
pub(crate) fn blend_alg(a: f32, wa: f32, b: f32, wb: f32) -> f32 {
    a.algebraic_mul(wa).algebraic_add(b.algebraic_mul(wb))
}

/// `a * wa - b * wb`
#[inline(always)]
pub(crate) fn diff_alg(a: f32, wa: f32, b: f32, wb: f32) -> f32 {
    a.algebraic_mul(wa).algebraic_sub(b.algebraic_mul(wb))
}

/// `conj(a) * b`, using algebraic float ops for both the real and
/// imaginary parts instead of `num_complex::Complex`'s operator impls
/// (which we don't control and can't add algebraic-ops permission to).
#[inline(always)]
pub fn conj_mul_alg(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
    Complex::new(
        blend_alg(a.re, b.re, a.im, b.im),
        diff_alg(a.re, b.im, a.im, b.re),
    )
}
