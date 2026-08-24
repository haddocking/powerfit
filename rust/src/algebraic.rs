//! Shared algebraic (fast-math-safe, stabilized Rust 1.98) float helpers.
//! Grant the compiler permission to reorder/fuse into an FMA, without the
//! UB-on-NaN/Inf risk of the older nightly-only `f*_fast` intrinsics.

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
