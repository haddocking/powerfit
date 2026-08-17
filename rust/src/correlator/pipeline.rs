//! Pure-math pieces of the LCC pipeline: template normalization, the LCC
//! mask, and the Laplace pre-filter. No FFT/PyO3 dependency.

use ndarray::{Array3, Zip};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    const ATOL: f32 = 1e-4;

    // -----------------------------------------------------------------------
    // Test: normalization factor counts non-zero mask voxels
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
    // Test: normalize_template output has zero mean and unit std within mask
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
    // Test: lcc_mask threshold is 5% of max
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
    // Test: laplace3d is approximately equivalent to scipy.ndimage.laplace
    // (wrap mode). We test that identity rotation gives laplace == 6-neighbour
    // finite difference.
    // -----------------------------------------------------------------------
    #[test]
    fn test_laplace3d_finite_difference() {
        let mut input = Array3::<f32>::zeros((4, 4, 4));
        input[[2, 2, 2]] = 1.0;
        let lap = laplace3d(&input);
        // center voxel surrounded by 6 zeros: lap = 0 + ... + 0 - 6*1 = -6
        assert!(
            (lap[[2, 2, 2]] + 6.0).abs() < ATOL,
            "center: {}",
            lap[[2, 2, 2]]
        );
        // each immediate neighbour: 0 + 1 - 0 - 0 - 0 - 0 - 0 = +1
        assert!(
            (lap[[1, 2, 2]] - 1.0).abs() < ATOL,
            "neighbour: {}",
            lap[[1, 2, 2]]
        );
    }
}
