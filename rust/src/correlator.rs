use ndarray::{Array3, s};
use num_complex::Complex;
use numpy::{PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

mod fft;
mod pipeline;
mod rotation;
mod scan;

use fft::{FftHandlers, make_fft_handlers, rfftn3};
use pipeline::{laplace3d, lcc_mask, mask_true_indices, normalization_factor, normalize_template};
use rotation::rot_slice_to_mat;
use scan::{
    RotationInput, ScanOutput, ScanWorkspace, TargetContext, merge_best_lcc_rot, scan_one_rotation,
};

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
            let target = TargetContext {
                target_ft: &self.target_ft,
                target2_ft: &self.target2_ft,
                lcc_mask_arr: &self.lcc_mask_arr,
                lcc_mask_indices: &self.lcc_mask_indices,
                norm_factor: self.norm_factor,
            };

            for n in 0..n_rot {
                let rotmat = rot_slice_to_mat(&self.rotations.slice(s![n, .., ..]));
                scan_one_rotation(
                    n,
                    RotationInput {
                        rotmat: &rotmat,
                        template: &self.template,
                        mask: &self.mask,
                    },
                    &target,
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
            }
            self.lcc = lcc;
            self.rot = rot;
        } else {
            // Parallel path: chunk rotations across nproc workers
            let nproc = self.nproc;
            let chunk_size = n_rot.div_ceil(nproc);

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
                        return (Array3::<f32>::zeros(shape), Array3::<i32>::zeros(shape));
                    }
                    let end = (start + chunk_size).min(n_rot);
                    let (mut hx, mut hy, mut hz) = make_fft_handlers(shape.0, shape.1, shape.2);
                    let mut lcc = Array3::<f32>::zeros(shape);
                    let mut rot = Array3::<i32>::zeros(shape);
                    let mut work = ScanWorkspace::new(shape);
                    let target = TargetContext {
                        target_ft,
                        target2_ft,
                        lcc_mask_arr,
                        lcc_mask_indices,
                        norm_factor,
                    };

                    for n in start..end {
                        let rotmat = rot_slice_to_mat(&rotations.slice(s![n, .., ..]));
                        scan_one_rotation(
                            n,
                            RotationInput {
                                rotmat: &rotmat,
                                template,
                                mask,
                            },
                            &target,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Zip;
    use numpy::PyArrayMethods;

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
    // Test: nproc=1 and nproc=2 produce identical lcc and rot
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

        // Drive the correlator entirely through its public API (`new` + `scan`
        // + the `lcc`/`rot` getters) rather than the private struct fields, so
        // this test doesn't need visibility into CpuRustCorrelator's
        // internals.
        let run = |nproc: usize| {
            Python::attach(|py| {
                let target_py = PyArray3::from_array(py, &target).readonly();
                let template_py = PyArray3::from_array(py, &template).readonly();
                let rotations_py = PyArray3::from_array(py, &rotations).readonly();
                let mask_py = PyArray3::from_array(py, &mask).readonly();

                let mut corr = CpuRustCorrelator::new(
                    target_py,
                    template_py,
                    rotations_py,
                    mask_py,
                    false,
                    nproc,
                )
                .unwrap();
                corr.scan().unwrap();
                let lcc = corr.lcc(py).readonly().as_array().to_owned();
                let rot = corr.rot(py).readonly().as_array().to_owned();
                (lcc, rot)
            })
        };

        let (lcc1, rot1) = run(1);
        let (lcc2, rot2) = run(2);

        assert_allclose(&lcc1, &lcc2, ATOL, "lcc nproc=1 vs nproc=2");
        // rot should be identical
        let max_diff_rot = Zip::from(&rot1)
            .and(&rot2)
            .map_collect(|&a, &b| (a - b).abs())
            .iter()
            .cloned()
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff_rot, 0,
            "rot arrays differ between nproc=1 and nproc=2"
        );
    }
}
