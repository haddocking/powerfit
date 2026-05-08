use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Convolve point atoms onto a 3D grid with a Gaussian kernel.
/// points: shape (3, n_atoms), row order: x, y, z coordinates.
/// out is modified in-place.
#[pyfunction]
pub fn blur_points<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    sigma: f64,
    out: &Bound<'py, PyArray3<f64>>,
    wraparound: bool,
) -> PyResult<()> {
    let points = points.as_array();
    let weights = weights.as_array();
    let mut out_arr = unsafe { out.as_array_mut() };

    if points.shape()[0] != 3 {
        return Err(PyValueError::new_err("points must have shape (3, n_atoms)"));
    }

    let s = out_arr.shape().to_owned();
    let sz = s[0] as isize;
    let sy = s[1] as isize;
    let sx = s[2] as isize;

    let extend = 4.0 * sigma;
    let extend2 = extend * extend;
    let dsigma2 = 2.0 * sigma * sigma;

    let (xmin_limit, ymin_limit, zmin_limit) = if wraparound {
        (-(sx - 1), -(sy - 1), -(sz - 1))
    } else {
        (0, 0, 0)
    };

    let n_atoms = points.shape()[1];
    let out_raw = out_arr.as_mut_ptr();

    for n in 0..n_atoms {
        let px = points[[0, n]];
        let py_p = points[[1, n]];
        let pz = points[[2, n]];
        let w = weights[n];

        let xmin = ((px - extend).ceil() as isize).max(xmin_limit);
        let ymin = ((py_p - extend).ceil() as isize).max(ymin_limit);
        let zmin = ((pz - extend).ceil() as isize).max(zmin_limit);
        let xmax = ((px + extend).floor() as isize).min(sx - 1);
        let ymax = ((py_p + extend).floor() as isize).min(sy - 1);
        let zmax = ((pz + extend).floor() as isize).min(sz - 1);

        for z in zmin..=zmax {
            let z2 = (z as f64 - pz).powi(2);
            let zi = z.rem_euclid(sz) as usize;
            for y in ymin..=ymax {
                let y2z2 = (y as f64 - py_p).powi(2) + z2;
                let yi = y.rem_euclid(sy) as usize;
                for x in xmin..=xmax {
                    let x2y2z2 = (x as f64 - px).powi(2) + y2z2;
                    if x2y2z2 <= extend2 {
                        let xi = x.rem_euclid(sx) as usize;
                        let idx = zi * s[1] * s[2] + yi * s[2] + xi;
                        unsafe {
                            *out_raw.add(idx) += w * (-x2y2z2 / dsigma2).exp();
                        }
                    }
                }
            }
        }
    }
    let _ = py;
    Ok(())
}

/// Dilate point atoms into binary sphere masks on a 3D grid.
/// points: shape (3, n_atoms), row order: x, y, z coordinates.
/// out is modified in-place (set to 1.0 within each sphere).
#[pyfunction]
pub fn dilate_points<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    out: &Bound<'py, PyArray3<f64>>,
    wraparound: bool,
) -> PyResult<()> {
    let points = points.as_array();
    let radii = radii.as_array();
    let mut out_arr = unsafe { out.as_array_mut() };

    if points.shape()[0] != 3 {
        return Err(PyValueError::new_err("points must have shape (3, n_atoms)"));
    }

    let s = out_arr.shape().to_owned();
    let sz = s[0] as isize;
    let sy = s[1] as isize;
    let sx = s[2] as isize;

    let (xmin_limit, ymin_limit, zmin_limit) = if wraparound {
        (-(sx - 1), -(sy - 1), -(sz - 1))
    } else {
        (0, 0, 0)
    };

    let n_atoms = points.shape()[1];
    let out_raw = out_arr.as_mut_ptr();

    for n in 0..n_atoms {
        let px = points[[0, n]];
        let py_p = points[[1, n]];
        let pz = points[[2, n]];
        let radius = radii[n];
        let radius2 = radius * radius;

        let xmin = ((px - radius).ceil() as isize).max(xmin_limit);
        let ymin = ((py_p - radius).ceil() as isize).max(ymin_limit);
        let zmin = ((pz - radius).ceil() as isize).max(zmin_limit);
        let xmax = ((px + radius).floor() as isize).min(sx - 1) + 1;
        let ymax = ((py_p + radius).floor() as isize).min(sy - 1) + 1;
        let zmax = ((pz + radius).floor() as isize).min(sz - 1) + 1;

        for z in zmin..zmax {
            let z2 = (z as f64 - pz).powi(2);
            let zi = z.rem_euclid(sz) as usize;
            for y in ymin..ymax {
                let y2z2 = (y as f64 - py_p).powi(2) + z2;
                let yi = y.rem_euclid(sy) as usize;
                for x in xmin..xmax {
                    let x2y2z2 = (x as f64 - px).powi(2) + y2z2;
                    if x2y2z2 <= radius2 {
                        let xi = x.rem_euclid(sx) as usize;
                        let idx = zi * s[1] * s[2] + yi * s[2] + xi;
                        unsafe {
                            *out_raw.add(idx) = 1.0;
                        }
                    }
                }
            }
        }
    }
    let _ = py;
    Ok(())
}
