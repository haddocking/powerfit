mod algebraic;
mod correlator;
mod helpers;
mod rotate;

use pyo3::prelude::*;

/// The crate's `Cargo.toml` version, exposed to Python as `cargo_version()`
/// (distinct from `__version__`, which is set from the same value below).
#[pyfunction]
fn cargo_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Rust extension module backing `powerfit_em`: rigid-body fitting of
/// high-resolution structures into cryo-EM density maps. Exports
/// `CpuRustCorrelator` (the LCC scan engine) plus the standalone
/// `blur_points`/`dilate_points`/`rotate_grid3d`/`rotate_grid3d_pair`
/// pyfunctions used to build its inputs.
#[pymodule]
mod powerfit_rs {
    use pyo3::types::PyModuleMethods;

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        m.add_function(pyo3::wrap_pyfunction!(crate::cargo_version, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }

    #[pymodule_export]
    use super::correlator::CpuRustCorrelator;
    #[pymodule_export]
    use super::helpers::blur_points;
    #[pymodule_export]
    use super::helpers::dilate_points;
    #[pymodule_export]
    use super::rotate::rotate_grid3d;
    #[pymodule_export]
    use super::rotate::rotate_grid3d_pair;
}
