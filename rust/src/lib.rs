mod correlator;
mod helpers;
mod rotate;

use pyo3::prelude::*;

#[pyfunction]
fn cargo_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
mod powerfitrs {
    use pyo3::types::PyModuleMethods;

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        m.add_function(pyo3::wrap_pyfunction!(crate::cargo_version, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }

    #[pymodule_export]
    use super::rotate::rotate_grid3d;
    #[pymodule_export]
    use super::rotate::rotate_grid3d_pair;
    #[pymodule_export]
    use super::helpers::blur_points;
    #[pymodule_export]
    use super::helpers::dilate_points;
    #[pymodule_export]
    use super::correlator::CpuRustCorrelator;
}
