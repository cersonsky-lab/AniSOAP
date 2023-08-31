mod calc_utils;
mod ellip_expansion;

use ellip_expansion::compute_moments::compute_moments_rust;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyAssertionError, prelude::*};

#[pymodule]
fn anisoap_rust_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn compute_moments<'py>(
        py: Python<'py>,
        mat: PyReadonlyArray2<'_, f64>,
        g_vec: PyReadonlyArray1<'_, f64>,
        max_deg: i32,
    ) -> PyResult<&'py PyArray3<f64>> {
        match compute_moments_rust(mat.as_array(), g_vec.as_array(), max_deg) {
            Ok(moment_array) => return Ok(moment_array.into_pyarray(py)),
            Err(message) => return Err(PyErr::new::<PyAssertionError, String>(message)),
        }
    }

    Ok(())
}
