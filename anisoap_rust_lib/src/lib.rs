mod ellip_expansion;

use ellip_expansion::compute_moments::compute_moments_rust;
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// TODO: This is an example function. Delete later.
#[pyfunction]
fn count_doubles(val: &str) -> PyResult<u64> {
    let mut total: u64 = 0;

    for (c1, c2) in val.chars().zip(val.chars().skip(1)) {
        if c1 == c2 {
            total += 1;
        }
    }
    Ok(total)
}

/// TODO: This is an example function. Delete later.
/// Returns zero-matrix for singular matrices
fn mat_inverse_2_rust(mat: ArrayView2<'_, f64>) -> Array2<f64> {
    let det = mat[[0, 0]] * mat[[1, 1]] - mat[[0, 1]] * mat[[1, 0]];
    if det.abs() < 1e-14 {
        Array2::<f64>::zeros((2, 2))
    } else {
        Array2::<f64>::from_shape_fn((2, 2), |(i, j)| {
            if i == j {
                mat[[(i + 1) % 2, (j + 1) % 2]] / det
            } else {
                -mat[[i, j]] / det
            }
        })
    }
}

#[pymodule]
fn anisoap_rust_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_doubles, m)?)?;

    #[pyfn(m)]
    fn mat_inverse_2<'py>(
        py: Python<'py>,
        mat: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        if mat.shape() != &[2, 2] {
            Err(PyErr::new::<PyTypeError, _>("Matrix must be size 2x2"))
        } else {
            Ok(mat_inverse_2_rust(mat.as_array()).into_pyarray(py))
        }
    }

    #[pyfn(m)]
    fn compute_moments<'py>(
        py: Python<'py>,
        mat: PyReadonlyArray2<'_, f64>,
        g_vec: PyReadonlyArray1<'_, f64>,
        max_deg: i32,
    ) -> PyResult<&'py PyArray3<f64>> {
        Ok(compute_moments_rust(mat.as_array(), g_vec.as_array(), max_deg)?.into_pyarray(py))
    }

    Ok(())
}
