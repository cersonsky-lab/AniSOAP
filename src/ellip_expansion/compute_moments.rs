use numpy::ndarray::{Array3, ArrayView1, ArrayView2};
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;

/// Compute all moments <x^n0 y^n1 z^n2> for a general dilation matrix.
/// Since this computes moments for all n0, n1, and n2, and stores 0 for some
/// impossible configurations, it may not be memory-efficient.
/// However, this implementation allows simple access to all moments with
/// [n0, n1, n2] indexing like normal arrays.
///
/// # Arguments
/// * `dil_mat` - A symmetric, 3x3 matrix, given by np.ndarray from python side.
///               This function will return Err (exception on Python side) if
///               the matrix is not of size 3x3, not symmetric, or not invertible.
/// * `gau_cen` - A 3-dimensional vector for center of tri-variate Gaussian.
/// * `max_deg` - An integer that represents the maximum degree for which moments
///               must be computed. The given number must be positive; otherwise,
///               it will return Err (exception on Python side).
pub fn compute_moments_rust(
    dil_mat: ArrayView2<'_, f64>,
    gau_cen: ArrayView1<'_, f64>,
    max_deg: i32,
) -> PyResult<Array3<f64>> {
    // Check if the dilation matrix is a 3x3 matrix.
    if dil_mat.shape() != &[3, 3] {
        return Err(PyErr::new::<PyAssertionError, _>(
            "Dilation matrix needs to be 3x3",
        ));
    }

    // Check if the dilation matrix is symmetric
    for i in 0..3 {
        for j in 0..3 {
            if (dil_mat[[i, j]] - dil_mat[[j, i]]).powi(2) >= 1e-14 {
                return Err(PyErr::new::<PyAssertionError, _>(
                    "Dilation matrix needs to be symmetric",
                ));
            }
        }
    }

    if gau_cen.shape() != &[3] {
        return Err(PyErr::new::<PyAssertionError, _>(
            "Center of Gaussian has to be given by a 3-dim. vector.",
        ));
    }

    if max_deg <= 0 {
        return Err(PyErr::new::<PyAssertionError, _>(
            "The maximum degree needs to be at least 1.",
        ));
    }

    // Unpack three values of Gaussian centers, as they will be frequently
    // accessed while calculating moments.
    let (a0, a1, a2) = (gau_cen[0], gau_cen[1], gau_cen[2]);

    // [a, b, c] <- This is how general symmetric 3x3 matrix look like
    // [b, d, e]    and we only need 6 out of 9 values to compute entire
    // [c, e, f]    determinant and inverse.
    //              These values are cached on stack to remove frequent address
    //              lookups required for indexing
    let (a, b, c, d, e, f) = (
        dil_mat[[0, 0]],
        dil_mat[[0, 1]],
        dil_mat[[0, 2]],
        dil_mat[[1, 1]],
        dil_mat[[1, 2]],
        dil_mat[[2, 2]],
    );

    // cofNM is determinant of resulting matrix after removing N-th row and
    // M-th column, with appropriate sign of (-1)^(row + col)
    // (i.e. (N, M) co-factor matrix)
    let (cof00, cof01, cof02) = (d * f - e * e, c * e - b * f, b * e - c * d);

    // Determinant of entire dilation matrix
    let det = a * cof00 + b * cof01 + c * cof02;
    if det.abs() < 1e-14 {
        return Err(PyErr::new::<PyAssertionError, _>(
            "The given dilation matrix is singular.",
        ));
    }

    // Compute inverse; but since each we use coefficients a lot for moments
    // calculation, each elements will be stored as individual variables.
    let (cov00, cov01, cov02, cov11, cov12, cov22) = (
        cof00 / det, // Use pre-computed co-factors
        cof01 / det,
        cof02 / det,
        (a * f - c * c) / det, // Computed with co-factors
        (b * c - a * e) / det,
        (a * d - b * b) / det,
    );

    // Compute global_factor, a number that must be multiplied by before returning.
    // global_factor = (2 PI)^1.5 / SQRT(det|dil_mat|)
    //               = SQRT(8 PI^3 / det|dil_mat|)
    let global_factor = (8.0 * (std::f64::consts::PI).powi(3) / det).sqrt();

    // Prepare an empty array to store answers
    let max_deg = max_deg as usize;
    let mut moments = Array3::<f64>::zeros((max_deg + 1, max_deg + 1, max_deg + 1));

    // Initialize degree-1 elements
    moments[[0, 0, 0]] = 1.0;
    moments[[1, 0, 0]] = a0;
    moments[[0, 1, 0]] = a1;
    moments[[0, 0, 1]] = a2;

    if max_deg > 1 {
        // Initialize degree-2 elements
        moments[[2, 0, 0]] = cov00 + a0 * a0;
        moments[[0, 2, 0]] = cov11 + a1 * a1;
        moments[[0, 0, 2]] = cov22 + a2 * a2;
        moments[[1, 1, 0]] = cov01 + a0 * a1;
        moments[[0, 1, 1]] = cov12 + a1 * a2;
        moments[[1, 0, 1]] = cov02 + a0 * a2;
    }

    if max_deg > 2 {
        for deg in 2..max_deg {
            for n0 in 0..=deg {
                for n1 in 0..=(deg - n0) {
                    let n2 = deg - n0 - n1; // Forces n0 + n1 + n2 = deg
                    let (n0_pos, n1_pos, n2_pos) = (n0 > 0, n1 > 0, n2 > 0);
                    let x_iter_add =
                        0.0 + if n0_pos {
                            cov00 * n0 as f64 * moments[[n0 - 1, n1, n2]]
                        } else {
                            0.0
                        } + if n1_pos {
                            cov01 * n1 as f64 * moments[[n0, n1 - 1, n2]]
                        } else {
                            0.0
                        } + if n2_pos {
                            cov02 * n2 as f64 * moments[[n0, n1, n2 - 1]]
                        } else {
                            0.0
                        };

                    // Run the x-iteration
                    moments[[n0 + 1, n1, n2]] = a0 * moments[[n0, n1, n2]] + x_iter_add;

                    // Run y-iteration if n0 is 0.
                    if !n0_pos {
                        let y_iter_add =
                            0.0 + if n1_pos {
                                cov11 * n1 as f64 * moments[[n0, n1 - 1, n2]]
                            } else {
                                0.0
                            } + if n2_pos {
                                cov12 * n2 as f64 * moments[[n0, n1, n2 - 1]]
                            } else {
                                0.0
                            };
                        moments[[n0, n1 + 1, n2]] = a1 * moments[[n0, n1, n2]] + y_iter_add;

                        // Run z-iteration if both n0 and n1 are 0.
                        if !n1_pos {
                            moments[[n0, n1, n2 + 1]] = a2 * moments[[n0, n1, n2]]
                                + if n2_pos {
                                    cov22 * n2 as f64 * moments[[n0, n1, n2 - 1]]
                                } else {
                                    0.0
                                }
                        }
                    }
                }
            }
        }
    }

    Ok(moments * global_factor)
}
