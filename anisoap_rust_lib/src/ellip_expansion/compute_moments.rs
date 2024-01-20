use numpy::ndarray::{Array3, ArrayView1, ArrayView2};

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
///
/// # Returns
/// A `Result` enum. `Ok` variant contains a 3-dimensional array (`Array3<f64>`)
/// of computed moments. The `Err(String)` is returned instead if there was an
/// error during parameter checks or computation. String inside the Err will
/// contain the error message.
pub fn compute_moments_rust(
    inv_dil_mat: ArrayView2<'_, f64>,
    gau_cen: ArrayView1<'_, f64>,
    max_deg: i32,
    det: f64,
) -> Result<Array3<f64>, String> {
    // Check if the dilation matrix is a 3x3 matrix.
    if inv_dil_mat.shape() != &[3, 3] {
        return Err("Inverse of dilation matrix needs to be 3x3".into());
    }

    // Check if gaussian center vector is 3-dimensional.
    if gau_cen.shape() != &[3] {
        return Err("Center of Gaussian has to be given by a 3-dim. vector.".into());
    }

    // Check if the dilation matrix is symmetric. We only need to check
    // [1, 0], [2, 0], and [2, 1] equal to [0, 1], [0, 2], and [1, 2] respectively.
    for i in 1..3 {
        for j in 0..i {
            // Inverse from numpy.linalg.inv causes some error even when the
            // matrix before inversion was perfectly symmetric. Setting tolerance
            // too small (ex: 1e-10) will cause error here.
            if (inv_dil_mat[[i, j]] - inv_dil_mat[[j, i]]).abs()
                > (1e-6 * inv_dil_mat[[i, j]]).abs()
            {
                return Err("Inverse of dilation matrix needs to be symmetric".into());
            }
        }
    }

    // Check if max_deg is strictly positive.
    if max_deg <= 0 {
        return Err("The maximum degree needs to be at least 1.".into());
    }

    // Check if determinant is strictly positive.
    if det <= 0.0 {
        return Err(
            "The matrix must be positive definite (thus must have positive determinant)".into(),
        );
    }

    Ok(compute_moments_rust_unchecked(
        inv_dil_mat,
        gau_cen,
        max_deg,
        det,
    ))
}

/// Contains the main computation logic for computing moments. The only difference
/// between checked version and unchecked version is that the unchecked version
/// will not check for any of the necessary conditions except checking whether
/// the given matrix is invertible.
/// This function is only intended to be used internally (not exposed to the FFI).
///
/// # Arguments
/// * `dil_mat` - Assumed to be a symmetric, 3x3 matrix, given by np.ndarray
///               from python side. This function will return Err (exception on
///               Python side) if the matrix is not invertible.
/// * `gau_cen` - A 3-dimensional vector for center of tri-variate Gaussian.
/// * `max_deg` - A positive integer that represents the maximum degree for
///               which moments must be computed.
///
/// # Returns
/// A `Result` enum. `Ok` variant contains a 3-dimensional array (`Array3<f64>`)
/// of computed moments. The `Err(String)` is returned instead if there was an
/// error during parameter checks or computation. String inside the Err will
/// contain the error message.
pub fn compute_moments_rust_unchecked(
    inv_dil_mat: ArrayView2<'_, f64>,
    gau_cen: ArrayView1<'_, f64>,
    max_deg: i32,
    det: f64,
) -> Array3<f64> {
    // Unpack three values of Gaussian centers, as they will be frequently
    // accessed while calculating moments.
    let (a0, a1, a2) = (gau_cen[0], gau_cen[1], gau_cen[2]);

    // Compute the determinant and inverses of the symmetric 3 x 3 matrix.
    let (inv00, inv01, inv02, inv11, inv12, inv22) = (
        inv_dil_mat[[0, 0]],
        inv_dil_mat[[0, 1]],
        inv_dil_mat[[0, 2]],
        inv_dil_mat[[1, 1]],
        inv_dil_mat[[1, 2]],
        inv_dil_mat[[2, 2]],
    );

    // Compute global_factor, a number that must be multiplied by before returning.
    // global_factor = (2 PI)^1.5 / SQRT(det|dil_mat|) = SQRT(8 PI^3 / det|dil_mat|)
    let global_factor = (8.0 * (std::f64::consts::PI).powi(3) / det).sqrt();

    // Prepare an empty array to store answers
    let max_deg = max_deg as usize;
    let mut moments = Array3::<f64>::zeros((max_deg + 1, max_deg + 1, max_deg + 1));

    // Initialize degree-0 and degree-1 elements
    moments[[0, 0, 0]] = 1.0;

    if max_deg >= 1 {
        moments[[1, 0, 0]] = a0;
        moments[[0, 1, 0]] = a1;
        moments[[0, 0, 1]] = a2;
    }

    if max_deg >= 2 {
        // Initialize degree-2 elements
        moments[[2, 0, 0]] = inv00 + a0 * a0;
        moments[[0, 2, 0]] = inv11 + a1 * a1;
        moments[[0, 0, 2]] = inv22 + a2 * a2;
        moments[[1, 1, 0]] = inv01 + a0 * a1;
        moments[[0, 1, 1]] = inv12 + a1 * a2;
        moments[[1, 0, 1]] = inv02 + a0 * a2;
    }

    // Compute the rest. Use iterative method to compute the rest. This for loop
    // is skipped if max_deg < 2.
    for deg in 2..max_deg {
        for n0 in 0..=deg {
            for n1 in 0..=(deg - n0) {
                let n2 = deg - n0 - n1; // Forces n0 + n1 + n2 = deg
                let flag = if n0 == 0 { 0 } else { 0b100 }
                    | if n1 == 0 { 0 } else { 0b010 }
                    | if n2 == 0 { 0 } else { 0b001 };

                // Memory access to current element is shared across all branches.
                let m_curr = moments[[n0, n1, n2]];
                match flag {
                    0 => {
                        // (0, 0, 0): Run all iterations
                        // Note this case is not likely to be encountered in the loop
                        // as n0 + n1 + n2 = deg != 0.
                        moments[[n0 + 1, n1, n2]] = a0 * m_curr;
                        moments[[n0, n1 + 1, n2]] = a1 * m_curr;
                        moments[[n0, n1, n2 + 1]] = a2 * m_curr;
                    }
                    1 => {
                        // (0, 0, +): run x, y, and z iterations
                        let z_mul = n2 as f64 * moments[[n0, n1, n2 - 1]];
                        moments[[n0 + 1, n1, n2]] = a0 * m_curr + inv02 * z_mul;
                        moments[[n0, n1 + 1, n2]] = a1 * m_curr + inv12 * z_mul;
                        moments[[n0, n1, n2 + 1]] = a2 * m_curr + inv22 * z_mul;
                    }
                    2 => {
                        // (0, +, 0): run x and y iterations.
                        let y_mul = n1 as f64 * moments[[n0, n1 - 1, n2]];
                        moments[[n0 + 1, n1, n2]] = a0 * m_curr + inv01 * y_mul;
                        moments[[n0, n1 + 1, n2]] = a1 * m_curr + inv11 * y_mul;
                    }
                    3 => {
                        // (0, +, +): run x and y iterations.
                        let y_mul = n1 as f64 * moments[[n0, n1 - 1, n2]];
                        let z_mul = n2 as f64 * moments[[n0, n1, n2 - 1]];
                        moments[[n0 + 1, n1, n2]] = a0 * m_curr + inv01 * y_mul + inv02 * z_mul;
                        moments[[n0, n1 + 1, n2]] = a1 * m_curr + inv11 * y_mul + inv12 * z_mul;
                    }
                    4 => {
                        // (+, 0, 0): run x iteration only.
                        moments[[n0 + 1, n1, n2]] =
                            a0 * m_curr + inv00 * n0 as f64 * moments[[n0 - 1, n1, n2]];
                    }
                    5 => {
                        // (+, 0, +): run x iteration only.
                        moments[[n0 + 1, n1, n2]] = a0 * m_curr
                            + inv00 * n0 as f64 * moments[[n0 - 1, n1, n2]]
                            + inv02 * n2 as f64 * moments[[n0, n1, n2 - 1]];
                    }
                    6 => {
                        // (+, +, 0): run x iteration only.
                        moments[[n0 + 1, n1, n2]] = a0 * m_curr
                            + inv00 * n0 as f64 * moments[[n0 - 1, n1, n2]]
                            + inv01 * n1 as f64 * moments[[n0, n1 - 1, n2]];
                    }
                    _ => {
                        // (+, +, +): run x iteration only.
                        moments[[n0 + 1, n1, n2]] = a0 * m_curr
                            + inv00 * n0 as f64 * moments[[n0 - 1, n1, n2]]
                            + inv01 * n1 as f64 * moments[[n0, n1 - 1, n2]]
                            + inv02 * n2 as f64 * moments[[n0, n1, n2 - 1]];
                    }
                }
            }
        }
    }

    // Return the final result.
    moments * global_factor
}
