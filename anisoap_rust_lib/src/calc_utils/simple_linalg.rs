use numpy::ndarray::{Array2, ArrayView2};

/// Computes the inverse of a general 3 x 3 matrix. However, this function will
/// *NOT* check for the necessary conditions (size of the matrix), as this
/// function is *NOT* intended to be used outside of the situation in which all
/// these conditions are already checked for.
///
/// # Arguments
/// * `mat` - A 3 x 3 matrix.
///
/// # Returns
/// A `Result` enum. `Ok` variant contains tuple `(det: f64, inv: Array2<f64>)`.
/// `det` is the determinant of the matrix, and `inv` is the inverse of the
/// provided matrix. The function will return `Err(String)` if the matrix is
/// singular, with string containing the basic error message.
#[allow(dead_code)] // Unused function for now.
pub fn mat33_inverse(mat: &ArrayView2<'_, f64>) -> Result<(f64, Array2<f64>), String> {
    let det = mat[[0, 0]] * (mat[[1, 1]] * mat[[2, 2]] - mat[[2, 1]] * mat[[1, 2]])
        - mat[[0, 1]] * (mat[[1, 0]] * mat[[2, 2]] - mat[[2, 0]] * mat[[1, 2]])
        + mat[[0, 2]] * (mat[[1, 0]] * mat[[2, 1]] - mat[[2, 0]] * mat[[1, 1]]);

    if det.abs() < 1e-14 {
        return Err("The given matrix is singular".into());
    }

    let mut inv = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let (cof_r1, cof_r2) = match i {
                0 => (1, 2),
                1 => (0, 2),
                2 => (0, 1),
                _ => (0, 0), // should not happen at all
            };
            let (cof_c1, cof_c2) = match j {
                0 => (1, 2),
                1 => (0, 2),
                2 => (0, 1),
                _ => (0, 0), // should not happen at all
            };

            inv[[i, j]] = (mat[[cof_r1, cof_c1]] * mat[[cof_r2, cof_c2]]
                - mat[[cof_r1, cof_c2]] * mat[[cof_r2, cof_c1]])
                / det;
        }
    }

    Ok((det, inv))
}

/// Computes the inverse of a 3 x 3 symmetric matrix. However, this function will
/// *NOT* check for the necessary conditions (size and symmetry of the matrix),
/// as this function is *NOT* intended to be used outside of the situation in
/// which all these conditions are already checked for.
///
/// # Arguments
/// * `mat` - A symmetric 3 x 3 matrix.
///
/// # Returns
/// A `Result` enum. `Ok` variant contains tuple `(det: f64, inv: [f64; 6])`.
/// `det` is the determinant of the matrix, and `inv` contains the inverse of
/// the matrix of indices (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), in order.
/// The function will return `Err(String)` if the matrix is singular, with string
/// containing the basic error message.
pub fn mat33_sym_inverse(mat: &ArrayView2<'_, f64>) -> Result<(f64, [f64; 6]), String> {
    // [a, b, c] <- This is how general symmetric 3x3 matrix look like
    // [b, d, e]    and we only need 6 out of 9 values to compute entire
    // [c, e, f]    determinant and inverse.
    //              These values are cached on stack to remove frequent address
    //              lookups required for indexing
    let (a, b, c, d, e, f) = (
        mat[[0, 0]],
        mat[[0, 1]],
        mat[[0, 2]],
        mat[[1, 1]],
        mat[[1, 2]],
        mat[[2, 2]],
    );

    // cofNM is determinant of resulting matrix after removing N-th row and
    // M-th column, with appropriate sign of (-1)^(row + col)
    // (i.e. (N, M) co-factor matrix)
    let (cof00, cof01, cof02) = (d * f - e * e, c * e - b * f, b * e - c * d);

    // Determinant of the symmetric matrix
    let det = a * cof00 + b * cof01 + c * cof02;

    if det.abs() < 1e-14 {
        return Err("The given matrix is singular.".into());
    }

    // Compute inverse; but since each we use coefficients a lot for moments
    // calculation, each elements will be stored as individual variables.
    let inverses = [
        cof00 / det, // Use pre-computed co-factors
        cof01 / det,
        cof02 / det,
        (a * f - c * c) / det, // Computed with co-factors
        (b * c - a * e) / det,
        (a * d - b * b) / det,
    ];

    Ok((det, inverses))
}
