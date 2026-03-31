use crate::matrix::Matrix;

use super::types::QrEigenpairsResult;

pub fn qr_wilkinson_eigenpairs(
    a: &Matrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<QrEigenpairsResult> {
    super::internal::qr_wilkinson_eigenpairs(a, max_iter, tol)
}
