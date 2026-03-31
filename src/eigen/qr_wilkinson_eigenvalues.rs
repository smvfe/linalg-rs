use crate::matrix::Matrix;

use super::types::QrEigenvaluesResult;

pub fn qr_wilkinson_eigenvalues(
    a: &Matrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<QrEigenvaluesResult> {
    super::internal::qr_wilkinson_eigenvalues(a, max_iter, tol)
}
