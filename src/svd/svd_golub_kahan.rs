use crate::matrix::Matrix;

use super::types::SVD;

pub fn svd(a: &Matrix<f64>) -> Option<SVD> {
    SVD::new(a)
}

pub fn svd_golub_kahan(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<SVD> {
    super::internal::svd_golub_kahan(a, max_iter, tol)
}
