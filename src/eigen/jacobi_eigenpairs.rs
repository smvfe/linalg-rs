use crate::matrix::Matrix;

use super::types::JacobiEigenResult;

pub fn jacobi_eigenpairs(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<JacobiEigenResult> {
    super::internal::jacobi_eigenpairs(a, max_iter, tol)
}
