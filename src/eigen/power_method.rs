use crate::matrix::Matrix;
use crate::vector::Vector;

use super::types::EigenResult;

pub fn power_method(
    a: &Matrix<f64>,
    x0: &Vector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<EigenResult> {
    super::internal::power_method(a, x0, max_iter, tol)
}
