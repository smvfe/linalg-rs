use crate::core::validate_linear_system;
use crate::dense::{Matrix, Vector};

use super::IterativeResult;

/// Solves `A*x = b` by the simple iteration (Richardson) method:
/// `x_{k+1} = x_k + tau * (b - A*x_k)`.
///
/// # Returns
/// - `Some(result)` for valid input dimensions and finite `tau`.
/// - `None` for invalid input.
pub fn simple_iteration(
    a: &Matrix<f64>,
    b: &Vector<f64>,
    x0: &Vector<f64>,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> Option<IterativeResult> {
    let n = validate_linear_system(a, b, x0, max_iter, tol)?;
    if !tau.is_finite() {
        return None;
    }

    let mut x = x0.clone();
    let mut x_next = Vector::zeros(n);
    let mut final_delta = f64::INFINITY;

    for iter in 1..=max_iter {
        let ax = a * &x;
        final_delta = 0.0;

        for i in 0..n {
            let next = x[i] + tau * (b[i] - ax[i]);
            final_delta = final_delta.max((next - x[i]).abs());
            x_next[i] = next;
        }

        if final_delta <= tol {
            return Some(IterativeResult {
                x: x_next,
                iterations: iter,
                converged: true,
                final_delta,
            });
        }

        std::mem::swap(&mut x, &mut x_next);
    }

    Some(IterativeResult {
        x,
        iterations: max_iter,
        converged: false,
        final_delta,
    })
}
