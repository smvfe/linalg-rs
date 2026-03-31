use crate::core::{check_nonzero_diagonal, validate_linear_system};
use crate::dense::{Matrix, Vector};

use super::IterativeResult;

/// Solves `A*x = b` by the Jacobi method.
///
/// # Returns
/// - `Some(result)` for valid input dimensions.
/// - `None` for invalid input or near-zero diagonal entries.
pub fn jacobi(
    a: &Matrix<f64>,
    b: &Vector<f64>,
    x0: &Vector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<IterativeResult> {
    let n = validate_linear_system(a, b, x0, max_iter, tol)?;
    if !check_nonzero_diagonal(a) {
        return None;
    }

    let mut x_prev = x0.clone();
    let mut x_next = Vector::zeros(n);
    let mut final_delta = f64::INFINITY;

    for iter in 1..=max_iter {
        for i in 0..n {
            let mut sigma = 0.0;
            for j in 0..n {
                if i != j {
                    sigma += a.data[i * n + j] * x_prev[j];
                }
            }
            x_next[i] = (b[i] - sigma) / a.data[i * n + i];
        }

        final_delta = 0.0;
        for i in 0..n {
            final_delta = final_delta.max((x_next[i] - x_prev[i]).abs());
        }

        if final_delta <= tol {
            return Some(IterativeResult {
                x: x_next,
                iterations: iter,
                converged: true,
                final_delta,
            });
        }

        std::mem::swap(&mut x_prev, &mut x_next);
    }

    Some(IterativeResult {
        x: x_prev,
        iterations: max_iter,
        converged: false,
        final_delta,
    })
}
