use crate::core::{check_nonzero_diagonal, validate_linear_system};
use crate::dense::{Matrix, Vector};

use super::IterativeResult;

/// Solves `A*x = b` by the Gauss-Seidel method.
///
/// # Returns
/// - `Some(result)` for valid input dimensions.
/// - `None` for invalid input or near-zero diagonal entries.
pub fn gauss_seidel(
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

    let mut x = x0.clone();
    let mut final_delta = f64::INFINITY;

    for iter in 1..=max_iter {
        let prev = x.clone();
        final_delta = 0.0;

        for i in 0..n {
            let mut sigma_lower = 0.0;
            for j in 0..i {
                sigma_lower += a.data[i * n + j] * x[j];
            }

            let mut sigma_upper = 0.0;
            for j in (i + 1)..n {
                sigma_upper += a.data[i * n + j] * prev[j];
            }

            let new_value = (b[i] - sigma_lower - sigma_upper) / a[(i, i)];
            final_delta = final_delta.max((new_value - prev[i]).abs());
            x[i] = new_value;
        }

        if final_delta <= tol {
            return Some(IterativeResult {
                x,
                iterations: iter,
                converged: true,
                final_delta,
            });
        }
    }

    Some(IterativeResult {
        x,
        iterations: max_iter,
        converged: false,
        final_delta,
    })
}
