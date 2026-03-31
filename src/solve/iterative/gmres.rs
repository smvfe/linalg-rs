use crate::core::numeric_utils::{arnoldi_process_step, givens_rotation_step};
use crate::core::{relative_tol, validate_linear_system};
use crate::dense::{Matrix, Vector};

use super::IterativeResult;

/// Solves `A*x = b` by restarted GMRES(m).
///
/// # Returns
/// - `Some(result)` for valid input dimensions.
/// - `None` for invalid input.
pub fn gmres_restarted(
    a: &Matrix<f64>,
    b: &Vector<f64>,
    x0: &Vector<f64>,
    restart: usize,
    max_iter: usize,
    tol: f64,
) -> Option<IterativeResult> {
    let _n = validate_linear_system(a, b, x0, max_iter, tol)?;
    if restart == 0 {
        return None;
    }

    let breakdown_tol = relative_tol(a).max(f64::EPSILON);

    let mut x = x0.clone();
    let mut iterations = 0usize;
    let mut final_delta = f64::INFINITY;

    while iterations < max_iter {
        let r = b - &(a * &x);
        let beta = r.norm_l2();
        final_delta = beta;

        if beta <= tol {
            return Some(IterativeResult {
                x,
                iterations,
                converged: true,
                final_delta,
            });
        }

        let cycle_len = restart.min(max_iter - iterations);
        let mut basis = Vec::with_capacity(cycle_len + 1);
        basis.push(&r * (1.0 / beta));

        let mut hessenberg = vec![vec![0.0_f64; cycle_len]; cycle_len + 1];
        let mut cosines = vec![0.0_f64; cycle_len];
        let mut sines = vec![0.0_f64; cycle_len];
        let mut g = vec![0.0_f64; cycle_len + 1];
        g[0] = beta;

        let mut x_cycle = x.clone();
        let mut used_steps = 0usize;

        for k in 0..cycle_len {
            let breakdown = arnoldi_process_step(a, &mut basis, &mut hessenberg, k, breakdown_tol);
            givens_rotation_step(
                &mut hessenberg,
                &mut cosines,
                &mut sines,
                &mut g,
                k,
                breakdown_tol,
            );

            used_steps = k + 1;
            final_delta = g[k + 1].abs();

            let mut y = vec![0.0_f64; used_steps];
            for idx in (0..used_steps).rev() {
                let mut rhs = g[idx];
                for j in (idx + 1)..used_steps {
                    rhs -= hessenberg[idx][j] * y[j];
                }

                let pivot = hessenberg[idx][idx];
                if pivot.abs() <= breakdown_tol {
                    return None;
                }
                y[idx] = rhs / pivot;
            }

            x_cycle = x.clone();
            for j in 0..used_steps {
                x_cycle += &(y[j] * &basis[j]);
            }

            if final_delta <= tol {
                return Some(IterativeResult {
                    x: x_cycle,
                    iterations: iterations + used_steps,
                    converged: true,
                    final_delta,
                });
            }

            if breakdown {
                break;
            }
        }

        if used_steps == 0 {
            break;
        }

        x = x_cycle;
        iterations += used_steps;
    }

    Some(IterativeResult {
        x,
        iterations,
        converged: false,
        final_delta,
    })
}

/// Solves `A*x = b` by GMRES without restart.
///
/// This is a convenience wrapper over [`gmres_restarted`] with
/// `restart = max_iter`.
pub fn gmres(
    a: &Matrix<f64>,
    b: &Vector<f64>,
    x0: &Vector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<IterativeResult> {
    gmres_restarted(a, b, x0, max_iter, max_iter, tol)
}
