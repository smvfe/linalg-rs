use crate::matrix::Matrix;
use crate::vector::Vector;

/// Computes a scale-aware floating-point tolerance for matrix algorithms.
///
/// # Arguments
/// - `a`: Input matrix used to estimate numerical scale.
///
/// # Returns
/// Relative tolerance defined as `EPSILON * max(rows, cols) * max(|a_ij|, 1)`.
///
/// # Notes
/// This threshold is preferred over a fixed absolute epsilon because it adapts
/// to both matrix size and value magnitude.
pub fn relative_tol(a: &Matrix<f64>) -> f64 {
    let max_abs = a
        .as_slice()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);

    f64::EPSILON * (a.rows().max(a.cols()) as f64) * max_abs
}

pub fn validate_linear_system(
    a: &Matrix<f64>,
    b: &Vector<f64>,
    x0: &Vector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<usize> {
    if !a.is_square() {
        return None;
    }

    let n = a.rows();
    if b.dim() != n || x0.dim() != n || max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
        return None;
    }

    Some(n)
}

pub fn check_nonzero_diagonal(a: &Matrix<f64>) -> bool {
    let n = a.rows();
    let diag_tol = relative_tol(a).max(f64::EPSILON);
    for i in 0..n {
        if a[(i, i)].abs() <= diag_tol {
            return false;
        }
    }
    true
}

pub(crate) fn arnoldi_process_step(
    a: &Matrix<f64>,
    basis: &mut Vec<Vector<f64>>,
    hessenberg: &mut [Vec<f64>],
    k: usize,
    breakdown_tol: f64,
) -> bool {
    let mut w = a * &basis[k];

    for j in 0..=k {
        let hij = w.dot(&basis[j]);
        hessenberg[j][k] = hij;
        w = &w - &(hij * &basis[j]);
    }

    let h_next = w.norm_l2();
    hessenberg[k + 1][k] = h_next;
    if h_next > breakdown_tol {
        basis.push(&w * (1.0 / h_next));
        false
    } else {
        true
    }
}

pub(crate) fn givens_rotation_step(
    hessenberg: &mut [Vec<f64>],
    cosines: &mut [f64],
    sines: &mut [f64],
    g: &mut [f64],
    k: usize,
    breakdown_tol: f64,
) {
    for j in 0..k {
        let temp = cosines[j] * hessenberg[j][k] + sines[j] * hessenberg[j + 1][k];
        hessenberg[j + 1][k] = -sines[j] * hessenberg[j][k] + cosines[j] * hessenberg[j + 1][k];
        hessenberg[j][k] = temp;
    }

    let diag = hessenberg[k][k];
    let subdiag = hessenberg[k + 1][k];
    let denom = (diag * diag + subdiag * subdiag).sqrt();
    if denom > breakdown_tol {
        cosines[k] = diag / denom;
        sines[k] = subdiag / denom;
    } else {
        cosines[k] = 1.0;
        sines[k] = 0.0;
    }

    hessenberg[k][k] = cosines[k] * diag + sines[k] * subdiag;
    hessenberg[k + 1][k] = 0.0;

    let gk = g[k];
    g[k] = cosines[k] * gk;
    g[k + 1] = -sines[k] * gk;
}
