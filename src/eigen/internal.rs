use crate::lu::lu;
use crate::matrix::Matrix;
use crate::numeric_utils::relative_tol;
use crate::vector::Vector;

const DEFAULT_EIGEN_MAX_ITER: usize = 1_000;
const DEFAULT_EIGEN_TOL: f64 = 1e-12;

/// Result of an eigenvalue iterative method.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenResult {
    pub eigenvalue: f64,
    pub eigenvector: Vector<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_delta: f64,
}

impl EigenResult {
    pub fn new_power(a: &Matrix<f64>, x0: &Vector<f64>) -> Option<Self> {
        power_method(a, x0, DEFAULT_EIGEN_MAX_ITER, DEFAULT_EIGEN_TOL)
    }

    pub fn new_power_with_params(
        a: &Matrix<f64>,
        x0: &Vector<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Option<Self> {
        power_method(a, x0, max_iter, tol)
    }

    pub fn new_inverse(a: &Matrix<f64>, x0: &Vector<f64>) -> Option<Self> {
        inverse_power_method(a, x0, DEFAULT_EIGEN_MAX_ITER, DEFAULT_EIGEN_TOL)
    }

    pub fn new_inverse_with_params(
        a: &Matrix<f64>,
        x0: &Vector<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Option<Self> {
        inverse_power_method(a, x0, max_iter, tol)
    }

    pub fn residual_norm(&self, a: &Matrix<f64>) -> f64 {
        residual_norm(a, &self.eigenvector, self.eigenvalue)
    }
}

fn validate_eigen_input(
    a: &Matrix<f64>,
    x0: &Vector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<usize> {
    if !a.is_square() {
        return None;
    }

    let n = a.rows();
    if n == 0 || x0.dim() != n || max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
        return None;
    }

    let x0_norm = x0.norm_l2();
    if !x0_norm.is_finite() || x0_norm <= f64::EPSILON {
        return None;
    }

    Some(n)
}

fn residual_norm(a: &Matrix<f64>, x: &Vector<f64>, lambda: f64) -> f64 {
    let ax = a * x;
    let lx = lambda * x;
    (&ax - &lx).norm_l2()
}

/// Computes dominant eigenpair by the power method.
///
/// # Returns
/// - `Some(result)` for valid input.
/// - `None` for invalid input or numerical breakdown.
pub fn power_method(
    a: &Matrix<f64>,
    x0: &Vector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<EigenResult> {
    validate_eigen_input(a, x0, max_iter, tol)?;

    let breakdown_tol = relative_tol(a).max(f64::EPSILON);
    let mut x = x0.normalize();
    let mut eigenvalue = x.dot(&(a * &x));
    let mut final_delta = f64::INFINITY;

    for iter in 1..=max_iter {
        let y = a * &x;
        let y_norm = y.norm_l2();
        if !y_norm.is_finite() || y_norm <= breakdown_tol {
            return None;
        }

        let x_next = &y * (1.0 / y_norm);
        eigenvalue = x_next.dot(&(a * &x_next));
        final_delta = residual_norm(a, &x_next, eigenvalue);

        if final_delta <= tol {
            return Some(EigenResult {
                eigenvalue,
                eigenvector: x_next,
                iterations: iter,
                converged: true,
                final_delta,
            });
        }

        x = x_next;
    }

    Some(EigenResult {
        eigenvalue,
        eigenvector: x,
        iterations: max_iter,
        converged: false,
        final_delta,
    })
}

/// Computes eigenpair nearest to zero by the inverse power method.
///
/// # Returns
/// - `Some(result)` for valid input and non-singular matrix.
/// - `None` for invalid input or numerical breakdown.
pub fn inverse_power_method(
    a: &Matrix<f64>,
    x0: &Vector<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<EigenResult> {
    validate_eigen_input(a, x0, max_iter, tol)?;

    let lu_decomposition = lu(a)?;
    let breakdown_tol = relative_tol(a).max(f64::EPSILON);

    let mut x = x0.normalize();
    let mut eigenvalue = x.dot(&(a * &x));
    let mut final_delta = f64::INFINITY;

    for iter in 1..=max_iter {
        let y = lu_decomposition.solve(&x)?;
        let y_norm = y.norm_l2();
        if !y_norm.is_finite() || y_norm <= breakdown_tol {
            return None;
        }

        let x_next = &y * (1.0 / y_norm);
        eigenvalue = x_next.dot(&(a * &x_next));
        final_delta = residual_norm(a, &x_next, eigenvalue);

        if final_delta <= tol {
            return Some(EigenResult {
                eigenvalue,
                eigenvector: x_next,
                iterations: iter,
                converged: true,
                final_delta,
            });
        }

        x = x_next;
    }

    Some(EigenResult {
        eigenvalue,
        eigenvector: x,
        iterations: max_iter,
        converged: false,
        final_delta,
    })
}

/// Result of QR eigenvalue-only algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct QrEigenvaluesResult {
    pub eigenvalues: Vector<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_delta: f64,
}

impl QrEigenvaluesResult {
    pub fn new(a: &Matrix<f64>) -> Option<Self> {
        qr_wilkinson_eigenvalues(a, DEFAULT_EIGEN_MAX_ITER, DEFAULT_EIGEN_TOL)
    }

    pub fn new_with_params(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<Self> {
        qr_wilkinson_eigenvalues(a, max_iter, tol)
    }

    pub fn diagonal_matrix(&self) -> Matrix<f64> {
        let n = self.eigenvalues.dim();
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            *d.get_mut(i, i) = self.eigenvalues[i];
        }
        d
    }
}

/// Result of QR eigenvalue/eigenvector algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct QrEigenpairsResult {
    pub eigenvalues: Vector<f64>,
    pub eigenvectors: Matrix<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_delta: f64,
}

impl QrEigenpairsResult {
    pub fn new(a: &Matrix<f64>) -> Option<Self> {
        qr_wilkinson_eigenpairs(a, DEFAULT_EIGEN_MAX_ITER, DEFAULT_EIGEN_TOL)
    }

    pub fn new_with_params(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<Self> {
        qr_wilkinson_eigenpairs(a, max_iter, tol)
    }

    pub fn diagonal_matrix(&self) -> Matrix<f64> {
        let n = self.eigenvalues.dim();
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            *d.get_mut(i, i) = self.eigenvalues[i];
        }
        d
    }

    pub fn reconstruct_symmetric(&self) -> Matrix<f64> {
        let d = self.diagonal_matrix();
        let vd = &self.eigenvectors * &d;
        &vd * &self.eigenvectors.transpose()
    }
}

/// Result of Jacobi eigenvalue/eigenvector method.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobiEigenResult {
    pub eigenvalues: Vector<f64>,
    pub eigenvectors: Matrix<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_delta: f64,
}

impl JacobiEigenResult {
    pub fn new(a: &Matrix<f64>) -> Option<Self> {
        jacobi_eigenpairs(a, DEFAULT_EIGEN_MAX_ITER, DEFAULT_EIGEN_TOL)
    }

    pub fn new_with_params(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<Self> {
        jacobi_eigenpairs(a, max_iter, tol)
    }

    pub fn diagonal_matrix(&self) -> Matrix<f64> {
        let n = self.eigenvalues.dim();
        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            *d.get_mut(i, i) = self.eigenvalues[i];
        }
        d
    }

    pub fn reconstruct_symmetric(&self) -> Matrix<f64> {
        let d = self.diagonal_matrix();
        let vd = &self.eigenvectors * &d;
        &vd * &self.eigenvectors.transpose()
    }
}

fn validate_qr_input(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<usize> {
    if !a.is_square() {
        return None;
    }

    let n = a.rows();
    if n == 0 || max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
        return None;
    }

    Some(n)
}

fn is_symmetric(a: &Matrix<f64>, tol: f64) -> bool {
    let n = a.rows();
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[(i, j)] - a[(j, i)]).abs() > tol {
                return false;
            }
        }
    }

    true
}

fn hessenberg_subdiag_norm(h: &Matrix<f64>) -> f64 {
    let n = h.rows();
    let mut acc = 0.0_f64;
    for i in 1..n {
        acc = acc.max(h[(i, i - 1)].abs());
    }
    acc
}

fn wilkinson_shift(h: &Matrix<f64>, m: usize) -> f64 {
    let a = h[(m - 1, m - 1)];
    let b = h[(m - 1, m)];
    let c = h[(m, m - 1)];
    let d = h[(m, m)];

    let trace_half = 0.5 * (a + d);
    let det = a * d - b * c;
    let disc = trace_half * trace_half - det;
    if disc < 0.0 {
        return d;
    }

    let sqrt_disc = disc.sqrt();
    let mu1 = trace_half + sqrt_disc;
    let mu2 = trace_half - sqrt_disc;

    if (mu1 - d).abs() < (mu2 - d).abs() {
        mu1
    } else {
        mu2
    }
}

fn hessenberg_reduction(a: &Matrix<f64>, tol: f64) -> (Matrix<f64>, Matrix<f64>) {
    let n = a.rows();
    let mut h = a.clone();
    let mut q = Matrix::identity(n);

    for k in 0..n.saturating_sub(2) {
        let m = n - k - 1;
        let mut v = vec![0.0_f64; m];
        let mut x_norm_sq = 0.0_f64;

        for i in 0..m {
            let value = h[(k + 1 + i, k)];
            v[i] = value;
            x_norm_sq += value * value;
        }

        let x_norm = x_norm_sq.sqrt();
        if x_norm <= tol {
            continue;
        }

        let alpha = if v[0] >= 0.0 { -x_norm } else { x_norm };
        v[0] -= alpha;

        let mut v_norm_sq = 0.0_f64;
        for &value in &v {
            v_norm_sq += value * value;
        }
        let v_norm = v_norm_sq.sqrt();
        if v_norm <= tol {
            continue;
        }

        for value in &mut v {
            *value /= v_norm;
        }

        for j in k..n {
            let mut dot = 0.0_f64;
            for i in 0..m {
                dot += v[i] * h[(k + 1 + i, j)];
            }

            let scale = 2.0 * dot;
            for i in 0..m {
                *h.get_mut(k + 1 + i, j) -= v[i] * scale;
            }
        }

        for i in 0..n {
            let mut dot = 0.0_f64;
            for j in 0..m {
                dot += h[(i, k + 1 + j)] * v[j];
            }

            let scale = 2.0 * dot;
            for j in 0..m {
                *h.get_mut(i, k + 1 + j) -= scale * v[j];
            }
        }

        for i in 0..n {
            let mut dot = 0.0_f64;
            for j in 0..m {
                dot += q[(i, k + 1 + j)] * v[j];
            }

            let scale = 2.0 * dot;
            for j in 0..m {
                *q.get_mut(i, k + 1 + j) -= scale * v[j];
            }
        }

        for i in (k + 2)..n {
            *h.get_mut(i, k) = 0.0;
        }
    }

    (h, q)
}

fn qr_step_hessenberg_wilkinson(
    h: &mut Matrix<f64>,
    active_end: usize,
    shift: f64,
    mut q_accum: Option<&mut Matrix<f64>>,
) {
    let n = h.rows();
    if active_end == 0 {
        return;
    }

    let mut x = h[(0, 0)] - shift;
    let mut z = h[(1, 0)];

    for k in 0..active_end {
        let r = (x * x + z * z).sqrt();
        let (c, s) = if r > f64::EPSILON {
            (x / r, -z / r)
        } else {
            (1.0, 0.0)
        };

        for j in k..=active_end {
            let top = h[(k, j)];
            let bottom = h[(k + 1, j)];
            *h.get_mut(k, j) = c * top - s * bottom;
            *h.get_mut(k + 1, j) = s * top + c * bottom;
        }

        let row_max = (k + 2).min(active_end);
        for i in 0..=row_max {
            let left = h[(i, k)];
            let right = h[(i, k + 1)];
            *h.get_mut(i, k) = c * left - s * right;
            *h.get_mut(i, k + 1) = s * left + c * right;
        }

        if let Some(q) = q_accum.as_deref_mut() {
            for i in 0..n {
                let left = q[(i, k)];
                let right = q[(i, k + 1)];
                *q.get_mut(i, k) = c * left - s * right;
                *q.get_mut(i, k + 1) = s * left + c * right;
            }
        }

        if k + 2 <= active_end {
            x = h[(k + 1, k)];
            z = h[(k + 2, k)];
        }
    }

    for j in 0..active_end {
        for i in (j + 2)..=active_end {
            if h[(i, j)].abs() <= f64::EPSILON {
                *h.get_mut(i, j) = 0.0;
            }
        }
    }
}

fn qr_iterate_hessenberg_wilkinson(
    h: &mut Matrix<f64>,
    max_iter: usize,
    tol: f64,
    mut q_accum: Option<&mut Matrix<f64>>,
) -> (usize, bool, f64) {
    let n = h.rows();
    if n <= 1 {
        return (0, true, 0.0);
    }

    let mut iterations = 0usize;
    let mut active_end = n - 1;

    while active_end > 0 {
        let sub = h[(active_end, active_end - 1)].abs();
        let scale =
            h[(active_end - 1, active_end - 1)].abs() + h[(active_end, active_end)].abs() + 1.0;
        if sub <= tol * scale {
            *h.get_mut(active_end, active_end - 1) = 0.0;
            active_end -= 1;
            continue;
        }

        if iterations >= max_iter {
            return (iterations, false, hessenberg_subdiag_norm(h));
        }

        let shift = wilkinson_shift(h, active_end);
        if let Some(q) = q_accum.as_deref_mut() {
            qr_step_hessenberg_wilkinson(h, active_end, shift, Some(q));
        } else {
            qr_step_hessenberg_wilkinson(h, active_end, shift, None);
        }
        iterations += 1;

        for i in 1..=active_end {
            let sub_i = h[(i, i - 1)].abs();
            let scale_i = h[(i - 1, i - 1)].abs() + h[(i, i)].abs() + 1.0;
            if sub_i <= tol * scale_i {
                *h.get_mut(i, i - 1) = 0.0;
            }
        }

        while active_end > 0 {
            let sub_tail = h[(active_end, active_end - 1)].abs();
            let scale_tail =
                h[(active_end - 1, active_end - 1)].abs() + h[(active_end, active_end)].abs() + 1.0;
            if sub_tail <= tol * scale_tail {
                *h.get_mut(active_end, active_end - 1) = 0.0;
                active_end -= 1;
            } else {
                break;
            }
        }
    }

    (iterations, true, hessenberg_subdiag_norm(h))
}

/// Computes eigenvalues by shifted QR algorithm with Hessenberg reduction.
///
/// # Returns
/// - `Some(result)` for valid input.
/// - `None` for invalid input.
pub fn qr_wilkinson_eigenvalues(
    a: &Matrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<QrEigenvaluesResult> {
    let n = validate_qr_input(a, max_iter, tol)?;
    let algo_tol = tol.max(relative_tol(a)).max(f64::EPSILON);

    let (mut h, _) = hessenberg_reduction(a, algo_tol);
    let (iterations, converged, final_delta) =
        qr_iterate_hessenberg_wilkinson(&mut h, max_iter, algo_tol, None);

    let eigenvalues = Vector::from_fn(n, |i| h[(i, i)]);
    Some(QrEigenvaluesResult {
        eigenvalues,
        iterations,
        converged,
        final_delta,
    })
}

/// Computes eigenvalues and eigenvectors by shifted QR algorithm with
/// Hessenberg reduction.
///
/// # Notes
/// This routine targets real symmetric matrices so that returned vectors are
/// true eigenvectors (not only Schur vectors).
///
/// # Returns
/// - `Some(result)` for valid symmetric input.
/// - `None` for invalid input or non-symmetric matrix.
pub fn qr_wilkinson_eigenpairs(
    a: &Matrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<QrEigenpairsResult> {
    let n = validate_qr_input(a, max_iter, tol)?;
    let algo_tol = tol.max(relative_tol(a)).max(f64::EPSILON);
    if !is_symmetric(a, 10.0 * algo_tol) {
        return None;
    }

    let (mut h, mut q_total) = hessenberg_reduction(a, algo_tol);
    let (iterations, converged, final_delta) =
        qr_iterate_hessenberg_wilkinson(&mut h, max_iter, algo_tol, Some(&mut q_total));

    let eigenvalues = Vector::from_fn(n, |i| h[(i, i)]);
    Some(QrEigenpairsResult {
        eigenvalues,
        eigenvectors: q_total,
        iterations,
        converged,
        final_delta,
    })
}

/// Computes eigenvalues and eigenvectors by the classical Jacobi rotation
/// method for real symmetric matrices.
///
/// # Returns
/// - `Some(result)` for valid symmetric input.
/// - `None` for invalid input or non-symmetric matrix.
pub fn jacobi_eigenpairs(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<JacobiEigenResult> {
    let n = validate_qr_input(a, max_iter, tol)?;
    let algo_tol = tol.max(relative_tol(a)).max(f64::EPSILON);
    if !is_symmetric(a, 10.0 * algo_tol) {
        return None;
    }

    if n == 1 {
        return Some(JacobiEigenResult {
            eigenvalues: Vector::from_vec(1, vec![a[(0, 0)]]),
            eigenvectors: Matrix::identity(1),
            iterations: 0,
            converged: true,
            final_delta: 0.0,
        });
    }

    let mut d = a.clone();
    let mut v = Matrix::identity(n);
    let mut final_delta = f64::INFINITY;

    for iter in 1..=max_iter {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_offdiag = 0.0_f64;

        for i in 0..n {
            for j in (i + 1)..n {
                let value = d[(i, j)].abs();
                if value > max_offdiag {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }

        final_delta = max_offdiag;
        if max_offdiag <= algo_tol {
            return Some(JacobiEigenResult {
                eigenvalues: Vector::from_fn(n, |i| d[(i, i)]),
                eigenvectors: v,
                iterations: iter - 1,
                converged: true,
                final_delta,
            });
        }

        let app = d[(p, p)];
        let aqq = d[(q, q)];
        let apq = d[(p, q)];

        let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = theta.cos();
        let s = theta.sin();

        for k in 0..n {
            if k == p || k == q {
                continue;
            }

            let dkp = d[(k, p)];
            let dkq = d[(k, q)];
            let new_kp = c * dkp - s * dkq;
            let new_kq = s * dkp + c * dkq;

            *d.get_mut(k, p) = new_kp;
            *d.get_mut(p, k) = new_kp;
            *d.get_mut(k, q) = new_kq;
            *d.get_mut(q, k) = new_kq;
        }

        let c2 = c * c;
        let s2 = s * s;
        let cs = c * s;
        *d.get_mut(p, p) = c2 * app - 2.0 * cs * apq + s2 * aqq;
        *d.get_mut(q, q) = s2 * app + 2.0 * cs * apq + c2 * aqq;
        *d.get_mut(p, q) = 0.0;
        *d.get_mut(q, p) = 0.0;

        for k in 0..n {
            let vkp = v[(k, p)];
            let vkq = v[(k, q)];
            *v.get_mut(k, p) = c * vkp - s * vkq;
            *v.get_mut(k, q) = s * vkp + c * vkq;
        }
    }

    Some(JacobiEigenResult {
        eigenvalues: Vector::from_fn(n, |i| d[(i, i)]),
        eigenvectors: v,
        iterations: max_iter,
        converged: false,
        final_delta,
    })
}
