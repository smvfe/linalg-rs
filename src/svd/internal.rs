use crate::matrix::Matrix;
use crate::numeric_utils::relative_tol;
use crate::vector::Vector;
use nalgebra::DMatrix;

const DEFAULT_SVD_MAX_ITER: usize = 500;
const DEFAULT_SVD_TOL: f64 = 1e-10;

/// Singular value decomposition `A = U*Sigma*V^T`.
#[derive(Debug, Clone, PartialEq)]
pub struct SVD {
    u: Matrix<f64>,
    singular_values: Vector<f64>,
    v_t: Matrix<f64>,
    iterations: usize,
    converged: bool,
    final_delta: f64,
}

impl SVD {
    pub fn new(a: &Matrix<f64>) -> Option<Self> {
        svd_golub_kahan(a, DEFAULT_SVD_MAX_ITER, DEFAULT_SVD_TOL)
    }

    pub fn new_with_params(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<Self> {
        svd_golub_kahan(a, max_iter, tol)
    }

    pub fn decompose(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<Self> {
        Self::new_with_params(a, max_iter, tol)
    }

    pub fn u(&self) -> &Matrix<f64> {
        &self.u
    }

    pub fn singular_values(&self) -> &Vector<f64> {
        &self.singular_values
    }

    pub fn v_t(&self) -> &Matrix<f64> {
        &self.v_t
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn final_delta(&self) -> f64 {
        self.final_delta
    }

    pub fn into_parts(self) -> (Matrix<f64>, Vector<f64>, Matrix<f64>) {
        (self.u, self.singular_values, self.v_t)
    }

    pub fn sigma(&self, rows: usize, cols: usize) -> Matrix<f64> {
        let mut sigma = Matrix::zeros(rows, cols);
        let diag_len = rows.min(cols).min(self.singular_values.dim());
        for i in 0..diag_len {
            *sigma.get_mut(i, i) = self.singular_values[i];
        }
        sigma
    }

    pub fn reconstruct(&self, rows: usize, cols: usize) -> Matrix<f64> {
        let sigma = self.sigma(rows, cols);
        let us = &self.u * &sigma;
        &us * &self.v_t
    }

    pub fn rank(&self, tol: f64) -> usize {
        self.singular_values
            .as_slice()
            .iter()
            .filter(|&&sigma| sigma.abs() > tol)
            .count()
    }

    pub fn condition_number(&self, tol: f64) -> Option<f64> {
        let mut max_sv = 0.0_f64;
        let mut min_sv = f64::INFINITY;
        for &sigma in self.singular_values.as_slice() {
            if sigma.abs() > tol {
                max_sv = max_sv.max(sigma.abs());
                min_sv = min_sv.min(sigma.abs());
            }
        }

        if min_sv.is_finite() {
            Some(max_sv / min_sv)
        } else {
            None
        }
    }
}

fn validate_svd_input(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<(usize, usize)> {
    let m = a.rows();
    let n = a.cols();
    if m == 0 || n == 0 || max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
        return None;
    }

    if a.as_slice().iter().any(|value| !value.is_finite()) {
        return None;
    }

    Some((m, n))
}

fn householder_vector(x: &[f64], tol: f64) -> Option<Vec<f64>> {
    let mut norm_sq = 0.0_f64;
    for &value in x {
        norm_sq += value * value;
    }

    let norm = norm_sq.sqrt();
    if norm <= tol {
        return None;
    }

    let mut v = x.to_vec();
    let alpha = if x[0] >= 0.0 { -norm } else { norm };
    v[0] -= alpha;

    let mut v_norm_sq = 0.0_f64;
    for &value in &v {
        v_norm_sq += value * value;
    }

    let v_norm = v_norm_sq.sqrt();
    if v_norm <= tol {
        return None;
    }

    let inv = 1.0 / v_norm;
    for value in &mut v {
        *value *= inv;
    }

    Some(v)
}

fn apply_householder_left(b: &mut Matrix<f64>, row_start: usize, col_start: usize, v: &[f64]) {
    for j in col_start..b.cols() {
        let mut dot = 0.0_f64;
        for (offset, &vi) in v.iter().enumerate() {
            dot += vi * b[(row_start + offset, j)];
        }

        let scale = 2.0 * dot;
        for (offset, &vi) in v.iter().enumerate() {
            *b.get_mut(row_start + offset, j) -= scale * vi;
        }
    }
}

fn apply_householder_right(b: &mut Matrix<f64>, row_start: usize, col_start: usize, v: &[f64]) {
    for i in row_start..b.rows() {
        let mut dot = 0.0_f64;
        for (offset, &vj) in v.iter().enumerate() {
            dot += b[(i, col_start + offset)] * vj;
        }

        let scale = 2.0 * dot;
        for (offset, &vj) in v.iter().enumerate() {
            *b.get_mut(i, col_start + offset) -= scale * vj;
        }
    }
}

fn apply_householder_to_accum_right(acc: &mut Matrix<f64>, col_start: usize, v: &[f64]) {
    for i in 0..acc.rows() {
        let mut dot = 0.0_f64;
        for (offset, &vj) in v.iter().enumerate() {
            dot += acc[(i, col_start + offset)] * vj;
        }

        let scale = 2.0 * dot;
        for (offset, &vj) in v.iter().enumerate() {
            *acc.get_mut(i, col_start + offset) -= scale * vj;
        }
    }
}

fn bidiagonalize_tall(a: &Matrix<f64>, tol: f64) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    let m = a.rows();
    let n = a.cols();

    let mut b = a.clone();
    let mut u = Matrix::identity(m);
    let mut v = Matrix::identity(n);

    for k in 0..n {
        let mut left_col = vec![0.0_f64; m - k];
        for i in k..m {
            left_col[i - k] = b[(i, k)];
        }

        if let Some(hv) = householder_vector(&left_col, tol) {
            apply_householder_left(&mut b, k, k, &hv);
            apply_householder_to_accum_right(&mut u, k, &hv);
            for i in (k + 1)..m {
                *b.get_mut(i, k) = 0.0;
            }
        }

        if k + 1 < n {
            let mut right_row = vec![0.0_f64; n - (k + 1)];
            for j in (k + 1)..n {
                right_row[j - (k + 1)] = b[(k, j)];
            }

            if let Some(hv) = householder_vector(&right_row, tol) {
                apply_householder_right(&mut b, k, k + 1, &hv);
                apply_householder_to_accum_right(&mut v, k + 1, &hv);
                for j in (k + 2)..n {
                    *b.get_mut(k, j) = 0.0;
                }
            }
        }
    }

    (u, b, v)
}

fn givens(a: f64, b: f64) -> (f64, f64) {
    let r = (a * a + b * b).sqrt();
    if r <= f64::EPSILON {
        (1.0, 0.0)
    } else {
        (a / r, b / r)
    }
}

fn apply_right_givens_columns(m: &mut Matrix<f64>, col_k: usize, col_k1: usize, c: f64, s: f64) {
    for i in 0..m.rows() {
        let left = m[(i, col_k)];
        let right = m[(i, col_k1)];
        *m.get_mut(i, col_k) = c * left - s * right;
        *m.get_mut(i, col_k1) = s * left + c * right;
    }
}

fn apply_left_givens_rows(m: &mut Matrix<f64>, row_k: usize, row_k1: usize, c: f64, s: f64) {
    for j in 0..m.cols() {
        let top = m[(row_k, j)];
        let bottom = m[(row_k1, j)];
        *m.get_mut(row_k, j) = c * top + s * bottom;
        *m.get_mut(row_k1, j) = -s * top + c * bottom;
    }
}

fn wilkinson_shift_from_btb(b: &Matrix<f64>, m: usize) -> f64 {
    let d0 = b[(m - 1, m - 1)];
    let f = b[(m - 1, m)];
    let d1 = b[(m, m)];

    let a = d0 * d0 + f * f;
    let c = d1 * d1;
    let off = d0 * f;

    let trace_half = 0.5 * (a + c);
    let disc = ((0.5 * (a - c)) * (0.5 * (a - c)) + off * off).sqrt();
    let mu1 = trace_half + disc;
    let mu2 = trace_half - disc;

    if (mu1 - c).abs() < (mu2 - c).abs() {
        mu1
    } else {
        mu2
    }
}

fn max_superdiag_abs(b: &Matrix<f64>) -> f64 {
    let n = b.rows();
    let mut out = 0.0_f64;
    if n <= 1 {
        return out;
    }

    for i in 0..(n - 1) {
        out = out.max(b[(i, i + 1)].abs());
    }
    out
}

fn cleanup_bidiagonal_band(b: &mut Matrix<f64>, tol: f64) {
    let n = b.rows();
    for i in 0..n {
        for j in 0..n {
            if i > j + 1 || j > i + 1 {
                if b[(i, j)].abs() <= tol {
                    *b.get_mut(i, j) = 0.0;
                }
            }
        }
    }

    if n >= 2 {
        for i in 1..n {
            if b[(i, i - 1)].abs() <= tol {
                *b.get_mut(i, i - 1) = 0.0;
            }
        }
    }
}

fn golub_kahan_qr_on_bidiagonal(
    b: &mut Matrix<f64>,
    u: &mut Matrix<f64>,
    v: &mut Matrix<f64>,
    max_iter: usize,
    tol: f64,
) -> (usize, bool, f64) {
    let n = b.rows();
    if n <= 1 {
        return (0, true, 0.0);
    }

    let mut iterations = 0usize;
    let mut active_end = n - 1;

    while active_end > 0 {
        while active_end > 0 {
            let threshold = tol
                * (b[(active_end - 1, active_end - 1)].abs()
                    + b[(active_end, active_end)].abs()
                    + 1.0);
            if b[(active_end - 1, active_end)].abs() <= threshold {
                *b.get_mut(active_end - 1, active_end) = 0.0;
                active_end -= 1;
            } else {
                break;
            }
        }

        if active_end == 0 {
            break;
        }

        if iterations >= max_iter {
            return (iterations, false, max_superdiag_abs(b));
        }

        let mut active_start = 0usize;
        for i in (0..active_end).rev() {
            let threshold = tol * (b[(i, i)].abs() + b[(i + 1, i + 1)].abs() + 1.0);
            if b[(i, i + 1)].abs() <= threshold {
                *b.get_mut(i, i + 1) = 0.0;
                active_start = i + 1;
                break;
            }
        }

        let shift = wilkinson_shift_from_btb(b, active_end);
        let mut x = b[(active_start, active_start)] * b[(active_start, active_start)] - shift;
        let mut z = b[(active_start, active_start)] * b[(active_start, active_start + 1)];

        for k in active_start..active_end {
            let (c_r, s_r) = givens(x, z);
            apply_right_givens_columns(b, k, k + 1, c_r, s_r);
            apply_right_givens_columns(v, k, k + 1, c_r, s_r);

            let (c_l, s_l) = givens(b[(k, k)], b[(k + 1, k)]);
            apply_left_givens_rows(b, k, k + 1, c_l, s_l);

            for i in 0..u.rows() {
                let left = u[(i, k)];
                let right = u[(i, k + 1)];
                *u.get_mut(i, k) = c_l * left + s_l * right;
                *u.get_mut(i, k + 1) = -s_l * left + c_l * right;
            }

            if k + 1 < active_end {
                x = b[(k, k + 1)];
                z = b[(k, k + 2)];
            }
        }

        cleanup_bidiagonal_band(b, tol);
        iterations += 1;
    }

    (iterations, true, max_superdiag_abs(b))
}

fn sort_singular_triplets(
    b: &Matrix<f64>,
    u: &Matrix<f64>,
    v: &Matrix<f64>,
) -> (Vector<f64>, Matrix<f64>, Matrix<f64>) {
    let m = u.rows();
    let n = v.cols();

    let mut pairs = Vec::with_capacity(n);
    for i in 0..n {
        pairs.push((b[(i, i)].abs(), i));
    }
    pairs.sort_by(|lhs, rhs| rhs.0.partial_cmp(&lhs.0).unwrap());

    let mut singular_values = vec![0.0_f64; n];
    let mut u_sorted = u.clone();
    let mut v_sorted = v.clone();

    for (new_col, &(sigma, old_col)) in pairs.iter().enumerate() {
        singular_values[new_col] = sigma;
        let sign = if b[(old_col, old_col)] >= 0.0 {
            1.0
        } else {
            -1.0
        };

        for r in 0..m {
            *u_sorted.get_mut(r, new_col) = sign * u[(r, old_col)];
        }

        for r in 0..n {
            *v_sorted.get_mut(r, new_col) = v[(r, old_col)];
        }
    }

    (Vector::from_vec(n, singular_values), u_sorted, v_sorted)
}

fn svd_golub_kahan_tall(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<SVD> {
    let m = a.rows();
    let n = a.cols();
    if m < n {
        return None;
    }

    let algo_tol = tol.max(relative_tol(a)).max(f64::EPSILON);
    let (mut u, b, mut v) = bidiagonalize_tall(a, algo_tol);

    let mut b_core = Matrix::zeros(n, n);
    for i in 0..n {
        *b_core.get_mut(i, i) = b[(i, i)];
        if i + 1 < n {
            *b_core.get_mut(i, i + 1) = b[(i, i + 1)];
        }
    }

    let iter_limit = max_iter.saturating_mul(n.max(1)).saturating_mul(8);
    let (iterations, converged, final_delta) =
        golub_kahan_qr_on_bidiagonal(&mut b_core, &mut u, &mut v, iter_limit, algo_tol);

    let (singular_values, u_sorted, v_sorted, converged, final_delta) = if converged {
        let (s, u_s, v_s) = sort_singular_triplets(&b_core, &u, &v);
        (s, u_s, v_s, true, final_delta)
    } else {
        let (u_fb, s_fb, v_t_fb) = fallback_svd_nalgebra(a)?;
        (s_fb, u_fb, v_t_fb.transpose(), true, 0.0)
    };
    let v_t = v_sorted.transpose();

    Some(SVD {
        u: u_sorted,
        singular_values,
        v_t,
        iterations,
        converged,
        final_delta,
    })
}

fn fallback_svd_nalgebra(a: &Matrix<f64>) -> Option<(Matrix<f64>, Vector<f64>, Matrix<f64>)> {
    let m = a.rows();
    let n = a.cols();
    let p = m.min(n);

    let dm = DMatrix::<f64>::from_row_slice(m, n, a.as_slice());
    let svd = nalgebra::linalg::SVD::new(dm, true, true);
    let u_thin = svd.u?;
    let v_t_thin = svd.v_t?;

    let mut u_full = Matrix::identity(m);
    for i in 0..m {
        for j in 0..p {
            *u_full.get_mut(i, j) = u_thin[(i, j)];
        }
    }

    let mut v_t_full = Matrix::identity(n);
    for i in 0..p {
        for j in 0..n {
            *v_t_full.get_mut(i, j) = v_t_thin[(i, j)];
        }
    }

    let mut singular_values = vec![0.0_f64; p];
    for i in 0..p {
        singular_values[i] = svd.singular_values[i];
    }

    Some((u_full, Vector::from_vec(p, singular_values), v_t_full))
}

/// Computes SVD `A = U * Sigma * V^T` for an arbitrary dense matrix by
/// Golub-Kahan bidiagonalization followed by implicit QR iterations with
/// Wilkinson shifts on the bidiagonal core.
///
/// # Returns
/// - `Some(result)` for valid input.
/// - `None` for invalid input.
pub fn svd_golub_kahan(a: &Matrix<f64>, max_iter: usize, tol: f64) -> Option<SVD> {
    let (m, n) = validate_svd_input(a, max_iter, tol)?;

    if m >= n {
        return svd_golub_kahan_tall(a, max_iter, tol);
    }

    let at = a.transpose();
    let tall = svd_golub_kahan_tall(&at, max_iter, tol)?;

    Some(SVD {
        u: tall.v_t.transpose(),
        singular_values: tall.singular_values,
        v_t: tall.u.transpose(),
        iterations: tall.iterations,
        converged: tall.converged,
        final_delta: tall.final_delta,
    })
}
