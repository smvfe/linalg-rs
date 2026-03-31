use crate::matrix::Matrix;
use crate::numeric_utils::relative_tol;

/// QR decomposition `A = Q*R`.
#[derive(Debug, Clone, PartialEq)]
pub struct QR {
    q: Matrix<f64>,
    r: Matrix<f64>,
}

impl QR {
    pub fn new(a: &Matrix<f64>) -> Option<Self> {
        let m = a.rows();
        let n = a.cols();
        let k_max = m.min(n);
        let tol = relative_tol(a);

        let mut q = Matrix::identity(m);
        let mut r = a.clone();
        let mut v = vec![0.0; m];

        for k in 0..k_max {
            let r_data = r.as_slice();
            let mut x_norm_sq = 0.0;
            for i in k..m {
                let val = r_data[i * n + k];
                x_norm_sq += val * val;
            }

            let x_norm = x_norm_sq.sqrt();
            if x_norm <= tol {
                continue;
            }

            let x0 = r_data[k * n + k];
            let alpha = if x0 >= 0.0 { -x_norm } else { x_norm };

            v[k] = x0 - alpha;
            for i in (k + 1)..m {
                v[i] = r_data[i * n + k];
            }

            let v_norm_sq = (-2.0 * alpha * v[k]).abs();
            if v_norm_sq <= tol * tol {
                continue;
            }

            let beta = 2.0 / v_norm_sq;

            let r_data = r.as_mut_slice();
            for j in k..n {
                let mut dot = 0.0;
                for i in k..m {
                    dot += v[i] * r_data[i * n + j];
                }

                let scale = beta * dot;
                for i in k..m {
                    let idx = i * n + j;
                    r_data[idx] -= v[i] * scale;
                }
            }

            let q_data = q.as_mut_slice();
            let q_cols = m;
            for i in 0..m {
                let mut dot = 0.0;
                let row_i = i * q_cols;
                for j in k..m {
                    dot += q_data[row_i + j] * v[j];
                }

                let scale = beta * dot;
                for j in k..m {
                    q_data[row_i + j] -= scale * v[j];
                }
            }
        }

        let r_cleanup_tol = {
            let max_abs = r
                .as_slice()
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);

            f64::EPSILON * (m.max(n) as f64) * max_abs
        };

        let r_data = r.as_mut_slice();
        for i in 0..m {
            for j in 0..i.min(n) {
                let idx = i * n + j;
                if r_data[idx].abs() <= r_cleanup_tol {
                    r_data[idx] = 0.0;
                }
            }
        }

        Some(Self { q, r })
    }

    pub fn q(&self) -> &Matrix<f64> {
        &self.q
    }

    pub fn r(&self) -> &Matrix<f64> {
        &self.r
    }

    pub fn into_parts(self) -> (Matrix<f64>, Matrix<f64>) {
        (self.q, self.r)
    }

    pub fn reconstruct(&self) -> Matrix<f64> {
        &self.q * &self.r
    }

    /// Solves `A*x=b` using `A=Q*R` (square full-rank case).
    pub fn solve(&self, b: &crate::vector::Vector<f64>) -> Option<crate::vector::Vector<f64>> {
        if self.q.rows() != self.q.cols()
            || self.r.rows() != self.r.cols()
            || b.dim() != self.q.rows()
        {
            return None;
        }

        let qt = self.q.transpose();
        let y = &qt * b;
        self.r.solve_upper_triangular(&y)
    }
}

/// Computes QR decomposition via Householder reflections.
///
/// # Arguments
/// - `a`: Input matrix of size `m x n`.
///
/// # Returns
/// - `Some(decomposition)` where `Q` is orthogonal (`m x m`) and `R` is upper
///   triangular (`m x n`).
/// - `None` is reserved for future failure modes; current implementation
///   returns `Some` for finite inputs.
///
/// # Notes
/// - Uses contiguous slice indexing in hot loops for lower overhead.
/// - Uses analytical `||v||²` computation for Householder vector to avoid one
///   extra pass per step.
pub fn qr(a: &Matrix<f64>) -> Option<QR> {
    QR::new(a)
}
