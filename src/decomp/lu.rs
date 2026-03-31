use crate::matrix::Matrix;
use crate::numeric_utils::relative_tol;
use crate::vector::Vector;

/// LU decomposition with partial pivoting, where `P*A = L*U`.
#[derive(Debug, Clone, PartialEq)]
pub struct LU {
    l: Matrix<f64>,
    u: Matrix<f64>,
    p: Vector<usize>,
}

impl LU {
    /// Constructs LU decomposition with partial pivoting from matrix `a`.
    pub fn new(a: &Matrix<f64>) -> Option<Self> {
        if a.rows() != a.cols() {
            return None;
        }

        let n = a.rows();
        let tol = relative_tol(a);
        let mut lu = a.clone();
        let mut p: Vec<usize> = (0..n).collect();

        for k in 0..n {
            let mut pivot_logical = k;
            let lu_data = lu.as_slice();
            let mut pivot_abs = lu_data[k * n + k].abs();

            for i in (k + 1)..n {
                let candidate_abs = lu_data[i * n + k].abs();
                if candidate_abs > pivot_abs {
                    pivot_abs = candidate_abs;
                    pivot_logical = i;
                }
            }

            if pivot_abs <= tol {
                return None;
            }

            lu.swap_rows(k, pivot_logical);
            p.swap(k, pivot_logical);

            let pivot_val = {
                let lu_data = lu.as_slice();
                lu_data[k * n + k]
            };
            if pivot_val.abs() <= tol {
                return None;
            }
            let inv_pivot = 1.0 / pivot_val;

            let pivot_row_start = k * n;
            for i in (k + 1)..n {
                let row_i_start = i * n;

                let lu_data = lu.as_mut_slice();
                let (before_i, from_i) = lu_data.split_at_mut(row_i_start);
                let pivot_row = &before_i[pivot_row_start..(pivot_row_start + n)];
                let row_i = &mut from_i[..n];

                let multiplier = row_i[k] * inv_pivot;
                row_i[k] = multiplier;

                for j in (k + 1)..n {
                    row_i[j] -= multiplier * pivot_row[j];
                }
            }
        }

        let mut l = Matrix::identity(n);
        let mut u = Matrix::zeros(n, n);
        let l_data = l.as_mut_slice();
        let u_data = u.as_mut_slice();
        let lu_data = lu.as_slice();

        for i in 0..n {
            let row_i = i * n;
            for j in 0..i {
                l_data[row_i + j] = lu_data[row_i + j];
            }
            for j in i..n {
                u_data[row_i + j] = lu_data[row_i + j];
            }
        }

        Some(Self {
            l,
            u,
            p: Vector::from_vec(n, p),
        })
    }

    pub fn l(&self) -> &Matrix<f64> {
        &self.l
    }

    pub fn u(&self) -> &Matrix<f64> {
        &self.u
    }

    pub fn permutation(&self) -> &Vector<usize> {
        &self.p
    }

    pub fn into_parts(self) -> (Matrix<f64>, Matrix<f64>, Vector<usize>) {
        (self.l, self.u, self.p)
    }

    /// Reconstructs `P*A` from decomposition as `L*U`.
    pub fn reconstruct_pa(&self) -> Matrix<f64> {
        &self.l * &self.u
    }

    /// Solves linear system `A*x=b` using LU factors.
    pub fn solve(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        let n = self.p.dim();
        if b.dim() != n {
            return None;
        }

        let mut pb = Vector::zeros(n);
        for i in 0..n {
            pb[i] = b[self.p[i]];
        }

        let y = self.l.solve_lower_triangular(&pb)?;
        self.u.solve_upper_triangular(&y)
    }
}

/// Computes LU decomposition with partial pivoting.
///
/// # Arguments
/// - `a`: Square input matrix.
///
/// # Returns
/// - `Some(decomposition)` where `P*A = L*U`, `p` is the row permutation
///   vector, `L` is unit lower-triangular, and `U` is upper-triangular.
/// - `None` when the matrix is non-square or numerically singular.
///
/// # Notes
/// - Uses physical row swapping for cache-friendly memory access.
/// - Uses scale-aware relative tolerance for singularity checks.
pub fn lu(a: &Matrix<f64>) -> Option<LU> {
    LU::new(a)
}
