use crate::matrix::Matrix;
use crate::numeric_utils::relative_tol;
use crate::vector::Vector;

/// LDLT decomposition `A = L*D*L^T` for symmetric matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct LDLT {
    l: Matrix<f64>,
    d: Vector<f64>,
}

impl LDLT {
    pub fn new(a: &Matrix<f64>) -> Option<Self> {
        ldlt_with_symmetry_check(a, true)
    }

    pub fn l(&self) -> &Matrix<f64> {
        &self.l
    }

    pub fn d(&self) -> &Vector<f64> {
        &self.d
    }

    pub fn into_parts(self) -> (Matrix<f64>, Vector<f64>) {
        (self.l, self.d)
    }

    pub fn reconstruct(&self) -> Matrix<f64> {
        let n = self.d.dim();
        let mut d_mat = Matrix::zeros(n, n);
        for i in 0..n {
            *d_mat.get_mut(i, i) = self.d[i];
        }

        let ld = &self.l * &d_mat;
        &ld * &self.l.transpose()
    }

    pub fn solve(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        let n = self.d.dim();
        if b.dim() != n {
            return None;
        }

        let y = self.l.solve_lower_triangular(b)?;

        let pivot_tol = relative_tol(&self.l).max(f64::EPSILON);

        let mut z = Vector::zeros(n);
        for i in 0..n {
            let diag = self.d[i];
            if diag.abs() <= pivot_tol {
                return None;
            }
            z[i] = y[i] / diag;
        }

        self.l.transpose().solve_upper_triangular(&z)
    }
}

/// Computes LDLᵀ decomposition for a symmetric square matrix.
///
/// # Arguments
/// - `a`: Square input matrix.
///
/// # Returns
/// - `Some(decomposition)` where `L` is unit lower-triangular and `D` is the
///   diagonal stored as a dense vector.
/// - `None` when the matrix is non-square, non-symmetric (within tolerance),
///   or numerically singular.
///
/// # Notes
/// Uses a scale-aware tolerance and stores only diagonal values of `D` to
/// reduce memory traffic.
pub fn ldlt(a: &Matrix<f64>) -> Option<LDLT> {
    LDLT::new(a)
}

/// Internal LDLᵀ implementation with optional symmetry validation.
///
/// # Arguments
/// - `a`: Square input matrix.
/// - `check_symmetry`: If `true`, verifies `a[i, j] ≈ a[j, i]` before factorization.
///
/// # Returns
/// Same contract as [`ldlt`].
pub(crate) fn ldlt_with_symmetry_check(a: &Matrix<f64>, check_symmetry: bool) -> Option<LDLT> {
    if a.rows() != a.cols() {
        return None;
    }

    let n = a.rows();
    let tol = relative_tol(a);

    if check_symmetry && !is_symmetric(a, tol) {
        return None;
    }

    let a_data = a.as_slice();
    let mut l = Matrix::identity(n);
    let mut d_diag = vec![0.0; n];
    // Optimization: reused cache for L[j, k] * D[k] terms to avoid recomputing
    // identical products across the inner update loops.
    let mut dl_cache = vec![0.0; n];
    let l_data = l.as_mut_slice();

    for j in 0..n {
        let row_j = j * n;
        let mut diag = a_data[row_j + j];

        for k in 0..j {
            let l_jk = l_data[row_j + k];
            let d_k = d_diag[k];
            let dl_val = l_jk * d_k;
            dl_cache[k] = dl_val;
            diag -= l_jk * dl_val;
        }

        if diag.abs() <= tol {
            return None;
        }
        d_diag[j] = diag;

        let inv_diag = 1.0 / diag;
        for i in (j + 1)..n {
            let row_i = i * n;
            let mut value = a_data[row_i + j];
            for k in 0..j {
                value -= l_data[row_i + k] * dl_cache[k];
            }
            l_data[row_i + j] = value * inv_diag;
        }
    }

    Some(LDLT {
        l,
        d: Vector::from_vec(n, d_diag),
    })
}

/// Checks matrix symmetry with a relative numerical tolerance.
///
/// # Arguments
/// - `a`: Matrix to validate.
/// - `tol`: Allowed absolute difference between mirrored entries.
///
/// # Returns
/// `true` if matrix is symmetric within tolerance; otherwise `false`.
fn is_symmetric(a: &Matrix<f64>, tol: f64) -> bool {
    let n = a.rows();
    let a_data = a.as_slice();
    for i in 0..n {
        let row_i = i * n;
        for j in (i + 1)..n {
            let row_j = j * n;
            if (a_data[row_i + j] - a_data[row_j + i]).abs() > tol {
                return false;
            }
        }
    }
    true
}
