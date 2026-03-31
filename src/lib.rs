extern crate nalgebra;

pub mod core;
pub mod decomp;
pub mod dense;
#[path = "eigen/mod.rs"]
pub mod eigen;
pub mod solve;
#[path = "sparse/mod.rs"]
pub mod sparse;
#[path = "svd/mod.rs"]
pub mod svd;

// Compatibility facades for legacy import paths.
pub mod matrix {
    pub use crate::dense::matrix::Matrix;
}

pub mod vector {
    pub use crate::dense::vector::Vector;
}

pub mod scalar {
    pub use crate::core::scalar::Scalar;
}

pub mod numeric_utils {
    pub use crate::core::numeric_utils::*;
}

pub mod lu {
    pub use crate::decomp::lu::{LU, lu};
}

pub mod cholesky {
    pub use crate::decomp::ldlt::{LDLT, ldlt};
}

pub mod qr {
    pub use crate::decomp::qr::{QR, qr};
}

pub mod iterative {
    pub use crate::solve::iterative::{
        IterativeResult, gauss_seidel, gmres, gmres_restarted, jacobi, simple_iteration,
    };
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use crate::add;
    use crate::decomp::ldlt::{ldlt, ldlt_with_symmetry_check};
    use crate::decomp::lu::{LU, lu};
    use crate::decomp::qr::{QR, qr};
    use crate::dense::{Matrix, Vector};
    use crate::eigen::{
        EigenResult, JacobiEigenResult, QrEigenpairsResult, QrEigenvaluesResult,
        inverse_power_method, jacobi_eigenpairs, power_method, qr_wilkinson_eigenpairs,
        qr_wilkinson_eigenvalues,
    };
    use crate::solve::iterative::{gauss_seidel, gmres, gmres_restarted, jacobi, simple_iteration};
    use crate::sparse::{CscMatrix, CsrMatrix};
    use crate::svd::{SVD, svd_golub_kahan};
    use nalgebra::{DMatrix, DVector};

    #[inline]
    fn matrix_at(m: &Matrix<f64>, i: usize, j: usize) -> f64 {
        m.data[i * m.cols + j]
    }

    #[inline]
    fn matrix_set(m: &mut Matrix<f64>, i: usize, j: usize, value: f64) {
        m.data[i * m.cols + j] = value;
    }

    fn csr_value(m: &CsrMatrix<f64>, i: usize, j: usize) -> f64 {
        let start = m.row_ptr[i];
        let end = m.row_ptr[i + 1];
        match m.col_idx[start..end].binary_search(&j) {
            Ok(local_idx) => m.data[start + local_idx],
            Err(_) => 0.0,
        }
    }

    fn csc_value(m: &CscMatrix<f64>, i: usize, j: usize) -> f64 {
        let start = m.col_ptr[j];
        let end = m.col_ptr[j + 1];
        match m.row_idx[start..end].binary_search(&i) {
            Ok(local_idx) => m.data[start + local_idx],
            Err(_) => 0.0,
        }
    }

    fn to_dmatrix(a: &Matrix<f64>) -> DMatrix<f64> {
        DMatrix::from_row_slice(a.rows(), a.cols(), &a.data)
    }

    fn to_dvector(v: &Vector<f64>) -> DVector<f64> {
        DVector::from_vec(v.as_slice().to_vec())
    }

    fn sorted_values(v: &Vector<f64>) -> Vec<f64> {
        let mut out = v.as_slice().to_vec();
        out.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        out
    }

    fn assert_slice_close(lhs: &[f64], rhs: &[f64], eps: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            let diff = (lhs[i] - rhs[i]).abs();
            assert!(
                diff <= eps,
                "Slice mismatch at {}: left={}, right={}, diff={}",
                i,
                lhs[i],
                rhs[i],
                diff
            );
        }
    }

    fn assert_matrix_close(lhs: &Matrix<f64>, rhs: &Matrix<f64>, eps: f64) {
        assert_eq!(lhs.rows(), rhs.rows());
        assert_eq!(lhs.cols(), rhs.cols());

        for i in 0..lhs.rows() {
            for j in 0..lhs.cols() {
                let diff = (matrix_at(lhs, i, j) - matrix_at(rhs, i, j)).abs();
                assert!(
                    diff <= eps,
                    "Matrix mismatch at ({}, {}): left={}, right={}, diff={}",
                    i,
                    j,
                    lhs[(i, j)],
                    rhs[(i, j)],
                    diff
                );
            }
        }
    }

    fn diag_from_vector(d: &Vector<f64>) -> Matrix<f64> {
        let mut out = Matrix::zeros(d.dim(), d.dim());
        for i in 0..d.dim() {
            matrix_set(&mut out, i, i, d[i]);
        }
        out
    }

    fn apply_permutation(p: &Vector<usize>, a: &Matrix<f64>) -> Matrix<f64> {
        let mut out = Matrix::zeros(a.rows(), a.cols());
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                matrix_set(&mut out, i, j, matrix_at(a, p[i], j));
            }
        }
        out
    }

    fn create_test_dense() -> Matrix<f64> {
        Matrix::from_vec(3, 3, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
    }

    fn create_test_dense2() -> Matrix<f64> {
        Matrix::from_vec(3, 2, vec![1.0, 0.0, 2.0, 3.0, 0.0, 4.0])
    }

    #[test]
    fn it_works() {
        assert_eq!(add(2, 2), 4);
    }

    // Vector tests
    #[test]
    fn test_add() {
        let a = Vector::from_array([1.0, 2.0, 3.0]);
        let b = Vector::from_array([4.0, 5.0, 6.0]);
        assert_eq!(&a + &b, Vector::from_array([5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_sub() {
        let a = Vector::from_array([4.0, 5.0, 6.0]);
        let b = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(&a - &b, Vector::from_array([3.0, 3.0, 3.0]));
    }

    #[test]
    fn test_neg() {
        let a = Vector::from_array([1.0, -2.0, 3.0]);
        assert_eq!(-&a, Vector::from_array([-1.0, 2.0, -3.0]));
    }

    #[test]
    fn test_mul_scalar() {
        let a = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(&a * 2.0, Vector::from_array([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_scalar_mul() {
        let a = Vector::from_array([1.0, 2.0, 3.0]);
        assert_eq!(2.0 * &a, Vector::from_array([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_dot() {
        let a = Vector::from_array([1.0, 2.0, 3.0]);
        let b = Vector::from_array([4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_zeros() {
        let v = Vector::<f64>::zeros(3);
        assert_eq!(v, Vector::from_array([0.0, 0.0, 0.0]));
    }

    #[test]
    #[should_panic]
    fn test_add_incompatible() {
        let a = Vector::from_array([1.0, 2.0]);
        let b = Vector::from_array([1.0, 2.0, 3.0]);
        let _ = &a + &b;
    }

    // LU tests
    #[test]
    fn test_lup_reconstructs_pa() {
        let a = Matrix::from_vec(3, 3, vec![2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0]);

        let lu_dec = lu(&a).expect("LU must exist for non-singular matrix");
        let pa = apply_permutation(lu_dec.permutation(), &a);
        let lu_prod = lu_dec.reconstruct_pa();

        assert_matrix_close(&pa, &lu_prod, 1e-10);
    }

    #[test]
    fn test_lup_requires_pivoting() {
        let a = Matrix::from_vec(3, 3, vec![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0]);

        let lu_dec = lu(&a).expect("LU with pivoting must exist");
        let pa = apply_permutation(lu_dec.permutation(), &a);
        let lu_prod = lu_dec.reconstruct_pa();

        assert_matrix_close(&pa, &lu_prod, 1e-10);
    }

    // LDLT tests
    #[test]
    fn test_ldlt_reconstructs_symmetric_matrix() {
        let a = Matrix::from_vec(3, 3, vec![4.0, 2.0, 2.0, 2.0, 5.0, 1.0, 2.0, 1.0, 3.0]);

        let ldlt_dec = ldlt(&a).expect("LDLT must exist for symmetric non-singular matrix");
        let l = ldlt_dec.l();
        let d_vec = ldlt_dec.d();
        let d = diag_from_vector(&d_vec);

        let lt = l.transpose();
        let ld = l * &d;
        let reconstructed = &ld * &lt;

        assert_matrix_close(&a, &reconstructed, 1e-10);
    }

    #[test]
    fn test_ldlt_unit_diagonal_l() {
        let a = Matrix::from_vec(3, 3, vec![6.0, 3.0, 2.0, 3.0, 5.0, 1.0, 2.0, 1.0, 4.0]);

        let ldlt_dec = ldlt(&a).expect("LDLT must exist for symmetric non-singular matrix");
        for i in 0..ldlt_dec.l().rows() {
            assert!((matrix_at(ldlt_dec.l(), i, i) - 1.0).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_ldlt_rejects_nonsymmetric_matrix() {
        let a = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        assert!(ldlt(&a).is_none());
        assert!(ldlt_with_symmetry_check(&a, false).is_some());
    }

    // QR tests
    #[test]
    fn test_qr_reconstructs_square_matrix() {
        let a = Matrix::from_vec(
            3,
            3,
            vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
        );

        let qr_dec = qr(&a).expect("QR must exist for finite matrix");
        let reconstructed = qr_dec.reconstruct();

        assert_matrix_close(&a, &reconstructed, 1e-9);
    }

    #[test]
    fn test_q_is_orthogonal() {
        let a = Matrix::from_vec(
            4,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 3.0, 4.0],
        );

        let qr_dec = qr(&a).expect("QR must exist for finite matrix");
        let qtq = &qr_dec.q().transpose() * qr_dec.q();
        let i = Matrix::identity(qr_dec.q().rows());

        assert_matrix_close(&qtq, &i, 1e-9);
    }

    #[test]
    fn test_r_is_upper_triangular_for_rectangular() {
        let a = Matrix::from_vec(
            4,
            3,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );

        let qr_dec = qr(&a).expect("QR must exist for finite matrix");
        for i in 0..qr_dec.r().rows() {
            for j in 0..qr_dec.r().cols() {
                if i > j {
                    assert!(matrix_at(qr_dec.r(), i, j).abs() <= 1e-9);
                }
            }
        }
    }

    // Sparse tests
    #[test]
    fn test_csr_from_dense() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);

        assert_eq!(csr.rows(), 3);
        assert_eq!(csr.cols(), 3);
        assert_eq!(csr.nnz(), 5);
    }

    #[test]
    fn test_csr_to_dense() {
        let original = create_test_dense();
        let csr = CsrMatrix::from_dense(&original);
        let dense = csr.to_dense();

        for i in 0..original.rows() {
            for j in 0..original.cols() {
                assert_eq!(matrix_at(&original, i, j), matrix_at(&dense, i, j));
            }
        }
    }

    #[test]
    fn test_csr_get_stored_element() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);

        assert_eq!(csr_value(&csr, 0, 0), 1.0);
        assert_eq!(csr_value(&csr, 0, 2), 2.0);
        assert_eq!(csr_value(&csr, 1, 1), 3.0);
        assert_eq!(csr_value(&csr, 2, 0), 4.0);
        assert_eq!(csr_value(&csr, 2, 2), 5.0);
    }

    #[test]
    fn test_csr_get_zero_element() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);

        assert_eq!(csr_value(&csr, 0, 1), 0.0);
        assert_eq!(csr_value(&csr, 1, 0), 0.0);
        assert_eq!(csr_value(&csr, 1, 2), 0.0);
        assert_eq!(csr_value(&csr, 2, 1), 0.0);
    }

    #[test]
    fn test_csr_index_trait() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);

        assert_eq!(csr[(0, 0)], 1.0);
        assert_eq!(csr[(2, 2)], 5.0);
    }

    #[test]
    #[should_panic(expected = "is zero")]
    fn test_csr_index_panics_on_zero() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);
        let _ = csr[(0, 1)];
    }

    #[test]
    fn test_csr_mul_scalar() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);
        let csr_scaled = &csr * 2.0;

        assert_eq!(csr_value(&csr_scaled, 0, 0), 2.0);
        assert_eq!(csr_value(&csr_scaled, 0, 2), 4.0);
        assert_eq!(csr_value(&csr_scaled, 1, 1), 6.0);
    }

    #[test]
    fn test_csr_neg() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);
        let csr_neg = -&csr;

        assert_eq!(csr_value(&csr_neg, 0, 0), -1.0);
        assert_eq!(csr_value(&csr_neg, 1, 1), -3.0);
    }

    #[test]
    fn test_csr_mul_vector() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);
        let v = Vector::from_vec(3, vec![1.0, 2.0, 3.0]);

        let result = &csr * &v;

        assert_eq!(result[0], 7.0);
        assert_eq!(result[1], 6.0);
        assert_eq!(result[2], 19.0);
    }

    #[test]
    fn test_csr_mul_matrix() {
        let m1 = create_test_dense();
        let m2 = create_test_dense2();
        let csr = CsrMatrix::from_dense(&m1);

        let result = &csr * &m2;

        assert_eq!(matrix_at(&result, 0, 0), 1.0);
        assert_eq!(matrix_at(&result, 0, 1), 8.0);
        assert_eq!(matrix_at(&result, 1, 0), 6.0);
        assert_eq!(matrix_at(&result, 1, 1), 9.0);
        assert_eq!(matrix_at(&result, 2, 0), 4.0);
        assert_eq!(matrix_at(&result, 2, 1), 20.0);
    }

    #[test]
    fn test_csr_density() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);
        let expected_density = 5.0 / 9.0;
        assert!((csr.density() - expected_density).abs() < 1e-10);
    }

    #[test]
    fn test_csc_from_dense() {
        let m = create_test_dense();
        let csc = CscMatrix::from_dense(&m);

        assert_eq!(csc.rows(), 3);
        assert_eq!(csc.cols(), 3);
        assert_eq!(csc.nnz(), 5);
    }

    #[test]
    fn test_csc_to_dense() {
        let original = create_test_dense();
        let csc = CscMatrix::from_dense(&original);
        let dense = csc.to_dense();

        for i in 0..original.rows() {
            for j in 0..original.cols() {
                assert_eq!(matrix_at(&original, i, j), matrix_at(&dense, i, j));
            }
        }
    }

    #[test]
    fn test_csc_get_stored_element() {
        let m = create_test_dense();
        let csc = CscMatrix::from_dense(&m);

        assert_eq!(csc_value(&csc, 0, 0), 1.0);
        assert_eq!(csc_value(&csc, 0, 2), 2.0);
        assert_eq!(csc_value(&csc, 1, 1), 3.0);
        assert_eq!(csc_value(&csc, 2, 0), 4.0);
        assert_eq!(csc_value(&csc, 2, 2), 5.0);
    }

    #[test]
    fn test_csc_get_zero_element() {
        let m = create_test_dense();
        let csc = CscMatrix::from_dense(&m);

        assert_eq!(csc_value(&csc, 0, 1), 0.0);
        assert_eq!(csc_value(&csc, 1, 0), 0.0);
        assert_eq!(csc_value(&csc, 1, 2), 0.0);
    }

    #[test]
    fn test_csc_index_trait() {
        let m = create_test_dense();
        let csc = CscMatrix::from_dense(&m);

        assert_eq!(csc[(0, 0)], 1.0);
        assert_eq!(csc[(2, 2)], 5.0);
    }

    #[test]
    #[should_panic(expected = "is zero")]
    fn test_csc_index_panics_on_zero() {
        let m = create_test_dense();
        let csc = CscMatrix::from_dense(&m);
        let _ = csc[(0, 1)];
    }

    #[test]
    fn test_csc_mul_scalar() {
        let m = create_test_dense();
        let csc = CscMatrix::from_dense(&m);
        let csc_scaled = &csc * 2.0;

        assert_eq!(csc_value(&csc_scaled, 0, 0), 2.0);
        assert_eq!(csc_value(&csc_scaled, 1, 1), 6.0);
    }

    #[test]
    fn test_csc_neg() {
        let m = create_test_dense();
        let csc = CscMatrix::from_dense(&m);
        let csc_neg = -&csc;

        assert_eq!(csc_value(&csc_neg, 0, 0), -1.0);
        assert_eq!(csc_value(&csc_neg, 2, 2), -5.0);
    }

    #[test]
    fn test_matrix_mul_csc() {
        let m1_real = Matrix::from_vec(2, 3, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        let m2 = create_test_dense();
        let csc = CscMatrix::from_dense(&m2);

        let result = &m1_real * &csc;

        assert_eq!(matrix_at(&result, 0, 0), 9.0);
        assert_eq!(matrix_at(&result, 0, 1), 0.0);
        assert_eq!(matrix_at(&result, 1, 1), 9.0);
    }

    #[test]
    fn test_csr_and_csc_equivalent() {
        let m = create_test_dense();
        let csr = CsrMatrix::from_dense(&m);
        let csc = CscMatrix::from_dense(&m);

        for i in 0..m.rows() {
            for j in 0..m.cols() {
                assert_eq!(csr_value(&csr, i, j), csc_value(&csc, i, j));
            }
        }
    }

    #[test]
    fn test_empty_matrix() {
        let m: Matrix<f64> = Matrix::zeros(3, 3);
        let csr = CsrMatrix::from_dense(&m);
        let csc = CscMatrix::from_dense(&m);

        assert_eq!(csr.nnz(), 0);
        assert_eq!(csc.nnz(), 0);
        assert_eq!(csr.density(), 0.0);
        assert_eq!(csc.density(), 0.0);
    }

    #[test]
    fn test_single_element() {
        let m = Matrix::from_vec(2, 2, vec![5.0, 0.0, 0.0, 0.0]);
        let csr = CsrMatrix::from_dense(&m);

        assert_eq!(csr.nnz(), 1);
        assert_eq!(csr_value(&csr, 0, 0), 5.0);
        assert_eq!(csr_value(&csr, 0, 1), 0.0);
    }

    // New larger and trickier tests
    #[test]
    fn test_lu_large_diagonal_dominant_reconstructs_pa() {
        let n = 48;
        let mut data = vec![0.0; n * n];

        for i in 0..n {
            let row = i * n;
            for j in 0..n {
                if i == j {
                    data[row + j] = 40.0 + (i as f64) * 0.5;
                } else if (i + j) % 11 == 0 {
                    data[row + j] = ((i as f64) - (j as f64)).sin() * 0.3;
                }
            }
        }

        let a = Matrix::from_vec(n, n, data);
        let lu_dec = lu(&a).expect("LU must exist for diagonally dominant matrix");
        let pa = apply_permutation(lu_dec.permutation(), &a);
        let lu_prod = lu_dec.reconstruct_pa();
        assert_matrix_close(&pa, &lu_prod, 1e-7);
    }

    #[test]
    fn test_ldlt_large_tridiagonal_spd() {
        let n = 80;
        let mut data = vec![0.0; n * n];

        for i in 0..n {
            let row = i * n;
            data[row + i] = 4.0;
            if i + 1 < n {
                data[row + i + 1] = -1.0;
                data[(i + 1) * n + i] = -1.0;
            }
        }

        let a = Matrix::from_vec(n, n, data);
        let ldlt_dec = ldlt(&a).expect("LDLT must exist for SPD matrix");
        let l = ldlt_dec.l();
        let d_vec = ldlt_dec.d();
        let d = diag_from_vector(&d_vec);
        let reconstructed = &(l * &d) * &l.transpose();
        assert_matrix_close(&a, &reconstructed, 1e-8);
    }

    #[test]
    fn test_qr_large_tall_matrix_properties() {
        let m = 96;
        let n = 40;
        let mut data = vec![0.0; m * n];

        for i in 0..m {
            let row = i * n;
            for j in 0..n {
                let base = ((i + 1) as f64 * (j + 2) as f64).sin();
                let tweak = if (i + 3 * j) % 13 == 0 { 0.25 } else { 0.0 };
                data[row + j] = base + tweak;
            }
        }

        let a = Matrix::from_vec(m, n, data);
        let qr_dec = qr(&a).expect("QR must exist for finite matrix");

        let reconstructed = qr_dec.reconstruct();
        assert_matrix_close(&a, &reconstructed, 1e-7);

        let qtq = &qr_dec.q().transpose() * qr_dec.q();
        let i = Matrix::identity(qr_dec.q().rows());
        assert_matrix_close(&qtq, &i, 1e-7);
    }

    #[test]
    fn test_sparse_large_pattern_roundtrip() {
        let rows = 120;
        let cols = 90;
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows {
            let row = i * cols;
            for j in 0..cols {
                if (i + 2 * j) % 17 == 0 || i == j.min(rows - 1) {
                    data[row + j] = ((i + j + 1) as f64).cos();
                }
            }
        }

        let dense = Matrix::from_vec(rows, cols, data);
        let csr = CsrMatrix::from_dense(&dense);
        let csc = CscMatrix::from_dense(&dense);
        let dense_from_csr = csr.to_dense();
        let dense_from_csc = csc.to_dense();

        assert_matrix_close(&dense, &dense_from_csr, 1e-12);
        assert_matrix_close(&dense, &dense_from_csc, 1e-12);
    }

    fn sample_system() -> (Matrix<f64>, Vector<f64>, Vector<f64>) {
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                10.0, -1.0, 2.0, 0.0, -1.0, 11.0, -1.0, 3.0, 2.0, -1.0, 10.0, -1.0, 0.0, 3.0, -1.0,
                8.0,
            ],
        );
        let b = Vector::from_array([6.0, 25.0, -11.0, 15.0]);
        let expected = Vector::from_array([1.0, 2.0, -1.0, 1.0]);
        (a, b, expected)
    }

    #[test]
    fn jacobi_converges_on_diagonally_dominant_system() {
        let (a, b, expected) = sample_system();
        let x0 = Vector::zeros(4);

        let result = jacobi(&a, &b, &x0, 500, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!(result.x.approx_eq(&expected, 1e-7));
    }

    #[test]
    fn gauss_seidel_converges_on_diagonally_dominant_system() {
        let (a, b, expected) = sample_system();
        let x0 = Vector::zeros(4);

        let result = gauss_seidel(&a, &b, &x0, 200, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!(result.x.approx_eq(&expected, 1e-9));
    }

    #[test]
    fn simple_iteration_converges_with_small_tau() {
        let (a, b, expected) = sample_system();
        let x0 = Vector::zeros(4);

        let result = simple_iteration(&a, &b, &x0, 0.05, 3_000, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!(result.x.approx_eq(&expected, 1e-6));
    }

    #[test]
    fn methods_reject_zero_diagonal_for_jacobi_and_seidel() {
        let a = Matrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 2.0]);
        let b = Vector::from_array([1.0, 1.0]);
        let x0 = Vector::zeros(2);

        assert!(jacobi(&a, &b, &x0, 100, 1e-8).is_none());
        assert!(gauss_seidel(&a, &b, &x0, 100, 1e-8).is_none());
    }

    #[test]
    fn gmres_converges_on_sample_system() {
        let (a, b, expected) = sample_system();
        let x0 = Vector::zeros(4);

        let result = gmres(&a, &b, &x0, 20, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!(result.x.approx_eq(&expected, 1e-10));
    }

    #[test]
    fn gmres_rejects_invalid_input() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 0.0, 2.0, 0.0, 1.0, 3.0]);
        let b = Vector::from_array([1.0, 2.0]);
        let x0 = Vector::zeros(2);

        assert!(gmres(&a, &b, &x0, 10, 1e-8).is_none());
    }

    #[test]
    fn restarted_gmres_converges_with_small_restart() {
        let (a, b, expected) = sample_system();
        let x0 = Vector::zeros(4);

        let result = gmres_restarted(&a, &b, &x0, 2, 40, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!(result.x.approx_eq(&expected, 1e-9));
    }

    #[test]
    fn restarted_gmres_rejects_zero_restart() {
        let (a, b, _) = sample_system();
        let x0 = Vector::zeros(4);

        assert!(gmres_restarted(&a, &b, &x0, 0, 20, 1e-8).is_none());
    }

    #[test]
    fn power_method_finds_dominant_eigenvalue() {
        let a = Matrix::from_vec(3, 3, vec![5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
        let x0 = Vector::from_array([1.0, 1.0, 1.0]);

        let result = power_method(&a, &x0, 100, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!((result.eigenvalue - 5.0).abs() <= 1e-9);
    }

    #[test]
    fn inverse_power_method_finds_smallest_eigenvalue() {
        let a = Matrix::from_vec(3, 3, vec![5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
        let x0 = Vector::from_array([1.0, 1.0, 1.0]);

        let result = inverse_power_method(&a, &x0, 100, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!((result.eigenvalue - 1.0).abs() <= 1e-9);
    }

    #[test]
    fn inverse_power_method_rejects_singular_matrix() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
        let x0 = Vector::from_array([1.0, 1.0]);

        assert!(inverse_power_method(&a, &x0, 20, 1e-8).is_none());
    }

    #[test]
    fn qr_wilkinson_eigenvalues_finds_diagonal_spectrum() {
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                7.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );

        let result = qr_wilkinson_eigenvalues(&a, 100, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!((result.eigenvalues[0] - 7.0).abs() <= 1e-10);
        assert!((result.eigenvalues[1] - 3.0).abs() <= 1e-10);
        assert!((result.eigenvalues[2] + 2.0).abs() <= 1e-10);
        assert!((result.eigenvalues[3] - 1.0).abs() <= 1e-10);
    }

    #[test]
    fn qr_wilkinson_eigenpairs_finds_vectors_for_symmetric_matrix() {
        let a = Matrix::from_vec(3, 3, vec![4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);

        let result = qr_wilkinson_eigenpairs(&a, 100, 1e-12).expect("valid input");
        assert!(result.converged);
        assert!((result.eigenvalues[0] - 4.0).abs() <= 1e-10);
        assert!((result.eigenvalues[1] - 2.0).abs() <= 1e-10);
        assert!((result.eigenvalues[2] - 1.0).abs() <= 1e-10);

        let v0 = Vector::from_vec(
            3,
            vec![
                matrix_at(&result.eigenvectors, 0, 0),
                matrix_at(&result.eigenvectors, 1, 0),
                matrix_at(&result.eigenvectors, 2, 0),
            ],
        );

        let av0 = &a * &v0;
        let lv0 = result.eigenvalues[0] * &v0;
        assert!(av0.approx_eq(&lv0, 1e-8));
    }

    #[test]
    fn qr_wilkinson_eigenpairs_rejects_nonsymmetric_matrix() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 0.0, 3.0]);

        assert!(qr_wilkinson_eigenpairs(&a, 50, 1e-10).is_none());
    }

    #[test]
    fn jacobi_eigenpairs_finds_symmetric_spectrum() {
        let a = Matrix::from_vec(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]);

        let result = jacobi_eigenpairs(&a, 200, 1e-12).expect("valid symmetric input");
        assert!(result.converged);

        for col in 0..3 {
            let lambda = result.eigenvalues[col];
            let v_col = Vector::from_vec(
                3,
                vec![
                    matrix_at(&result.eigenvectors, 0, col),
                    matrix_at(&result.eigenvectors, 1, col),
                    matrix_at(&result.eigenvectors, 2, col),
                ],
            );

            let av = &a * &v_col;
            let lv = lambda * &v_col;
            assert!(av.approx_eq(&lv, 1e-7));
        }
    }

    #[test]
    fn jacobi_eigenpairs_rejects_nonsymmetric_matrix() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 0.0, 3.0]);

        assert!(jacobi_eigenpairs(&a, 100, 1e-10).is_none());
    }

    fn sigma_matrix(rows: usize, cols: usize, singular_values: &Vector<f64>) -> Matrix<f64> {
        let mut sigma = Matrix::zeros(rows, cols);
        let diag_len = rows.min(cols).min(singular_values.dim());
        for i in 0..diag_len {
            matrix_set(&mut sigma, i, i, singular_values[i]);
        }
        sigma
    }

    #[test]
    fn svd_golub_kahan_reconstructs_tall_matrix() {
        let a = Matrix::from_vec(
            4,
            3,
            vec![3.0, 1.0, 1.0, -1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0],
        );

        let result = svd_golub_kahan(&a, 500, 1e-10).expect("valid input");
        assert!(result.converged());

        let sigma = sigma_matrix(a.rows(), a.cols(), result.singular_values());
        let us = result.u() * &sigma;
        let reconstructed = &us * result.v_t();
        assert_matrix_close(&a, &reconstructed, 1e-6);
    }

    #[test]
    fn svd_golub_kahan_reconstructs_wide_matrix() {
        let a = Matrix::from_vec(
            3,
            5,
            vec![
                2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
            ],
        );

        let result = svd_golub_kahan(&a, 500, 1e-10).expect("valid input");
        assert!(result.converged());

        let sigma = sigma_matrix(a.rows(), a.cols(), result.singular_values());
        let us = result.u() * &sigma;
        let reconstructed = &us * result.v_t();
        assert_matrix_close(&a, &reconstructed, 1e-6);
    }

    #[test]
    fn svd_golub_kahan_returns_sorted_nonnegative_singular_values() {
        let a = Matrix::from_vec(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);

        let result = svd_golub_kahan(&a, 500, 1e-10).expect("valid input");
        assert!(result.converged());

        for i in 0..result.singular_values().dim() {
            assert!(result.singular_values()[i] >= -1e-12);
            if i + 1 < result.singular_values().dim() {
                assert!(result.singular_values()[i] + 1e-12 >= result.singular_values()[i + 1]);
            }
        }
    }

    #[test]
    fn nalgebra_parity_iterative_methods() {
        let (a, b, _) = sample_system();
        let x0 = Vector::zeros(4);

        let dm = to_dmatrix(&a);
        let db = to_dvector(&b);
        let x_ref = dm
            .lu()
            .solve(&db)
            .expect("nalgebra LU solve should succeed");
        let x_ref_vec = Vector::from_vec(4, x_ref.iter().copied().collect());

        let jacobi_res = jacobi(&a, &b, &x0, 500, 1e-12).expect("valid input");
        assert!(jacobi_res.converged);
        assert!(jacobi_res.x.approx_eq(&x_ref_vec, 1e-7));

        let seidel_res = gauss_seidel(&a, &b, &x0, 300, 1e-12).expect("valid input");
        assert!(seidel_res.converged);
        assert!(seidel_res.x.approx_eq(&x_ref_vec, 1e-9));

        let simple_res = simple_iteration(&a, &b, &x0, 0.05, 3_000, 1e-12).expect("valid input");
        assert!(simple_res.converged);
        assert!(simple_res.x.approx_eq(&x_ref_vec, 1e-6));

        let gmres_res = gmres(&a, &b, &x0, 30, 1e-12).expect("valid input");
        assert!(gmres_res.converged);
        assert!(gmres_res.x.approx_eq(&x_ref_vec, 1e-10));

        let restarted_res = gmres_restarted(&a, &b, &x0, 2, 50, 1e-12).expect("valid input");
        assert!(restarted_res.converged);
        assert!(restarted_res.x.approx_eq(&x_ref_vec, 1e-9));
    }

    #[test]
    fn nalgebra_parity_power_and_inverse_methods() {
        let a = Matrix::from_vec(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]);
        let x0 = Vector::from_array([1.0, 1.0, 1.0]);

        let se = nalgebra::linalg::SymmetricEigen::new(to_dmatrix(&a));
        let mut eigs = se.eigenvalues.iter().copied().collect::<Vec<_>>();
        eigs.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        let lambda_min = eigs[0];
        let lambda_max = *eigs.last().unwrap();

        let power_res = power_method(&a, &x0, 200, 1e-12).expect("valid input");
        assert!(power_res.converged);
        assert!((power_res.eigenvalue - lambda_max).abs() <= 1e-8);

        let inverse_res = inverse_power_method(&a, &x0, 200, 1e-12).expect("valid input");
        assert!(inverse_res.converged);
        assert!((inverse_res.eigenvalue - lambda_min).abs() <= 1e-8);
    }

    #[test]
    fn nalgebra_parity_qr_wilkinson_and_jacobi_eigen() {
        let a_qr = Matrix::from_vec(
            4,
            4,
            vec![
                7.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );

        let se_qr = nalgebra::linalg::SymmetricEigen::new(to_dmatrix(&a_qr));
        let mut eig_ref_qr = se_qr.eigenvalues.iter().copied().collect::<Vec<_>>();
        eig_ref_qr.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());

        let qr_vals = qr_wilkinson_eigenvalues(&a_qr, 2_000, 1e-12).expect("valid input");
        assert_slice_close(&sorted_values(&qr_vals.eigenvalues), &eig_ref_qr, 1e-8);

        let qr_pairs = qr_wilkinson_eigenpairs(&a_qr, 2_000, 1e-12).expect("valid input");
        assert_slice_close(&sorted_values(&qr_pairs.eigenvalues), &eig_ref_qr, 1e-8);

        let a_jacobi = Matrix::from_vec(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]);
        let se_jacobi = nalgebra::linalg::SymmetricEigen::new(to_dmatrix(&a_jacobi));
        let mut eig_ref_jacobi = se_jacobi.eigenvalues.iter().copied().collect::<Vec<_>>();
        eig_ref_jacobi.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());

        let jacobi_pairs = jacobi_eigenpairs(&a_jacobi, 500, 1e-12).expect("valid input");
        assert!(jacobi_pairs.converged);
        assert_slice_close(
            &sorted_values(&jacobi_pairs.eigenvalues),
            &eig_ref_jacobi,
            1e-8,
        );
    }

    #[test]
    fn nalgebra_parity_svd_golub_kahan() {
        let a = Matrix::from_vec(
            4,
            3,
            vec![3.0, 1.0, 1.0, -1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0],
        );

        let ours = svd_golub_kahan(&a, 500, 1e-10).expect("valid input");
        assert!(ours.converged());

        let na = nalgebra::linalg::SVD::new(to_dmatrix(&a), true, true);
        let sv_ref = na.singular_values.iter().copied().collect::<Vec<_>>();

        assert_slice_close(ours.singular_values().as_slice(), &sv_ref, 1e-6);

        let sigma = sigma_matrix(a.rows(), a.cols(), ours.singular_values());
        let us = ours.u() * &sigma;
        let reconstructed = &us * ours.v_t();
        assert_matrix_close(&a, &reconstructed, 1e-6);
    }

    #[test]
    fn decomposition_structs_support_new_style_constructor() {
        let a_square = Matrix::from_vec(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]);
        let b = Vector::from_array([1.0, 2.0, 3.0]);
        let x0 = Vector::from_array([1.0, 1.0, 1.0]);

        let lu_dec = LU::new(&a_square).expect("LU::new should succeed");
        assert!(lu_dec.solve(&b).is_some());

        let ldlt_dec = crate::decomp::ldlt::LDLT::new(&a_square).expect("LDLT::new should succeed");
        assert!(ldlt_dec.solve(&b).is_some());

        let qr_dec = QR::new(&a_square).expect("QR::new should succeed");
        assert!(qr_dec.solve(&b).is_some());

        let _power =
            EigenResult::new_power(&a_square, &x0).expect("EigenResult::new_power should succeed");
        let _inverse = EigenResult::new_inverse(&a_square, &x0)
            .expect("EigenResult::new_inverse should succeed");

        let _qr_vals =
            QrEigenvaluesResult::new(&a_square).expect("QrEigenvaluesResult::new should succeed");
        let _qr_pairs =
            QrEigenpairsResult::new(&a_square).expect("QrEigenpairsResult::new should succeed");
        let _jacobi =
            JacobiEigenResult::new(&a_square).expect("JacobiEigenResult::new should succeed");

        let a_rect = Matrix::from_vec(
            4,
            3,
            vec![3.0, 1.0, 1.0, -1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0],
        );
        let _svd = SVD::new(&a_rect).expect("SVD::new should succeed");
    }
}
