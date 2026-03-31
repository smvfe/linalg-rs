use crate::matrix::Matrix;
use crate::scalar::Scalar;
use crate::vector::Vector;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Index, Mul, Neg};

use super::csr::CsrMatrix;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct CscMatrix<T = f64> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) col_ptr: Vec<usize>,
    pub(crate) row_idx: Vec<usize>,
    pub(crate) data: Vec<T>,
}

impl<T> CscMatrix<T> {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn density(&self) -> f64 {
        if self.rows == 0 || self.cols == 0 {
            return 0.0;
        }
        self.nnz() as f64 / (self.rows * self.cols) as f64
    }

    pub fn into_transpose_csr(self) -> CsrMatrix<T> {
        CsrMatrix {
            rows: self.cols,
            cols: self.rows,
            row_ptr: self.col_ptr,
            col_idx: self.row_idx,
            data: self.data,
        }
    }

    pub fn transpose_csr(self) -> CsrMatrix<T> {
        self.into_transpose_csr()
    }

    #[inline]
    pub fn col_data(&self, j: usize) -> (&[usize], &[T]) {
        let start = self.col_ptr[j];
        let end = self.col_ptr[j + 1];
        (&self.row_idx[start..end], &self.data[start..end])
    }

    #[inline]
    pub fn col_nnz(&self, j: usize) -> usize {
        self.col_ptr[j + 1] - self.col_ptr[j]
    }

    pub fn is_valid(&self) -> bool {
        if self.col_ptr.len() != self.cols + 1 {
            return false;
        }
        if self.row_idx.len() != self.data.len() {
            return false;
        }
        if *self.col_ptr.last().unwrap_or(&0) != self.data.len() {
            return false;
        }

        for j in 0..self.cols {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            if start > end || end > self.data.len() {
                return false;
            }
            for k in start..end {
                if self.row_idx[k] >= self.rows {
                    return false;
                }
            }
            for k in (start + 1)..end {
                if self.row_idx[k] <= self.row_idx[k - 1] {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: Scalar> CscMatrix<T> {
    pub fn from_dense(matrix: &Matrix<T>) -> Self {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let dense = matrix.as_slice();

        let mut col_ptr = Vec::with_capacity(cols + 1);
        let mut row_idx = Vec::new();
        let mut data = Vec::new();

        let zero = T::default();

        col_ptr.push(0);

        for j in 0..cols {
            for i in 0..rows {
                let value = dense[i * cols + j];
                if value != zero {
                    row_idx.push(i);
                    data.push(value);
                }
            }
            col_ptr.push(row_idx.len());
        }

        Self {
            rows,
            cols,
            col_ptr,
            row_idx,
            data,
        }
    }

    pub fn empty(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptr: vec![0; cols + 1],
            row_idx: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn to_csr(&self) -> CsrMatrix<T> {
        let nnz = self.nnz();
        let mut row_ptr = vec![0usize; self.rows + 1];
        let mut col_idx = vec![0usize; nnz];
        let mut data = vec![T::default(); nnz];

        for &row in &self.row_idx {
            row_ptr[row + 1] += 1;
        }
        for i in 1..=self.rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        let mut pos = row_ptr[..self.rows].to_vec();
        for j in 0..self.cols {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let row = self.row_idx[k];
                let dest = pos[row];
                col_idx[dest] = j;
                data[dest] = self.data[k];
                pos[row] += 1;
            }
        }

        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            row_ptr,
            col_idx,
            data,
        }
    }

    pub fn transpose(&self) -> CscMatrix<T> {
        self.to_csr().into_transpose_csc()
    }
}

impl<T: Scalar> CscMatrix<T> {
    pub fn to_dense(&self) -> Matrix<T> {
        let mut result = Matrix::zeros(self.rows, self.cols);
        let result_data = result.as_mut_slice();
        let cols = self.cols;

        for j in 0..self.cols {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];

            for k in start..end {
                result_data[self.row_idx[k] * cols + j] = self.data[k];
            }
        }

        result
    }
}

impl<T: Scalar> CscMatrix<T> {
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");

        let col_start = self.col_ptr[j];
        let col_end = self.col_ptr[j + 1];

        match self.row_idx[col_start..col_end].binary_search(&i) {
            Ok(local_idx) => self.data[col_start + local_idx],
            Err(_) => T::default(),
        }
    }
}

impl<T: Scalar> Index<(usize, usize)> for CscMatrix<T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");

        let col_start = self.col_ptr[j];
        let col_end = self.col_ptr[j + 1];

        match self.row_idx[col_start..col_end].binary_search(&i) {
            Ok(local_idx) => &self.data[col_start + local_idx],
            Err(_) => panic!(
                "Element at ({}, {}) is zero (not stored in sparse matrix)",
                i, j
            ),
        }
    }
}

impl<T: Scalar> Neg for &CscMatrix<T> {
    type Output = CscMatrix<T>;

    fn neg(self) -> CscMatrix<T> {
        let mut result = self.clone();
        for value in &mut result.data {
            *value = -(*value);
        }
        result
    }
}

impl<T: Scalar> Mul<T> for &CscMatrix<T> {
    type Output = CscMatrix<T>;

    fn mul(self, scalar: T) -> CscMatrix<T> {
        let mut result = self.clone();
        for value in &mut result.data {
            *value = scalar * *value;
        }
        result
    }
}

impl<T: Scalar> CscMatrix<T> {
    pub fn negate_inplace(&mut self) {
        for value in &mut self.data {
            *value = -(*value);
        }
    }

    pub fn scale_inplace(&mut self, scalar: T) {
        for value in &mut self.data {
            *value = *value * scalar;
        }
    }
}

impl<T: Scalar> Mul<&Vector<T>> for &CscMatrix<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: &Vector<T>) -> Vector<T> {
        assert_eq!(
            self.cols,
            rhs.dim(),
            "Dim mismatch for matrix-vector multiplication"
        );

        let x = rhs.as_slice();
        let mut acc = vec![T::default(); self.rows];
        let zero = T::default();

        for j in 0..self.cols {
            let xj = x[j];
            if xj == zero {
                continue;
            }

            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                acc[self.row_idx[k]] += self.data[k] * xj;
            }
        }

        Vector::from_vec(self.rows, acc)
    }
}

impl<T: Scalar> Mul<&CscMatrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &CscMatrix<T>) -> Matrix<T> {
        assert_eq!(self.cols(), rhs.rows(), "Dim mismatch for multiplication");

        let m = self.rows();
        let n = rhs.cols;
        let lhs_cols = self.cols();
        let lhs_data = self.as_slice();

        let mut acc: Matrix<T> = Matrix::zeros(m, n);
        let acc_data = acc.as_mut_slice();

        for j in 0..n {
            for k in rhs.col_ptr[j]..rhs.col_ptr[j + 1] {
                let sparse_row = rhs.row_idx[k];
                let sparse_value = rhs.data[k];

                for i in 0..m {
                    acc_data[i * n + j] += lhs_data[i * lhs_cols + sparse_row] * sparse_value;
                }
            }
        }

        acc
    }
}

impl CscMatrix<f64> {
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.col_ptr == other.col_ptr
            && self.row_idx == other.row_idx
            && self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() <= tol)
    }
}

impl Display for CscMatrix<f64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(
            f,
            "CscMatrix {}x{}, nnz={}, density={:.2}%",
            self.rows,
            self.cols,
            self.nnz(),
            self.density() * 100.0
        )?;

        for j in 0..self.cols {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                writeln!(f, "  ({}, {}) = {:.6}", self.row_idx[k], j, self.data[k])?;
            }
        }

        Ok(())
    }
}
