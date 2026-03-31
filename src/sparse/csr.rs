use crate::matrix::Matrix;
use crate::scalar::Scalar;
use crate::vector::Vector;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Index, Mul, Neg};

use super::csc::CscMatrix;

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct CsrMatrix<T = f64> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_ptr: Vec<usize>,
    pub(crate) col_idx: Vec<usize>,
    pub(crate) data: Vec<T>,
}

impl<T> CsrMatrix<T> {
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

    pub fn into_transpose_csc(self) -> CscMatrix<T> {
        CscMatrix {
            rows: self.cols,
            cols: self.rows,
            col_ptr: self.row_ptr,
            row_idx: self.col_idx,
            data: self.data,
        }
    }

    pub fn transpose_csc(self) -> CscMatrix<T> {
        self.into_transpose_csc()
    }

    #[inline]
    pub fn row_data(&self, i: usize) -> (&[usize], &[T]) {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        (&self.col_idx[start..end], &self.data[start..end])
    }

    #[inline]
    pub fn row_nnz(&self, i: usize) -> usize {
        self.row_ptr[i + 1] - self.row_ptr[i]
    }

    pub fn is_valid(&self) -> bool {
        if self.row_ptr.len() != self.rows + 1 {
            return false;
        }
        if self.col_idx.len() != self.data.len() {
            return false;
        }
        if *self.row_ptr.last().unwrap_or(&0) != self.data.len() {
            return false;
        }

        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            if start > end || end > self.data.len() {
                return false;
            }
            for k in start..end {
                if self.col_idx[k] >= self.cols {
                    return false;
                }
            }
            for k in (start + 1)..end {
                if self.col_idx[k] <= self.col_idx[k - 1] {
                    return false;
                }
            }
        }

        true
    }

    pub fn iter_nonzero(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.rows).flat_map(move |i| {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            (start..end).map(move |k| (i, self.col_idx[k], &self.data[k]))
        })
    }
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn from_dense(matrix: &Matrix<T>) -> Self {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let dense = matrix.as_slice();

        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::new();
        let mut data = Vec::new();

        let zero = T::default();

        row_ptr.push(0);

        for i in 0..rows {
            let row_start = i * cols;
            for j in 0..cols {
                let value = dense[row_start + j];
                if value != zero {
                    col_idx.push(j);
                    data.push(value);
                }
            }
            row_ptr.push(col_idx.len());
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            data,
        }
    }

    pub fn empty(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn to_csc(&self) -> CscMatrix<T> {
        let nnz = self.nnz();
        let mut col_ptr = vec![0usize; self.cols + 1];
        let mut row_idx = vec![0usize; nnz];
        let mut data = vec![T::default(); nnz];

        for &col in &self.col_idx {
            col_ptr[col + 1] += 1;
        }
        for j in 1..=self.cols {
            col_ptr[j] += col_ptr[j - 1];
        }

        let mut pos = col_ptr[..self.cols].to_vec();

        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let col = self.col_idx[k];
                let dest = pos[col];
                row_idx[dest] = i;
                data[dest] = self.data[k];
                pos[col] += 1;
            }
        }

        CscMatrix {
            rows: self.rows,
            cols: self.cols,
            col_ptr,
            row_idx,
            data,
        }
    }

    pub fn transpose(&self) -> CsrMatrix<T> {
        self.to_csc().into_transpose_csr()
    }
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn to_dense(&self) -> Matrix<T> {
        let mut result = Matrix::zeros(self.rows, self.cols);
        let result_data = result.as_mut_slice();
        let cols = self.cols;

        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            for k in start..end {
                result_data[i * cols + self.col_idx[k]] = self.data[k];
            }
        }

        result
    }
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");

        let row_start = self.row_ptr[i];
        let row_end = self.row_ptr[i + 1];

        match self.col_idx[row_start..row_end].binary_search(&j) {
            Ok(local_idx) => self.data[row_start + local_idx],
            Err(_) => T::default(),
        }
    }
}

impl<T: Scalar> Index<(usize, usize)> for CsrMatrix<T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");

        let row_start = self.row_ptr[i];
        let row_end = self.row_ptr[i + 1];

        match self.col_idx[row_start..row_end].binary_search(&j) {
            Ok(local_idx) => &self.data[row_start + local_idx],
            Err(_) => panic!(
                "Element at ({}, {}) is zero (not stored in sparse matrix)",
                i, j
            ),
        }
    }
}

impl<T: Scalar> Neg for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn neg(self) -> CsrMatrix<T> {
        let mut result = self.clone();
        for value in &mut result.data {
            *value = -(*value);
        }
        result
    }
}

impl<T: Scalar> Mul<T> for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn mul(self, scalar: T) -> CsrMatrix<T> {
        let mut result = self.clone();
        for value in &mut result.data {
            *value = scalar * *value;
        }
        result
    }
}

impl<T: Scalar> CsrMatrix<T> {
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

impl<T: Scalar> Mul<&Vector<T>> for &CsrMatrix<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: &Vector<T>) -> Vector<T> {
        assert_eq!(
            self.cols,
            rhs.dim(),
            "Dim mismatch for matrix-vector multiplication"
        );

        let mut acc = vec![T::default(); self.rows];

        let x = rhs.as_slice();
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = T::default();

            for k in start..end {
                sum += self.data[k] * x[self.col_idx[k]];
            }
            acc[i] = sum;
        }

        Vector::from_vec(self.rows, acc)
    }
}

impl<T: Scalar> Mul<&Matrix<T>> for &CsrMatrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, rhs.rows(), "Dim mismatch for multiplication");

        let m = self.rows;
        let n = rhs.cols();
        let rhs_data = rhs.as_slice();
        let mut acc = Matrix::zeros(m, n);
        let acc_data = acc.as_mut_slice();

        for i in 0..m {
            let c_row = &mut acc_data[i * n..(i + 1) * n];
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let col_k = self.col_idx[k];
                let val = self.data[k];

                let b_row = &rhs_data[col_k * n..(col_k + 1) * n];
                for j in 0..n {
                    c_row[j] += val * b_row[j];
                }
            }
        }

        acc
    }
}

impl CsrMatrix<f64> {
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.row_ptr == other.row_ptr
            && self.col_idx == other.col_idx
            && self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() <= tol)
    }
}

impl Display for CsrMatrix<f64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(
            f,
            "CsrMatrix {}x{}, nnz={}, density={:.2}%",
            self.rows,
            self.cols,
            self.nnz(),
            self.density() * 100.0
        )?;

        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                writeln!(f, "  ({}, {}) = {:.6}", i, self.col_idx[k], self.data[k])?;
            }
        }

        Ok(())
    }
}
