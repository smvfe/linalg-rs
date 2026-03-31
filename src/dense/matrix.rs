use crate::scalar::Scalar;
use crate::vector::Vector;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, Index, Mul, Neg, Sub};

// ======================================================
// Dense matrix (row-major)
// ======================================================
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct Matrix<T = f64> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<T>,
}

// IndexMut intentionally not implemented.
// Use get_mut() instead to avoid borrow checker issues
// with simultaneous reads and writes.
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        let linear_idx = self.cols * row + col;
        &self.data[linear_idx]
    }
}
// ======================================================

// Constructors ==========================================
impl<T: Scalar> Matrix<T> {
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "Dimensions mismatch");

        Self { rows, cols, data }
    }

    pub fn from_slice(rows: usize, cols: usize, data: &[T]) -> Self {
        assert_eq!(data.len(), rows * cols, "Dimensions mismatch");

        Self {
            rows,
            cols,
            data: data.to_vec(),
        }
    }

    pub fn from_array<const N: usize>(rows: usize, cols: usize, data: [T; N]) -> Self {
        assert_eq!(N, rows * cols, "Dimensions mismatch");

        Self {
            rows,
            cols,
            data: Vec::from(data),
        }
    }

    pub fn from_fn<F: Fn(usize, usize) -> T>(rows: usize, cols: usize, f: F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }

        Self { rows, cols, data }
    }

    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub fn from_diagonal(diag: &[T]) -> Self {
        let n = diag.len();
        let mut m = Self::zeros(n, n);
        for (i, &value) in diag.iter().enumerate() {
            m.data[i * n + i] = value;
        }
        m
    }

    // Utility constructors ==============================
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1u8.into();
        }
        m
    }
}

// Basic access ==========================================
impl<T: Scalar> Matrix<T> {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub(crate) fn swap_rows(&mut self, lhs: usize, rhs: usize) {
        debug_assert!(
            lhs < self.rows && rhs < self.rows,
            "Row index out of bounds"
        );
        if lhs == rhs {
            return;
        }

        let cols = self.cols;
        let (low, high) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };

        let split_at = high * cols;
        let (left, right) = self.data.split_at_mut(split_at);
        let low_row = &mut left[(low * cols)..((low + 1) * cols)];
        let high_row = &mut right[..cols];
        low_row.swap_with_slice(high_row);
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.cols + j]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.cols + j]
    }

    #[inline]
    pub fn row_slice(&self, i: usize) -> &[T] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row(&self, i: usize) -> Vec<T> {
        self.row_slice(i).to_vec()
    }

    pub fn col(&self, j: usize) -> Vec<T> {
        (0..self.rows)
            .map(|i| self.data[i * self.cols + j])
            .collect()
    }

    pub fn diagonal(&self) -> Vec<T> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.data[i * self.cols + i]).collect()
    }
}

// Matrix transforms and reductions ======================
impl<T: Scalar> Matrix<T> {
    pub fn trace(&self) -> T {
        // Sum of diagonal elements.
        assert_eq!(self.rows, self.cols, "Trace defined only for square matrix");
        let mut acc = T::default();

        for i in 0..self.rows {
            acc += self.data[i * self.cols + i];
        }

        acc
    }

    pub fn transpose(&self) -> Matrix<T> {
        // Build transposed matrix by swapping indices (i, j) -> (j, i).
        let mut result = Matrix::zeros(self.cols, self.rows);

        for j in 0..self.cols {
            let dst_start = j * self.rows;
            for i in 0..self.rows {
                result.data[dst_start + i] = self.data[i * self.cols + j];
            }
        }

        result
    }

    pub fn transpose_inplace(&mut self) {
        assert!(
            self.is_square(),
            "In-place transpose requires square matrix"
        );
        let n = self.rows;

        for i in 0..n {
            for j in (i + 1)..n {
                self.data.swap(i * n + j, j * n + i);
            }
        }
    }

    pub fn map<F: Fn(T) -> T>(&self, f: F) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    pub fn map_inplace<F: Fn(T) -> T>(&mut self, f: F) {
        for value in &mut self.data {
            *value = f(*value);
        }
    }

    pub fn zip_map<F: Fn(T, T) -> T>(&self, other: &Matrix<T>, f: F) -> Matrix<T> {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }
}

impl Matrix<f64> {
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() <= tol)
    }

    pub fn norm_frobenius(&self) -> f64 {
        self.data
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    pub fn norm_max(&self) -> f64 {
        self.data
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Solves `L*x = b` for lower-triangular `L` by forward substitution.
    pub fn solve_lower_triangular(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        if !self.is_square() || b.dim() != self.rows {
            return None;
        }

        let n = self.rows;
        let max_abs = self
            .data
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let tol = f64::EPSILON * (n as f64) * max_abs;

        for i in 0..n {
            for j in (i + 1)..n {
                if self[(i, j)].abs() > tol {
                    return None;
                }
            }
        }

        let mut x = Vector::zeros(n);
        for i in 0..n {
            let mut rhs = b[i];
            for j in 0..i {
                rhs -= self[(i, j)] * x[j];
            }

            let diag = self[(i, i)];
            if diag.abs() <= tol {
                return None;
            }
            x[i] = rhs / diag;
        }

        Some(x)
    }

    /// Solves `U*x = b` for upper-triangular `U` by backward substitution.
    pub fn solve_upper_triangular(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        if !self.is_square() || b.dim() != self.rows {
            return None;
        }

        let n = self.rows;
        let max_abs = self
            .data
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let tol = f64::EPSILON * (n as f64) * max_abs;

        for i in 1..n {
            for j in 0..i {
                if self[(i, j)].abs() > tol {
                    return None;
                }
            }
        }

        let mut x = Vector::zeros(n);
        for rev_i in 0..n {
            let i = n - 1 - rev_i;
            let mut rhs = b[i];
            for j in (i + 1)..n {
                rhs -= self[(i, j)] * x[j];
            }

            let diag = self[(i, i)];
            if diag.abs() <= tol {
                return None;
            }
            x[i] = rhs / diag;
        }

        Some(x)
    }
}

impl Display for Matrix<f64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for i in 0..self.rows {
            write!(f, "│")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:8.4}", self.data[i * self.cols + j])?;
            }
            writeln!(f, " │")?;
        }
        Ok(())
    }
}

// ======================================================

// Arithmetic ops ========================================
impl<T: Scalar> Add for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, rhs.rows, "Rows mismatch");
        assert_eq!(self.cols, rhs.cols, "Cols mismatch");

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&x, &y)| x + y)
                .collect(),
        }
    }
}

impl<T: Scalar> Sub for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, rhs.rows, "Rows mismatch");
        assert_eq!(self.cols, rhs.cols, "Cols mismatch");

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&x, &y)| x - y)
                .collect(),
        }
    }
}

impl<T: Scalar> Neg for &Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| -x).collect(),
        }
    }
}

// Mul: matrix * scalar ----------------------------------
impl<T: Scalar> Mul<T> for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, scalar: T) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| scalar * x).collect(),
        }
    }
}

impl Mul<&Matrix<f64>> for f64 {
    type Output = Matrix<f64>;
    fn mul(self, m: &Matrix<f64>) -> Matrix<f64> {
        m * self
    }
}

// Mul: matrix * vector ----------------------------------
impl<T: Scalar> Mul<&Vector<T>> for &Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: &Vector<T>) -> Vector<T> {
        assert_eq!(
            self.cols,
            rhs.dim(),
            "Dim mismatch for matrix-vector multiplication"
        );

        let mut acc = vec![T::default(); self.rows];
        let rhs_data = rhs.as_slice();
        let cols = self.cols;

        for i in 0..self.rows {
            let row_start = i * cols;
            let mut sum = T::default();
            for j in 0..cols {
                sum += self.data[row_start + j] * rhs_data[j];
            }
            acc[i] = sum;
        }

        Vector::from_vec(self.rows, acc)
    }
}

// Mul: matrix * matrix ----------------------------------
impl<T: Scalar> Mul for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, rhs.rows, "Dim mismatch for multiplication");

        let m = self.rows;
        let k_dim = self.cols;
        let n = rhs.cols;
        let mut data = vec![T::default(); m * n];

        for i in 0..m {
            let c_row = i * n;
            let a_row = i * k_dim;
            for k in 0..k_dim {
                let a_ik = self.data[a_row + k];
                let b_row = k * n;
                for j in 0..n {
                    data[c_row + j] += a_ik * rhs.data[b_row + j];
                }
            }
        }

        Matrix {
            rows: m,
            cols: n,
            data,
        }
    }
}

// =======================================================
