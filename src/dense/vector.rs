use crate::scalar::Scalar;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

// ======================================================
// Dense vector
// ======================================================
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct Vector<T = f64> {
    pub(crate) data: Vec<T>,
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}
// =======================================================

// Constructors ==========================================
impl<T> Vector<T> {
    /// Builds a vector from owned data.
    ///
    /// # Arguments
    /// - `data`: Backing data in dense contiguous layout.
    ///
    /// # Returns
    /// A new vector with dimension `data.len()`.
    pub fn from_data(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Builds a vector from owned data and checks expected dimension.
    ///
    /// # Arguments
    /// - `dim`: Expected dimension.
    /// - `data`: Backing data.
    ///
    /// # Returns
    /// A new vector when dimensions match.
    ///
    /// # Panics
    /// Panics if `data.len() != dim`.
    pub fn from_vec(dim: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), dim, "Dimensions mismatch");
        Self { data }
    }

    /// Builds a vector by evaluating function `f(i)` for each index.
    ///
    /// # Arguments
    /// - `dim`: Dimension of the vector.
    /// - `f`: Element generator.
    ///
    /// # Returns
    /// A new vector with generated values.
    pub fn from_fn<F: Fn(usize) -> T>(dim: usize, f: F) -> Self {
        let mut data = Vec::with_capacity(dim);
        for i in 0..dim {
            data.push(f(i));
        }
        Self { data }
    }
}

// Additional constructors ===============================
impl<T: Clone> Vector<T> {
    /// Builds a vector from slice with explicit dimension check.
    ///
    /// # Panics
    /// Panics if `data.len() != dim`.
    pub fn from_slice(dim: usize, data: &[T]) -> Self {
        assert_eq!(data.len(), dim, "Dimensions mismatch");
        Self {
            data: data.to_vec(),
        }
    }

    /// Builds a vector from slice.
    pub fn from_data_slice(data: &[T]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Builds a vector from fixed-size array.
    pub fn from_array<const N: usize>(data: [T; N]) -> Self {
        Self {
            data: Vec::from(data),
        }
    }

    /// Creates a vector filled with the same value.
    pub fn filled(dim: usize, value: T) -> Self {
        Self {
            data: vec![value; dim],
        }
    }
}

// Basic access ==========================================
impl<T> Vector<T> {
    /// Returns vector dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if vector has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns immutable contiguous view of vector data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns mutable contiguous view of vector data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns immutable reference to element `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> &T {
        &self.data[i]
    }

    /// Returns mutable reference to element `i`.
    #[inline(always)]
    pub fn get_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

// Utility constructors ==================================
impl<T: Scalar> Vector<T> {
    /// Creates all-zero vector of given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![T::default(); dim],
        }
    }

    /// Creates all-ones vector of given dimension.
    pub fn ones(dim: usize) -> Self {
        Self {
            data: vec![T::from(1u8); dim],
        }
    }

    /// Creates unit basis vector `e_index`.
    ///
    /// # Panics
    /// Panics if `index >= dim`.
    pub fn unit(dim: usize, index: usize) -> Self {
        assert!(index < dim, "Index out of bounds for unit vector");
        let mut out = Self::zeros(dim);
        out.data[index] = T::from(1u8);
        out
    }
}

// Dot product ===========================================
impl<T: Scalar> Vector<T> {
    /// Computes dot product.
    ///
    /// # Notes
    /// Uses a tight index-based loop with local accumulator. This improves
    /// autovectorization chances and avoids iterator adaptor overhead.
    pub fn dot(&self, rhs: &Vector<T>) -> T {
        assert_eq!(self.data.len(), rhs.data.len(), "Dimensions mismatch");
        let lhs_data = &self.data;
        let rhs_data = &rhs.data;
        let mut sum = T::default();

        for i in 0..lhs_data.len() {
            sum += lhs_data[i] * rhs_data[i];
        }

        sum
    }

    /// Computes sum of all elements.
    pub fn sum(&self) -> T {
        let mut acc = T::default();
        for &value in &self.data {
            acc += value;
        }
        acc
    }

    /// Returns mapped copy of the vector.
    pub fn map<F: Fn(T) -> T>(&self, f: F) -> Vector<T> {
        Vector {
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Maps all elements in place.
    pub fn map_inplace<F: Fn(T) -> T>(&mut self, f: F) {
        for value in &mut self.data {
            *value = f(*value);
        }
    }

    /// Maps pairwise elements from two vectors.
    pub fn zip_map<F: Fn(T, T) -> T>(&self, other: &Vector<T>, f: F) -> Vector<T> {
        debug_assert_eq!(self.data.len(), other.data.len());
        Vector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }
}

impl Vector<f64> {
    /// Checks approximate equality with absolute tolerance.
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() <= tol)
    }

    /// L2 norm.
    pub fn norm_l2(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// L1 norm.
    pub fn norm_l1(&self) -> f64 {
        self.data.iter().map(|value| value.abs()).sum()
    }

    /// Infinity norm.
    pub fn norm_inf(&self) -> f64 {
        self.data
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Returns normalized vector copy.
    pub fn normalize(&self) -> Vector<f64> {
        let norm = self.norm_l2();
        assert!(norm > f64::EPSILON, "Cannot normalize zero vector");
        self * (1.0 / norm)
    }

    /// Normalizes vector in place.
    ///
    /// # Notes
    /// Uses precomputed reciprocal to replace repeated division by multiplication.
    pub fn normalize_inplace(&mut self) {
        let norm = self.norm_l2();
        assert!(norm > f64::EPSILON, "Cannot normalize zero vector");
        let inv = 1.0 / norm;
        for value in &mut self.data {
            *value *= inv;
        }
    }

    /// Returns minimum element.
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Returns maximum element.
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Returns arithmetic mean.
    pub fn mean(&self) -> f64 {
        self.sum() / self.dim() as f64
    }

    /// Returns index of minimum element.
    pub fn argmin(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .min_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
            .unwrap()
            .0
    }

    /// Returns index of maximum element.
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
            .unwrap()
            .0
    }

    /// Computes Euclidean distance between vectors.
    pub fn distance(&self, other: &Vector<f64>) -> f64 {
        debug_assert_eq!(self.dim(), other.dim());

        let lhs = &self.data;
        let rhs = &other.data;
        let mut sum = 0.0;

        for i in 0..lhs.len() {
            let diff = lhs[i] - rhs[i];
            sum += diff * diff;
        }

        sum.sqrt()
    }

    /// Computes 3D cross product.
    pub fn cross(&self, rhs: &Vector<f64>) -> Vector<f64> {
        assert_eq!(
            self.dim(),
            3,
            "Cross product is defined only for 3D vectors"
        );
        assert_eq!(rhs.dim(), 3, "Cross product is defined only for 3D vectors");

        Vector {
            data: vec![
                self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1],
                self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2],
                self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0],
            ],
        }
    }
}

impl Display for Vector<f64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "[")?;
        for (index, value) in self.data.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", value)?;
        }
        write!(f, "]")
    }
}

// =======================================================

// Arithmetic ops ========================================
impl<T: Scalar> Add for &Vector<T> {
    type Output = Vector<T>;

    fn add(self, rhs: &Vector<T>) -> Vector<T> {
        assert_eq!(self.data.len(), rhs.data.len(), "Dimensions mismatch");
        Vector {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&x, &y)| x + y)
                .collect(),
        }
    }
}

impl<T: Scalar> AddAssign<&Vector<T>> for Vector<T> {
    fn add_assign(&mut self, rhs: &Vector<T>) {
        debug_assert_eq!(self.data.len(), rhs.data.len());
        // Optimization: in-place loop avoids temporary allocation.
        for (lhs, &rhs_value) in self.data.iter_mut().zip(rhs.data.iter()) {
            *lhs += rhs_value;
        }
    }
}

impl<T: Scalar> Sub for &Vector<T> {
    type Output = Vector<T>;

    fn sub(self, rhs: &Vector<T>) -> Vector<T> {
        assert_eq!(self.data.len(), rhs.data.len(), "Dimensions mismatch");
        Vector {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&x, &y)| x - y)
                .collect(),
        }
    }
}

impl<T: Scalar> SubAssign<&Vector<T>> for Vector<T> {
    fn sub_assign(&mut self, rhs: &Vector<T>) {
        debug_assert_eq!(self.data.len(), rhs.data.len());
        // Optimization: in-place loop avoids temporary allocation.
        for (lhs, &rhs_value) in self.data.iter_mut().zip(rhs.data.iter()) {
            *lhs = *lhs - rhs_value;
        }
    }
}

impl<T: Scalar> Neg for &Vector<T> {
    type Output = Vector<T>;

    fn neg(self) -> Vector<T> {
        Vector {
            data: self.data.iter().map(|&x| -x).collect(),
        }
    }
}

impl<T: Scalar> Mul<T> for &Vector<T> {
    type Output = Vector<T>;

    fn mul(self, scalar: T) -> Vector<T> {
        Vector {
            data: self.data.iter().map(|&x| scalar * x).collect(),
        }
    }
}

impl<T: Scalar> MulAssign<T> for Vector<T> {
    fn mul_assign(&mut self, scalar: T) {
        // Optimization: single pass in-place scaling.
        for value in &mut self.data {
            *value = *value * scalar;
        }
    }
}

impl<T: Scalar> Add for Vector<T> {
    type Output = Vector<T>;
    fn add(self, rhs: Vector<T>) -> Vector<T> {
        &self + &rhs
    }
}

impl<T: Scalar> Add<&Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn add(self, rhs: &Vector<T>) -> Vector<T> {
        &self + rhs
    }
}

impl<T: Scalar> Add<Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    fn add(self, rhs: Vector<T>) -> Vector<T> {
        self + &rhs
    }
}

impl<T: Scalar> Sub for Vector<T> {
    type Output = Vector<T>;
    fn sub(self, rhs: Vector<T>) -> Vector<T> {
        &self - &rhs
    }
}

impl<T: Scalar> Sub<&Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn sub(self, rhs: &Vector<T>) -> Vector<T> {
        &self - rhs
    }
}

impl<T: Scalar> Sub<Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    fn sub(self, rhs: Vector<T>) -> Vector<T> {
        self - &rhs
    }
}

impl Mul<&Vector<f64>> for f64 {
    type Output = Vector<f64>;
    fn mul(self, v: &Vector<f64>) -> Vector<f64> {
        v * self
    }
}

// =======================================================
