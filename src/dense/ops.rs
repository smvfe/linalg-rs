use crate::dense::{Matrix, Vector};

pub fn matmul(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
    a * b
}

pub fn matvec(a: &Matrix<f64>, x: &Vector<f64>) -> Vector<f64> {
    a * x
}
