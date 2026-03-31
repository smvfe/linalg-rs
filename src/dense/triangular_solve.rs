use crate::dense::{Matrix, Vector};

pub fn solve_lower_triangular(l: &Matrix<f64>, b: &Vector<f64>) -> Option<Vector<f64>> {
    l.solve_lower_triangular(b)
}

pub fn solve_upper_triangular(u: &Matrix<f64>, b: &Vector<f64>) -> Option<Vector<f64>> {
    u.solve_upper_triangular(b)
}
