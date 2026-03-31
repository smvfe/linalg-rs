use crate::dense::Vector;

pub trait DecompositionSolve {
    fn solve(&self, b: &Vector<f64>) -> Option<Vector<f64>>;
}

impl DecompositionSolve for crate::lu::LU {
    fn solve(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        crate::lu::LU::solve(self, b)
    }
}

impl DecompositionSolve for crate::cholesky::LDLT {
    fn solve(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        crate::cholesky::LDLT::solve(self, b)
    }
}

impl DecompositionSolve for crate::qr::QR {
    fn solve(&self, b: &Vector<f64>) -> Option<Vector<f64>> {
        crate::qr::QR::solve(self, b)
    }
}
