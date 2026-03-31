use crate::dense::Vector;

/// Result of an iterative linear-system solver.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeResult {
    pub x: Vector<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_delta: f64,
}
