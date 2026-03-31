pub mod numeric_utils;
pub mod scalar;
pub mod tolerance;
pub mod validation;

pub use scalar::Scalar;
pub use tolerance::relative_tol;
pub use validation::{check_nonzero_diagonal, validate_linear_system};
