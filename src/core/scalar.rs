use std::ops::{Add, AddAssign, Mul, Neg, Sub};

pub trait Scalar:
    Copy
    + Clone
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + From<u8>
    + AddAssign
{
}

impl<T> Scalar for T where
    T: Copy
        + Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Neg<Output = T>
        + Mul<Output = T>
        + From<u8>
        + AddAssign
{
}
