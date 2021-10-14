//! Provides a generic one-dimensional [`Array`] which wraps [`arrayfire::Array`] and supports
//! all [`number_general::NumberType`] variants. `Array` and `ArrayExt` support basic math like
//! `Add`, `Sub`, `Mul`, `Div`, and `Rem`, with hardware acceleration on systems which support CUDA
//! or OpenCL. `ArrayExt<bool>` supports common logical operations `and`, `not`, `or`, and `xor`.
//!
//! N-dimensional array functionality can be implemented using `Coords` and `Offsets`, which
//! provide methods for indexing a one-dimensional `Array` or `ArrayExt` as an n-dimensional
//! tensor.
//!
//! `Array` supports (de)serialization without type hinting. `ArrayExt<T>` supports serialization
//! for `T: Serialize` and deserialization for `T: Deserialize`.
//!
//! Example usage:
//! ```
//! # use std::iter::FromIterator;
//! # use afarray::Array;
//! # use number_general::Number;
//! let a = [1, 2, 3];
//! let b = [5];
//!
//! let product = &Array::from(&a[..]) * &Array::from(&b[..]);
//! assert_eq!(product, Array::from_iter(vec![5, 10, 15]));
//! assert_eq!(product.sum(), Number::from(30))
//! ```
//!
//! This crate depends on ArrayFire version 3.8. You will have to install ArrayFire separately by
//! following the instructions at
//! [https://arrayfire.org/docs/installing.htm](https://arrayfire.org/docs/installing.htm)
//! in order to build this crate.
//!
//! You can find detailed instructions for building the Rust `arrayfire` crate from crates.io at
//! [https://crates.io/crates/arrayfire](https://crates.io/crates/arrayfire).

use std::fmt;

pub use array::*;
pub use coords::*;
pub use ext::*;
pub use stream::*;

mod array;
mod coords;
mod ext;
mod stream;

pub type Complex<T> = num_complex::Complex<T>;

pub trait Sum {
    fn sum(self, other: Self) -> Self;
}

impl Sum for bool {
    fn sum(self, other: Self) -> Self {
        self || other
    }
}

macro_rules! sum {
    ($t:ty) => {
        impl Sum for $t {
            fn sum(self, other: Self) -> Self {
                self + other
            }
        }
    };
}

sum!(u8);
sum!(u16);
sum!(u32);
sum!(u64);
sum!(i16);
sum!(i32);
sum!(i64);
sum!(f32);
sum!(f64);
sum!(Complex<f32>);
sum!(Complex<f64>);

pub trait Product {
    fn product(self, other: Self) -> Self;
}

impl Product for bool {
    fn product(self, other: Self) -> Self {
        self && other
    }
}

macro_rules! product {
    ($t:ty) => {
        impl Product for $t {
            fn product(self, other: Self) -> Self {
                self * other
            }
        }
    };
}

product!(u8);
product!(u16);
product!(u32);
product!(u64);
product!(i16);
product!(i32);
product!(i64);
product!(f32);
product!(f64);
product!(Complex<f32>);
product!(Complex<f64>);

/// The error type used for Array which may fail recoverably.
pub struct ArrayError {
    message: String,
}

impl fmt::Debug for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// The result of an `Array` operation which may fail recoverably.
pub type Result<T> = std::result::Result<T, ArrayError>;

/// Call [`arrayfire::info`].
pub fn print_af_info() {
    arrayfire::info()
}

fn error<I: fmt::Display>(message: I) -> ArrayError {
    ArrayError {
        message: message.to_string(),
    }
}

#[inline]
fn dim4(size: usize) -> arrayfire::Dim4 {
    arrayfire::Dim4::new(&[size as u64, 1, 1, 1])
}

#[inline]
fn coord_bounds(shape: &[u64]) -> Vec<u64> {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}
