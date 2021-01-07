//! Provides a generic one-dimensional [`Array`] which wraps [`arrayfire::Array`] and supports
//! all [`number_general::NumberType`] variants. `Array` supports basic math like `Add`, `Sub`,
//! `Mul`, and `Div`, with hardware acceleration on systems which support CUDA or OpenCL.
//! `ArrayExt<bool>` supports common logical operations `and`, `not`, `or`, and `xor`.
//!
//! N-dimensional array functionality can be implemented using `ArrayExt<u64>`, which provides
//! methods `to_coords` and `from_coords` and can be used to index an `Array`.
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

use std::fmt;

mod array;
mod ext;

pub use array::*;
pub use ext::*;

type _Complex<T> = num_complex::Complex<T>;

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

fn error<I: fmt::Display>(message: I) -> ArrayError {
    ArrayError {
        message: message.to_string(),
    }
}

fn dim4(size: usize) -> arrayfire::Dim4 {
    arrayfire::Dim4::new(&[size as u64, 1, 1, 1])
}
