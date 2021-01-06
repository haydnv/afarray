use std::fmt;

mod array;
mod ext;

pub use array::*;
pub use ext::*;

type _Complex<T> = num_complex::Complex<T>;

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

pub type Result<T> = std::result::Result<T, ArrayError>;

fn error<I: fmt::Display>(message: I) -> ArrayError {
    ArrayError {
        message: message.to_string(),
    }
}

fn dim4(size: usize) -> arrayfire::Dim4 {
    arrayfire::Dim4::new(&[size as u64, 1, 1, 1])
}
