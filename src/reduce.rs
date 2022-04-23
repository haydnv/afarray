use collate;

use crate::Complex;

/// Trait defining the maximum of two numbers.
///
/// By default this is the same as `Ord::cmp`, but floating-point and complex numbers use the
/// collation functions from the `collate` crate.
pub trait Max {
    fn max(self, other: Self) -> Self;
}

macro_rules! max_ord {
    ($t:ty) => {
        impl Max for $t {
            fn max(self, other: Self) -> Self {
                Ord::max(self, other)
            }
        }
    };
}

max_ord!(bool);
max_ord!(i16);
max_ord!(i32);
max_ord!(i64);
max_ord!(u8);
max_ord!(u16);
max_ord!(u32);
max_ord!(u64);

macro_rules! max_collate {
    ($t:ty, $cmp:expr) => {
        impl Max for $t {
            fn max(self, other: Self) -> Self {
                use std::cmp::Ordering;

                match $cmp(&self, &other) {
                    Ordering::Less => other,
                    Ordering::Equal => self,
                    Ordering::Greater => self,
                }
            }
        }
    };
}

max_collate!(f32, collate::compare_f32);
max_collate!(f64, collate::compare_f64);
max_collate!(Complex<f32>, collate::compare_c32);
max_collate!(Complex<f64>, collate::compare_c64);

/// Trait defining the minimum of two numbers.
///
/// By default this is the same as `Ord::cmp`, but floating-point and complex numbers use the
/// collation functions from the `collate` crate.
pub trait Min {
    fn min(self, other: Self) -> Self;
}

macro_rules! min_ord {
    ($t:ty) => {
        impl Min for $t {
            fn min(self, other: Self) -> Self {
                Ord::min(self, other)
            }
        }
    };
}

min_ord!(bool);
min_ord!(i16);
min_ord!(i32);
min_ord!(i64);
min_ord!(u8);
min_ord!(u16);
min_ord!(u32);
min_ord!(u64);

macro_rules! min_collate {
    ($t:ty, $cmp:expr) => {
        impl Min for $t {
            fn min(self, other: Self) -> Self {
                use std::cmp::Ordering;

                match $cmp(&self, &other) {
                    Ordering::Less => self,
                    Ordering::Equal => self,
                    Ordering::Greater => other,
                }
            }
        }
    };
}

min_collate!(f32, collate::compare_f32);
min_collate!(f64, collate::compare_f64);
min_collate!(Complex<f32>, collate::compare_c32);
min_collate!(Complex<f64>, collate::compare_c64);

/// Trait defining the sum of two numbers.
///
/// For `bool` this is the same as boolean OR; otherwise it's the same as `Add`.
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

/// Trait defining the product of two numbers.
///
/// For `bool` this is the same as boolean AND; otherwise it's the same as `Mul`.
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
