use std::fmt;
use std::iter::FromIterator;
use std::ops::*;

use arrayfire as af;
use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use num_traits::{FromPrimitive, ToPrimitive};
use number_general::*;
use safecast::{CastFrom, CastInto};
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

use super::ext::*;
use super::{error, Complex, Result};

/// The [`NumberType`] of the product of an [`Array`] with the given `array_dtype`.
pub fn product_dtype(array_dtype: NumberType) -> NumberType {
    use {ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT};

    match array_dtype {
        NT::Bool => ArrayExt::<bool>::product_dtype(),
        NT::Complex(ct) => match ct {
            CT::C32 => ArrayExt::<Complex<f32>>::product_dtype(),
            CT::C64 => ArrayExt::<Complex<f64>>::product_dtype(),
            CT::Complex => ArrayExt::<Complex<f64>>::product_dtype(),
        },
        NT::Float(ft) => match ft {
            FT::F32 => ArrayExt::<f32>::product_dtype(),
            FT::F64 => ArrayExt::<f64>::product_dtype(),
            FT::Float => ArrayExt::<f64>::product_dtype(),
        },
        NT::Int(it) => match it {
            IT::I8 => ArrayExt::<i16>::product_dtype(),
            IT::I16 => ArrayExt::<i16>::product_dtype(),
            IT::I32 => ArrayExt::<i32>::product_dtype(),
            IT::I64 => ArrayExt::<i64>::product_dtype(),
            IT::Int => ArrayExt::<i64>::product_dtype(),
        },
        NT::UInt(ut) => match ut {
            UT::U8 => ArrayExt::<u8>::product_dtype(),
            UT::U16 => ArrayExt::<u16>::product_dtype(),
            UT::U32 => ArrayExt::<u32>::product_dtype(),
            UT::U64 => ArrayExt::<u64>::product_dtype(),
            UT::UInt => ArrayExt::<u64>::product_dtype(),
        },
        NT::Number => ArrayExt::<f64>::product_dtype(),
    }
}

/// The [`NumberType`] of the sum of an [`Array`] with the given `array_dtype`.
pub fn sum_dtype(array_dtype: NumberType) -> NumberType {
    use {ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT};

    match array_dtype {
        NT::Bool => ArrayExt::<bool>::sum_dtype(),
        NT::Complex(ct) => match ct {
            CT::C32 => ArrayExt::<Complex<f32>>::sum_dtype(),
            CT::C64 => ArrayExt::<Complex<f64>>::sum_dtype(),
            CT::Complex => ArrayExt::<Complex<f64>>::sum_dtype(),
        },
        NT::Float(ft) => match ft {
            FT::F32 => ArrayExt::<f32>::sum_dtype(),
            FT::F64 => ArrayExt::<f64>::sum_dtype(),
            FT::Float => ArrayExt::<f64>::sum_dtype(),
        },
        NT::Int(it) => match it {
            IT::I8 => ArrayExt::<i16>::sum_dtype(),
            IT::I16 => ArrayExt::<i16>::sum_dtype(),
            IT::I32 => ArrayExt::<i32>::sum_dtype(),
            IT::I64 => ArrayExt::<i64>::sum_dtype(),
            IT::Int => ArrayExt::<i64>::sum_dtype(),
        },
        NT::UInt(ut) => match ut {
            UT::U8 => ArrayExt::<u8>::sum_dtype(),
            UT::U16 => ArrayExt::<u16>::sum_dtype(),
            UT::U32 => ArrayExt::<u32>::sum_dtype(),
            UT::U64 => ArrayExt::<u64>::sum_dtype(),
            UT::UInt => ArrayExt::<u64>::sum_dtype(),
        },
        NT::Number => ArrayExt::<f64>::sum_dtype(),
    }
}

macro_rules! dispatch {
    ($this:expr, $call:expr) => {
        match $this {
            Array::Bool(this) => $call(this),
            Array::C32(this) => $call(this),
            Array::C64(this) => $call(this),
            Array::F32(this) => $call(this),
            Array::F64(this) => $call(this),
            Array::I16(this) => $call(this),
            Array::I32(this) => $call(this),
            Array::I64(this) => $call(this),
            Array::U8(this) => $call(this),
            Array::U16(this) => $call(this),
            Array::U32(this) => $call(this),
            Array::U64(this) => $call(this),
        }
    };
}

macro_rules! trig {
    ($fun:ident) => {
        pub fn $fun(&self) -> Array {
            fn $fun<T>(this: &ArrayExt<T>) -> Array
            where
                T: af::HasAfEnum + Default,
                ArrayExt<T>: ArrayInstanceTrig<T>,
                Array: From<ArrayExt<T::UnaryOutType>>,
            {
                this.$fun().into()
            }

            dispatch!(self, $fun)
        }
    };
}

/// A generic one-dimensional array which supports all variants of [`NumberType`].
#[derive(Clone)]
pub enum Array {
    Bool(ArrayExt<bool>),
    C32(ArrayExt<Complex<f32>>),
    C64(ArrayExt<Complex<f64>>),
    F32(ArrayExt<f32>),
    F64(ArrayExt<f64>),
    I16(ArrayExt<i16>),
    I32(ArrayExt<i32>),
    I64(ArrayExt<i64>),
    U8(ArrayExt<u8>),
    U16(ArrayExt<u16>),
    U32(ArrayExt<u32>),
    U64(ArrayExt<u64>),
}

impl Array {
    /// Cast the values of this array into an `ArrayExt<T>`.
    pub fn type_cast<T: af::HasAfEnum>(&self) -> ArrayExt<T> {
        dispatch!(self, ArrayExt::type_cast)
    }

    /// Concatenate two `Array`s.
    pub fn concatenate(left: &Array, right: &Array) -> Array {
        use Array::*;
        match (left, right) {
            (Bool(l), Bool(r)) => Bool(ArrayExt::concatenate(l, r)),

            (F32(l), F32(r)) => F32(ArrayExt::concatenate(l, r)),
            (F64(l), F64(r)) => F64(ArrayExt::concatenate(l, r)),

            (C32(l), C32(r)) => C32(ArrayExt::concatenate(l, r)),
            (C64(l), C64(r)) => C64(ArrayExt::concatenate(l, r)),

            (I16(l), I16(r)) => I16(ArrayExt::concatenate(l, r)),
            (I32(l), I32(r)) => I32(ArrayExt::concatenate(l, r)),
            (I64(l), I64(r)) => I64(ArrayExt::concatenate(l, r)),

            (U8(l), U8(r)) => U8(ArrayExt::concatenate(l, r)),
            (U16(l), U16(r)) => U16(ArrayExt::concatenate(l, r)),
            (U32(l), U32(r)) => U32(ArrayExt::concatenate(l, r)),
            (U64(l), U64(r)) => U64(ArrayExt::concatenate(l, r)),

            (l, r) if l.dtype() > r.dtype() => Array::concatenate(l, &r.cast_into(l.dtype())),
            (l, r) if l.dtype() < r.dtype() => Array::concatenate(&l.cast_into(r.dtype()), r),

            (l, r) => unreachable!("concatenate {}, {}", l, r),
        }
    }

    /// Construct a new array with the given constant value and length.
    pub fn constant(value: Number, length: usize) -> Array {
        use number_general::Complex;
        use Array::*;

        match value {
            Number::Bool(b) => {
                let b: bool = b.into();
                Bool(ArrayExt::constant(b, length))
            }
            Number::Complex(c) => match c {
                Complex::C32(c) => C32(ArrayExt::constant(c, length)),
                Complex::C64(c) => C64(ArrayExt::constant(c, length)),
            },
            Number::Float(f) => match f {
                Float::F32(f) => F32(ArrayExt::constant(f, length)),
                Float::F64(f) => F64(ArrayExt::constant(f, length)),
            },
            Number::Int(i) => match i {
                Int::I16(i) => I16(ArrayExt::constant(i, length)),
                Int::I32(i) => I32(ArrayExt::constant(i, length)),
                Int::I64(i) => I64(ArrayExt::constant(i, length)),
                other => panic!("ArrayFire does not support {}", other),
            },
            Number::UInt(u) => match u {
                UInt::U8(u) => U8(ArrayExt::constant(u, length)),
                UInt::U16(u) => U16(ArrayExt::constant(u, length)),
                UInt::U32(u) => U32(ArrayExt::constant(u, length)),
                UInt::U64(u) => U64(ArrayExt::constant(u, length)),
            },
        }
    }

    /// The [`NumberType`] of this `Array`.
    pub fn dtype(&self) -> NumberType {
        use number_general::DType;
        use Array::*;

        match self {
            Bool(_) => bool::dtype(),
            C32(_) => Complex::<f32>::dtype(),
            C64(_) => Complex::<f64>::dtype(),
            F32(_) => f32::dtype(),
            F64(_) => f64::dtype(),
            I16(_) => i16::dtype(),
            I32(_) => i32::dtype(),
            I64(_) => i64::dtype(),
            U8(_) => u8::dtype(),
            U16(_) => u16::dtype(),
            U32(_) => u32::dtype(),
            U64(_) => u64::dtype(),
        }
    }

    /// Cast into an `Array` of a different `NumberType`.
    pub fn cast_into(&self, dtype: NumberType) -> Array {
        use {ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT};

        match dtype {
            NT::Bool => Self::Bool(self.type_cast()),
            NT::Complex(ct) => match ct {
                CT::C32 => Self::C32(self.type_cast()),
                CT::C64 => Self::C64(self.type_cast()),
                CT::Complex => Self::C64(self.type_cast()),
            },
            NT::Float(ft) => match ft {
                FT::F32 => Self::F32(self.type_cast()),
                FT::F64 => Self::F64(self.type_cast()),
                FT::Float => Self::F64(self.type_cast()),
            },
            NT::Int(it) => match it {
                IT::I16 => Self::I16(self.type_cast()),
                IT::I32 => Self::I32(self.type_cast()),
                IT::I64 => Self::I64(self.type_cast()),
                IT::Int => Self::I64(self.type_cast()),
                other => panic!("ArrayFire does not support {}", other),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => Self::U8(self.type_cast()),
                UT::U16 => Self::U16(self.type_cast()),
                UT::U32 => Self::U32(self.type_cast()),
                UT::U64 => Self::U64(self.type_cast()),
                UT::UInt => Self::U64(self.type_cast()),
            },
            NT::Number => self.clone(),
        }
    }

    /// Copy the contents of this `Array` into a new `Vec`.
    pub fn to_vec(&self) -> Vec<Number> {
        fn to_vec<T>(this: &ArrayExt<T>) -> Vec<Number>
        where
            T: af::HasAfEnum + Clone + Default,
            Number: From<T>,
        {
            this.to_vec().into_iter().map(Number::from).collect()
        }

        dispatch!(self, to_vec)
    }

    /// Calculate the element-wise absolute value.
    pub fn abs(&self) -> Array {
        use Array::*;
        match self {
            C32(c) => F32(c.abs()),
            C64(c) => F64(c.abs()),
            F32(f) => F32(f.abs()),
            F64(f) => F64(f.abs()),
            I16(i) => I16(i.abs()),
            I32(i) => I32(i.abs()),
            I64(i) => I64(i.abs()),
            other => other.clone(),
        }
    }

    /// Returns `true` if all elements of this `Array` are nonzero.
    pub fn all(&self) -> bool {
        dispatch!(self, ArrayExt::all)
    }

    /// Returns `true` if any element of this `Array` is nonzero.
    pub fn any(&self) -> bool {
        dispatch!(self, ArrayExt::any)
    }

    /// Element-wise logical and.
    pub fn and(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = other.type_cast();
        Array::Bool(this.and(&that))
    }

    /// Element-wise logical and, relative to a constant `other`.
    pub fn and_const(&self, other: Number) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = ArrayExt::from(&[other.cast_into()][..]);
        Array::Bool(this.and(&that))
    }

    /// Find the maximum value in this `Array` and its offset.
    pub fn argmax(&self) -> (usize, Number) {
        fn imax<T: af::HasAfEnum>(x: &ArrayExt<T>) -> (usize, Number)
        where
            ArrayExt<T>: ArrayInstanceIndex,
            Number: From<<ArrayExt<T> as ArrayInstance>::DType>,
        {
            let (i, max) = x.argmax();
            (i, max.into())
        }

        dispatch!(self, imax)
    }

    /// Element-wise equality comparison.
    pub fn eq(&self, other: &Array) -> Array {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l.eq(r.deref())),
            (C32(l), C32(r)) => Bool(l.eq(r.deref())),
            (C64(l), C64(r)) => Bool(l.eq(r.deref())),
            (F32(l), F32(r)) => Bool(l.eq(r.deref())),
            (F64(l), F64(r)) => Bool(l.eq(r.deref())),
            (I16(l), I16(r)) => Bool(l.eq(r.deref())),
            (I32(l), I32(r)) => Bool(l.eq(r.deref())),
            (I64(l), I64(r)) => Bool(l.eq(r.deref())),
            (U8(l), U8(r)) => Bool(l.eq(r.deref())),
            (U16(l), U16(r)) => Bool(l.eq(r.deref())),
            (U32(l), U32(r)) => Bool(l.eq(r.deref())),
            (U64(l), U64(r)) => Bool(l.eq(r.deref())),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.eq(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).eq(r),
                (l, r) => unreachable!("{} equal to {}", l, r),
            },
        }
    }

    /// Element-wise equality comparison.
    pub fn eq_const(&self, other: Number) -> Array {
        use number_general::Complex;
        match (self, other) {
            (Self::Bool(l), Number::Bool(r)) => Self::Bool(l.eq(&bool::from(r))),

            (Self::C32(l), Number::Complex(Complex::C32(r))) => Self::Bool(l.eq(&r)),
            (Self::C64(l), Number::Complex(Complex::C64(r))) => Self::Bool(l.eq(&r)),

            (Self::F32(l), Number::Float(Float::F32(r))) => Self::Bool(l.eq(&r)),
            (Self::F64(l), Number::Float(Float::F64(r))) => Self::Bool(l.eq(&r)),

            (Self::I16(l), Number::Int(Int::I16(r))) => Self::Bool(l.eq(&r)),
            (Self::I32(l), Number::Int(Int::I32(r))) => Self::Bool(l.eq(&r)),
            (Self::I64(l), Number::Int(Int::I64(r))) => Self::Bool(l.eq(&r)),

            (Self::U8(l), Number::UInt(UInt::U8(r))) => Self::Bool(l.eq(&r)),
            (Self::U16(l), Number::UInt(UInt::U16(r))) => Self::Bool(l.eq(&r)),
            (Self::U32(l), Number::UInt(UInt::U32(r))) => Self::Bool(l.eq(&r)),
            (Self::U64(l), Number::UInt(UInt::U64(r))) => Self::Bool(l.eq(&r)),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.eq_const(r.into_type(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).eq_const(r),
                (l, r) => unreachable!("{} equal to {}", l, r),
            },
        }
    }

    /// Raise `e` to the power of `self`.
    pub fn exp(&self) -> Array {
        fn exp<T>(this: &ArrayExt<T>) -> Array
        where
            T: af::HasAfEnum + Default,
            Array: From<ArrayExt<T::UnaryOutType>>,
        {
            this.exp().into()
        }

        dispatch!(self, exp)
    }

    /// Element-wise greater-than comparison.
    pub fn gt(&self, other: &Array) -> Array {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l.gt(r.deref())),
            (C32(l), C32(r)) => Bool(l.gt(r.deref())),
            (C64(l), C64(r)) => Bool(l.gt(r.deref())),
            (F32(l), F32(r)) => Bool(l.gt(r.deref())),
            (F64(l), F64(r)) => Bool(l.gt(r.deref())),
            (I16(l), I16(r)) => Bool(l.gt(r.deref())),
            (I32(l), I32(r)) => Bool(l.gt(r.deref())),
            (I64(l), I64(r)) => Bool(l.gt(r.deref())),
            (U8(l), U8(r)) => Bool(l.gt(r.deref())),
            (U16(l), U16(r)) => Bool(l.gt(r.deref())),
            (U32(l), U32(r)) => Bool(l.gt(r.deref())),
            (U64(l), U64(r)) => Bool(l.gt(r.deref())),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.gt(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).gt(r),
                (l, r) => unreachable!("{} greater than {}", l, r),
            },
        }
    }

    /// Element-wise greater-than comparison.
    pub fn gt_const(&self, other: Number) -> Array {
        use number_general::Complex;
        match (self, other) {
            (Self::Bool(l), Number::Bool(r)) => Self::Bool(l.gt(&bool::from(r))),
            (Self::C32(l), Number::Complex(Complex::C32(r))) => Self::Bool(l.gt(&r)),
            (Self::C64(l), Number::Complex(Complex::C64(r))) => Self::Bool(l.gt(&r)),
            (Self::F32(l), Number::Float(Float::F32(r))) => Self::Bool(l.gt(&r)),
            (Self::F64(l), Number::Float(Float::F64(r))) => Self::Bool(l.gt(&r)),
            (Self::I16(l), Number::Int(Int::I16(r))) => Self::Bool(l.gt(&r)),
            (Self::I32(l), Number::Int(Int::I32(r))) => Self::Bool(l.gt(&r)),
            (Self::I64(l), Number::Int(Int::I64(r))) => Self::Bool(l.gt(&r)),
            (Self::U8(l), Number::UInt(UInt::U8(r))) => Self::Bool(l.gt(&r)),
            (Self::U16(l), Number::UInt(UInt::U16(r))) => Self::Bool(l.gt(&r)),
            (Self::U32(l), Number::UInt(UInt::U32(r))) => Self::Bool(l.gt(&r)),
            (Self::U64(l), Number::UInt(UInt::U64(r))) => Self::Bool(l.gt(&r)),
            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.gt_const(r.into_type(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).gt_const(r),
                (l, r) => unreachable!("{} greater than {}", l, r),
            },
        }
    }

    /// Element-wise greater-or-equal comparison.
    pub fn gte(&self, other: &Array) -> Array {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l.gte(r.deref())),
            (C32(l), C32(r)) => Bool(l.gte(r.deref())),
            (C64(l), C64(r)) => Bool(l.gte(r.deref())),
            (F32(l), F32(r)) => Bool(l.gte(r.deref())),
            (F64(l), F64(r)) => Bool(l.gte(r.deref())),
            (I16(l), I16(r)) => Bool(l.gte(r.deref())),
            (I32(l), I32(r)) => Bool(l.gte(r.deref())),
            (I64(l), I64(r)) => Bool(l.gte(r.deref())),
            (U8(l), U8(r)) => Bool(l.gte(r.deref())),
            (U16(l), U16(r)) => Bool(l.gte(r.deref())),
            (U32(l), U32(r)) => Bool(l.gte(r.deref())),
            (U64(l), U64(r)) => Bool(l.gte(r.deref())),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.gte(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).gte(r),
                (l, r) => unreachable!("{} greater than or equal to {}", l, r),
            },
        }
    }

    /// Element-wise greater-than-or-equal comparison.
    pub fn gte_const(&self, other: Number) -> Array {
        use number_general::Complex;
        match (self, other) {
            (Self::Bool(l), Number::Bool(r)) => Self::Bool(l.gte(&bool::from(r))),

            (Self::C32(l), Number::Complex(Complex::C32(r))) => Self::Bool(l.gte(&r)),
            (Self::C64(l), Number::Complex(Complex::C64(r))) => Self::Bool(l.gte(&r)),

            (Self::F32(l), Number::Float(Float::F32(r))) => Self::Bool(l.gte(&r)),
            (Self::F64(l), Number::Float(Float::F64(r))) => Self::Bool(l.gte(&r)),

            (Self::I16(l), Number::Int(Int::I16(r))) => Self::Bool(l.gte(&r)),
            (Self::I32(l), Number::Int(Int::I32(r))) => Self::Bool(l.gte(&r)),
            (Self::I64(l), Number::Int(Int::I64(r))) => Self::Bool(l.gte(&r)),

            (Self::U8(l), Number::UInt(UInt::U8(r))) => Self::Bool(l.gte(&r)),
            (Self::U16(l), Number::UInt(UInt::U16(r))) => Self::Bool(l.gte(&r)),
            (Self::U32(l), Number::UInt(UInt::U32(r))) => Self::Bool(l.gte(&r)),
            (Self::U64(l), Number::UInt(UInt::U64(r))) => Self::Bool(l.gte(&r)),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.gte_const(r.into_type(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).gte_const(r),
                (l, r) => unreachable!("{} greater than or equal to {}", l, r),
            },
        }
    }

    /// Element-wise check for infinite values.
    pub fn is_infinite(&self) -> Array {
        fn is_infinite<T>(this: &ArrayExt<T>) -> Array
        where
            T: af::HasAfEnum + Default,
            ArrayExt<T>: ArrayInstanceUnreal,
        {
            this.is_infinite().into()
        }

        dispatch!(self, is_infinite)
    }

    /// Element-wise check for non-numeric (NaN) values.
    pub fn is_nan(&self) -> Array {
        fn is_nan<T>(this: &ArrayExt<T>) -> Array
        where
            T: af::HasAfEnum + Default,
            ArrayExt<T>: ArrayInstanceUnreal,
        {
            this.is_nan().into()
        }

        dispatch!(self, is_nan)
    }

    /// Compute the natural log of this `Array`.
    pub fn ln(&self) -> Array {
        fn ln<T>(this: &ArrayExt<T>) -> Array
        where
            T: af::HasAfEnum + Default,
            Array: From<ArrayExt<T::UnaryOutType>>,
        {
            this.ln().into()
        }

        dispatch!(self, ln)
    }

    /// Compute the logarithm of this `Array` with respect to the given `base`.
    pub fn log(&self, base: &Array) -> Array {
        use Array::*;
        match (self, base) {
            (Bool(l), Bool(r)) => l.log(r).into(),
            (C32(l), C32(r)) => l.log(r).into(),
            (C64(l), C64(r)) => l.log(r).into(),
            (F32(l), F32(r)) => l.log(r).into(),
            (F64(l), F64(r)) => l.log(r).into(),
            (I16(l), I16(r)) => l.log(r).into(),
            (I32(l), I32(r)) => l.log(r).into(),
            (I64(l), I64(r)) => l.log(r).into(),
            (U8(l), U8(r)) => l.log(r).into(),
            (U16(l), U16(r)) => l.log(r).into(),
            (U32(l), U32(r)) => l.log(r).into(),
            (U64(l), U64(r)) => l.log(r).into(),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.log(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).log(&r),
                (l, r) => unreachable!("{} log {}", l, r),
            },
        }
    }

    /// Compute the logarithm of this `Array` with respect to the given constant `base`.
    pub fn log_const(&self, base: Number) -> Array {
        (&self.ln()) / base.ln()
    }

    /// Element-wise less-than comparison.
    pub fn lt(&self, other: &Array) -> Array {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l.lt(r.deref())),
            (C32(l), C32(r)) => Bool(l.lt(r.deref())),
            (C64(l), C64(r)) => Bool(l.lt(r.deref())),
            (F32(l), F32(r)) => Bool(l.lt(r.deref())),
            (F64(l), F64(r)) => Bool(l.lt(r.deref())),
            (I16(l), I16(r)) => Bool(l.lt(r.deref())),
            (I32(l), I32(r)) => Bool(l.lt(r.deref())),
            (I64(l), I64(r)) => Bool(l.lt(r.deref())),
            (U8(l), U8(r)) => Bool(l.lt(r.deref())),
            (U16(l), U16(r)) => Bool(l.lt(r.deref())),
            (U32(l), U32(r)) => Bool(l.lt(r.deref())),
            (U64(l), U64(r)) => Bool(l.lt(r.deref())),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.lt(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).lt(r),
                (l, r) => unreachable!("{} less than {}", l, r),
            },
        }
    }

    /// Element-wise less-than comparison.
    pub fn lt_const(&self, other: Number) -> Array {
        use number_general::Complex;
        match (self, other) {
            (Self::Bool(l), Number::Bool(r)) => Self::Bool(l.lt(&bool::from(r))),

            (Self::C32(l), Number::Complex(Complex::C32(r))) => Self::Bool(l.lt(&r)),
            (Self::C64(l), Number::Complex(Complex::C64(r))) => Self::Bool(l.lt(&r)),

            (Self::F32(l), Number::Float(Float::F32(r))) => Self::Bool(l.lt(&r)),
            (Self::F64(l), Number::Float(Float::F64(r))) => Self::Bool(l.lt(&r)),

            (Self::I16(l), Number::Int(Int::I16(r))) => Self::Bool(l.lt(&r)),
            (Self::I32(l), Number::Int(Int::I32(r))) => Self::Bool(l.lt(&r)),
            (Self::I64(l), Number::Int(Int::I64(r))) => Self::Bool(l.lt(&r)),

            (Self::U8(l), Number::UInt(UInt::U8(r))) => Self::Bool(l.lt(&r)),
            (Self::U16(l), Number::UInt(UInt::U16(r))) => Self::Bool(l.lt(&r)),
            (Self::U32(l), Number::UInt(UInt::U32(r))) => Self::Bool(l.lt(&r)),
            (Self::U64(l), Number::UInt(UInt::U64(r))) => Self::Bool(l.lt(&r)),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.lt_const(r.into_type(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).lt_const(r),
                (l, r) => unreachable!("{} less than {}", l, r),
            },
        }
    }

    /// Element-wise less-or-equal comparison.
    pub fn lte(&self, other: &Array) -> Array {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l.lte(r.deref())),
            (C32(l), C32(r)) => Bool(l.lte(r.deref())),
            (C64(l), C64(r)) => Bool(l.lte(r.deref())),
            (F32(l), F32(r)) => Bool(l.lte(r.deref())),
            (F64(l), F64(r)) => Bool(l.lte(r.deref())),
            (I16(l), I16(r)) => Bool(l.lte(r.deref())),
            (I32(l), I32(r)) => Bool(l.lte(r.deref())),
            (I64(l), I64(r)) => Bool(l.lte(r.deref())),
            (U8(l), U8(r)) => Bool(l.lte(r.deref())),
            (U16(l), U16(r)) => Bool(l.lte(r.deref())),
            (U32(l), U32(r)) => Bool(l.lte(r.deref())),
            (U64(l), U64(r)) => Bool(l.lte(r.deref())),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.lte(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).lte(r),
                (l, r) => unreachable!("{} less than or equal to {}", l, r),
            },
        }
    }

    /// Element-wise less-than-or-equal comparison.
    pub fn lte_const(&self, other: Number) -> Array {
        use number_general::Complex;
        match (self, other) {
            (Self::Bool(l), Number::Bool(r)) => Self::Bool(l.lte(&bool::from(r))),

            (Self::C32(l), Number::Complex(Complex::C32(r))) => Self::Bool(l.lte(&r)),
            (Self::C64(l), Number::Complex(Complex::C64(r))) => Self::Bool(l.lte(&r)),

            (Self::F32(l), Number::Float(Float::F32(r))) => Self::Bool(l.lte(&r)),
            (Self::F64(l), Number::Float(Float::F64(r))) => Self::Bool(l.lte(&r)),

            (Self::I16(l), Number::Int(Int::I16(r))) => Self::Bool(l.lte(&r)),
            (Self::I32(l), Number::Int(Int::I32(r))) => Self::Bool(l.lte(&r)),
            (Self::I64(l), Number::Int(Int::I64(r))) => Self::Bool(l.lte(&r)),

            (Self::U8(l), Number::UInt(UInt::U8(r))) => Self::Bool(l.lte(&r)),
            (Self::U16(l), Number::UInt(UInt::U16(r))) => Self::Bool(l.lte(&r)),
            (Self::U32(l), Number::UInt(UInt::U32(r))) => Self::Bool(l.lte(&r)),
            (Self::U64(l), Number::UInt(UInt::U64(r))) => Self::Bool(l.lte(&r)),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.lte_const(r.into_type(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).lte_const(r),
                (l, r) => unreachable!("{} less than or equal to {}", l, r),
            },
        }
    }

    /// Element-wise inequality comparison.
    pub fn ne(&self, other: &Array) -> Array {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l.ne(r.deref())),
            (C32(l), C32(r)) => Bool(l.ne(r.deref())),
            (C64(l), C64(r)) => Bool(l.ne(r.deref())),
            (F32(l), F32(r)) => Bool(l.ne(r.deref())),
            (F64(l), F64(r)) => Bool(l.ne(r.deref())),
            (I16(l), I16(r)) => Bool(l.ne(r.deref())),
            (I32(l), I32(r)) => Bool(l.ne(r.deref())),
            (I64(l), I64(r)) => Bool(l.ne(r.deref())),
            (U8(l), U8(r)) => Bool(l.ne(r.deref())),
            (U16(l), U16(r)) => Bool(l.ne(r.deref())),
            (U32(l), U32(r)) => Bool(l.ne(r.deref())),
            (U64(l), U64(r)) => Bool(l.ne(r.deref())),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.ne(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).ne(r),
                (l, r) => unreachable!("{} not equal to {}", l, r),
            },
        }
    }

    /// Element-wise not-equal comparison.
    pub fn ne_const(&self, other: Number) -> Array {
        use number_general::Complex;
        match (self, other) {
            (Self::Bool(l), Number::Bool(r)) => Self::Bool(l.ne(&bool::from(r))),

            (Self::C32(l), Number::Complex(Complex::C32(r))) => Self::Bool(l.ne(&r)),
            (Self::C64(l), Number::Complex(Complex::C64(r))) => Self::Bool(l.ne(&r)),

            (Self::F32(l), Number::Float(Float::F32(r))) => Self::Bool(l.ne(&r)),
            (Self::F64(l), Number::Float(Float::F64(r))) => Self::Bool(l.ne(&r)),

            (Self::I16(l), Number::Int(Int::I16(r))) => Self::Bool(l.ne(&r)),
            (Self::I32(l), Number::Int(Int::I32(r))) => Self::Bool(l.ne(&r)),
            (Self::I64(l), Number::Int(Int::I64(r))) => Self::Bool(l.ne(&r)),

            (Self::U8(l), Number::UInt(UInt::U8(r))) => Self::Bool(l.ne(&r)),
            (Self::U16(l), Number::UInt(UInt::U16(r))) => Self::Bool(l.ne(&r)),
            (Self::U32(l), Number::UInt(UInt::U32(r))) => Self::Bool(l.ne(&r)),
            (Self::U64(l), Number::UInt(UInt::U64(r))) => Self::Bool(l.ne(&r)),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.ne_const(r.into_type(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).ne_const(r),
                (l, r) => unreachable!("{} not equal to {}", l, r),
            },
        }
    }

    /// Element-wise logical not.
    pub fn not(&self) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        Array::Bool(this.not())
    }

    /// Element-wise logical or.
    pub fn or(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = other.type_cast();
        Array::Bool(this.or(&that))
    }

    /// Element-wise logical or, relative to a constant `other`.
    pub fn or_const(&self, other: Number) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = ArrayExt::from(&[other.cast_into()][..]);
        Array::Bool(this.or(&that))
    }

    /// Calculate the cumulative product of this `Array`.
    pub fn product(&self) -> Number {
        fn product<T>(this: &ArrayExt<T>) -> Number
        where
            T: af::HasAfEnum + Default,
            T::AggregateOutType: number_general::DType,
            T::ProductOutType: number_general::DType,
            ArrayExt<T>: ArrayInstanceReduce<T>,
            Number: From<T::ProductOutType>,
        {
            this.product().into()
        }

        dispatch!(self, product)
    }

    /// Calculate the cumulative sum of this `Array`.
    pub fn sum(&self) -> Number {
        fn sum<T>(this: &ArrayExt<T>) -> Number
        where
            T: af::HasAfEnum + Default,
            T::AggregateOutType: number_general::DType,
            T::ProductOutType: number_general::DType,
            ArrayExt<T>: ArrayInstanceReduce<T>,
            Number: From<T::AggregateOutType>,
        {
            this.sum().into()
        }

        dispatch!(self, sum)
    }

    /// The number of elements in this `Array`.
    pub fn len(&self) -> usize {
        dispatch!(self, ArrayExt::len)
    }

    /// Get the value at the specified index.
    pub fn get_value(&self, index: usize) -> Number {
        debug_assert!(index < self.len());

        use number_general::Complex;
        use Array::*;
        match self {
            Bool(b) => b.get_value(index).into(),
            C32(c) => Complex::from(c.get_value(index)).into(),
            C64(c) => Complex::from(c.get_value(index)).into(),
            F32(f) => Float::from(f.get_value(index)).into(),
            F64(f) => Float::from(f.get_value(index)).into(),
            I16(i) => Int::from(i.get_value(index)).into(),
            I32(i) => Int::from(i.get_value(index)).into(),
            I64(i) => Int::from(i.get_value(index)).into(),
            U8(u) => UInt::from(u.get_value(index)).into(),
            U16(u) => UInt::from(u.get_value(index)).into(),
            U32(u) => UInt::from(u.get_value(index)).into(),
            U64(u) => UInt::from(u.get_value(index)).into(),
        }
    }

    /// Get the values at the specified coordinates.
    pub fn get(&self, index: &ArrayExt<u64>) -> Self {
        let mut indexer = af::Indexer::default();
        indexer.set_index(index.deref(), 0, None);
        self.get_at(indexer)
    }

    fn get_at(&self, index: af::Indexer) -> Self {
        use Array::*;
        match self {
            Bool(b) => Bool(b.get(index)),
            C32(c) => C32(c.get(index)),
            C64(c) => C64(c.get(index)),
            F32(f) => F32(f.get(index)),
            F64(f) => F64(f.get(index)),
            I16(i) => I16(i.get(index)),
            I32(i) => I32(i.get(index)),
            I64(i) => I64(i.get(index)),
            U8(i) => U8(i.get(index)),
            U16(i) => U16(i.get(index)),
            U32(i) => U32(i.get(index)),
            U64(i) => U64(i.get(index)),
        }
    }

    /// Return this `Array` raised to the power of `other`.
    pub fn pow(&self, other: &Self) -> Self {
        // af::pow only works with floating point numbers!
        use Array::*;
        match (self, other) {
            (C32(l), C32(r)) => C32(l.pow(r.deref())),
            (C64(l), C64(r)) => C64(l.pow(r.deref())),
            (F32(l), F32(r)) => F32(l.pow(r.deref())),
            (F64(l), F64(r)) => F64(l.pow(r.deref())),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.pow(&r.cast_into(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).pow(r),
                _ => Self::F64(l.type_cast()).pow(r),
            },
        }
    }

    /// Return this `Array` raised to the power of `other`.
    pub fn pow_const(&self, other: Number) -> Self {
        // af::pow only works with floating point numbers!
        use number_general::Complex;
        match (self, other) {
            (Self::C32(l), Number::Complex(Complex::C32(r))) => Self::C32(l.pow(&r)),
            (Self::C64(l), Number::Complex(Complex::C64(r))) => Self::C64(l.pow(&r)),
            (Self::F32(l), Number::Float(Float::F32(r))) => Self::F32(l.pow(&r)),
            (Self::F64(l), Number::Float(Float::F64(r))) => Self::F64(l.pow(&r)),
            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l.pow_const(r.into_type(l_dtype)),
                (l_dtype, r_dtype) if l_dtype < r_dtype => l.cast_into(r_dtype).pow_const(r),
                _ => Self::F64(l.type_cast()).pow_const(r),
            },
        }
    }

    /// Round this `Array` to the nearest integer, element-wise
    pub fn round(&self) -> Self {
        fn round<T: af::HasAfEnum>(x: &ArrayExt<T>) -> Array
        where
            Array: From<ArrayExt<<ArrayExt<T> as ArrayInstanceRound>::Round>>,
        {
            x.round().into()
        }

        dispatch!(self, round)
    }

    /// Set the values at the specified coordinates to the corresponding values in `other`.
    pub fn set(&mut self, index: &ArrayExt<u64>, other: &Array) -> Result<()> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(index.deref(), 0, None);
        self.set_at(indexer, other)
    }

    /// Set the value at the specified coordinate to `value`.
    pub fn set_value(&mut self, offset: usize, value: Number) -> Result<()> {
        use Array::*;
        match self {
            Bool(b) => {
                let value: Boolean = value.cast_into();
                b.set_at(offset, value.cast_into());
            }
            C32(c) => {
                let value: Complex<f32> = value.cast_into();
                c.set_at(offset, value.cast_into())
            }
            C64(c) => {
                let value: Complex<f64> = value.cast_into();
                c.set_at(offset, value.cast_into())
            }
            F32(f) => {
                let value: Float = value.cast_into();
                f.set_at(offset, value.cast_into())
            }
            F64(f) => {
                let value: Float = value.cast_into();
                f.set_at(offset, value.cast_into())
            }
            I16(i) => {
                let value: Int = value.cast_into();
                i.set_at(offset, value.cast_into())
            }
            I32(i) => {
                let value: Int = value.cast_into();
                i.set_at(offset, value.cast_into())
            }
            I64(i) => {
                let value: Int = value.cast_into();
                i.set_at(offset, value.cast_into())
            }
            U8(u) => {
                let value: UInt = value.cast_into();
                u.set_at(offset, value.cast_into())
            }
            U16(u) => {
                let value: UInt = value.cast_into();
                u.set_at(offset, value.cast_into())
            }
            U32(u) => {
                let value: UInt = value.cast_into();
                u.set_at(offset, value.cast_into())
            }
            U64(u) => {
                let value: UInt = value.cast_into();
                u.set_at(offset, value.cast_into())
            }
        }

        Ok(())
    }

    fn set_at(&mut self, index: af::Indexer, value: &Array) -> Result<()> {
        use Array::*;
        match self {
            Bool(l) => l.set(&index, &value.type_cast()),
            C32(l) => l.set(&index, &value.type_cast()),
            C64(l) => l.set(&index, &value.type_cast()),
            F32(l) => l.set(&index, &value.type_cast()),
            F64(l) => l.set(&index, &value.type_cast()),
            I16(l) => l.set(&index, &value.type_cast()),
            I32(l) => l.set(&index, &value.type_cast()),
            I64(l) => l.set(&index, &value.type_cast()),
            U8(l) => l.set(&index, &value.type_cast()),
            U16(l) => l.set(&index, &value.type_cast()),
            U32(l) => l.set(&index, &value.type_cast()),
            U64(l) => l.set(&index, &value.type_cast()),
        }

        Ok(())
    }

    /// Return a slice of this `Array`.
    pub fn slice(&self, start: usize, end: usize) -> Result<Self> {
        if start > self.len() {
            return Err(error(format!(
                "invalid start index for array slice: {}",
                start
            )));
        }

        if end > self.len() {
            return Err(error(format!(
                "invalid start index for array slice: {}",
                end
            )));
        }

        use Array::*;
        let slice = match self {
            Bool(b) => b.slice(start, end).into(),
            C32(c) => c.slice(start, end).into(),
            C64(c) => c.slice(start, end).into(),
            F32(f) => f.slice(start, end).into(),
            F64(f) => f.slice(start, end).into(),
            I16(i) => i.slice(start, end).into(),
            I32(i) => i.slice(start, end).into(),
            I64(i) => i.slice(start, end).into(),
            U8(u) => u.slice(start, end).into(),
            U16(u) => u.slice(start, end).into(),
            U32(u) => u.slice(start, end).into(),
            U64(u) => u.slice(start, end).into(),
        };

        Ok(slice)
    }

    /// Sort this `Array` in-place.
    pub fn sort(&mut self, ascending: bool) -> Result<()> {
        use Array::*;
        match self {
            Bool(b) => b.sort(ascending),
            F32(f) => f.sort(ascending),
            F64(f) => f.sort(ascending),
            I16(i) => i.sort(ascending),
            I32(i) => i.sort(ascending),
            I64(i) => i.sort(ascending),
            U8(u) => u.sort(ascending),
            U16(u) => u.sort(ascending),
            U32(u) => u.sort(ascending),
            U64(u) => u.sort(ascending),
            other => {
                return Err(error(format!(
                    "{} does not support ordering",
                    other.dtype()
                )))
            }
        }

        Ok(())
    }

    /// Split this `Array` into two new instances at the given pivot.
    pub fn split(&self, at: usize) -> Result<(Array, Array)> {
        if at > self.len() {
            return Err(error(format!(
                "Invalid pivot for Array of length {}",
                self.len()
            )));
        }

        use Array::*;
        match self {
            Bool(u) => {
                let (l, r) = u.split(at);
                Ok((Bool(l), Bool(r)))
            }
            C32(u) => {
                let (l, r) = u.split(at);
                Ok((C32(l), C32(r)))
            }
            C64(u) => {
                let (l, r) = u.split(at);
                Ok((C64(l), C64(r)))
            }
            F32(u) => {
                let (l, r) = u.split(at);
                Ok((F32(l), F32(r)))
            }
            F64(u) => {
                let (l, r) = u.split(at);
                Ok((F64(l), F64(r)))
            }
            I16(u) => {
                let (l, r) = u.split(at);
                Ok((I16(l), I16(r)))
            }
            I32(u) => {
                let (l, r) = u.split(at);
                Ok((I32(l), I32(r)))
            }
            I64(u) => {
                let (l, r) = u.split(at);
                Ok((I64(l), I64(r)))
            }
            U8(u) => {
                let (l, r) = u.split(at);
                Ok((U8(l), U8(r)))
            }
            U16(u) => {
                let (l, r) = u.split(at);
                Ok((U16(l), U16(r)))
            }
            U32(u) => {
                let (l, r) = u.split(at);
                Ok((U32(l), U32(r)))
            }
            U64(u) => {
                let (l, r) = u.split(at);
                Ok((U64(l), U64(r)))
            }
        }
    }

    // TODO: how to include documentation in macro invocations?

    trig! {sin}
    trig! {asin}
    trig! {sinh}
    trig! {asinh}
    trig! {cos}
    trig! {acos}
    trig! {cosh}
    trig! {acosh}
    trig! {tan}
    trig! {atan}
    trig! {tanh}
    trig! {atanh}

    /// Element-wise logical xor.
    pub fn xor(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = other.type_cast();
        Array::Bool(this.xor(&that))
    }

    /// Element-wise logical xor, relative to a constant `other`.
    pub fn xor_const(&self, other: Number) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = ArrayExt::from(&[other.cast_into()][..]);
        Array::Bool(this.xor(&that))
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Array) -> bool {
        if self.len() != other.len() {
            return false;
        } else {
            Array::eq(self, other).all()
        }
    }
}

impl Add for &Array {
    type Output = Array;

    fn add(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l + r),
            (C32(l), C32(r)) => C32(l + r),
            (C64(l), C64(r)) => C64(l + r),
            (F32(l), F32(r)) => F32(l + r),
            (F64(l), F64(r)) => F64(l + r),
            (I16(l), I16(r)) => I16(l + r),
            (I32(l), I32(r)) => I32(l + r),
            (I64(l), I64(r)) => I64(l + r),
            (U8(l), U8(r)) => U8(l + r),
            (U16(l), U16(r)) => U16(l + r),
            (U32(l), U32(r)) => U32(l + r),
            (U64(l), U64(r)) => U64(l + r),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l + &r.cast_into(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) + r,
                (l, r) => unreachable!("add {}, {}", l, r),
            },
        }
    }
}

impl Add<Number> for &Array {
    type Output = Array;

    fn add(self, rhs: Number) -> Self::Output {
        use number_general::Complex;
        match (self, rhs) {
            (Array::Bool(l), Number::Bool(r)) => Array::Bool((l.deref() + bool::from(r)).into()),

            (Array::F32(l), Number::Float(Float::F32(r))) => Array::F32((l.deref() + r).into()),
            (Array::F64(l), Number::Float(Float::F32(r))) => Array::F64((l.deref() + r).into()),
            (Array::F64(l), Number::Float(Float::F64(r))) => Array::F64((l.deref() + r).into()),

            (Array::C32(l), Number::Complex(Complex::C32(r))) => Array::C32((l.deref() + r).into()),
            (Array::C64(l), Number::Complex(Complex::C64(r))) => Array::C64((l.deref() + r).into()),

            (Array::I16(l), Number::Int(Int::I16(r))) => Array::I16((l.deref() + r).into()),
            (Array::I32(l), Number::Int(Int::I32(r))) => Array::I32((l.deref() + r).into()),
            (Array::I64(l), Number::Int(Int::I64(r))) => Array::I64((l.deref() + r).into()),

            (Array::U8(l), Number::UInt(UInt::U8(r))) => Array::U8((l.deref() + r).into()),
            (Array::U16(l), Number::UInt(UInt::U16(r))) => Array::U16((l.deref() + r).into()),
            (Array::U32(l), Number::UInt(UInt::U32(r))) => Array::U32((l.deref() + r).into()),
            (Array::U64(l), Number::UInt(UInt::U64(r))) => Array::U64((l.deref() + r).into()),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l + r.into_type(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) + r,
                (l, r) => unreachable!("add {}, {}", l, r),
            },
        }
    }
}

impl AddAssign<&Array> for Array {
    fn add_assign(&mut self, other: &Array) {
        let sum = &*self + other;
        *self = sum;
    }
}

impl AddAssign<Number> for Array {
    fn add_assign(&mut self, rhs: Number) {
        *self = &*self + rhs;
    }
}

impl Sub for &Array {
    type Output = Array;

    fn sub(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l - r),
            (C32(l), C32(r)) => C32(l - r),
            (C64(l), C64(r)) => C64(l - r),
            (F32(l), F32(r)) => F32(l - r),
            (F64(l), F64(r)) => F64(l - r),
            (I16(l), I16(r)) => I16(l - r),
            (I32(l), I32(r)) => I32(l - r),
            (I64(l), I64(r)) => I64(l - r),
            (U8(l), U8(r)) => U8(l - r),
            (U16(l), U16(r)) => U16(l - r),
            (U32(l), U32(r)) => U32(l - r),
            (U64(l), U64(r)) => U64(l - r),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l - &r.cast_into(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) - r,
                (l, r) => unreachable!("subtract {}, {}", l, r),
            },
        }
    }
}

impl Sub<Number> for &Array {
    type Output = Array;

    fn sub(self, rhs: Number) -> Self::Output {
        use number_general::Complex;
        match (self, rhs) {
            (Array::Bool(l), Number::Bool(r)) => Array::Bool((l.deref() - bool::from(r)).into()),

            (Array::F32(l), Number::Float(Float::F32(r))) => Array::F32((l.deref() - r).into()),
            (Array::F64(l), Number::Float(Float::F64(r))) => Array::F64((l.deref() - r).into()),

            (Array::C32(l), Number::Complex(Complex::C32(r))) => Array::C32((l.deref() - r).into()),
            (Array::C64(l), Number::Complex(Complex::C64(r))) => Array::C64((l.deref() - r).into()),

            (Array::I16(l), Number::Int(Int::I16(r))) => Array::I16((l.deref() - r).into()),
            (Array::I32(l), Number::Int(Int::I32(r))) => Array::I32((l.deref() - r).into()),
            (Array::I64(l), Number::Int(Int::I64(r))) => Array::I64((l.deref() - r).into()),

            (Array::U8(l), Number::UInt(UInt::U8(r))) => Array::U8((l.deref() - r).into()),
            (Array::U16(l), Number::UInt(UInt::U16(r))) => Array::U16((l.deref() - r).into()),
            (Array::U32(l), Number::UInt(UInt::U32(r))) => Array::U32((l.deref() - r).into()),
            (Array::U64(l), Number::UInt(UInt::U64(r))) => Array::U64((l.deref() - r).into()),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l - r.into_type(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) - r,
                (l, r) => unreachable!("subtract {}, {}", l, r),
            },
        }
    }
}

impl SubAssign<&Array> for Array {
    fn sub_assign(&mut self, other: &Array) {
        let diff = &*self - other;
        *self = diff;
    }
}

impl SubAssign<Number> for Array {
    fn sub_assign(&mut self, rhs: Number) {
        *self = &*self - rhs;
    }
}

impl Mul for &Array {
    type Output = Array;

    fn mul(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l * r),
            (C32(l), C32(r)) => C32(l * r),
            (C64(l), C64(r)) => C64(l * r),
            (F32(l), F32(r)) => F32(l * r),
            (F64(l), F64(r)) => F64(l * r),
            (I16(l), I16(r)) => I16(l * r),
            (I32(l), I32(r)) => I32(l * r),
            (I64(l), I64(r)) => I64(l * r),
            (U8(l), U8(r)) => U8(l * r),
            (U16(l), U16(r)) => U16(l * r),
            (U32(l), U32(r)) => U32(l * r),
            (U64(l), U64(r)) => U64(l * r),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l * &r.cast_into(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) * r,
                (l, r) => unreachable!("multiply {}, {}", l, r),
            },
        }
    }
}

impl Mul<Number> for &Array {
    type Output = Array;

    fn mul(self, rhs: Number) -> Self::Output {
        use number_general::Complex;
        match (self, rhs) {
            (Array::Bool(l), Number::Bool(r)) => Array::Bool((l.deref() * bool::from(r)).into()),

            (Array::F32(l), Number::Float(Float::F32(r))) => Array::F32((l.deref() * r).into()),
            (Array::F64(l), Number::Float(Float::F64(r))) => Array::F64((l.deref() * r).into()),

            (Array::C32(l), Number::Complex(Complex::C32(r))) => Array::C32((l.deref() * r).into()),
            (Array::C64(l), Number::Complex(Complex::C64(r))) => Array::C64((l.deref() * r).into()),

            (Array::I16(l), Number::Int(Int::I16(r))) => Array::I16((l.deref() * r).into()),
            (Array::I32(l), Number::Int(Int::I32(r))) => Array::I32((l.deref() * r).into()),
            (Array::I64(l), Number::Int(Int::I64(r))) => Array::I64((l.deref() * r).into()),

            (Array::U8(l), Number::UInt(UInt::U8(r))) => Array::U8((l.deref() * r).into()),
            (Array::U16(l), Number::UInt(UInt::U16(r))) => Array::U16((l.deref() * r).into()),
            (Array::U32(l), Number::UInt(UInt::U32(r))) => Array::U32((l.deref() * r).into()),
            (Array::U64(l), Number::UInt(UInt::U64(r))) => Array::U64((l.deref() * r).into()),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l * r.into_type(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) * r,
                (l, r) => unreachable!("subtract {}, {}", l, r),
            },
        }
    }
}

impl MulAssign<&Array> for Array {
    fn mul_assign(&mut self, other: &Array) {
        let product = &*self * other;
        *self = product;
    }
}

impl MulAssign<Number> for Array {
    fn mul_assign(&mut self, rhs: Number) {
        *self = &*self * rhs;
    }
}

impl Div for &Array {
    type Output = Array;

    fn div(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l / r),
            (C32(l), C32(r)) => C32(l / r),
            (C64(l), C64(r)) => C64(l / r),
            (F32(l), F32(r)) => F32(l / r),
            (F64(l), F64(r)) => F64(l / r),
            (I16(l), I16(r)) => I16(l / r),
            (I32(l), I32(r)) => I32(l / r),
            (I64(l), I64(r)) => I64(l / r),
            (U8(l), U8(r)) => U8(l / r),
            (U16(l), U16(r)) => U16(l / r),
            (U32(l), U32(r)) => U32(l / r),
            (U64(l), U64(r)) => U64(l / r),
            (l, r) => match (l.dtype(), r.dtype()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l / &r.cast_into(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) / r,
                (l, r) => unreachable!("divide {}, {}", l, r),
            },
        }
    }
}

impl Div<Number> for &Array {
    type Output = Array;

    fn div(self, rhs: Number) -> Self::Output {
        use number_general::Complex;
        match (self, rhs) {
            (Array::Bool(l), Number::Bool(r)) => Array::Bool((l.deref() / bool::from(r)).into()),

            (Array::F32(l), Number::Float(Float::F32(r))) => Array::F32((l.deref() / r).into()),
            (Array::F64(l), Number::Float(Float::F64(r))) => Array::F64((l.deref() / r).into()),

            (Array::C32(l), Number::Complex(Complex::C32(r))) => Array::C32((l.deref() / r).into()),
            (Array::C64(l), Number::Complex(Complex::C64(r))) => Array::C64((l.deref() / r).into()),

            (Array::I16(l), Number::Int(Int::I16(r))) => Array::I16((l.deref() / r).into()),
            (Array::I32(l), Number::Int(Int::I32(r))) => Array::I32((l.deref() / r).into()),
            (Array::I64(l), Number::Int(Int::I64(r))) => Array::I64((l.deref() / r).into()),

            (Array::U8(l), Number::UInt(UInt::U8(r))) => Array::U8((l.deref() / r).into()),
            (Array::U16(l), Number::UInt(UInt::U16(r))) => Array::U16((l.deref() / r).into()),
            (Array::U32(l), Number::UInt(UInt::U32(r))) => Array::U32((l.deref() / r).into()),
            (Array::U64(l), Number::UInt(UInt::U64(r))) => Array::U64((l.deref() / r).into()),

            (l, r) => match (l.dtype(), r.class()) {
                (l_dtype, r_dtype) if l_dtype > r_dtype => l / r.into_type(l_dtype),
                (l_dtype, r_dtype) if l_dtype < r_dtype => &l.cast_into(r_dtype) / r,
                (l, r) => unreachable!("subtract {}, {}", l, r),
            },
        }
    }
}

impl DivAssign<&Array> for Array {
    fn div_assign(&mut self, other: &Array) {
        let div = &*self / other;
        *self = div;
    }
}

impl DivAssign<Number> for Array {
    fn div_assign(&mut self, rhs: Number) {
        *self = &*self / rhs;
    }
}

impl<T: af::HasAfEnum> CastFrom<Array> for ArrayExt<T> {
    fn cast_from(array: Array) -> ArrayExt<T> {
        use Array::*;
        match array {
            Bool(b) => b.type_cast(),
            C32(c) => c.type_cast(),
            C64(c) => c.type_cast(),
            F32(f) => f.type_cast(),
            F64(f) => f.type_cast(),
            I16(i) => i.type_cast(),
            I32(i) => i.type_cast(),
            I64(i) => i.type_cast(),
            U8(u) => u.type_cast(),
            U16(u) => u.type_cast(),
            U32(u) => u.type_cast(),
            U64(u) => u.type_cast(),
        }
    }
}

impl From<ArrayExt<bool>> for Array {
    fn from(i: ArrayExt<bool>) -> Array {
        Array::Bool(i)
    }
}

impl From<ArrayExt<Complex<f32>>> for Array {
    fn from(c: ArrayExt<Complex<f32>>) -> Array {
        Array::C32(c)
    }
}

impl From<ArrayExt<Complex<f64>>> for Array {
    fn from(c: ArrayExt<Complex<f64>>) -> Array {
        Array::C64(c)
    }
}

impl From<ArrayExt<f32>> for Array {
    fn from(f: ArrayExt<f32>) -> Array {
        Array::F32(f)
    }
}

impl From<ArrayExt<f64>> for Array {
    fn from(f: ArrayExt<f64>) -> Array {
        Array::F64(f)
    }
}

impl From<ArrayExt<i16>> for Array {
    fn from(i: ArrayExt<i16>) -> Array {
        Array::I16(i)
    }
}

impl From<ArrayExt<i32>> for Array {
    fn from(i: ArrayExt<i32>) -> Array {
        Array::I32(i)
    }
}

impl From<ArrayExt<i64>> for Array {
    fn from(i: ArrayExt<i64>) -> Array {
        Array::I64(i)
    }
}

impl From<ArrayExt<u8>> for Array {
    fn from(u: ArrayExt<u8>) -> Array {
        Array::U8(u)
    }
}

impl From<ArrayExt<u16>> for Array {
    fn from(u: ArrayExt<u16>) -> Array {
        Array::U16(u)
    }
}

impl From<ArrayExt<u32>> for Array {
    fn from(u: ArrayExt<u32>) -> Array {
        Array::U32(u)
    }
}

impl From<ArrayExt<u64>> for Array {
    fn from(u: ArrayExt<u64>) -> Array {
        Array::U64(u)
    }
}

impl<T: af::HasAfEnum> From<Vec<T>> for Array
where
    Array: From<ArrayExt<T>>,
{
    fn from(values: Vec<T>) -> Self {
        ArrayExt::from(values.as_slice()).into()
    }
}

impl<T: af::HasAfEnum> From<&[T]> for Array
where
    Array: From<ArrayExt<T>>,
{
    fn from(values: &[T]) -> Self {
        ArrayExt::from(values).into()
    }
}

impl<T: af::HasAfEnum> FromIterator<T> for Array
where
    Array: From<ArrayExt<T>>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        ArrayExt::from_iter(iter).into()
    }
}

impl From<Vec<Number>> for Array {
    fn from(elements: Vec<Number>) -> Self {
        use {ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT};

        let dtype = elements.iter().map(|n| n.class()).fold(NT::Bool, Ord::max);

        let array = match dtype {
            NT::Bool => Self::Bool(array_from(elements)),
            NT::Complex(ct) => match ct {
                CT::C32 => Self::C32(array_from(elements)),
                _ => Self::C64(array_from(elements)),
            },
            NT::Float(ft) => match ft {
                FT::F32 => Self::F32(array_from(elements)),
                _ => Self::F64(array_from(elements)),
            },
            NT::Int(it) => match it {
                IT::I8 => Self::I16(array_from(elements)),
                IT::I16 => Self::I16(array_from(elements)),
                IT::I32 => Self::I32(array_from(elements)),
                _ => Self::I64(array_from(elements)),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => Self::U8(array_from(elements)),
                UT::U16 => Self::U16(array_from(elements)),
                UT::U32 => Self::U32(array_from(elements)),
                _ => Self::U64(array_from(elements)),
            },
            NT::Number => Self::F64(array_from(elements)),
        };

        array
    }
}

impl<'de> Deserialize<'de> for Array {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Vec::<Number>::deserialize(deserializer).map(Self::from)
    }
}

fn array_from<T: af::HasAfEnum + CastFrom<Number>>(elements: Vec<Number>) -> ArrayExt<T> {
    elements
        .into_iter()
        .map(|n| n.cast_into())
        .collect::<Vec<T>>()
        .as_slice()
        .into()
}

impl Serialize for Array {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.to_vec().serialize(serializer)
    }
}

#[async_trait]
impl de::FromStream for Array {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        _: (),
        decoder: &mut D,
    ) -> std::result::Result<Self, D::Error> {
        decoder.decode_seq(ArrayVisitor).await
    }
}

impl<'en> en::ToStream<'en> for Array {
    fn to_stream<E: en::Encoder<'en>>(
        &'en self,
        encoder: E,
    ) -> std::result::Result<E::Ok, E::Error> {
        use en::IntoStream;

        match self {
            Self::Bool(array) => (DType::Bool, array).into_stream(encoder),
            Self::C32(array) => (DType::C32, array.re(), array.im()).into_stream(encoder),
            Self::C64(array) => (DType::C64, array.re(), array.im()).into_stream(encoder),
            Self::F32(array) => (DType::F32, array).into_stream(encoder),
            Self::F64(array) => (DType::F64, array).into_stream(encoder),
            Self::I16(array) => (DType::I16, array).into_stream(encoder),
            Self::I32(array) => (DType::I32, array).into_stream(encoder),
            Self::I64(array) => (DType::I64, array).into_stream(encoder),
            Self::U8(array) => (DType::U8, array).into_stream(encoder),
            Self::U16(array) => (DType::U16, array).into_stream(encoder),
            Self::U32(array) => (DType::U32, array).into_stream(encoder),
            Self::U64(array) => (DType::U64, array).into_stream(encoder),
        }
    }
}

impl<'en> en::IntoStream<'en> for Array {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> std::result::Result<E::Ok, E::Error> {
        match self {
            Self::Bool(array) => (DType::Bool, array).into_stream(encoder),
            Self::C32(array) => (DType::C32, array.re(), array.im()).into_stream(encoder),
            Self::C64(array) => (DType::C64, array.re(), array.im()).into_stream(encoder),
            Self::F32(array) => (DType::F32, array).into_stream(encoder),
            Self::F64(array) => (DType::F64, array).into_stream(encoder),
            Self::I16(array) => (DType::I16, array).into_stream(encoder),
            Self::I32(array) => (DType::I32, array).into_stream(encoder),
            Self::I64(array) => (DType::I64, array).into_stream(encoder),
            Self::U8(array) => (DType::U8, array).into_stream(encoder),
            Self::U16(array) => (DType::U16, array).into_stream(encoder),
            Self::U32(array) => (DType::U32, array).into_stream(encoder),
            Self::U64(array) => (DType::U64, array).into_stream(encoder),
        }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bool(array) => fmt::Debug::fmt(array, f),
            Self::C32(array) => fmt::Debug::fmt(array, f),
            Self::C64(array) => fmt::Debug::fmt(array, f),
            Self::F32(array) => fmt::Debug::fmt(array, f),
            Self::F64(array) => fmt::Debug::fmt(array, f),
            Self::I16(array) => fmt::Debug::fmt(array, f),
            Self::I32(array) => fmt::Debug::fmt(array, f),
            Self::I64(array) => fmt::Debug::fmt(array, f),
            Self::U8(array) => fmt::Debug::fmt(array, f),
            Self::U16(array) => fmt::Debug::fmt(array, f),
            Self::U32(array) => fmt::Debug::fmt(array, f),
            Self::U64(array) => fmt::Debug::fmt(array, f),
        }
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bool(array) => fmt::Display::fmt(array, f),
            Self::C32(array) => fmt::Display::fmt(array, f),
            Self::C64(array) => fmt::Display::fmt(array, f),
            Self::F32(array) => fmt::Display::fmt(array, f),
            Self::F64(array) => fmt::Display::fmt(array, f),
            Self::I16(array) => fmt::Display::fmt(array, f),
            Self::I32(array) => fmt::Display::fmt(array, f),
            Self::I64(array) => fmt::Display::fmt(array, f),
            Self::U8(array) => fmt::Display::fmt(array, f),
            Self::U16(array) => fmt::Display::fmt(array, f),
            Self::U32(array) => fmt::Display::fmt(array, f),
            Self::U64(array) => fmt::Display::fmt(array, f),
        }
    }
}

struct ArrayVisitor;

impl ArrayVisitor {
    async fn visit_array<A: de::SeqAccess, T: af::HasAfEnum>(
        seq: &mut A,
    ) -> std::result::Result<ArrayExt<T>, A::Error>
    where
        ArrayExt<T>: de::FromStream<Context = ()>,
    {
        seq.next_element(())
            .await?
            .ok_or_else(|| de::Error::custom("missing array"))
    }
}

#[async_trait]
impl de::Visitor for ArrayVisitor {
    type Value = Array;

    fn expecting() -> &'static str {
        "a numeric array"
    }

    async fn visit_seq<A: de::SeqAccess>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let dtype = seq
            .next_element::<DType>(())
            .await?
            .ok_or_else(|| de::Error::custom("missing array data type"))?;

        match dtype {
            DType::Bool => Self::visit_array(&mut seq).map_ok(Array::Bool).await,
            DType::C32 => {
                let re = Self::visit_array(&mut seq).await?;
                let im = Self::visit_array(&mut seq).await?;
                Ok(Array::C32(ArrayExt::from((re, im))))
            }
            DType::C64 => {
                let re = Self::visit_array(&mut seq).await?;
                let im = Self::visit_array(&mut seq).await?;
                Ok(Array::C64(ArrayExt::from((re, im))))
            }
            DType::F32 => Self::visit_array(&mut seq).map_ok(Array::F32).await,
            DType::F64 => Self::visit_array(&mut seq).map_ok(Array::F64).await,
            DType::I16 => Self::visit_array(&mut seq).map_ok(Array::I16).await,
            DType::I32 => Self::visit_array(&mut seq).map_ok(Array::I32).await,
            DType::I64 => Self::visit_array(&mut seq).map_ok(Array::I64).await,
            DType::U8 => Self::visit_array(&mut seq).map_ok(Array::U8).await,
            DType::U16 => Self::visit_array(&mut seq).map_ok(Array::U16).await,
            DType::U32 => Self::visit_array(&mut seq).map_ok(Array::U32).await,
            DType::U64 => Self::visit_array(&mut seq).map_ok(Array::U64).await,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, num_derive::FromPrimitive, num_derive::ToPrimitive)]
enum DType {
    Bool,
    C32,
    C64,
    F32,
    F64,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[async_trait]
impl de::FromStream for DType {
    type Context = ();

    async fn from_stream<D: de::Decoder>(
        cxt: (),
        decoder: &mut D,
    ) -> std::result::Result<Self, D::Error> {
        let dtype = u8::from_stream(cxt, decoder).await?;
        Self::from_u8(dtype).ok_or_else(|| de::Error::invalid_value(dtype, "an array data type"))
    }
}

impl<'en> en::IntoStream<'en> for DType {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> std::result::Result<E::Ok, E::Error> {
        self.to_u8().into_stream(encoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_value() {
        assert_eq!(Array::from(&[1, 2, 3][..]).get_value(1), Number::from(2));
    }

    #[test]
    fn test_get() {
        let arr = Array::from(vec![1, 2, 3].as_slice());
        let actual = arr.get(&(&[1, 2][..]).into());
        let expected = Array::from(&[2, 3][..]);
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_set() {
        let mut actual = Array::from(&[1, 2, 3][..]);
        actual
            .set(&(&[1, 2][..]).into(), &Array::from(&[4, 5][..]))
            .unwrap();

        let expected = Array::from(&[1, 4, 5][..]);
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_add() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [1][..].into();
        assert_eq!(&a + &b, [2, 3, 4][..].into());

        let b: Array = [3, 2, 1][..].into();
        assert_eq!(&a + &b, [4, 4, 4][..].into());

        assert_eq!(&b + Number::from(1), [4, 3, 2][..].into());
    }

    #[test]
    fn test_add_float() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2.0][..].into();
        assert_eq!(&a + &b, [3.0, 4.0, 5.0][..].into());

        let b: Array = [-1., -4., 4.][..].into();
        assert_eq!(&a + &b, [0., -2., 7.][..].into());

        assert_eq!(&b + Number::from(3), [2, -1, 7][..].into());
    }

    #[test]
    fn test_gte() {
        let a: Array = [0, 1, 2][..].into();
        let b: Array = [1][..].into();
        assert_eq!(a.gte(&b), [false, true, true][..].into());
        assert_eq!(a.gte_const(Number::from(1)), [false, true, true][..].into());
    }

    #[test]
    fn test_sub() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [1][..].into();
        assert_eq!(&a - &b, [0, 1, 2][..].into());

        let b: Array = [3, 2, 1][..].into();
        assert_eq!(&a - &b, [-2, 0, 2][..].into());
    }

    #[test]
    fn test_sub_float() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2.0][..].into();
        assert_eq!(&a - &b, [-1.0, 0., 1.0][..].into());

        let b: Array = [-1., -4., 4.][..].into();
        assert_eq!(&a - &b, [2., 6., -1.][..].into());
    }

    #[test]
    fn test_mul() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2][..].into();
        assert_eq!(&a * &b, [2, 4, 6][..].into());

        let b: Array = [5, 4, 3][..].into();
        assert_eq!(&a * &b, [5, 8, 9][..].into());
    }

    #[test]
    fn test_mul_const() {
        let a: Array = [1, 2, 3][..].into();
        let b: Number = 2f32.into();
        assert_eq!(&a * b, [2.0, 4.0, 6.0][..].into());
    }

    #[test]
    fn test_mul_float() {
        let a: Array = [1.0f32, 2.0f32, 3.0f32][..].into();
        let b: Array = [2.0f32][..].into();
        assert_eq!(&a * &b, [2.0, 4.0, 6.0][..].into());

        let b: Array = [-1., -4., 4.][..].into();
        assert_eq!(&a * &b, [-1., -8., 12.][..].into());
    }

    #[test]
    fn test_div() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2.0][..].into();
        assert_eq!(&a / &b, [0.5, 1.0, 1.5][..].into());

        let b: Array = [-1., -4., 4.][..].into();
        assert_eq!(&a / &b, [-1., -0.5, 0.75][..].into());
    }

    #[test]
    fn test_pow() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2][..].into();
        assert_eq!(a.pow(&b), [1.0, 4.0, 9.0][..].into());

        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2.0][..].into();
        assert_eq!(a.pow(&b), [1.0, 4.0, 9.0][..].into());

        let a: Array = [1.0, 2.0, 3.0][..].into();
        let b: Array = [2][..].into();
        assert_eq!(a.pow(&b), [1.0, 4.0, 9.0][..].into());
    }

    #[test]
    fn test_sum() {
        let a: Array = [1, 2, 3, 4][..].into();
        assert_eq!(a.sum(), 10.into());
    }

    #[test]
    fn test_product() {
        let a: Array = [1, 2, 3, 4][..].into();
        assert_eq!(a.product(), 24.into());
    }

    #[tokio::test]
    async fn test_serialization() {
        let expected: Array = [1, 2, 3, 4][..].into();
        let serialized = tbon::en::encode(&expected).expect("encode");
        let actual = tbon::de::try_decode((), serialized).await.expect("decode");
        assert!(expected.eq(&actual).all());
    }
}
