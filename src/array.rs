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
        use Array::*;
        match self {
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

    /// Concatenate two `Array`s.
    pub fn concatenate(left: &Array, right: &Array) -> Result<Array> {
        use Array::*;
        match (left, right) {
            (Bool(l), Bool(r)) => Ok(Bool(ArrayExt::concatenate(l, r))),

            (F32(l), F32(r)) => Ok(F32(ArrayExt::concatenate(l, r))),
            (F64(l), F64(r)) => Ok(F64(ArrayExt::concatenate(l, r))),

            (C32(l), C32(r)) => Ok(C32(ArrayExt::concatenate(l, r))),
            (C64(l), C64(r)) => Ok(C64(ArrayExt::concatenate(l, r))),

            (I16(l), I16(r)) => Ok(I16(ArrayExt::concatenate(l, r))),
            (I32(l), I32(r)) => Ok(I32(ArrayExt::concatenate(l, r))),
            (I64(l), I64(r)) => Ok(I64(ArrayExt::concatenate(l, r))),

            (U8(l), U8(r)) => Ok(U8(ArrayExt::concatenate(l, r))),
            (U16(l), U16(r)) => Ok(U16(ArrayExt::concatenate(l, r))),
            (U32(l), U32(r)) => Ok(U32(ArrayExt::concatenate(l, r))),
            (U64(l), U64(r)) => Ok(U64(ArrayExt::concatenate(l, r))),

            (l, r) => Err(error(format!(
                "Cannot concatenate arrays with different data types: {}, {}",
                l.dtype(),
                r.dtype()
            ))),
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
        use Array::*;
        match self {
            Bool(_) => NumberType::Bool,
            C32(_) => ComplexType::C32.into(),
            C64(_) => ComplexType::C32.into(),
            F32(_) => FloatType::F32.into(),
            F64(_) => FloatType::F32.into(),
            I16(_) => IntType::I16.into(),
            I32(_) => IntType::I32.into(),
            I64(_) => IntType::I64.into(),
            U8(_) => UIntType::U16.into(),
            U16(_) => UIntType::U16.into(),
            U32(_) => UIntType::U32.into(),
            U64(_) => UIntType::U64.into(),
        }
    }

    /// Cast into an `Array` of a different `NumberType`.
    pub fn cast_into(&self, dtype: NumberType) -> Array {
        use ComplexType as CT;
        use FloatType as FT;
        use IntType as IT;
        use NumberType as NT;
        use UIntType as UT;

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
        use Array::*;
        match self {
            Bool(b) => b.to_vec().into_iter().map(Number::from).collect(),
            C32(c) => c.to_vec().into_iter().map(Number::from).collect(),
            C64(c) => c.to_vec().into_iter().map(Number::from).collect(),
            F32(f) => f.to_vec().into_iter().map(Number::from).collect(),
            F64(f) => f.to_vec().into_iter().map(Number::from).collect(),
            I16(i) => i.to_vec().into_iter().map(Number::from).collect(),
            I32(i) => i.to_vec().into_iter().map(Number::from).collect(),
            I64(i) => i.to_vec().into_iter().map(Number::from).collect(),
            U8(u) => u.to_vec().into_iter().map(Number::from).collect(),
            U16(u) => u.to_vec().into_iter().map(Number::from).collect(),
            U32(u) => u.to_vec().into_iter().map(Number::from).collect(),
            U64(u) => u.to_vec().into_iter().map(Number::from).collect(),
        }
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
        use Array::*;
        match self {
            Bool(b) => b.all(),
            C32(c) => c.all(),
            C64(c) => c.all(),
            F32(f) => f.all(),
            F64(f) => f.all(),
            I16(i) => i.all(),
            I32(i) => i.all(),
            I64(i) => i.all(),
            U8(u) => u.all(),
            U16(u) => u.all(),
            U32(u) => u.all(),
            U64(u) => u.all(),
        }
    }

    /// Returns `true` if any element of this `Array` is nonzero.
    pub fn any(&self) -> bool {
        use Array::*;
        match self {
            Bool(b) => b.any(),
            C32(c) => c.any(),
            C64(c) => c.any(),
            F32(f) => f.any(),
            F64(f) => f.any(),
            I16(i) => i.any(),
            I32(i) => i.any(),
            I64(i) => i.any(),
            U8(u) => u.any(),
            U16(u) => u.any(),
            U32(u) => u.any(),
            U64(u) => u.any(),
        }
    }

    /// Element-wise logical and.
    pub fn and(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = other.type_cast();
        Array::Bool(this.and(&that))
    }

    /// Element-wise equality comparison.
    pub fn eq(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.eq(&other.type_cast())),
            C32(l) => Bool(l.eq(&other.type_cast())),
            C64(l) => Bool(l.eq(&other.type_cast())),
            F32(l) => Bool(l.eq(&other.type_cast())),
            F64(l) => Bool(l.eq(&other.type_cast())),
            I16(l) => Bool(l.eq(&other.type_cast())),
            I32(l) => Bool(l.eq(&other.type_cast())),
            I64(l) => Bool(l.eq(&other.type_cast())),
            U8(l) => Bool(l.eq(&other.type_cast())),
            U16(l) => Bool(l.eq(&other.type_cast())),
            U32(l) => Bool(l.eq(&other.type_cast())),
            U64(l) => Bool(l.eq(&other.type_cast())),
        }
    }

    /// Element-wise greater-than comparison.
    pub fn gt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gt(&other.type_cast())),
            C32(l) => Bool(l.gt(&other.type_cast())),
            C64(l) => Bool(l.gt(&other.type_cast())),
            F32(l) => Bool(l.gt(&other.type_cast())),
            F64(l) => Bool(l.gt(&other.type_cast())),
            I16(l) => Bool(l.gt(&other.type_cast())),
            I32(l) => Bool(l.gt(&other.type_cast())),
            I64(l) => Bool(l.gt(&other.type_cast())),
            U8(l) => Bool(l.gt(&other.type_cast())),
            U16(l) => Bool(l.gt(&other.type_cast())),
            U32(l) => Bool(l.gt(&other.type_cast())),
            U64(l) => Bool(l.gt(&other.type_cast())),
        }
    }

    /// Element-wise greater-or-equal comparison.
    pub fn gte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gte(&other.type_cast())),
            C32(l) => Bool(l.gte(&other.type_cast())),
            C64(l) => Bool(l.gte(&other.type_cast())),
            F32(l) => Bool(l.gte(&other.type_cast())),
            F64(l) => Bool(l.gte(&other.type_cast())),
            I16(l) => Bool(l.gte(&other.type_cast())),
            I32(l) => Bool(l.gte(&other.type_cast())),
            I64(l) => Bool(l.gte(&other.type_cast())),
            U8(l) => Bool(l.gte(&other.type_cast())),
            U16(l) => Bool(l.gte(&other.type_cast())),
            U32(l) => Bool(l.gte(&other.type_cast())),
            U64(l) => Bool(l.gte(&other.type_cast())),
        }
    }

    /// Element-wise check for infinite values.
    pub fn is_infinite(&self) -> Array {
        use Array::*;
        match self {
            Bool(b) => b.is_infinite().into(),
            C32(c) => c.is_infinite().into(),
            C64(c) => c.is_infinite().into(),
            F32(f) => f.is_infinite().into(),
            F64(f) => f.is_infinite().into(),
            I16(i) => i.is_infinite().into(),
            I32(i) => i.is_infinite().into(),
            I64(i) => i.is_infinite().into(),
            U8(u) => u.is_infinite().into(),
            U16(u) => u.is_infinite().into(),
            U32(u) => u.is_infinite().into(),
            U64(u) => u.is_infinite().into(),
        }
    }

    /// Element-wise check for non-numeric (NaN) values.
    pub fn is_nan(&self) -> Array {
        use Array::*;
        match self {
            Bool(b) => b.is_nan().into(),
            C32(c) => c.is_nan().into(),
            C64(c) => c.is_nan().into(),
            F32(f) => f.is_nan().into(),
            F64(f) => f.is_nan().into(),
            I16(i) => i.is_nan().into(),
            I32(i) => i.is_nan().into(),
            I64(i) => i.is_nan().into(),
            U8(u) => u.is_nan().into(),
            U16(u) => u.is_nan().into(),
            U32(u) => u.is_nan().into(),
            U64(u) => u.is_nan().into(),
        }
    }

    /// Element-wise less-than comparison.
    pub fn lt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lt(&other.type_cast())),
            C32(l) => Bool(l.lt(&other.type_cast())),
            C64(l) => Bool(l.lt(&other.type_cast())),
            F32(l) => Bool(l.lt(&other.type_cast())),
            F64(l) => Bool(l.lt(&other.type_cast())),
            I16(l) => Bool(l.lt(&other.type_cast())),
            I32(l) => Bool(l.lt(&other.type_cast())),
            I64(l) => Bool(l.lt(&other.type_cast())),
            U8(l) => Bool(l.lt(&other.type_cast())),
            U16(l) => Bool(l.lt(&other.type_cast())),
            U32(l) => Bool(l.lt(&other.type_cast())),
            U64(l) => Bool(l.lt(&other.type_cast())),
        }
    }

    /// Element-wise less-or-equal comparison.
    pub fn lte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lte(&other.type_cast())),
            C32(l) => Bool(l.lte(&other.type_cast())),
            C64(l) => Bool(l.lte(&other.type_cast())),
            F32(l) => Bool(l.lte(&other.type_cast())),
            F64(l) => Bool(l.lte(&other.type_cast())),
            I16(l) => Bool(l.lte(&other.type_cast())),
            I32(l) => Bool(l.lte(&other.type_cast())),
            I64(l) => Bool(l.lte(&other.type_cast())),
            U8(l) => Bool(l.lte(&other.type_cast())),
            U16(l) => Bool(l.lte(&other.type_cast())),
            U32(l) => Bool(l.lte(&other.type_cast())),
            U64(l) => Bool(l.lte(&other.type_cast())),
        }
    }

    /// Element-wise inequality comparison.
    pub fn ne(&self, other: &Array) -> Array {
        use Array::*;

        match self {
            Bool(l) => Bool(l.ne(&other.type_cast())),
            C32(l) => Bool(l.ne(&other.type_cast())),
            C64(l) => Bool(l.ne(&other.type_cast())),
            F32(l) => Bool(l.ne(&other.type_cast())),
            F64(l) => Bool(l.ne(&other.type_cast())),
            I16(l) => Bool(l.ne(&other.type_cast())),
            I32(l) => Bool(l.ne(&other.type_cast())),
            I64(l) => Bool(l.ne(&other.type_cast())),
            U8(l) => Bool(l.ne(&other.type_cast())),
            U16(l) => Bool(l.ne(&other.type_cast())),
            U32(l) => Bool(l.ne(&other.type_cast())),
            U64(l) => Bool(l.ne(&other.type_cast())),
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

    /// Calculate the cumulative product of this `Array`.
    pub fn product(&self) -> Number {
        use Array::*;
        match self {
            Bool(b) => Number::UInt(b.product().into()),
            C32(c) => Number::Complex(c.product().into()),
            C64(c) => Number::Complex(c.product().into()),
            F32(f) => Number::Float(f.product().into()),
            F64(f) => Number::Float(f.product().into()),
            I16(i) => Number::Int(i.product().into()),
            I32(i) => Number::Int(i.product().into()),
            I64(i) => Number::Int(i.product().into()),
            U8(u) => Number::UInt(u.product().into()),
            U16(u) => Number::UInt(u.product().into()),
            U32(u) => Number::UInt(u.product().into()),
            U64(u) => Number::UInt(u.product().into()),
        }
    }

    /// Calculate the cumulative sum of this `Array`.
    pub fn sum(&self) -> Number {
        use Array::*;
        match self {
            Bool(b) => Number::UInt(b.sum().into()),
            C32(c) => Number::Complex(c.sum().into()),
            C64(c) => Number::Complex(c.sum().into()),
            F32(f) => Number::Float(f.sum().into()),
            F64(f) => Number::Float(f.sum().into()),
            I16(i) => Number::Int(i.sum().into()),
            I32(i) => Number::Int(i.sum().into()),
            I64(i) => Number::Int(i.sum().into()),
            U8(u) => Number::UInt(u.sum().into()),
            U16(u) => Number::UInt(u.sum().into()),
            U32(u) => Number::UInt(u.sum().into()),
            U64(u) => Number::UInt(u.sum().into()),
        }
    }

    /// The number of elements in this `Array`.
    pub fn len(&self) -> usize {
        use Array::*;
        match self {
            Bool(b) => b.len(),
            C32(c) => c.len(),
            C64(c) => c.len(),
            F32(f) => f.len(),
            F64(f) => f.len(),
            I16(i) => i.len(),
            I32(i) => i.len(),
            I64(i) => i.len(),
            U8(u) => u.len(),
            U16(u) => u.len(),
            U32(u) => u.len(),
            U64(u) => u.len(),
        }
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
        indexer.set_index(index.af(), 0, None);
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
        if self.dtype() != other.dtype() {
            let dtype = Ord::max(self.dtype(), other.dtype());
            return self.cast_into(dtype).pow(&other.cast_into(dtype));
        }

        // af::pow only works with floating point numbers!
        // raising an f32 af::Array to a power causes a stack overflow!
        use Array::*;
        match (self, other) {
            (C64(l), C64(r)) => C64(l.pow(r)),
            (C32(l), C32(r)) => C64(l.type_cast()).pow(&C64(r.type_cast())),
            (F64(l), F64(r)) => F64(l.pow(r)),
            (l, r) => {
                let l = F64(l.type_cast());
                let r = F64(r.type_cast());
                l.pow(&r)
            }
        }
    }

    /// Set the values at the specified coordinates to the corresponding values in `other`.
    pub fn set(&mut self, index: &ArrayExt<u64>, other: &Array) -> Result<()> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(index.af(), 0, None);
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

    /// Element-wise logical xor.
    pub fn xor(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.type_cast();
        let that: ArrayExt<bool> = other.type_cast();
        Array::Bool(this.xor(&that))
    }
}

impl PartialEq for Array {
    fn eq(&self, other: &Array) -> bool {
        if self.len() != other.len() {
            return false;
        }

        use Array::*;
        let eq = match (self, other) {
            (Bool(l), Bool(r)) => l.eq(r),
            (C32(l), C32(r)) => l.eq(r),
            (C64(l), C64(r)) => l.eq(r),
            (F32(l), F32(r)) => l.eq(r),
            (F64(l), F64(r)) => l.eq(r),
            (I16(l), I16(r)) => l.eq(r),
            (I32(l), I32(r)) => l.eq(r),
            (I64(l), I64(r)) => l.eq(r),
            (U8(l), U8(r)) => l.eq(r),
            (U16(l), U16(r)) => l.eq(r),
            (U32(l), U32(r)) => l.eq(r),
            (U64(l), U64(r)) => l.eq(r),
            (l, r) if l.dtype() > r.dtype() => {
                let r = r.cast_into(l.dtype());
                return l == &r;
            }
            (l, r) => {
                let l = l.cast_into(r.dtype());
                return &l == r;
            }
        };

        eq.all()
    }
}

impl Add for &Array {
    type Output = Array;

    fn add(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l + r),
            // Adding an F32 array causes a stack overflow
            (C32(l), C32(r)) => &C64(l.type_cast()) + &C64(r.type_cast()),
            (C64(l), C64(r)) => C64(l + r),
            // Adding an F32 array causes a stack overflow
            (F32(l), F32(r)) => &F64(l.type_cast()) + &F64(r.type_cast()),
            (F64(l), F64(r)) => F64(l + r),
            (I16(l), I16(r)) => I16(l + r),
            (I32(l), I32(r)) => I32(l + r),
            (I64(l), I64(r)) => I64(l + r),
            (U8(l), U8(r)) => U8(l + r),
            (U16(l), U16(r)) => U16(l + r),
            (U32(l), U32(r)) => U32(l + r),
            (U64(l), U64(r)) => U64(l + r),
            (l, r) => {
                let dtype = Ord::max(l.dtype(), r.dtype());
                let l = l.cast_into(dtype);
                let r = r.cast_into(dtype);
                &l + &r
            }
        }
    }
}

impl AddAssign for Array {
    fn add_assign(&mut self, other: Self) {
        let sum = &*self + &other;
        *self = sum;
    }
}

impl Sub for &Array {
    type Output = Array;

    fn sub(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l - r),
            // Subtracting a 32-bit float causes a stack overflow
            (C32(l), C32(r)) => &C64(l.type_cast()) - &C64(r.type_cast()),
            (C64(l), C64(r)) => C64(l - r),
            // Subtracting a 32-bit float causes a stack overflow
            (F32(l), F32(r)) => &F64(l.type_cast()) - &F64(r.type_cast()),
            (F64(l), F64(r)) => F64(l - r),
            (I16(l), I16(r)) => I16(l - r),
            (I32(l), I32(r)) => I32(l - r),
            (I64(l), I64(r)) => I64(l - r),
            (U8(l), U8(r)) => U8(l - r),
            (U16(l), U16(r)) => U16(l - r),
            (U32(l), U32(r)) => U32(l - r),
            (U64(l), U64(r)) => U64(l - r),
            (l, r) => {
                let dtype = Ord::max(l.dtype(), r.dtype());
                let l = l.cast_into(dtype);
                let r = r.cast_into(dtype);
                &l - &r
            }
        }
    }
}

impl SubAssign for Array {
    fn sub_assign(&mut self, other: Self) {
        let diff = &*self - &other;
        *self = diff;
    }
}

impl Mul for &Array {
    type Output = Array;

    fn mul(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l * r),
            // Multiplying a 32-bit float causes a stack overflow
            (C32(l), C32(r)) => &C64(l.type_cast()) * &C64(r.type_cast()),
            (C64(l), C64(r)) => C64(l * r),
            // Multiplying a 32-bit float causes a stack overflow
            (F32(l), F32(r)) => &F64(l.type_cast()) * &F64(r.type_cast()),
            (F64(l), F64(r)) => F64(l * r),
            (I16(l), I16(r)) => I16(l * r),
            (I32(l), I32(r)) => I32(l * r),
            (I64(l), I64(r)) => I64(l * r),
            (U8(l), U8(r)) => U8(l * r),
            (U16(l), U16(r)) => U16(l * r),
            (U32(l), U32(r)) => U32(l * r),
            (U64(l), U64(r)) => U64(l * r),
            (l, r) => {
                let dtype = Ord::max(l.dtype(), r.dtype());
                let l = l.cast_into(dtype);
                let r = r.cast_into(dtype);
                &l * &r
            }
        }
    }
}

impl MulAssign for Array {
    fn mul_assign(&mut self, other: Array) {
        let product = &*self * &other;
        *self = product;
    }
}

impl Div for &Array {
    type Output = Array;

    fn div(self, other: &Array) -> Self::Output {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Bool(l / r),
            // dividing an F32 array causes a stack overflow
            (C32(l), C32(r)) => &C64(l.type_cast()) / &C64(r.type_cast()),
            (C64(l), C64(r)) => C64(l / r),
            // dividing an F32 array causes a stack overflow
            (F32(l), F32(r)) => &F64(l.type_cast()) / &F64(r.type_cast()),
            (F64(l), F64(r)) => F64(l / r),
            (I16(l), I16(r)) => I16(l / r),
            (I32(l), I32(r)) => I32(l / r),
            (I64(l), I64(r)) => I64(l / r),
            (U8(l), U8(r)) => U8(l / r),
            (U16(l), U16(r)) => U16(l / r),
            (U32(l), U32(r)) => U32(l / r),
            (U64(l), U64(r)) => U64(l / r),
            (l, r) => {
                let dtype = Ord::max(l.dtype(), r.dtype());
                let l = l.cast_into(dtype);
                let r = r.cast_into(dtype);
                &l / &r
            }
        }
    }
}

impl DivAssign for Array {
    fn div_assign(&mut self, other: Array) {
        let div = &*self / &other;
        *self = div;
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
        use ComplexType as CT;
        use FloatType as FT;
        use IntType as IT;
        use NumberType as NT;
        use UIntType as UT;

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
    }

    #[test]
    fn test_add_float() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2.0][..].into();
        assert_eq!(&a + &b, [3.0, 4.0, 5.0][..].into());

        let b: Array = [-1., -4., 4.][..].into();
        assert_eq!(&a + &b, [0., -2., 7.][..].into());
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
    fn test_mul_float() {
        let a: Array = [1, 2, 3][..].into();
        let b: Array = [2.0][..].into();
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
