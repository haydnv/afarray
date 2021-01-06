use std::fmt;
use std::ops::*;

use arrayfire as af;
use number_general::*;
use safecast::{CastFrom, CastInto};

use super::ext::*;
use super::{dim4, error, Result, _Complex};
use std::iter::FromIterator;

#[derive(Clone)]
pub enum Array {
    Bool(ArrayExt<bool>),
    C32(ArrayExt<_Complex<f32>>),
    C64(ArrayExt<_Complex<f64>>),
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
    fn cast_inner<T: af::HasAfEnum>(&self) -> ArrayExt<T> {
        use Array::*;
        match self {
            Bool(b) => b.cast_to(),
            C32(c) => c.cast_to(),
            C64(c) => c.cast_to(),
            F32(f) => f.cast_to(),
            F64(f) => f.cast_to(),
            I16(i) => i.cast_to(),
            I32(i) => i.cast_to(),
            I64(i) => i.cast_to(),
            U8(u) => u.cast_to(),
            U16(u) => u.cast_to(),
            U32(u) => u.cast_to(),
            U64(u) => u.cast_to(),
        }
    }

    pub fn concatenate(left: &Array, right: &Array) -> Result<Array> {
        use Array::*;
        match (left, right) {
            (U64(l), U64(r)) => Ok(U64(ArrayExt::concatenate(&l, &r))),
            (l, r) => Err(error(format!(
                "Cannot concatenate arrays with different data types: {}, {}",
                l.dtype(),
                r.dtype()
            ))),
        }
    }

    pub fn constant(value: Number, len: usize) -> Array {
        let dim = dim4(len);

        use Array::*;
        match value {
            Number::Bool(b) => {
                let b: bool = b.into();
                Bool(af::constant(b, dim).into())
            }
            Number::Complex(c) => match c {
                Complex::C32(c) => C32(af::constant(c, dim).into()),
                Complex::C64(c) => C64(af::constant(c, dim).into()),
            },
            Number::Float(f) => match f {
                Float::F32(f) => F32(af::constant(f, dim).into()),
                Float::F64(f) => F64(af::constant(f, dim).into()),
            },
            Number::Int(i) => match i {
                Int::I16(i) => I16(af::constant(i, dim).into()),
                Int::I32(i) => I32(af::constant(i, dim).into()),
                Int::I64(i) => I64(af::constant(i, dim).into()),
            },
            Number::UInt(u) => match u {
                UInt::U8(i) => U8(af::constant(i, dim).into()),
                UInt::U16(u) => U16(af::constant(u, dim).into()),
                UInt::U32(u) => U32(af::constant(u, dim).into()),
                UInt::U64(u) => U64(af::constant(u, dim).into()),
            },
        }
    }

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

    pub fn cast_into(&self, dtype: NumberType) -> Array {
        use ComplexType as CT;
        use FloatType as FT;
        use IntType as IT;
        use NumberType as NT;
        use UIntType as UT;

        match dtype {
            NT::Bool => Self::Bool(self.cast_inner()),
            NT::Complex(ct) => match ct {
                CT::C32 => Self::C32(self.cast_inner()),
                CT::C64 => Self::C64(self.cast_inner()),
                CT::Complex => Self::C64(self.cast_inner()),
            },
            NT::Float(ft) => match ft {
                FT::F32 => Self::F32(self.cast_inner()),
                FT::F64 => Self::F64(self.cast_inner()),
                FT::Float => Self::F64(self.cast_inner()),
            },
            NT::Int(it) => match it {
                IT::I16 => Self::I16(self.cast_inner()),
                IT::I32 => Self::I32(self.cast_inner()),
                IT::I64 => Self::I64(self.cast_inner()),
                IT::Int => Self::I64(self.cast_inner()),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => Self::U8(self.cast_inner()),
                UT::U16 => Self::U16(self.cast_inner()),
                UT::U32 => Self::U32(self.cast_inner()),
                UT::U64 => Self::U64(self.cast_inner()),
                UT::UInt => Self::U64(self.cast_inner()),
            },
            NT::Number => self.clone(),
        }
    }

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

    pub fn and(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.cast_inner();
        let that: ArrayExt<bool> = other.cast_inner();
        Array::Bool(this.and(&that))
    }

    pub fn eq(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.eq(&other.cast_inner())),
            C32(l) => Bool(l.eq(&other.cast_inner())),
            C64(l) => Bool(l.eq(&other.cast_inner())),
            F32(l) => Bool(l.eq(&other.cast_inner())),
            F64(l) => Bool(l.eq(&other.cast_inner())),
            I16(l) => Bool(l.eq(&other.cast_inner())),
            I32(l) => Bool(l.eq(&other.cast_inner())),
            I64(l) => Bool(l.eq(&other.cast_inner())),
            U8(l) => Bool(l.eq(&other.cast_inner())),
            U16(l) => Bool(l.eq(&other.cast_inner())),
            U32(l) => Bool(l.eq(&other.cast_inner())),
            U64(l) => Bool(l.eq(&other.cast_inner())),
        }
    }

    pub fn gt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gt(&other.cast_inner())),
            C32(l) => Bool(l.gt(&other.cast_inner())),
            C64(l) => Bool(l.gt(&other.cast_inner())),
            F32(l) => Bool(l.gt(&other.cast_inner())),
            F64(l) => Bool(l.gt(&other.cast_inner())),
            I16(l) => Bool(l.gt(&other.cast_inner())),
            I32(l) => Bool(l.gt(&other.cast_inner())),
            I64(l) => Bool(l.gt(&other.cast_inner())),
            U8(l) => Bool(l.gt(&other.cast_inner())),
            U16(l) => Bool(l.gt(&other.cast_inner())),
            U32(l) => Bool(l.gt(&other.cast_inner())),
            U64(l) => Bool(l.gt(&other.cast_inner())),
        }
    }

    pub fn gte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gte(&other.cast_inner())),
            C32(l) => Bool(l.gte(&other.cast_inner())),
            C64(l) => Bool(l.gte(&other.cast_inner())),
            F32(l) => Bool(l.gte(&other.cast_inner())),
            F64(l) => Bool(l.gte(&other.cast_inner())),
            I16(l) => Bool(l.gte(&other.cast_inner())),
            I32(l) => Bool(l.gte(&other.cast_inner())),
            I64(l) => Bool(l.gte(&other.cast_inner())),
            U8(l) => Bool(l.gte(&other.cast_inner())),
            U16(l) => Bool(l.gte(&other.cast_inner())),
            U32(l) => Bool(l.gte(&other.cast_inner())),
            U64(l) => Bool(l.gte(&other.cast_inner())),
        }
    }

    pub fn lt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lt(&other.cast_inner())),
            C32(l) => Bool(l.lt(&other.cast_inner())),
            C64(l) => Bool(l.lt(&other.cast_inner())),
            F32(l) => Bool(l.lt(&other.cast_inner())),
            F64(l) => Bool(l.lt(&other.cast_inner())),
            I16(l) => Bool(l.lt(&other.cast_inner())),
            I32(l) => Bool(l.lt(&other.cast_inner())),
            I64(l) => Bool(l.lt(&other.cast_inner())),
            U8(l) => Bool(l.lt(&other.cast_inner())),
            U16(l) => Bool(l.lt(&other.cast_inner())),
            U32(l) => Bool(l.lt(&other.cast_inner())),
            U64(l) => Bool(l.lt(&other.cast_inner())),
        }
    }

    pub fn lte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lte(&other.cast_inner())),
            C32(l) => Bool(l.lte(&other.cast_inner())),
            C64(l) => Bool(l.lte(&other.cast_inner())),
            F32(l) => Bool(l.lte(&other.cast_inner())),
            F64(l) => Bool(l.lte(&other.cast_inner())),
            I16(l) => Bool(l.lte(&other.cast_inner())),
            I32(l) => Bool(l.lte(&other.cast_inner())),
            I64(l) => Bool(l.lte(&other.cast_inner())),
            U8(l) => Bool(l.lte(&other.cast_inner())),
            U16(l) => Bool(l.lte(&other.cast_inner())),
            U32(l) => Bool(l.lte(&other.cast_inner())),
            U64(l) => Bool(l.lte(&other.cast_inner())),
        }
    }

    pub fn ne(&self, other: &Array) -> Array {
        use Array::*;

        match self {
            Bool(l) => Bool(l.ne(&other.cast_inner())),
            C32(l) => Bool(l.ne(&other.cast_inner())),
            C64(l) => Bool(l.ne(&other.cast_inner())),
            F32(l) => Bool(l.ne(&other.cast_inner())),
            F64(l) => Bool(l.ne(&other.cast_inner())),
            I16(l) => Bool(l.ne(&other.cast_inner())),
            I32(l) => Bool(l.ne(&other.cast_inner())),
            I64(l) => Bool(l.ne(&other.cast_inner())),
            U8(l) => Bool(l.ne(&other.cast_inner())),
            U16(l) => Bool(l.ne(&other.cast_inner())),
            U32(l) => Bool(l.ne(&other.cast_inner())),
            U64(l) => Bool(l.ne(&other.cast_inner())),
        }
    }

    pub fn not(&self) -> Array {
        let this: ArrayExt<bool> = self.cast_inner();
        Array::Bool(this.not())
    }

    pub fn or(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.cast_inner();
        let that: ArrayExt<bool> = other.cast_inner();
        Array::Bool(this.or(&that))
    }

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

    pub fn get_value(&self, index: usize) -> Number {
        debug_assert!(index < self.len());

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

    pub fn get(&self, index: af::Array<u64>) -> Self {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
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

    pub fn set(&mut self, index: af::Array<u64>, other: &Array) -> Result<()> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
        self.set_at(indexer, other)
    }

    pub fn set_value(&mut self, offset: usize, value: Number) -> Result<()> {
        use Array::*;
        match self {
            Bool(b) => {
                let value: Boolean = value.cast_into();
                b.set_at(offset, value.cast_into());
            }
            C32(c) => {
                let value: Complex = value.cast_into();
                c.set_at(offset, value.cast_into())
            }
            C64(c) => {
                let value: Complex = value.cast_into();
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
            Bool(l) => l.set(&index, &value.cast_inner()),
            C32(l) => l.set(&index, &value.cast_inner()),
            C64(l) => l.set(&index, &value.cast_inner()),
            F32(l) => l.set(&index, &value.cast_inner()),
            F64(l) => l.set(&index, &value.cast_inner()),
            I16(l) => l.set(&index, &value.cast_inner()),
            I32(l) => l.set(&index, &value.cast_inner()),
            I64(l) => l.set(&index, &value.cast_inner()),
            U8(l) => l.set(&index, &value.cast_inner()),
            U16(l) => l.set(&index, &value.cast_inner()),
            U32(l) => l.set(&index, &value.cast_inner()),
            U64(l) => l.set(&index, &value.cast_inner()),
        }

        Ok(())
    }

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

    pub fn split(&self, at: usize) -> Result<(Array, Array)> {
        if at < self.len() {
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
        } else {
            Err(error(format!(
                "Invalid pivot for Array of length {}",
                self.len()
            )))
        }
    }

    pub fn xor(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.cast_inner();
        let that: ArrayExt<bool> = other.cast_inner();
        Array::Bool(this.xor(&that))
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
            Bool(b) => b.cast_to(),
            C32(c) => c.cast_to(),
            C64(c) => c.cast_to(),
            F32(f) => f.cast_to(),
            F64(f) => f.cast_to(),
            I16(i) => i.cast_to(),
            I32(i) => i.cast_to(),
            I64(i) => i.cast_to(),
            U8(u) => u.cast_to(),
            U16(u) => u.cast_to(),
            U32(u) => u.cast_to(),
            U64(u) => u.cast_to(),
        }
    }
}

impl From<ArrayExt<bool>> for Array {
    fn from(i: ArrayExt<bool>) -> Array {
        Array::Bool(i)
    }
}

impl From<ArrayExt<_Complex<f32>>> for Array {
    fn from(c: ArrayExt<_Complex<f32>>) -> Array {
        Array::C32(c)
    }
}

impl From<ArrayExt<_Complex<f64>>> for Array {
    fn from(c: ArrayExt<_Complex<f64>>) -> Array {
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
        let array = ArrayExt::from_iter(iter);
        Array::from(array)
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Array<{}>", self.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_value() {
        let arr = Array::from(&[1, 2, 3][..]);
        assert_eq!(arr.get_value(1), Number::from(2))
    }

    #[test]
    fn test_get() {
        let arr = Array::from(vec![1, 2, 3].as_slice());
        let indices = af::Array::new(&[1, 2], af::Dim4::new(&[2, 1, 1, 1]));
        let actual = arr.get(indices);
        let expected = Array::from(&[2, 3][..]);
        assert_eq!(
            actual.eq(&expected).to_vec(),
            vec![true.into(), true.into()]
        )
    }

    #[test]
    fn test_set() {
        let mut actual = Array::from(&[1, 2, 3][..]);
        let indices = af::Array::new(&[1, 2], af::Dim4::new(&[2, 1, 1, 1]));
        actual.set(indices, &Array::from(&[4, 5][..])).unwrap();
        let expected = Array::from(&[1, 4, 5][..]);
        assert_eq!(
            actual.eq(&expected).to_vec(),
            vec![true.into(), true.into(), true.into()]
        )
    }
}
