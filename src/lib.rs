use std::convert::TryFrom;
use std::fmt;
use std::ops::{Add, Mul};

use arrayfire as af;
use number_general::*;
use safecast::{CastFrom, CastInto};

const BATCH: bool = true;

type _Complex<T> = num::Complex<T>;

pub struct ArrayError {
    message: String
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
    ArrayError { message: message.to_string() }
}

pub trait ArrayInstance {
    type DType: af::HasAfEnum;

    fn af(&'_ self) -> &'_ af::Array<Self::DType>;

    fn af_mut(&'_ mut self) -> &'_ mut af::Array<Self::DType>;

    fn len(&self) -> usize {
        self.af().elements()
    }

    fn as_type<T: af::HasAfEnum>(&self) -> ArrayExt<T> {
        ArrayExt(self.af().cast())
    }

    fn get(&self, index: af::Indexer) -> ArrayExt<Self::DType> {
        ArrayExt(af::index_gen(self.af(), index))
    }

    fn set<T: ArrayInstance<DType = Self::DType>>(&mut self, index: &af::Indexer, other: &T) {
        af::assign_gen(self.af_mut(), index, other.af());
    }

    fn set_at(&mut self, offset: usize, value: Self::DType) {
        let seq = af::Seq::new(offset as f32, offset as f32, 1.0f32);
        af::assign_seq(self.af_mut(), &[seq], &af::Array::new(&[value], dim4(1)));
    }
}

#[derive(Clone)]
pub struct ArrayExt<T: af::HasAfEnum>(af::Array<T>);

impl<T: af::HasAfEnum + Default> ArrayExt<T> {
    fn concatenate(left: &ArrayExt<T>, right: &ArrayExt<T>) -> ArrayExt<T> {
        af::join(0, left.af(), right.af()).into()
    }

    fn get_value(&self, index: usize) -> T {
        debug_assert!(index < self.af().elements());

        let af_value = af::index(self.af(), &[af::Seq::new(index as f64, index as f64, 1f64)]);
        if af_value.elements() == 1 {
            let mut value = vec![T::default()];
            af_value.host(&mut value);
            value.pop().unwrap()
        } else {
            panic!("no value at {}", index)
        }
    }

    fn split(&self, at: usize) -> (ArrayExt<T>, ArrayExt<T>) {
        let left = af::Seq::new(0.0, at as f32, 1.0);
        let right = af::Seq::new(at as f32, self.len() as f32, 1.0);
        (
            ArrayExt(af::index(self.af(), &[left, af::Seq::default()])),
            ArrayExt(af::index(self.af(), &[right, af::Seq::default()])),
        )
    }
}

impl<T: af::HasAfEnum> ArrayInstance for ArrayExt<T> {
    type DType = T;

    fn af(&'_ self) -> &'_ af::Array<T> {
        &self.0
    }

    fn af_mut(&mut self) -> &mut af::Array<Self::DType> {
        &mut self.0
    }
}

impl<T: af::HasAfEnum + Clone + Default> From<ArrayExt<T>> for Vec<T> {
    fn from(array: ArrayExt<T>) -> Vec<T> {
        let len = array.0.elements();
        let mut v = vec![T::default(); len];
        array.0.host(&mut v);
        v
    }
}

impl From<ArrayExt<bool>> for Vec<Number> {
    fn from(array: ArrayExt<bool>) -> Vec<Number> {
        let array: Vec<bool> = array.into();
        let array: Vec<Boolean> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<_Complex<f32>>> for Vec<Number> {
    fn from(array: ArrayExt<_Complex<f32>>) -> Vec<Number> {
        let array: Vec<_Complex<f32>> = array.into();
        let array: Vec<Complex> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<_Complex<f64>>> for Vec<Number> {
    fn from(array: ArrayExt<_Complex<f64>>) -> Vec<Number> {
        let array: Vec<_Complex<f64>> = array.into();
        let array: Vec<Complex> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<f32>> for Vec<Number> {
    fn from(array: ArrayExt<f32>) -> Vec<Number> {
        let array: Vec<f32> = array.into();
        let array: Vec<Float> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<f64>> for Vec<Number> {
    fn from(array: ArrayExt<f64>) -> Vec<Number> {
        let array: Vec<f64> = array.into();
        let array: Vec<Float> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<i16>> for Vec<Number> {
    fn from(array: ArrayExt<i16>) -> Vec<Number> {
        let array: Vec<i16> = array.into();
        let array: Vec<Int> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<i32>> for Vec<Number> {
    fn from(array: ArrayExt<i32>) -> Vec<Number> {
        let array: Vec<i32> = array.into();
        let array: Vec<Int> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<i64>> for Vec<Number> {
    fn from(array: ArrayExt<i64>) -> Vec<Number> {
        let array: Vec<i64> = array.into();
        let array: Vec<Int> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u8>> for Vec<Number> {
    fn from(array: ArrayExt<u8>) -> Vec<Number> {
        let array: Vec<u8> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u16>> for Vec<Number> {
    fn from(array: ArrayExt<u16>) -> Vec<Number> {
        let array: Vec<u16> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u32>> for Vec<Number> {
    fn from(array: ArrayExt<u32>) -> Vec<Number> {
        let array: Vec<u32> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u64>> for Vec<Number> {
    fn from(array: ArrayExt<u64>) -> Vec<Number> {
        let array: Vec<u64> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl<T: af::HasAfEnum> From<af::Array<T>> for ArrayExt<T> {
    fn from(array: af::Array<T>) -> ArrayExt<T> {
        ArrayExt(array)
    }
}

impl<T: af::HasAfEnum> From<Vec<T>> for ArrayExt<T> {
    fn from(values: Vec<T>) -> ArrayExt<T> {
        let dim = dim4(values.len());
        ArrayExt(af::Array::new(&values, dim))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Add for ArrayExt<T>
where
    <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
    <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
    <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn add(self, other: Self) -> Self::Output {
        ArrayExt(af::add(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Mul for ArrayExt<T>
where
    <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
    <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
    <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn mul(self, other: Self) -> Self::Output {
        ArrayExt(af::mul(&self.0, &other.0, BATCH))
    }
}

pub trait ArrayInstanceAbs: ArrayInstance {
    type AbsValue: af::HasAfEnum;

    fn abs(&self) -> ArrayExt<Self::AbsValue>;
}

impl ArrayInstanceAbs for ArrayExt<_Complex<f32>> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<_Complex<f64>> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<f32> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<f64> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<i16> {
    type AbsValue = i16;

    fn abs(&self) -> ArrayExt<i16> {
        ArrayExt(af::abs(self.af()).cast())
    }
}

impl ArrayInstanceAbs for ArrayExt<i32> {
    type AbsValue = i32;

    fn abs(&self) -> ArrayExt<i32> {
        ArrayExt(af::abs(self.af()).cast())
    }
}

impl ArrayInstanceAbs for ArrayExt<i64> {
    type AbsValue = i64;

    fn abs(&self) -> ArrayExt<i64> {
        ArrayExt(af::abs(self.af()).cast())
    }
}

pub trait ArrayInstanceAnyAll: ArrayInstance {
    fn all(&self) -> bool {
        af::all_true_all(self.af()).0 > 0.0f64
    }

    fn any(&self) -> bool {
        af::any_true_all(self.af()).0 > 0.0f64
    }
}

impl ArrayInstanceAnyAll for ArrayExt<bool> {}
impl ArrayInstanceAnyAll for ArrayExt<f32> {}
impl ArrayInstanceAnyAll for ArrayExt<f64> {}
impl ArrayInstanceAnyAll for ArrayExt<i16> {}
impl ArrayInstanceAnyAll for ArrayExt<i32> {}
impl ArrayInstanceAnyAll for ArrayExt<i64> {}
impl ArrayInstanceAnyAll for ArrayExt<u8> {}
impl ArrayInstanceAnyAll for ArrayExt<u16> {}
impl ArrayInstanceAnyAll for ArrayExt<u32> {}
impl ArrayInstanceAnyAll for ArrayExt<u64> {}

impl ArrayInstanceAnyAll for ArrayExt<_Complex<f32>> {
    fn all(&self) -> bool {
        af::all_true_all(&af::abs(self.af())).0 > 0f64
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.af());
        any.0 > 0.0f64 || any.1 > 0.0f64
    }
}

impl ArrayInstanceAnyAll for ArrayExt<_Complex<f64>> {
    fn all(&self) -> bool {
        af::all_true_all(&af::abs(self.af())).0 > 0f64
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.af());
        any.0 > 0.0f64 || any.1 > 0.0f64
    }
}

trait ArrayInstanceNot: ArrayInstance {
    fn not(&self) -> ArrayExt<bool>;
}

impl ArrayInstanceNot for ArrayExt<bool> {
    fn not(&self) -> ArrayExt<bool> {
        ArrayExt(!self.af())
    }
}

trait ArrayInstanceBool: ArrayInstance {
    fn and(&self, other: &Self) -> ArrayExt<bool>;

    fn or(&self, other: &Self) -> ArrayExt<bool>;

    fn xor(&self, other: &Self) -> ArrayExt<bool>;
}

impl<T: af::HasAfEnum> ArrayInstanceBool for ArrayExt<T> {
    fn and(&self, other: &Self) -> ArrayExt<bool> {
        let l: af::Array<bool> = self.af().cast();
        let r: af::Array<bool> = other.af().cast();
        ArrayExt(af::and(&l, &r, BATCH))
    }

    fn or(&self, other: &Self) -> ArrayExt<bool> {
        let l: af::Array<bool> = self.af().cast();
        let r: af::Array<bool> = other.af().cast();
        ArrayExt(af::and(&l, &r, BATCH))
    }

    fn xor(&self, other: &Self) -> ArrayExt<bool> {
        let l: af::Array<bool> = self.af().cast();
        let r: af::Array<bool> = other.af().cast();

        let one = af::or(&l, &r, BATCH);
        let not_both = !(&af::and(&l, &r, BATCH));
        let one_and_not_both = af::and(&one, &not_both, BATCH);
        ArrayExt(one_and_not_both.cast())
    }
}

trait ArrayInstanceCompare {
    fn eq(&self, other: &Self) -> ArrayExt<bool>;

    fn gt(&self, other: &Self) -> ArrayExt<bool>;

    fn gte(&self, other: &Self) -> ArrayExt<bool>;

    fn lt(&self, other: &Self) -> ArrayExt<bool>;

    fn lte(&self, other: &Self) -> ArrayExt<bool>;

    fn ne(&self, other: &Self) -> ArrayExt<bool>;
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T>> ArrayInstanceCompare for ArrayExt<T> {
    fn eq(&self, other: &Self) -> ArrayExt<bool> {
        af::eq(self.af(), other.af(), BATCH).into()
    }

    fn gt(&self, other: &Self) -> ArrayExt<bool> {
        af::gt(self.af(), other.af(), BATCH).into()
    }

    fn gte(&self, other: &Self) -> ArrayExt<bool> {
        af::ge(self.af(), other.af(), BATCH).into()
    }

    fn lt(&self, other: &Self) -> ArrayExt<bool> {
        af::lt(self.af(), other.af(), BATCH).into()
    }

    fn lte(&self, other: &Self) -> ArrayExt<bool> {
        af::le(self.af(), other.af(), BATCH).into()
    }

    fn ne(&self, other: &Self) -> ArrayExt<bool> {
        self.eq(other).not()
    }
}

trait ArrayInstanceReduce: ArrayInstance {
    type Product: af::HasAfEnum;
    type Sum: af::HasAfEnum;

    fn product(&self) -> Self::Product;

    fn sum(&self) -> Self::Sum;
}

impl ArrayInstanceReduce for ArrayExt<bool> {
    type Product = u64;
    type Sum = u64;

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<_Complex<f32>> {
    type Product = _Complex<f64>;
    type Sum = _Complex<f64>;

    fn product(&self) -> Self::Product {
        let product = af::product_all(self.af());
        _Complex::new(product.0, product.1)
    }

    fn sum(&self) -> Self::Sum {
        let sum = af::sum_all(self.af());
        _Complex::new(sum.0, sum.1)
    }
}

impl ArrayInstanceReduce for ArrayExt<_Complex<f64>> {
    type Product = _Complex<f64>;
    type Sum = _Complex<f64>;

    fn product(&self) -> Self::Product {
        let product = af::product_all(self.af());
        _Complex::new(product.0, product.1)
    }

    fn sum(&self) -> Self::Sum {
        let sum = af::sum_all(self.af());
        _Complex::new(sum.0, sum.1)
    }
}

impl ArrayInstanceReduce for ArrayExt<f32> {
    type Product = f64;
    type Sum = f64;

    fn product(&self) -> f64 {
        af::product_all(self.af()).0
    }

    fn sum(&self) -> f64 {
        af::sum_all(self.af()).0
    }
}

impl ArrayInstanceReduce for ArrayExt<f64> {
    type Product = f64;
    type Sum = f64;

    fn product(&self) -> f64 {
        af::product_all(self.af()).0
    }

    fn sum(&self) -> f64 {
        af::sum_all(self.af()).0
    }
}

impl ArrayInstanceReduce for ArrayExt<i16> {
    type Product = i64;
    type Sum = i64;

    fn product(&self) -> i64 {
        af::product_all(self.af()).0 as i64
    }

    fn sum(&self) -> i64 {
        af::sum_all(self.af()).0 as i64
    }
}

impl ArrayInstanceReduce for ArrayExt<i32> {
    type Product = i64;
    type Sum = i64;

    fn product(&self) -> i64 {
        af::product_all(self.af()).0 as i64
    }

    fn sum(&self) -> i64 {
        af::sum_all(self.af()).0 as i64
    }
}

impl ArrayInstanceReduce for ArrayExt<i64> {
    type Product = i64;
    type Sum = i64;

    fn product(&self) -> i64 {
        af::product_all(self.af()).0 as i64
    }

    fn sum(&self) -> i64 {
        af::sum_all(self.af()).0 as i64
    }
}

impl ArrayInstanceReduce for ArrayExt<u8> {
    type Product = u64;
    type Sum = u64;

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<u16> {
    type Product = u64;
    type Sum = u64;

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<u32> {
    type Product = u64;
    type Sum = u64;

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<u64> {
    type Product = u64;
    type Sum = u64;

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }
}

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
    fn af_cast<T: af::HasAfEnum>(&self) -> ArrayExt<T> {
        use Array::*;
        match self {
            Bool(b) => b.as_type(),
            C32(c) => c.as_type(),
            C64(c) => c.as_type(),
            F32(f) => f.as_type(),
            F64(f) => f.as_type(),
            I16(i) => i.as_type(),
            I32(i) => i.as_type(),
            I64(i) => i.as_type(),
            U8(u) => u.as_type(),
            U16(u) => u.as_type(),
            U32(u) => u.as_type(),
            U64(u) => u.as_type(),
        }
    }

    pub fn concatenate(left: &Array, right: &Array) -> Result<Array> {
        use Array::*;
        match (left, right) {
            (U64(l), U64(r)) => Ok(U64(ArrayExt::concatenate(&l, &r))),
            (l, r) => Err(error(format!("Cannot concatenate arrays with different data types: {}, {}", l.dtype(), r.dtype()))),
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

    pub fn try_cast_from<N>(values: Vec<N>, dtype: NumberType) -> Result<Array> where Number: From<N> {
        const UNSUPPORTED: &str = "Array requires a uniform type of Number";
        let values = values.into_iter().map(Number::from);

        use Array::*;
        let chunk = match dtype {
            NumberType::Bool => Bool(vec_cast_into(values).into()),
            NumberType::Complex(c) => match c {
                ComplexType::C32 => C32(vec_cast_into(values).into()),
                ComplexType::C64 => C32(vec_cast_into(values).into()),
                ComplexType::Complex => return Err(error(UNSUPPORTED))
            },
            NumberType::Float(f) => match f {
                FloatType::F32 => F32(vec_cast_into(values).into()),
                FloatType::F64 => F32(vec_cast_into(values).into()),
                FloatType::Float => return Err(error(UNSUPPORTED))
            },
            NumberType::Int(i) => match i {
                IntType::I16 => I16(vec_cast_into(values).into()),
                IntType::I32 => I32(vec_cast_into(values).into()),
                IntType::I64 => I64(vec_cast_into(values).into()),
                IntType::Int => return Err(error(UNSUPPORTED))
            },
            NumberType::UInt(u) => match u {
                UIntType::U8 => U8(vec_cast_into(values).into()),
                UIntType::U16 => U16(vec_cast_into(values).into()),
                UIntType::U32 => U32(vec_cast_into(values).into()),
                UIntType::U64 => U64(vec_cast_into(values).into()),
                UIntType::UInt => return Err(error(UNSUPPORTED))
            },
            NumberType::Number => return Err(error(UNSUPPORTED))
        };

        Ok(chunk)
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

    pub fn into_af_array<T: af::HasAfEnum>(self) -> af::Array<T> {
        self.af_cast().0
    }

    pub fn into_type(self, dtype: NumberType) -> Array {
        use ComplexType as CT;
        use FloatType as FT;
        use IntType as IT;
        use NumberType as NT;
        use UIntType as UT;

        match dtype {
            NT::Bool => Self::Bool(self.af_cast()),
            NT::Complex(ct) => match ct {
                CT::C32 => Self::C32(self.af_cast()),
                CT::C64 => Self::C64(self.af_cast()),
                CT::Complex => Self::C64(self.af_cast()),
            },
            NT::Float(ft) => match ft {
                FT::F32 => Self::F32(self.af_cast()),
                FT::F64 => Self::F64(self.af_cast()),
                FT::Float => Self::F64(self.af_cast()),
            },
            NT::Int(it) => match it {
                IT::I16 => Self::I16(self.af_cast()),
                IT::I32 => Self::I32(self.af_cast()),
                IT::I64 => Self::I64(self.af_cast()),
                IT::Int => Self::I64(self.af_cast()),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => Self::U8(self.af_cast()),
                UT::U16 => Self::U16(self.af_cast()),
                UT::U32 => Self::U32(self.af_cast()),
                UT::U64 => Self::U64(self.af_cast()),
                UT::UInt => Self::U64(self.af_cast()),
            },
            NT::Number => self,
        }
    }

    pub fn into_values(self) -> Vec<Number> {
        use Array::*;
        match self {
            Bool(b) => b.into(),
            C32(c) => c.into(),
            C64(c) => c.into(),
            F32(f) => f.into(),
            F64(f) => f.into(),
            I16(i) => i.into(),
            I32(i) => i.into(),
            I64(i) => i.into(),
            U8(u) => u.into(),
            U16(u) => u.into(),
            U32(u) => u.into(),
            U64(u) => u.into(),
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

    pub fn add(&self, other: &Array) -> Array {
        let dtype = Ord::max(self.dtype(), other.dtype());

        use ComplexType as CT;
        use FloatType as FT;
        use IntType as IT;
        use NumberType as NT;
        use UIntType as UT;

        match dtype {
            NT::Bool => Self::Bool(self.af_cast::<bool>() + other.af_cast()),
            NT::Complex(ct) => match ct {
                CT::C32 => Self::C32(self.af_cast::<_Complex<f32>>() + other.af_cast()),
                CT::C64 => Self::C64(self.af_cast::<_Complex<f64>>() + other.af_cast()),
                CT::Complex => Self::C64(self.af_cast::<_Complex<f64>>() + other.af_cast::<_Complex<f64>>())
            },
            NT::Float(ft) => match ft {
                FT::F32 => Self::F32(self.af_cast::<f32>() + other.af_cast()),
                FT::F64 => Self::F64(self.af_cast::<f64>() + other.af_cast()),
                FT::Float => Self::F64(self.af_cast::<f64>() + other.af_cast::<f64>())
            },
            NT::Int(it) => match it {
                IT::I16 => Self::I16(self.af_cast::<i16>() + other.af_cast()),
                IT::I32 => Self::I32(self.af_cast::<i32>() + other.af_cast()),
                IT::I64 => Self::I64(self.af_cast::<i64>() + other.af_cast()),
                IT::Int => Self::I64(self.af_cast::<i64>() + other.af_cast::<i64>())
            },
            NT::UInt(ut) => match ut {
                UT::U8 => Self::U8(self.af_cast::<u8>() + other.af_cast()),
                UT::U16 => Self::U16(self.af_cast::<u16>() + other.af_cast()),
                UT::U32 => Self::U32(self.af_cast::<u32>() + other.af_cast()),
                UT::U64 => Self::U64(self.af_cast::<u64>() + other.af_cast()),
                UT::UInt => Self::U64(self.af_cast::<u64>() + other.af_cast::<u64>())
            },
            NT::Number => panic!("Array does not support generic type Number"),
        }
    }

    pub fn and(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        let that: ArrayExt<bool> = other.af_cast();
        Array::Bool(this.and(&that))
    }

    pub fn eq(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.eq(&other.af_cast())),
            C32(l) => Bool(l.eq(&other.af_cast())),
            C64(l) => Bool(l.eq(&other.af_cast())),
            F32(l) => Bool(l.eq(&other.af_cast())),
            F64(l) => Bool(l.eq(&other.af_cast())),
            I16(l) => Bool(l.eq(&other.af_cast())),
            I32(l) => Bool(l.eq(&other.af_cast())),
            I64(l) => Bool(l.eq(&other.af_cast())),
            U8(l) => Bool(l.eq(&other.af_cast())),
            U16(l) => Bool(l.eq(&other.af_cast())),
            U32(l) => Bool(l.eq(&other.af_cast())),
            U64(l) => Bool(l.eq(&other.af_cast())),
        }
    }

    pub fn gt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gt(&other.af_cast())),
            C32(l) => Bool(l.gt(&other.af_cast())),
            C64(l) => Bool(l.gt(&other.af_cast())),
            F32(l) => Bool(l.gt(&other.af_cast())),
            F64(l) => Bool(l.gt(&other.af_cast())),
            I16(l) => Bool(l.gt(&other.af_cast())),
            I32(l) => Bool(l.gt(&other.af_cast())),
            I64(l) => Bool(l.gt(&other.af_cast())),
            U8(l) => Bool(l.gt(&other.af_cast())),
            U16(l) => Bool(l.gt(&other.af_cast())),
            U32(l) => Bool(l.gt(&other.af_cast())),
            U64(l) => Bool(l.gt(&other.af_cast())),
        }
    }

    pub fn gte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gte(&other.af_cast())),
            C32(l) => Bool(l.gte(&other.af_cast())),
            C64(l) => Bool(l.gte(&other.af_cast())),
            F32(l) => Bool(l.gte(&other.af_cast())),
            F64(l) => Bool(l.gte(&other.af_cast())),
            I16(l) => Bool(l.gte(&other.af_cast())),
            I32(l) => Bool(l.gte(&other.af_cast())),
            I64(l) => Bool(l.gte(&other.af_cast())),
            U8(l) => Bool(l.gte(&other.af_cast())),
            U16(l) => Bool(l.gte(&other.af_cast())),
            U32(l) => Bool(l.gte(&other.af_cast())),
            U64(l) => Bool(l.gte(&other.af_cast())),
        }
    }

    pub fn lt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lt(&other.af_cast())),
            C32(l) => Bool(l.lt(&other.af_cast())),
            C64(l) => Bool(l.lt(&other.af_cast())),
            F32(l) => Bool(l.lt(&other.af_cast())),
            F64(l) => Bool(l.lt(&other.af_cast())),
            I16(l) => Bool(l.lt(&other.af_cast())),
            I32(l) => Bool(l.lt(&other.af_cast())),
            I64(l) => Bool(l.lt(&other.af_cast())),
            U8(l) => Bool(l.lt(&other.af_cast())),
            U16(l) => Bool(l.lt(&other.af_cast())),
            U32(l) => Bool(l.lt(&other.af_cast())),
            U64(l) => Bool(l.lt(&other.af_cast())),
        }
    }

    pub fn lte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lte(&other.af_cast())),
            C32(l) => Bool(l.lte(&other.af_cast())),
            C64(l) => Bool(l.lte(&other.af_cast())),
            F32(l) => Bool(l.lte(&other.af_cast())),
            F64(l) => Bool(l.lte(&other.af_cast())),
            I16(l) => Bool(l.lte(&other.af_cast())),
            I32(l) => Bool(l.lte(&other.af_cast())),
            I64(l) => Bool(l.lte(&other.af_cast())),
            U8(l) => Bool(l.lte(&other.af_cast())),
            U16(l) => Bool(l.lte(&other.af_cast())),
            U32(l) => Bool(l.lte(&other.af_cast())),
            U64(l) => Bool(l.lte(&other.af_cast())),
        }
    }

    pub fn ne(&self, other: &Array) -> Array {
        use Array::*;

        match self {
            Bool(l) => Bool(l.ne(&other.af_cast())),
            C32(l) => Bool(l.ne(&other.af_cast())),
            C64(l) => Bool(l.ne(&other.af_cast())),
            F32(l) => Bool(l.ne(&other.af_cast())),
            F64(l) => Bool(l.ne(&other.af_cast())),
            I16(l) => Bool(l.ne(&other.af_cast())),
            I32(l) => Bool(l.ne(&other.af_cast())),
            I64(l) => Bool(l.ne(&other.af_cast())),
            U8(l) => Bool(l.ne(&other.af_cast())),
            U16(l) => Bool(l.ne(&other.af_cast())),
            U32(l) => Bool(l.ne(&other.af_cast())),
            U64(l) => Bool(l.ne(&other.af_cast())),
        }
    }

    pub fn multiply(&self, other: &Array) -> Array {
        let dtype = Ord::max(self.dtype(), other.dtype());

        use ComplexType as CT;
        use FloatType as FT;
        use IntType as IT;
        use NumberType as NT;
        use UIntType as UT;

        match dtype {
            NT::Bool => Self::Bool(self.af_cast::<bool>() * other.af_cast()),
            NT::Complex(ct) => match ct {
                CT::C32 => Self::C32(self.af_cast::<_Complex<f32>>() * other.af_cast()),
                CT::C64 => Self::C64(self.af_cast::<_Complex<f64>>() * other.af_cast()),
                CT::Complex => Self::C64(self.af_cast::<_Complex<f64>>() * other.af_cast::<_Complex<f64>>())
            },
            NT::Float(ft) => match ft {
                FT::F32 => Self::F32(self.af_cast::<f32>() * other.af_cast()),
                FT::F64 => Self::F64(self.af_cast::<f64>() * other.af_cast()),
                FT::Float => Self::F64(self.af_cast::<f64>() * other.af_cast::<f64>())
            },
            NT::Int(it) => match it {
                IT::I16 => Self::I16(self.af_cast::<i16>() * other.af_cast()),
                IT::I32 => Self::I32(self.af_cast::<i32>() * other.af_cast()),
                IT::I64 => Self::I64(self.af_cast::<i64>() * other.af_cast()),
                IT::Int => Self::I64(self.af_cast::<i64>() * other.af_cast::<i64>())
            },
            NT::UInt(ut) => match ut {
                UT::U8 => Self::U8(self.af_cast::<u8>() * other.af_cast()),
                UT::U16 => Self::U16(self.af_cast::<u16>() * other.af_cast()),
                UT::U32 => Self::U32(self.af_cast::<u32>() * other.af_cast()),
                UT::U64 => Self::U64(self.af_cast::<u64>() * other.af_cast()),
                UT::UInt => Self::U64(self.af_cast::<u64>() * other.af_cast::<u64>())
            },
            NT::Number => panic!("Array does not support generic type Number"),
        }
    }

    pub fn not(&self) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        Array::Bool(this.not())
    }

    pub fn or(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        let that: ArrayExt<bool> = other.af_cast();
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
            Bool(l) => l.set(&index, &value.af_cast()),
            C32(l) => l.set(&index, &value.af_cast()),
            C64(l) => l.set(&index, &value.af_cast()),
            F32(l) => l.set(&index, &value.af_cast()),
            F64(l) => l.set(&index, &value.af_cast()),
            I16(l) => l.set(&index, &value.af_cast()),
            I32(l) => l.set(&index, &value.af_cast()),
            I64(l) => l.set(&index, &value.af_cast()),
            U8(l) => l.set(&index, &value.af_cast()),
            U16(l) => l.set(&index, &value.af_cast()),
            U32(l) => l.set(&index, &value.af_cast()),
            U64(l) => l.set(&index, &value.af_cast()),
        }

        Ok(())
    }

    pub fn sort(&mut self, ascending: bool) -> Result<()> {
        use Array::*;
        match self {
            Bool(ArrayExt(u)) => *self = Array::Bool(ArrayExt(af::sort(u, 0, ascending))),
            F32(ArrayExt(u)) => *self = Array::F32(ArrayExt(af::sort(u, 0, ascending))),
            F64(ArrayExt(u)) => *self = Array::F64(ArrayExt(af::sort(u, 0, ascending))),
            I16(ArrayExt(u)) => *self = Array::I16(ArrayExt(af::sort(u, 0, ascending))),
            I32(ArrayExt(u)) => *self = Array::I32(ArrayExt(af::sort(u, 0, ascending))),
            I64(ArrayExt(u)) => *self = Array::I64(ArrayExt(af::sort(u, 0, ascending))),
            U8(ArrayExt(u)) => *self = Array::U8(ArrayExt(af::sort(u, 0, ascending))),
            U16(ArrayExt(u)) => *self = Array::U16(ArrayExt(af::sort(u, 0, ascending))),
            U32(ArrayExt(u)) => *self = Array::U32(ArrayExt(af::sort(u, 0, ascending))),
            U64(ArrayExt(u)) => *self = Array::U64(ArrayExt(af::sort(u, 0, ascending))),
            other => return Err(error(format!("{} does not support ordering", other.dtype())))
        };

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
            Err(error(format!("Invalid pivot for Array of length {}", self.len())))
        }
    }

    pub fn xor(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        let that: ArrayExt<bool> = other.af_cast();
        Array::Bool(this.xor(&that))
    }
}

impl TryFrom<af::Array<u64>> for Array {
    type Error = ArrayError;

    fn try_from(arr: af::Array<u64>) -> Result<Array> {
        let size = arr.elements() as u64;
        if arr.dims() == af::Dim4::new(&[size, 1, 1, 1]) {
            Ok(Array::U64(ArrayExt(arr)))
        } else {
            Err(error("Array only supports a single dimension"))
        }
    }
}

impl From<Vec<bool>> for Array {
    fn from(b: Vec<bool>) -> Array {
        let data = af::Array::new(&b, dim4(b.len()));
        Array::Bool(data.into())
    }
}

impl From<Vec<_Complex<f32>>> for Array {
    fn from(c: Vec<_Complex<f32>>) -> Array {
        let data = af::Array::new(&c, dim4(c.len()));
        Array::C32(data.into())
    }
}

impl From<Vec<_Complex<f64>>> for Array {
    fn from(c: Vec<_Complex<f64>>) -> Array {
        let data = af::Array::new(&c, dim4(c.len()));
        Array::C64(data.into())
    }
}

impl From<Vec<f32>> for Array {
    fn from(f: Vec<f32>) -> Array {
        let data = af::Array::new(&f, dim4(f.len()));
        Array::F32(data.into())
    }
}

impl From<Vec<f64>> for Array {
    fn from(f: Vec<f64>) -> Array {
        let data = af::Array::new(&f, dim4(f.len()));
        Array::F64(data.into())
    }
}

impl From<Vec<i16>> for Array {
    fn from(i: Vec<i16>) -> Array {
        let data = af::Array::new(&i, dim4(i.len()));
        Array::I16(data.into())
    }
}

impl From<Vec<i32>> for Array {
    fn from(i: Vec<i32>) -> Array {
        let data = af::Array::new(&i, dim4(i.len()));
        Array::I32(data.into())
    }
}

impl From<Vec<i64>> for Array {
    fn from(i: Vec<i64>) -> Array {
        let data = af::Array::new(&i, dim4(i.len()));
        Array::I64(data.into())
    }
}

impl From<Vec<u8>> for Array {
    fn from(u: Vec<u8>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U8(data.into())
    }
}

impl From<Vec<u16>> for Array {
    fn from(u: Vec<u16>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U16(data.into())
    }
}

impl From<Vec<u32>> for Array {
    fn from(u: Vec<u32>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U32(data.into())
    }
}

impl From<Vec<u64>> for Array {
    fn from(u: Vec<u64>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U64(data.into())
    }
}

impl From<Array> for Vec<Number> {
    fn from(chunk: Array) -> Vec<Number> {
        use Array::*;
        match chunk {
            Bool(b) => b.into(),
            C32(c) => c.into(),
            C64(c) => c.into(),
            F32(f) => f.into(),
            F64(f) => f.into(),
            I16(i) => i.into(),
            I32(i) => i.into(),
            I64(i) => i.into(),
            U8(u) => u.into(),
            U16(u) => u.into(),
            U32(u) => u.into(),
            U64(u) => u.into(),
        }
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dtype = match self {
            Self::Bool(_) => "Bool",
            Self::C32(_) => "Complex::C32",
            Self::C64(_) => "Complex::C64",
            Self::F32(_) => "Float::F32",
            Self::F64(_) => "Float::F64",
            Self::I16(_) => "Int::I16",
            Self::I32(_) => "Int::I32",
            Self::I64(_) => "Int::I64",
            Self::U8(_) => "UInt::U8",
            Self::U16(_) => "UInt::U16",
            Self::U32(_) => "UInt::U32",
            Self::U64(_) => "UInt::U64",
        };

        write!(f, "ArrayInstance<{}>", dtype)
    }
}

fn dim4(size: usize) -> af::Dim4 {
    af::Dim4::new(&[size as u64, 1, 1, 1])
}

fn vec_into<S, D>(source: Vec<S>) -> Vec<D> where D: From<S> {
    source.into_iter().map(D::from).collect()
}

fn vec_cast_into<I: Iterator<Item = S>, D: CastFrom<S>, S>(source: I) -> Vec<D> {
    source.map(D::cast_from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_value() {
        let arr = Array::from(vec![1, 2, 3]);
        assert_eq!(arr.get_value(1), Number::from(2))
    }

    #[test]
    fn test_get() {
        let arr = Array::from(vec![1, 2, 3]);
        let indices = af::Array::new(&[1, 2], af::Dim4::new(&[2, 1, 1, 1]));
        let actual = arr.get(indices);
        let expected = Array::from(vec![2, 3]);
        assert_eq!(
            actual.eq(&expected).into_values(),
            vec![true.into(), true.into()]
        )
    }
    #[test]
    fn test_set() {
        let mut actual = Array::from(vec![1, 2, 3]);
        let indices = af::Array::new(&[1, 2], af::Dim4::new(&[2, 1, 1, 1]));
        actual.set(indices, &Array::from(vec![4, 5])).unwrap();
        let expected = Array::from(vec![1, 4, 5]);
        assert_eq!(
            actual.eq(&expected).into_values(),
            vec![true.into(), true.into(), true.into()]
        )
    }
}
