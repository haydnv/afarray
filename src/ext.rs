use std::ops::*;

use arrayfire as af;
use number_general::*;

use super::{_Complex, dim4};

const BATCH: bool = true;

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
    pub fn concatenate(left: &ArrayExt<T>, right: &ArrayExt<T>) -> ArrayExt<T> {
        af::join(0, left.af(), right.af()).into()
    }

    pub fn get_value(&self, index: usize) -> T {
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

    pub fn into_inner(self) -> af::Array<T> {
        self.0
    }

    pub fn sort(&mut self, ascending: bool)
    where
        T: af::RealNumber,
    {
        *self = Self(af::sort(&self.0, 0, ascending))
    }

    pub fn split(&self, at: usize) -> (ArrayExt<T>, ArrayExt<T>) {
        let left = af::Seq::new(0.0, at as f32, 1.0);
        let right = af::Seq::new(at as f32, self.len() as f32, 1.0);
        (
            ArrayExt(af::index(self.af(), &[left, af::Seq::default()])),
            ArrayExt(af::index(self.af(), &[right, af::Seq::default()])),
        )
    }
}

impl ArrayExt<u64> {
    pub fn from_coords<C: IntoIterator<Item = Vec<u64>>>(shape: &[u64], coords: C) -> Self {
        let ndim = shape.len();
        let coord_bounds = coord_bounds(shape);
        let af_coord_bounds: af::Array<u64> =
            af::Array::new(&coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));

        let coords: Vec<u64> = coords.into_iter().flatten().collect();
        let num_coords = coords.len() / ndim;
        assert_eq!(coords.len(), num_coords * ndim);

        let coords = af::Array::new(
            &coords,
            af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]),
        );
        let offsets = af::mul(&coords, &af_coord_bounds, true);
        let offsets = af::sum(&offsets, 0);

        Self(offsets)
    }

    pub fn to_coords(&self, shape: &[u64]) -> Vec<Vec<u64>> {
        let ndim = shape.len();
        assert_eq!(self.len() % ndim, 0);

        let coord_bounds = coord_bounds(shape);
        let af_coord_bounds: af::Array<u64> =
            af::Array::new(&coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));
        let af_shape: af::Array<u64> =
            af::Array::new(&shape.to_vec(), af::Dim4::new(&[1, ndim as u64, 1, 1]));

        let offsets = af::div(self.af(), &af_coord_bounds, true);
        let offsets = af::modulo(&offsets, &af_shape, true);

        let mut coords = vec![0u64; offsets.elements()];
        af::transpose(&offsets, false).host(&mut coords);
        coords.chunks(ndim).map(|coord| coord.to_vec()).collect()
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

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Add for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn add(self, other: Self) -> Self::Output {
        ArrayExt(af::add(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> AddAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn add_assign(&mut self, other: Self) {
        let sum = &*self + &other;
        *self = sum.as_type();
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Sub for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn sub(self, other: Self) -> Self::Output {
        ArrayExt(af::sub(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> SubAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn sub_assign(&mut self, other: Self) {
        let diff = &*self - &other;
        *self = diff.as_type();
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Mul for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn mul(self, other: Self) -> Self::Output {
        ArrayExt(af::mul(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> MulAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn mul_assign(&mut self, other: Self) {
        let product = &*self * &other;
        *self = product.as_type();
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Div for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn div(self, other: Self) -> Self::Output {
        ArrayExt(af::div(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> DivAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn div_assign(&mut self, other: Self) {
        let div = &*self / &other;
        *self = div.as_type();
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

pub trait ArrayInstanceNot: ArrayInstance {
    fn not(&self) -> ArrayExt<bool>;
}

impl ArrayInstanceNot for ArrayExt<bool> {
    fn not(&self) -> ArrayExt<bool> {
        ArrayExt(!self.af())
    }
}

pub trait ArrayInstanceBool: ArrayInstance {
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

pub trait ArrayInstanceCompare {
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

pub trait ArrayInstanceReduce: ArrayInstance {
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

fn coord_bounds(shape: &[u64]) -> Vec<u64> {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}

fn vec_into<S, D>(source: Vec<S>) -> Vec<D>
where
    D: From<S>,
{
    source.into_iter().map(D::from).collect()
}
