use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::*;

use arrayfire as af;
use async_trait::async_trait;
use destream::{de, en};
use futures::{future, stream, Stream};
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

use super::{dim4, Complex};

const BATCH: bool = true;

/// Defines common access methods for instance of [`ArrayExt`].
pub trait ArrayInstance {
    type DType: af::HasAfEnum;

    /// Borrow this instance as an [`af::Array`].
    fn af(&'_ self) -> &'_ af::Array<Self::DType>;

    /// Borrow this instance as a a mutable [`af::Array`].
    fn af_mut(&'_ mut self) -> &'_ mut af::Array<Self::DType>;

    /// Cast this instance into an [`af::Array`] with type `T`.
    fn af_cast<T: af::HasAfEnum>(&self) -> af::Array<T> {
        self.af().cast()
    }

    /// How many elements are in this `ArrayInstance`.
    fn len(&self) -> usize {
        self.af().elements()
    }

    /// Get the values specified by the given [`af::Indexer`].
    fn get(&self, index: af::Indexer) -> af::Array<Self::DType> {
        af::index_gen(self.af(), index)
    }

    /// Set the values specified by the given [`af::Indexer`] to the corresponding values in `T`.
    fn set<T: ArrayInstance<DType = Self::DType>>(&mut self, index: &af::Indexer, other: &T) {
        af::assign_gen(self.af_mut(), index, other.af());
    }

    /// Set the value at the specified index to `value`.
    fn set_at(&mut self, offset: usize, value: Self::DType) {
        let seq = af::Seq::new(offset as f32, offset as f32, 1.0f32);
        af::assign_seq(self.af_mut(), &[seq], &af::Array::new(&[value], dim4(1)));
    }

    /// Copy the data in this [`af::Array`] into a new `Vec`.
    fn to_vec(&self) -> Vec<Self::DType>
    where
        Self::DType: Clone + Default,
    {
        let mut v = vec![Self::DType::default(); self.len()];
        self.af().host(&mut v);
        v
    }
}

/// A wrapper around [`af::Array`] which defines common operations.
#[derive(Clone)]
pub struct ArrayExt<T: af::HasAfEnum>(af::Array<T>);

impl<T: af::HasAfEnum + Default> ArrayExt<T> {
    /// Construct a new ArrayExt with the given value and length.
    pub fn constant(value: T, length: usize) -> Self
    where
        T: af::ConstGenerator<OutType = T>,
    {
        let dim = dim4(length);
        Self(af::constant(value, dim))
    }

    /// Concatenate two instances of `ArrayExt<T>`.
    pub fn concatenate(left: &Self, right: &Self) -> Self {
        af::join(0, left.af(), right.af()).into()
    }

    /// Return `true` if the elements of this `ArrayExt` are in sorted order.
    pub fn is_sorted(&self) -> bool
    where
        T: af::RealNumber + Clone,
    {
        af::all_true_all(&af::eq(self.af(), self.sorted(true).af(), false)).0
    }

    /// Cast the values of this `ArrayExt` into a destination type `D`.
    pub fn type_cast<D: af::HasAfEnum>(&self) -> ArrayExt<D> {
        ArrayExt(self.af_cast())
    }

    /// Get the values specified by the given [`af::Indexer`].
    pub fn get(&self, index: af::Indexer) -> Self {
        Self(ArrayInstance::get(self, index))
    }

    /// Get the value at the given index.
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

    /// Deconstruct this `ArrayExt<T>` into its underlying [`af::Array`].
    pub fn into_inner(self) -> af::Array<T> {
        self.0
    }

    /// Return a slice of this `ArrayExt`.
    ///
    /// Panics: if `end` is out of bounds
    pub fn slice(&self, start: usize, end: usize) -> Self {
        af::index(
            self.af(),
            &[af::Seq::new(start as f32, (end - 1) as f32, 1.)],
        )
        .into()
    }

    /// Sort this `ArrayExt`.
    pub fn sort(&mut self, ascending: bool)
    where
        T: af::RealNumber,
    {
        *self = self.sorted(ascending)
    }

    /// Return a sorted copy of this `ArrayExt`.
    pub fn sorted(&self, ascending: bool) -> Self
    where
        T: af::RealNumber,
    {
        debug_assert_eq!(self.0.dims(), dim4(self.len()));
        Self(af::sort(&self.0, 0, ascending))
    }

    /// Split this `ArrayExt<T>` into two new instances at the given pivot.
    pub fn split(&self, at: usize) -> (Self, Self) {
        let left = af::Seq::new(0.0, at as f32, 1.0);
        let right = af::Seq::new(at as f32, self.len() as f32, 1.0);
        (
            ArrayExt(af::index(self.af(), &[left, af::Seq::default()])),
            ArrayExt(af::index(self.af(), &[right, af::Seq::default()])),
        )
    }

    /// Return only the unique values from this `ArrayExt`.
    ///
    /// Pass `true` for `sorted` if this `ArrayExt` is known to be in sorted order.
    pub fn unique(&self, sorted: bool) -> Self
    where
        T: af::RealNumber,
    {
        Self(af::set_unique(self.af(), sorted))
    }

    fn into_stream(self) -> impl Stream<Item = Vec<T>>
    where
        T: Clone,
    {
        stream::once(future::ready(self.to_vec()))
    }

    fn to_stream(&self) -> impl Stream<Item = Vec<T>>
    where
        T: Clone,
    {
        stream::once(future::ready(self.to_vec()))
    }
}

impl ArrayExt<bool> {
    /// Logical not.
    pub fn not(&self) -> Self {
        ArrayExt(!self.af())
    }

    /// Logical and.
    pub fn and(&self, other: &Self) -> Self {
        ArrayExt(af::and(self.af(), other.af(), BATCH))
    }

    /// Logical or.
    pub fn or(&self, other: &Self) -> Self {
        ArrayExt(af::or(self.af(), other.af(), BATCH))
    }

    /// Logical xor.
    pub fn xor(&self, other: &Self) -> Self {
        let one = af::or(self.af(), other.af(), BATCH);
        let not_both = !(&af::and(self.af(), other.af(), BATCH));
        let one_and_not_both = af::and(&one, &not_both, BATCH);
        ArrayExt(one_and_not_both)
    }
}

impl ArrayExt<Complex<f32>> {
    /// Get the real component of this array.
    pub fn re(&self) -> ArrayExt<f32> {
        af::real(&self.0).into()
    }

    /// Get the imaginary component of this array.
    pub fn im(&self) -> ArrayExt<f32> {
        af::imag(&self.0).into()
    }
}

impl ArrayExt<Complex<f64>> {
    /// Get the real component of this array.
    pub fn re(&self) -> ArrayExt<f64> {
        af::real(&self.0).into()
    }

    /// Get the imaginary component of this array.
    pub fn im(&self) -> ArrayExt<f64> {
        af::imag(&self.0).into()
    }
}

impl ArrayExt<u64> {
    /// Construct a new `ArrayExt<u64>` with elements `start..end`.
    pub fn range(start: u64, end: u64) -> Self {
        let dims = dim4((end - start) as usize);
        let tile = dim4(1);
        let range: af::Array<u64> = af::iota(dims, tile).into();
        if start == 0 {
            range.into()
        } else {
            af::add(&range, &af::Array::new(&[start], dim4(1)), true).into()
        }
    }
}

impl<T: af::HasAfEnum> ArrayInstance for ArrayExt<T> {
    type DType = T;

    #[inline]
    fn af(&'_ self) -> &'_ af::Array<T> {
        &self.0
    }

    #[inline]
    fn af_mut(&mut self) -> &mut af::Array<Self::DType> {
        &mut self.0
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Add for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn add(self, other: Self) -> Self::Output {
        ArrayExt(af::add(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Add for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn add(self, other: Self) -> Self::Output {
        ArrayExt(af::add(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> AddAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn add_assign(&mut self, other: Self) {
        let sum = &*self + &other;
        *self = sum.type_cast();
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Mul for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn mul(self, other: Self) -> Self::Output {
        ArrayExt(af::mul(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Mul for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn mul(self, other: Self) -> Self::Output {
        ArrayExt(af::mul(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> MulAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn mul_assign(&mut self, other: Self) {
        let product = &*self * &other;
        *self = product.type_cast();
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Div for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn div(self, other: Self) -> Self::Output {
        ArrayExt(af::div(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Div for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn div(self, other: Self) -> Self::Output {
        ArrayExt(af::div(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Sub for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn sub(self, other: Self) -> Self::Output {
        ArrayExt(af::sub(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Sub for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn sub(self, other: Self) -> Self::Output {
        ArrayExt(af::sub(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> SubAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn sub_assign(&mut self, other: Self) {
        let diff = &*self - &other;
        *self = diff.type_cast();
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Rem for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn rem(self, other: Self) -> Self::Output {
        ArrayExt(af::modulo(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Rem for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn rem(self, other: Self) -> Self::Output {
        ArrayExt(af::modulo(&self.0, &other.0, BATCH))
    }
}

impl From<(ArrayExt<f32>, ArrayExt<f32>)> for ArrayExt<Complex<f32>> {
    fn from(elements: (ArrayExt<f32>, ArrayExt<f32>)) -> Self {
        let (re, im) = elements;
        Self(af::cplx2(re.af(), im.af(), false).cast())
    }
}

impl From<(ArrayExt<f64>, ArrayExt<f64>)> for ArrayExt<Complex<f64>> {
    fn from(elements: (ArrayExt<f64>, ArrayExt<f64>)) -> Self {
        let (re, im) = elements;
        Self(af::cplx2(re.af(), im.af(), false).cast())
    }
}

impl<T: af::HasAfEnum> From<af::Array<T>> for ArrayExt<T> {
    fn from(array: af::Array<T>) -> ArrayExt<T> {
        ArrayExt(array)
    }
}

impl<T: af::HasAfEnum> From<&[T]> for ArrayExt<T> {
    fn from(values: &[T]) -> ArrayExt<T> {
        let dim = dim4(values.len());
        ArrayExt(af::Array::new(values, dim))
    }
}

impl<T: af::HasAfEnum> From<Vec<T>> for ArrayExt<T> {
    fn from(values: Vec<T>) -> ArrayExt<T> {
        Self::from(&values[..])
    }
}

impl<T: af::HasAfEnum> FromIterator<T> for ArrayExt<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values = Vec::from_iter(iter);
        let dim = dim4(values.len());
        ArrayExt(af::Array::new(&values, dim))
    }
}

/// Defines an absolute value method `abs`.
pub trait ArrayInstanceAbs: ArrayInstance {
    type AbsValue: af::HasAfEnum;

    /// Calculate the element-wise absolute value.
    fn abs(&self) -> ArrayExt<Self::AbsValue>;
}

impl ArrayInstanceAbs for ArrayExt<Complex<f32>> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<Complex<f64>> {
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

/// Defines cumulative boolean operations `any` and `all`.
pub trait ArrayInstanceAnyAll: ArrayInstance {
    /// Returns `true` if all elements are nonzero.
    fn all(&self) -> bool {
        af::all_true_all(self.af()).0
    }

    /// Returns `true` if any element is nonzero.
    fn any(&self) -> bool {
        af::any_true_all(self.af()).0
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

impl ArrayInstanceAnyAll for ArrayExt<Complex<f32>> {
    fn all(&self) -> bool {
        af::all_true_all(&af::abs(self.af())).0
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.af());
        any.0 || any.1
    }
}

impl ArrayInstanceAnyAll for ArrayExt<Complex<f64>> {
    fn all(&self) -> bool {
        af::all_true_all(&af::abs(self.af())).0
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.af());
        any.0 || any.1
    }
}

/// Defines element-wise comparison operations.
pub trait ArrayInstanceCompare {
    /// Element-wise check for infinite values.
    fn is_infinite(&self) -> ArrayExt<bool>;

    /// Element-wise check for non-numeric (NaN) values.
    fn is_nan(&self) -> ArrayExt<bool>;

    /// Element-wise equality.
    fn eq(&self, other: &Self) -> ArrayExt<bool>;

    /// Element-wise greater-than comparison.
    fn gt(&self, other: &Self) -> ArrayExt<bool>;

    /// Element-wise greater-or-equal comparison
    fn gte(&self, other: &Self) -> ArrayExt<bool>;

    /// Element-wise less-than comparison.
    fn lt(&self, other: &Self) -> ArrayExt<bool>;

    /// Element-wise less-or-equal comparison.
    fn lte(&self, other: &Self) -> ArrayExt<bool>;

    /// Element-wise inequality.
    fn ne(&self, other: &Self) -> ArrayExt<bool>;
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T>> ArrayInstanceCompare for ArrayExt<T> {
    fn is_infinite(&self) -> ArrayExt<bool> {
        af::isinf(self.af()).into()
    }

    fn is_nan(&self) -> ArrayExt<bool> {
        af::isnan(self.af()).into()
    }

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

/// Defines common reduction operations `product` and `sum`.
pub trait ArrayInstanceReduce: ArrayInstance {
    type Product: af::HasAfEnum;
    type Sum: af::HasAfEnum;

    /// Calculate the cumulative product.
    fn product(&self) -> Self::Product;

    /// Calculate the cumulative sum.
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

impl ArrayInstanceReduce for ArrayExt<Complex<f32>> {
    type Product = Complex<f32>;
    type Sum = Complex<f32>;

    fn product(&self) -> Self::Product {
        let product = af::product_all(self.af());
        Complex::new(product.0, product.1)
    }

    fn sum(&self) -> Self::Sum {
        let sum = af::sum_all(self.af());
        Complex::new(sum.0, sum.1)
    }
}

impl ArrayInstanceReduce for ArrayExt<Complex<f64>> {
    type Product = Complex<f64>;
    type Sum = Complex<f64>;

    fn product(&self) -> Self::Product {
        let product = af::product_all(self.af());
        Complex::new(product.0, product.1)
    }

    fn sum(&self) -> Self::Sum {
        let sum = af::sum_all(self.af());
        Complex::new(sum.0, sum.1)
    }
}

impl ArrayInstanceReduce for ArrayExt<f32> {
    type Product = f32;
    type Sum = f32;

    fn product(&self) -> f32 {
        af::product_all(self.af()).0
    }

    fn sum(&self) -> f32 {
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

impl<'de, T: af::HasAfEnum + Deserialize<'de> + 'de> Deserialize<'de> for ArrayExt<T>
where
    ArrayExt<T>: From<Vec<T>>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<T>::deserialize(deserializer).map(ArrayExt::from)
    }
}

impl<T: af::HasAfEnum + Clone + Default + Serialize> Serialize for ArrayExt<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.to_vec().serialize(s)
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<bool> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_bool(ArrayExtVisitor::<bool>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<f32> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_f32(ArrayExtVisitor::<f32>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<f64> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_f64(ArrayExtVisitor::<f64>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<u8> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_u8(ArrayExtVisitor::<u8>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<u16> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_u16(ArrayExtVisitor::<u16>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<u32> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_u32(ArrayExtVisitor::<u32>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<u64> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_u64(ArrayExtVisitor::<u64>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<i16> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_i16(ArrayExtVisitor::<i16>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<i32> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_i32(ArrayExtVisitor::<i32>::default())
            .await
    }
}

#[async_trait]
impl de::FromStream for ArrayExt<i64> {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_array_i64(ArrayExtVisitor::<i64>::default())
            .await
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<bool> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_bool(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<bool> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_bool(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<f32> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_f32(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<f32> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_f32(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<f64> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_f64(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<f64> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_f64(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<u8> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u8(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<u8> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u8(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<u16> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u16(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<u16> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u16(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<u32> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u32(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<u32> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u32(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<u64> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u64(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<u64> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_u64(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<i16> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_i16(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<i16> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_i16(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<i32> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_i32(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<i32> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_i32(self.to_stream())
    }
}

impl<'en> en::IntoStream<'en> for ArrayExt<i64> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_i64(self.into_stream())
    }
}

impl<'en> en::ToStream<'en> for ArrayExt<i64> {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_array_i64(self.to_stream())
    }
}

impl<T: af::HasAfEnum + fmt::Display + Default + Clone> fmt::Debug for ArrayExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ArrayExt<{}>({}): {}",
            std::any::type_name::<T>(),
            self.af().dims(),
            self
        )
    }
}

impl<T: af::HasAfEnum + fmt::Display + Default + Clone> fmt::Display for ArrayExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let as_str: String = self
            .to_vec()
            .into_iter()
            .map(|n| n.to_string())
            .collect::<Vec<String>>()
            .join(",");

        write!(f, "[{}]", as_str)
    }
}

#[derive(Default)]
struct ArrayExtVisitor<T> {
    phantom: PhantomData<T>,
}

impl<'a, T: af::HasAfEnum + Clone + Copy + Default + Send + 'a> ArrayExtVisitor<T>
where
    ArrayExt<T>: From<Vec<T>>,
{
    async fn visit_array<A: de::ArrayAccess<T>>(mut access: A) -> Result<ArrayExt<T>, A::Error> {
        let mut buf = [T::default(); 4096];
        let mut elements = Vec::new();

        loop {
            let num = access.buffer(&mut buf).await?;
            if num == 0 {
                break;
            } else {
                elements.extend_from_slice(&buf[..num]);
            }
        }

        Ok(ArrayExt::from(elements))
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<bool> {
    type Value = ArrayExt<bool>;

    fn expecting() -> &'static str {
        "a boolean array"
    }

    async fn visit_array_bool<A: de::ArrayAccess<bool>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<f32> {
    type Value = ArrayExt<f32>;

    fn expecting() -> &'static str {
        "a 32-bit float array"
    }

    async fn visit_array_f32<A: de::ArrayAccess<f32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<f64> {
    type Value = ArrayExt<f64>;

    fn expecting() -> &'static str {
        "a 64-bit float array"
    }

    async fn visit_array_f64<A: de::ArrayAccess<f64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<u8> {
    type Value = ArrayExt<u8>;

    fn expecting() -> &'static str {
        "an 8-bit unsigned integer array"
    }

    async fn visit_array_u8<A: de::ArrayAccess<u8>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<u16> {
    type Value = ArrayExt<u16>;

    fn expecting() -> &'static str {
        "a 16-bit unsigned integer array"
    }

    async fn visit_array_u16<A: de::ArrayAccess<u16>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<u32> {
    type Value = ArrayExt<u32>;

    fn expecting() -> &'static str {
        "a 32-bit unsigned integer array"
    }

    async fn visit_array_u32<A: de::ArrayAccess<u32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<u64> {
    type Value = ArrayExt<u64>;

    fn expecting() -> &'static str {
        "a 64-bit unsigned integer array"
    }

    async fn visit_array_u64<A: de::ArrayAccess<u64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<i16> {
    type Value = ArrayExt<i16>;

    fn expecting() -> &'static str {
        "a 16-bit integer array"
    }

    async fn visit_array_i16<A: de::ArrayAccess<i16>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<i32> {
    type Value = ArrayExt<i32>;

    fn expecting() -> &'static str {
        "a 32-bit integer array"
    }

    async fn visit_array_i32<A: de::ArrayAccess<i32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[async_trait]
impl de::Visitor for ArrayExtVisitor<i64> {
    type Value = ArrayExt<i64>;

    fn expecting() -> &'static str {
        "a 64-bit integer array"
    }

    async fn visit_array_i64<A: de::ArrayAccess<i64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        Self::visit_array(access).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range() {
        let range = ArrayExt::range(1, 10);
        assert_eq!(range.to_vec(), (1..10).collect::<Vec<u64>>())
    }
}
