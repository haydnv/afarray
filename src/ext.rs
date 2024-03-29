use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::*;

use arrayfire as af;
use async_trait::async_trait;
use destream::{de, en};
use futures::{future, stream, Stream};
use number_general::{DType, NumberType};
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

use super::Complex;

/// Convenience methods defining the base value of a reduce operation on an `ArrayExt`.
pub trait HasArrayExt
where
    Self: af::HasAfEnum,
{
    /// The base value of a `product` operation.
    fn product_one() -> <Self as af::HasAfEnum>::ProductOutType;

    /// The base value of a `sum` operation.
    fn zero_sum() -> <Self as af::HasAfEnum>::AggregateOutType;

    /// The base value of a comparison (min or max).
    fn zero_cmp() -> Self;
}

macro_rules! has_array_ext {
    ($t:ty, $one:expr, $zero_sum:expr, $zero_cmp:expr) => {
        impl HasArrayExt for $t {
            fn product_one() -> <Self as af::HasAfEnum>::ProductOutType {
                $one
            }

            fn zero_sum() -> <Self as af::HasAfEnum>::AggregateOutType {
                $zero_sum
            }

            fn zero_cmp() -> Self {
                $zero_cmp
            }
        }
    };
}

has_array_ext!(bool, true, 0, false);
has_array_ext!(u8, 1, 0, 0);
has_array_ext!(u16, 1, 0, 0);
has_array_ext!(u32, 1, 0, 0);
has_array_ext!(u64, 1, 0, 0);
has_array_ext!(i16, 1, 0, 0);
has_array_ext!(i32, 1, 0, 0);
has_array_ext!(i64, 1, 0, 0);
has_array_ext!(f32, 1., 0., 0.);
has_array_ext!(f64, 1., 0., 0.);
has_array_ext!(
    Complex<f32>,
    Complex::new(1., 1.),
    Complex::new(0., 0.),
    Complex::new(0., 0.)
);
has_array_ext!(
    Complex<f64>,
    Complex::new(1., 1.),
    Complex::new(0., 0.),
    Complex::new(0., 0.)
);

/// Defines common access methods for instance of [`ArrayExt`].
pub trait ArrayInstance: Deref<Target = af::Array<Self::DType>> + DerefMut {
    type DType: af::HasAfEnum;

    /// Cast this instance into an [`af::Array`] with type `T`.
    fn af_cast<T: af::HasAfEnum>(&self) -> af::Array<T> {
        self.cast()
    }

    /// How many elements are in this `ArrayInstance`.
    fn len(&self) -> usize {
        self.elements()
    }

    /// Get the values specified by the given [`af::Indexer`].
    fn get(&self, index: af::Indexer) -> af::Array<Self::DType> {
        af::index_gen(self, index)
    }

    /// Set the values specified by the given [`af::Indexer`] to the corresponding values in `T`.
    fn set<T: ArrayInstance<DType = Self::DType>>(&mut self, index: &af::Indexer, other: &T) {
        af::assign_gen(self, index, other);
    }

    /// Set the value at the specified index to `value`.
    fn set_at(&mut self, offset: usize, value: Self::DType) {
        let seq = af::seq!(offset as i32, offset as i32, 1);
        af::assign_seq(self, &[seq], &af::Array::new(&[value], af::dim4!(1)));
    }

    /// Copy the data in this [`af::Array`] into a new `Vec`.
    fn to_vec(&self) -> Vec<Self::DType>
    where
        Self::DType: Clone + Default,
    {
        let mut v = vec![Self::DType::default(); self.len()];
        self.host(&mut v);
        v
    }
}

/// A wrapper around [`af::Array`] which defines common operations.
#[derive(Clone)]
pub struct ArrayExt<T: af::HasAfEnum>(af::Array<T>);

impl<T: af::HasAfEnum + Default> ArrayExt<T> {
    /// Construct a new `ArrayExt` with the given value and length.
    pub fn constant(value: T, length: usize) -> Self
    where
        T: af::ConstGenerator<OutType = T>,
    {
        let dim = af::dim4!(length as u64);
        Self(af::constant(value, dim))
    }

    /// Concatenate two instances of `ArrayExt<T>`.
    pub fn concatenate(left: &Self, right: &Self) -> Self {
        af::join(0, left, right).into()
    }

    /// Raise `e` to the power of `self`.
    pub fn exp(&self) -> ArrayExt<T::UnaryOutType> {
        af::exp(self).into()
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
        debug_assert!(index < self.elements());

        let af_value = af::index(self, &[af::seq!(index as i32, index as i32, 1)]);
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
        af::index(self, &[af::seq!(start as i32, (end - 1) as i32, 1)]).into()
    }

    /// Split this `ArrayExt<T>` into two new instances at the given pivot.
    pub fn split(&self, at: usize) -> (Self, Self) {
        let left = af::seq!(0, at as i32, 1);
        let right = af::seq!(at as i32, self.len() as i32, 1);
        (
            ArrayExt(af::index(self, &[left, af::seq!()])),
            ArrayExt(af::index(self, &[right, af::seq!()])),
        )
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

impl<T: af::HasAfEnum + af::RealNumber + Clone + Default> ArrayExt<T> {
    /// Return `true` if the elements of this `ArrayExt` are in sorted order.
    pub fn is_sorted(&self) -> bool
    where
        T: af::RealNumber + Clone,
    {
        let sorted = self.sorted(true);
        af::all_true_all(&af::eq(self.deref(), sorted.deref(), false)).0
    }

    /// Sort this `ArrayExt`.
    pub fn sort(&mut self, ascending: bool) {
        *self = self.sorted(ascending)
    }

    /// Compute the indices needed to sort this `ArrayExt`.
    pub fn sort_index(&self, ascending: bool) -> (ArrayExt<T>, ArrayExt<u32>) {
        let (sorted, indices) = af::sort_index(self, 0, ascending);
        (sorted.into(), indices.into())
    }

    /// Return a sorted copy of this `ArrayExt`.
    pub fn sorted(&self, ascending: bool) -> Self {
        debug_assert_eq!(self.dims(), af::dim4!(self.len() as u64));
        Self(af::sort(self, 0, ascending))
    }

    /// Return only the unique values from this `ArrayExt`.
    ///
    /// Pass `true` for `sorted` if this `ArrayExt` is known to be in sorted order.
    pub fn unique(&self, sorted: bool) -> Self {
        Self(af::set_unique(self, sorted))
    }
}

impl<T: af::HasAfEnum + af::FloatingPoint + Default> ArrayExt<T> {
    /// Construct a new `ArrayExt` with a random normal distribution.
    pub fn random_normal(length: usize) -> Self {
        let dim = af::dim4!(length as u64);
        let engine = af::get_default_random_engine();
        Self(af::random_normal(dim, &engine))
    }

    /// Construct a new `ArrayExt` with a uniform random distribution.
    pub fn random_uniform(length: usize) -> Self {
        let dim = af::dim4!(length as u64);
        let engine = af::get_default_random_engine();
        Self(af::random_uniform(dim, &engine))
    }
}

impl<T: af::HasAfEnum> ArrayInstance for ArrayExt<T> {
    type DType = T;
}

impl<T: af::HasAfEnum> Deref for ArrayExt<T> {
    type Target = af::Array<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: af::HasAfEnum> DerefMut for ArrayExt<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ArrayExt<bool> {
    /// Logical not.
    pub fn not(&self) -> Self {
        ArrayExt(!self.deref())
    }

    /// Logical and.
    pub fn and(&self, other: &Self) -> Self {
        ArrayExt(af::and(self, other, batch(self, other)))
    }

    /// Logical or.
    pub fn or(&self, other: &Self) -> Self {
        ArrayExt(af::or(self, other, batch(self, other)))
    }

    /// Logical xor.
    pub fn xor(&self, other: &Self) -> Self {
        let batch = batch(self, other);
        let one = af::or(self, other, batch);
        let not_both = !(&af::and(self, other, batch));
        let one_and_not_both = af::and(&one, &not_both, batch);
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
        let dims = af::dim4!((end - start));
        let tile = af::dim4!(1);
        let range: af::Array<u64> = af::iota(dims, tile).into();
        if start == 0 {
            range.into()
        } else {
            af::add(&range, &af::Array::new(&[start], af::dim4!(1)), true).into()
        }
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Add for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn add(self, other: Self) -> Self::Output {
        ArrayExt(af::add(&self.0, &other.0, batch(&self, &other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Add for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn add(self, other: Self) -> Self::Output {
        ArrayExt(af::add(&self.0, &other.0, batch(self, other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Mul for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn mul(self, other: Self) -> Self::Output {
        ArrayExt(af::mul(&self.0, &other.0, batch(&self, &other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Mul for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn mul(self, other: Self) -> Self::Output {
        ArrayExt(af::mul(&self.0, &other.0, batch(self, other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Div for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn div(self, other: Self) -> Self::Output {
        ArrayExt(af::div(&self.0, &other.0, batch(&self, &other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Div for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn div(self, other: Self) -> Self::Output {
        ArrayExt(af::div(&self.0, &other.0, batch(self, other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> DivAssign for ArrayExt<T>
where
    <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
    <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
    <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum
{

    fn div_assign(&mut self, other: Self) {
        let quotient = &*self / &other;
        *self = quotient.type_cast();
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Sub for ArrayExt<T>
where
    <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
    <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
    <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum,
{

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn sub(self, other: Self) -> Self::Output {
        ArrayExt(af::sub(&self.0, &other.0, batch(&self, &other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Sub for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn sub(self, other: Self) -> Self::Output {
        ArrayExt(af::sub(&self.0, &other.0, batch(self, other)))
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
        ArrayExt(af::modulo(&self.0, &other.0, batch(&self, &other)))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Rem for &ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn rem(self, other: Self) -> Self::Output {
        ArrayExt(af::modulo(&self.0, &other.0, batch(self, other)))
    }
}

impl From<(ArrayExt<f32>, ArrayExt<f32>)> for ArrayExt<Complex<f32>> {
    fn from(elements: (ArrayExt<f32>, ArrayExt<f32>)) -> Self {
        let (re, im) = elements;
        Self(af::cplx2(&*re, &*im, false).cast())
    }
}

impl From<(ArrayExt<f64>, ArrayExt<f64>)> for ArrayExt<Complex<f64>> {
    fn from(elements: (ArrayExt<f64>, ArrayExt<f64>)) -> Self {
        let (re, im) = elements;
        Self(af::cplx2(&*re, &*im, false).cast())
    }
}

impl<T: af::HasAfEnum> From<af::Array<T>> for ArrayExt<T> {
    fn from(array: af::Array<T>) -> ArrayExt<T> {
        ArrayExt(array)
    }
}

impl<T: af::HasAfEnum> From<&[T]> for ArrayExt<T> {
    fn from(values: &[T]) -> ArrayExt<T> {
        let dim = af::dim4!(values.len() as u64);
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
        let dim = af::dim4!(values.len() as u64);
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
        ArrayExt(af::abs(&*self))
    }
}

impl ArrayInstanceAbs for ArrayExt<Complex<f64>> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(&*self))
    }
}

impl ArrayInstanceAbs for ArrayExt<f32> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(&*self))
    }
}

impl ArrayInstanceAbs for ArrayExt<f64> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(&*self))
    }
}

impl ArrayInstanceAbs for ArrayExt<i16> {
    type AbsValue = i16;

    fn abs(&self) -> ArrayExt<i16> {
        ArrayExt(af::abs(&*self).cast())
    }
}

impl ArrayInstanceAbs for ArrayExt<i32> {
    type AbsValue = i32;

    fn abs(&self) -> ArrayExt<i32> {
        ArrayExt(af::abs(&*self).cast())
    }
}

impl ArrayInstanceAbs for ArrayExt<i64> {
    type AbsValue = i64;

    fn abs(&self) -> ArrayExt<i64> {
        ArrayExt(af::abs(&*self).cast())
    }
}

/// Defines a rounding method `round`.
pub trait ArrayInstanceRound: ArrayInstance {
    type Round: af::HasAfEnum;

    /// Round to the nearest integer, element-wise.
    fn round(&self) -> ArrayExt<Self::Round>;
}

impl<T: af::HasAfEnum> ArrayInstanceRound for ArrayExt<T> {
    type Round = T::AbsOutType;

    fn round(&self) -> ArrayExt<Self::Round> {
        ArrayExt(af::round(self))
    }
}

/// Defines cumulative boolean operations `any` and `all`.
pub trait ArrayInstanceAnyAll: ArrayInstance {
    /// Returns `true` if all elements are nonzero.
    fn all(&self) -> bool {
        af::all_true_all(&*self).0
    }

    /// Returns `true` if any element is nonzero.
    fn any(&self) -> bool {
        af::any_true_all(&*self).0
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
        af::all_true_all(&af::abs(&*self)).0
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(&*self);
        any.0 || any.1
    }
}

impl ArrayInstanceAnyAll for ArrayExt<Complex<f64>> {
    fn all(&self) -> bool {
        af::all_true_all(&af::abs(&*self)).0
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(&*self);
        any.0 || any.1
    }
}

/// Indexing operations for `ArrayExt`
pub trait ArrayInstanceIndex: ArrayInstance {
    fn argmax(&self) -> (usize, Self::DType);
}

macro_rules! array_index_real {
    ($t:ty) => {
        impl ArrayInstanceIndex for ArrayExt<$t> {
            fn argmax(&self) -> (usize, $t) {
                let (max, _, i) = af::imax_all(self);
                (i as usize, max)
            }
        }
    };
}

array_index_real!(bool);
array_index_real!(f32);
array_index_real!(f64);
array_index_real!(i16);
array_index_real!(i32);
array_index_real!(i64);
array_index_real!(u8);
array_index_real!(u16);
array_index_real!(u32);
array_index_real!(u64);

macro_rules! array_index_complex {
    ($t:ty) => {
        impl ArrayInstanceIndex for ArrayExt<$t> {
            fn argmax(&self) -> (usize, $t) {
                let (max_re, max_im, i) = af::imax_all(self);
                (i as usize, <$t>::new(max_re, max_im))
            }
        }
    };
}

array_index_complex!(Complex<f32>);
array_index_complex!(Complex<f64>);

/// Methods to check for infinite or non-numeric array elements.
pub trait ArrayInstanceUnreal {
    /// Element-wise check for infinite values.
    fn is_infinite(&self) -> ArrayExt<bool>;

    /// Element-wise check for non-numeric (NaN) values.
    fn is_nan(&self) -> ArrayExt<bool>;
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T>> ArrayInstanceUnreal for ArrayExt<T> {
    fn is_infinite(&self) -> ArrayExt<bool> {
        af::isinf(&*self).into()
    }

    fn is_nan(&self) -> ArrayExt<bool> {
        af::isnan(&*self).into()
    }
}

/// Defines element-wise comparison operations.
pub trait ArrayInstanceCompare<U> {
    /// Element-wise equality.
    fn eq(&self, other: &U) -> ArrayExt<bool>;

    /// Element-wise greater-than comparison.
    fn gt(&self, other: &U) -> ArrayExt<bool>;

    /// Element-wise greater-or-equal comparison
    fn gte(&self, other: &U) -> ArrayExt<bool>;

    /// Element-wise less-than comparison.
    fn lt(&self, other: &U) -> ArrayExt<bool>;

    /// Element-wise less-or-equal comparison.
    fn lte(&self, other: &U) -> ArrayExt<bool>;

    /// Element-wise inequality.
    fn ne(&self, other: &U) -> ArrayExt<bool>;
}

impl<T, U> ArrayInstanceCompare<U> for ArrayExt<T>
where
    T: Clone + af::HasAfEnum + af::Convertable<OutType = T>,
    U: af::Convertable<OutType = T>,
    <T as af::Convertable>::OutType: af::ImplicitPromote<<U as af::Convertable>::OutType>,
    <U as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
{
    fn eq(&self, other: &U) -> ArrayExt<bool> {
        af::eq(self.deref(), other, true).into()
    }

    fn gt(&self, other: &U) -> ArrayExt<bool> {
        af::gt(self.deref(), other, true).into()
    }

    fn gte(&self, other: &U) -> ArrayExt<bool> {
        af::ge(self.deref(), other, true).into()
    }

    fn lt(&self, other: &U) -> ArrayExt<bool> {
        af::lt(self.deref(), other, true).into()
    }

    fn lte(&self, other: &U) -> ArrayExt<bool> {
        af::le(self.deref(), other, true).into()
    }

    fn ne(&self, other: &U) -> ArrayExt<bool> {
        self.eq(other).not()
    }
}

/// Defines the natural logarithm.
pub trait ArrayInstanceNaturalLog<T>: ArrayInstance
where
    T: af::HasAfEnum,
{
    /// Calculate the element-wise natural logarithm.
    fn ln(&self) -> ArrayExt<T::UnaryOutType>;
}

impl<T> ArrayInstanceNaturalLog<T> for ArrayExt<T>
where
    T: af::HasAfEnum,
{
    fn ln(&self) -> ArrayExt<T::UnaryOutType> {
        af::log(self).into()
    }
}

/// Defines a general logarithm.
pub trait ArrayInstanceLog<T, U>: ArrayInstance
where
    T: af::HasAfEnum,
    U: af::HasAfEnum,
    ArrayExt<T>: ArrayInstanceNaturalLog<T>,
    ArrayExt<U>: ArrayInstanceNaturalLog<U>,
    ArrayExt<T::UnaryOutType>: Div<ArrayExt<U::UnaryOutType>>,
{
    /// Calculate the element-wise logarithm.
    fn log(
        &self,
        base: &ArrayExt<U>,
    ) -> <ArrayExt<T::UnaryOutType> as Div<ArrayExt<U::UnaryOutType>>>::Output;
}

impl<T, U> ArrayInstanceLog<T, U> for ArrayExt<T>
where
    T: af::HasAfEnum,
    U: af::HasAfEnum,
    Self: ArrayInstanceNaturalLog<T>,
    ArrayExt<U>: ArrayInstanceNaturalLog<U>,
    ArrayExt<T::UnaryOutType>: Div<ArrayExt<U::UnaryOutType>>,
{
    fn log(
        &self,
        base: &ArrayExt<U>,
    ) -> <ArrayExt<T::UnaryOutType> as Div<ArrayExt<U::UnaryOutType>>>::Output {
        self.ln() / base.ln()
    }
}

/// Defines an exponentiation method `pow`.
pub trait ArrayInstancePow<U>: ArrayInstance {
    type Pow: af::HasAfEnum;

    /// Calculate the element-wise exponentiation.
    fn pow(&self, other: &U) -> ArrayExt<Self::Pow>;
}

impl<T, U> ArrayInstancePow<U> for ArrayExt<T>
where
    T: af::HasAfEnum + af::Convertable<OutType = T>,
    U: af::Convertable<OutType = T>,
    <T as af::Convertable>::OutType: af::ImplicitPromote<T> + af::ImplicitPromote<<U as af::Convertable>::OutType>,
    <U as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
    <<T as af::Convertable>::OutType as af::ImplicitPromote<<U as af::Convertable>::OutType>>::Output: af::HasAfEnum,
{
    type Pow = <<T as af::Convertable>::OutType as af::ImplicitPromote<<U as af::Convertable>::OutType>>::Output;

    fn pow(&self, other: &U) -> ArrayExt<Self::Pow> {
        ArrayExt(af::pow(self.deref(), other, true))
    }
}

/// Defines common reduction operation `sum`.
pub trait ArrayInstanceSum<T>: ArrayInstance
where
    T: af::HasAfEnum,
    T::AggregateOutType: DType,
{
    type Sum: af::HasAfEnum;

    /// Calculate the cumulative sum.
    fn sum(&self) -> T::AggregateOutType;

    /// The `NumberType` of the sum of this array.
    fn sum_dtype() -> NumberType {
        T::AggregateOutType::dtype()
    }
}

/// Defines common reduction operation `product`.
pub trait ArrayInstanceProduct<T>: ArrayInstance
where
    T: af::HasAfEnum,
    T::ProductOutType: DType,
{
    type Product: af::HasAfEnum;

    /// Calculate the cumulative product.
    fn product(&self) -> T::ProductOutType;

    /// The `NumberType` of the product of this array.
    fn product_dtype() -> NumberType {
        T::ProductOutType::dtype()
    }
}

/// Defines common reduction operations `min` and `max`.
pub trait ArrayInstanceMinMax<T>: ArrayInstance
where
    T: af::HasAfEnum,
{
    /// Find the maximum element.
    fn max(&self) -> T;

    /// Find the minimum element.
    fn min(&self) -> T;
}

macro_rules! reduce_real {
    ($t:ty) => {
        impl ArrayInstanceSum<$t> for ArrayExt<$t> {
            type Sum = <$t as af::HasAfEnum>::AggregateOutType;

            fn sum(&self) -> Self::Sum {
                af::sum_all(self).0
            }
        }

        impl ArrayInstanceProduct<$t> for ArrayExt<$t> {
            type Product = <$t as af::HasAfEnum>::ProductOutType;

            fn product(&self) -> Self::Product {
                af::product_all(self).0
            }
        }

        impl ArrayInstanceMinMax<$t> for ArrayExt<$t> {
            fn max(&self) -> $t {
                af::max_all(self).0
            }

            fn min(&self) -> $t {
                af::min_all(self).0
            }
        }
    };
}

reduce_real!(bool);
reduce_real!(u8);
reduce_real!(u16);
reduce_real!(u32);
reduce_real!(u64);
reduce_real!(i16);
reduce_real!(i32);
reduce_real!(i64);
reduce_real!(f32);
reduce_real!(f64);

macro_rules! reduce_complex {
    ($t:ty) => {
        impl ArrayInstanceSum<Complex<$t>> for ArrayExt<Complex<$t>> {
            type Sum = Complex<$t>;

            fn sum(&self) -> Self::Sum {
                let sum = af::sum_all(self);
                Complex::new(sum.0, sum.1)
            }
        }

        impl ArrayInstanceProduct<Complex<$t>> for ArrayExt<Complex<$t>> {
            type Product = Complex<$t>;

            fn product(&self) -> Self::Product {
                let product = af::product_all(self);
                Complex::new(product.0, product.1)
            }
        }

        impl ArrayInstanceMinMax<Complex<$t>> for ArrayExt<Complex<$t>> {
            fn max(&self) -> Complex<$t> {
                let max = af::min_all(self);
                Complex::new(max.0, max.1)
            }

            fn min(&self) -> Complex<$t> {
                let min = af::min_all(self);
                Complex::new(min.0, min.1)
            }
        }
    };
}

reduce_complex!(f32);
reduce_complex!(f64);

macro_rules! unary {
    ($t:ty, $f:ident) => {
        fn $f(&self) -> ArrayExt<<$t>::UnaryOutType> {
            af::$f(self).into()
        }
    };
}

pub trait ArrayInstanceTrig<T>: ArrayInstance<DType = T>
where
    T: af::HasAfEnum,
{
    unary!(T, sin);
    unary!(T, asin);
    unary!(T, sinh);
    unary!(T, asinh);

    unary!(T, cos);
    unary!(T, acos);
    unary!(T, cosh);
    unary!(T, acosh);

    unary!(T, tan);
    unary!(T, atan);
    unary!(T, tanh);
    unary!(T, atanh);
}

impl<T: af::HasAfEnum + Default> ArrayInstanceTrig<T> for ArrayExt<T> {}

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
            self.dims(),
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

#[inline]
fn batch<T: af::HasAfEnum>(this: &ArrayExt<T>, that: &ArrayExt<T>) -> bool {
    this.0.dims() != that.0.dims()
}
