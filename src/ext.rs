use std::fmt;
use std::iter::FromIterator;
use std::ops::*;

use arrayfire as af;
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

use super::{_Complex, dim4};

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
    /// Concatenate two instances of `ArrayExt<T>`.
    pub fn concatenate(left: &ArrayExt<T>, right: &ArrayExt<T>) -> ArrayExt<T> {
        af::join(0, left.af(), right.af()).into()
    }

    /// Cast the values of this `ArrayExt` into a destination type `D`.
    pub fn cast_to<D: af::HasAfEnum>(&self) -> ArrayExt<D> {
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

    /// Sort this `ArrayExt`.
    pub fn sort(&mut self, ascending: bool)
    where
        T: af::RealNumber,
    {
        *self = Self(af::sort(&self.0, 0, ascending))
    }

    /// Split this `ArrayExt<T>` into two new instances at the given pivot.
    pub fn split(&self, at: usize) -> (ArrayExt<T>, ArrayExt<T>) {
        let left = af::Seq::new(0.0, at as f32, 1.0);
        let right = af::Seq::new(at as f32, self.len() as f32, 1.0);
        (
            ArrayExt(af::index(self.af(), &[left, af::Seq::default()])),
            ArrayExt(af::index(self.af(), &[right, af::Seq::default()])),
        )
    }
}

impl ArrayExt<bool> {
    pub fn not(&self) -> Self {
        ArrayExt(!self.af())
    }

    pub fn and(&self, other: &Self) -> Self {
        ArrayExt(af::and(self.af(), other.af(), BATCH))
    }

    pub fn or(&self, other: &Self) -> Self {
        ArrayExt(af::or(self.af(), other.af(), BATCH))
    }

    pub fn xor(&self, other: &Self) -> Self {
        let one = af::or(self.af(), other.af(), BATCH);
        let not_both = !(&af::and(self.af(), other.af(), BATCH));
        let one_and_not_both = af::and(&one, &not_both, BATCH);
        ArrayExt(one_and_not_both)
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
        *self = sum.cast_to();
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
        *self = diff.cast_to();
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
        *self = product.cast_to();
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

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> DivAssign for ArrayExt<T>
    where
        <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum + Default,
        <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
        <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    fn div_assign(&mut self, other: Self) {
        let div = &*self / &other;
        *self = div.cast_to();
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

impl ArrayInstanceAnyAll for ArrayExt<_Complex<f32>> {
    fn all(&self) -> bool {
        af::all_true_all(&af::abs(self.af())).0
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.af());
        any.0 || any.1
    }
}

impl ArrayInstanceAnyAll for ArrayExt<_Complex<f64>> {
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

impl ArrayInstanceReduce for ArrayExt<_Complex<f32>> {
    type Product = _Complex<f32>;
    type Sum = _Complex<f32>;

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

impl<'de, T: af::HasAfEnum + Deserialize<'de>> Deserialize<'de> for ArrayExt<T>
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

impl<T: af::HasAfEnum> fmt::Debug for ArrayExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<T: af::HasAfEnum> fmt::Display for ArrayExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ArrayExt<{}>", std::any::type_name::<T>())
    }
}

fn coord_bounds(shape: &[u64]) -> Vec<u64> {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}
