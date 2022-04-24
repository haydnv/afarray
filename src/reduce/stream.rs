use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt, TryStreamExt};
use number_general::*;
use pin_project::pin_project;

use crate::array::{reduce_block, Array};
use crate::ext::{
    ArrayExt, ArrayInstance, ArrayInstanceMinMax, ArrayInstanceProduct, ArrayInstanceSum,
    HasArrayExt,
};
use crate::stream::{Aggregate, ArrayTryStream, Resize};
use crate::Complex;

use super::{Product, Sum};

/// Trait to reduce the product of a [`Stream`] of [`ArrayExt`]s.
pub trait ArrayProductTryStream<'a, T, E>
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    Self: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    ArrayExt<T>: ArrayInstanceProduct<T> + From<af::Array<T>>,
{
    /// Compute the product of each `stride` of a [`Stream`] of [`ArrayExt`]s.
    fn reduce_product(
        self,
        block_size: usize,
        stride: u64,
    ) -> Box<dyn Stream<Item = Result<ArrayExt<T::ProductOutType>, E>> + Send + Unpin + 'a> {
        reduce(
            self,
            block_size,
            stride,
            T::one(),
            Product::product,
            |block| af::product(&block, 0).into(),
            |block| block.product(),
        )
    }
}

impl<'a, T, E, S> ArrayProductTryStream<'a, T, E> for S
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    S: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceProduct<T>,
{
}

/// Methods for handling a [`Stream`] of [`ArrayExt`]s.
pub trait ArraySumTryStream<'a, T, E>
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    T::AggregateOutType: DType + Sum + Copy + Default + Send + 'a,
    Self: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    ArrayExt<T>: ArrayInstanceSum<T> + From<af::Array<T>>,
{
    /// Compute the sum of each `stride` of a [`Stream`] of [`ArrayExt`]s.
    fn reduce_sum(
        self,
        block_size: usize,
        stride: u64,
    ) -> Box<dyn Stream<Item = Result<ArrayExt<T::AggregateOutType>, E>> + Send + Unpin + 'a> {
        reduce(
            self,
            block_size,
            stride,
            T::zero(),
            Sum::sum,
            |block| af::sum(&block, 0).into(),
            |block| block.sum(),
        )
    }
}

impl<'a, T, E, S> ArraySumTryStream<'a, T, E> for S
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    S: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    T::AggregateOutType: DType + Sum + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceSum<T>,
{
}

macro_rules! dispatch {
    ($call:ident, $dtype:ident, $blocks:ident, $block_size:ident, $stride: ident) => {
        match $dtype {
            NT::Bool => $call::<bool, E, S>($blocks, $block_size, $stride),
            NT::UInt(ut) => match ut {
                UT::U8 => $call::<u8, E, S>($blocks, $block_size, $stride),
                UT::U16 => $call::<u16, E, S>($blocks, $block_size, $stride),
                UT::U32 => $call::<u32, E, S>($blocks, $block_size, $stride),
                UT::U64 => $call::<u64, E, S>($blocks, $block_size, $stride),
                UT::UInt => $call::<u64, E, S>($blocks, $block_size, $stride),
            },
            NT::Int(it) => match it {
                IT::I8 => $call::<i16, E, S>($blocks, $block_size, $stride),
                IT::I16 => $call::<i16, E, S>($blocks, $block_size, $stride),
                IT::I32 => $call::<i32, E, S>($blocks, $block_size, $stride),
                IT::I64 => $call::<i64, E, S>($blocks, $block_size, $stride),
                IT::Int => $call::<i64, E, S>($blocks, $block_size, $stride),
            },
            NT::Float(ft) => match ft {
                FT::F32 => $call::<f32, E, S>($blocks, $block_size, $stride),
                FT::F64 => $call::<f64, E, S>($blocks, $block_size, $stride),
                FT::Float => $call::<f64, E, S>($blocks, $block_size, $stride),
            },
            NT::Complex(ct) => match ct {
                CT::C32 => $call::<Complex<f32>, E, S>($blocks, $block_size, $stride),
                CT::C64 => $call::<Complex<f64>, E, S>($blocks, $block_size, $stride),
                CT::Complex => $call::<Complex<f64>, E, S>($blocks, $block_size, $stride),
            },
            NT::Number => $call::<f64, E, S>($blocks, $block_size, $stride),
        }
    };
}

/// Compute the product of each `stride` of a [`Stream`] of [`Array`]s.
pub fn reduce_product<'a, E, S>(
    blocks: S,
    dtype: NumberType,
    block_size: usize,
    stride: u64,
) -> Box<dyn Stream<Item = Result<Array, E>> + Send + Unpin + 'a>
where
    E: Send + 'a,
    S: Stream<Item = Result<Array, E>> + Send + Unpin + 'a,
{
    use {ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT};

    dispatch!(reduce_product_inner, dtype, blocks, block_size, stride)
}

fn reduce_product_inner<'a, T, E, S>(
    blocks: S,
    block_size: usize,
    stride: u64,
) -> Box<dyn Stream<Item = Result<Array, E>> + Send + Unpin + 'a>
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    S: Stream<Item = Result<Array, E>> + Sized + Send + Unpin + 'a,
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceProduct<T>,
    Array: From<ArrayExt<T::ProductOutType>>,
{
    let blocks = blocks.map_ok(|block| block.type_cast::<T>());
    let reduced = blocks.reduce_product(block_size, stride);
    Box::new(reduced.map_ok(Array::from))
}

/// Compute the sum of each `stride` of a [`Stream`] of [`Array`]s.
pub fn reduce_sum<'a, E, S>(
    blocks: S,
    dtype: NumberType,
    block_size: usize,
    stride: u64,
) -> Box<dyn Stream<Item = Result<Array, E>> + Send + Unpin + 'a>
where
    E: Send + 'a,
    S: Stream<Item = Result<Array, E>> + Send + Unpin + 'a,
{
    use {ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT};

    dispatch!(reduce_sum_inner, dtype, blocks, block_size, stride)
}

fn reduce_sum_inner<'a, T, E, S>(
    blocks: S,
    block_size: usize,
    stride: u64,
) -> Box<dyn Stream<Item = Result<Array, E>> + Send + Unpin + 'a>
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    S: Stream<Item = Result<Array, E>> + Sized + Send + Unpin + 'a,
    T::AggregateOutType: DType + Sum + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceSum<T>,
    Array: From<ArrayExt<T::AggregateOutType>>,
{
    let blocks = blocks.map_ok(|block| block.type_cast::<T>());
    let reduced = blocks.reduce_sum(block_size, stride);
    Box::new(reduced.map_ok(Array::from))
}

fn reduce<'a, T, B, E, S, RV, RM, RB>(
    blocks: S,
    block_size: usize,
    stride: u64,
    base: B,
    reduce_value: RV,
    reduce_multi: RM,
    reduce_block: RB,
) -> Box<dyn Stream<Item = Result<ArrayExt<B>, E>> + Send + Unpin + 'a>
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    B: af::HasAfEnum + Copy + Default + Send + 'a,
    E: Send + 'a,
    S: ArrayTryStream<'a, T, E>,
    RV: Fn(B, B) -> B + Send + 'a,
    RM: Fn(af::Array<T>) -> ArrayExt<B> + Send + 'a,
    RB: Fn(ArrayExt<T>) -> B + Send + 'a,
{
    if stride < (block_size / 2) as u64 {
        if block_size as u64 % stride == 0 {
            let per_block = block_size as u64 / stride;
            debug_assert_eq!(per_block % stride, 0);
            let reduced = reduce_small(blocks, stride, reduce_multi);
            Box::new(Resize::new(reduced, block_size))
        } else {
            let reduce_block_size = block_size - (block_size % stride as usize);
            debug_assert_eq!(reduce_block_size % stride as usize, 0);
            let blocks = blocks.resize(reduce_block_size);
            let reduced = reduce_small(blocks, stride, reduce_multi);
            Box::new(Resize::new(reduced, block_size))
        }
    } else if stride < block_size as u64 {
        let reduced = blocks.resize(stride as usize).map_ok(reduce_block);
        Box::new(Aggregate::new(reduced, block_size))
    } else if stride == block_size as u64 {
        let reduced = blocks.map_ok(reduce_block);
        Box::new(Aggregate::new(reduced, block_size))
    } else {
        let reduced = ReduceLarge::new(blocks, stride, base, reduce_block, reduce_value);
        Box::new(Aggregate::new(reduced, block_size))
    }
}

fn reduce_small<T, B, E, S, R>(
    blocks: S,
    stride: u64,
    mut reduce: R,
) -> impl Stream<Item = Result<ArrayExt<B>, E>>
where
    T: af::HasAfEnum,
    B: af::HasAfEnum,
    S: Stream<Item = Result<ArrayExt<T>, E>>,
    R: FnMut(af::Array<T>) -> ArrayExt<B>,
{
    blocks.map_ok(move |block| reduce_block(&block, stride, &mut reduce))
}

#[pin_project]
struct ReduceLarge<S, B, RB, RV> {
    #[pin]
    source: Fuse<S>,
    reduced: B,
    reduced_size: u64,
    reduce_block: RB,
    reduce_value: RV,
    base: B,
    stride: u64,
}

impl<S: Stream, B: Copy, RB, RV> ReduceLarge<S, B, RB, RV> {
    fn new(source: S, stride: u64, base: B, reduce_block: RB, reduce_value: RV) -> Self {
        Self {
            source: source.fuse(),
            reduced: base,
            reduced_size: 0,
            base,
            reduce_block,
            reduce_value,
            stride,
        }
    }
}

impl<T, E, S, B, RB, RV> Stream for ReduceLarge<S, B, RB, RV>
where
    T: af::HasAfEnum + Copy + Default,
    B: Copy,
    S: Stream<Item = Result<ArrayExt<T>, E>>,
    RB: Fn(ArrayExt<T>) -> B,
    RV: Fn(B, B) -> B,
{
    type Item = Result<B, E>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        let rb = this.reduce_block;
        let rv = this.reduce_value;

        Poll::Ready(loop {
            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok(block)) if *this.reduced_size + (block.len() as u64) < *this.stride => {
                    *this.reduced_size += block.len() as u64;
                    *this.reduced = rv(*this.reduced, rb(block));
                }
                Some(Ok(block)) if *this.reduced_size + (block.len() as u64) > *this.stride => {
                    let (l, r) = block.split((*this.stride - *this.reduced_size) as usize);
                    let reduced = rv(*this.reduced, rb(l));
                    *this.reduced_size = r.len() as u64;
                    *this.reduced = rb(r);
                    break Some(Ok(reduced));
                }
                Some(Ok(block)) => {
                    debug_assert_eq!(*this.reduced_size + (block.len() as u64), *this.stride);
                    let reduced = rv(*this.reduced, rb(block));
                    *this.reduced = *this.base;
                    *this.reduced_size = 0;
                    break Some(Ok(reduced));
                }
                Some(Err(cause)) => break Some(Err(cause)),
                None => {
                    let reduced = *this.reduced;
                    *this.reduced = *this.base;
                    *this.reduced_size = 0;
                    break Some(Ok(reduced));
                }
            }
        })
    }
}
