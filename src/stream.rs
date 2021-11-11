use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt, TryStreamExt};
use number_general::*;
use pin_project::pin_project;

use crate::{
    Array, ArrayExt, ArrayInstance, ArrayInstanceReduce, Complex, HasArrayExt, Product, Sum,
};

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

    match dtype {
        NT::Bool => reduce_product_inner::<bool, E, S>(blocks, block_size, stride),
        NT::UInt(ut) => match ut {
            UT::U8 => reduce_product_inner::<u8, E, S>(blocks, block_size, stride),
            UT::U16 => reduce_product_inner::<u16, E, S>(blocks, block_size, stride),
            UT::U32 => reduce_product_inner::<u32, E, S>(blocks, block_size, stride),
            UT::U64 => reduce_product_inner::<u64, E, S>(blocks, block_size, stride),
            UT::UInt => reduce_product_inner::<u64, E, S>(blocks, block_size, stride),
        },
        NT::Int(it) => match it {
            IT::I8 => reduce_product_inner::<i16, E, S>(blocks, block_size, stride),
            IT::I16 => reduce_product_inner::<i16, E, S>(blocks, block_size, stride),
            IT::I32 => reduce_product_inner::<i32, E, S>(blocks, block_size, stride),
            IT::I64 => reduce_product_inner::<i64, E, S>(blocks, block_size, stride),
            IT::Int => reduce_product_inner::<i64, E, S>(blocks, block_size, stride),
        },
        NT::Float(ft) => match ft {
            FT::F32 => reduce_product_inner::<f32, E, S>(blocks, block_size, stride),
            FT::F64 => reduce_product_inner::<f64, E, S>(blocks, block_size, stride),
            FT::Float => reduce_product_inner::<f64, E, S>(blocks, block_size, stride),
        },
        NT::Complex(ct) => match ct {
            CT::C32 => reduce_product_inner::<Complex<f32>, E, S>(blocks, block_size, stride),
            CT::C64 => reduce_product_inner::<Complex<f64>, E, S>(blocks, block_size, stride),
            CT::Complex => reduce_product_inner::<Complex<f64>, E, S>(blocks, block_size, stride),
        },
        NT::Number => reduce_product_inner::<f64, E, S>(blocks, block_size, stride),
    }
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
    T::AggregateOutType: DType + Sum + Copy + Default + Send + 'a,
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceReduce<T>,
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

    match dtype {
        NT::Bool => reduce_sum_inner::<bool, E, S>(blocks, block_size, stride),
        NT::UInt(ut) => match ut {
            UT::U8 => reduce_sum_inner::<u8, E, S>(blocks, block_size, stride),
            UT::U16 => reduce_sum_inner::<u16, E, S>(blocks, block_size, stride),
            UT::U32 => reduce_sum_inner::<u32, E, S>(blocks, block_size, stride),
            UT::U64 => reduce_sum_inner::<u64, E, S>(blocks, block_size, stride),
            UT::UInt => reduce_sum_inner::<u64, E, S>(blocks, block_size, stride),
        },
        NT::Int(it) => match it {
            IT::I8 => reduce_sum_inner::<i16, E, S>(blocks, block_size, stride),
            IT::I16 => reduce_sum_inner::<i16, E, S>(blocks, block_size, stride),
            IT::I32 => reduce_sum_inner::<i32, E, S>(blocks, block_size, stride),
            IT::I64 => reduce_sum_inner::<i64, E, S>(blocks, block_size, stride),
            IT::Int => reduce_sum_inner::<i64, E, S>(blocks, block_size, stride),
        },
        NT::Float(ft) => match ft {
            FT::F32 => reduce_sum_inner::<f32, E, S>(blocks, block_size, stride),
            FT::F64 => reduce_sum_inner::<f64, E, S>(blocks, block_size, stride),
            FT::Float => reduce_sum_inner::<f64, E, S>(blocks, block_size, stride),
        },
        NT::Complex(ct) => match ct {
            CT::C32 => reduce_sum_inner::<Complex<f32>, E, S>(blocks, block_size, stride),
            CT::C64 => reduce_sum_inner::<Complex<f64>, E, S>(blocks, block_size, stride),
            CT::Complex => reduce_sum_inner::<Complex<f64>, E, S>(blocks, block_size, stride),
        },
        NT::Number => reduce_sum_inner::<f64, E, S>(blocks, block_size, stride),
    }
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
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceReduce<T>,
    Array: From<ArrayExt<T::AggregateOutType>>,
{
    let blocks = blocks.map_ok(|block| block.type_cast::<T>());
    let reduced = blocks.reduce_sum(block_size, stride);
    Box::new(reduced.map_ok(Array::from))
}

/// Methods for handling a [`Stream`] of [`ArrayExt`]s.
pub trait ArrayTryStream<'a, T, E>
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    T::AggregateOutType: DType + Sum + Copy + Default + Send + 'a,
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    Self: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    ArrayExt<T>: ArrayInstanceReduce<T> + From<af::Array<T>>,
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

    /// Change the block size of a stream of [`ArrayExt`]s.
    fn resize(self, block_size: usize) -> Resize<Self, T> {
        Resize::new(self, block_size)
    }
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
    T::AggregateOutType: DType + Sum + Copy + Default + Send + 'a,
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceReduce<T>,
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

impl<'a, T, E, S> ArrayTryStream<'a, T, E> for S
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    S: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    T::AggregateOutType: DType + Sum + Copy + Default + Send + 'a,
    T::ProductOutType: DType + Product + Copy + Default + Send + 'a,
    ArrayExt<T>: ArrayInstanceReduce<T>,
{
}

fn reduce_small<T, B, E, S, R>(
    blocks: S,
    stride: u64,
    reduce: R,
) -> impl Stream<Item = Result<ArrayExt<B>, E>>
where
    T: af::HasAfEnum,
    B: af::HasAfEnum,
    S: Stream<Item = Result<ArrayExt<T>, E>>,
    R: FnMut(af::Array<T>) -> ArrayExt<B>,
{
    blocks
        .map_ok(move |block| {
            assert_eq!(block.len() as u64 % stride, 0);
            let shape = af::Dim4::new(&[stride, block.len() as u64 / stride, 1, 1]);
            let block = af::moddims(&block, shape);
            block
        })
        .map_ok(reduce)
        .map_ok(move |reduced| {
            let shape = af::Dim4::new(&[reduced.len() as u64, 1, 1, 1]);
            af::moddims(&reduced, shape)
        })
        .map_ok(ArrayExt::from)
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

#[pin_project]
/// Aggregate a [`Stream`] of numbers into a [`Stream`] of [`ArrayExt`]s.
pub struct Aggregate<S, T: af::HasAfEnum> {
    #[pin]
    source: Fuse<S>,
    buffer: Vec<T>,
    block_size: usize,
}

impl<S: Stream, T: af::HasAfEnum> Aggregate<S, T> {
    /// Construct a new [`Aggregate`] of numbers into [`ArrayExt`]s.
    pub fn new(source: S, block_size: usize) -> Self {
        Self {
            source: source.fuse(),
            buffer: Vec::with_capacity(block_size),
            block_size,
        }
    }
}

impl<T, E, S> Stream for Aggregate<S, T>
where
    T: af::HasAfEnum + Default,
    S: Stream<Item = Result<T, E>>,
{
    type Item = Result<ArrayExt<T>, E>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            if this.buffer.len() == *this.block_size {
                let array = ArrayExt::from(this.buffer.as_slice());
                this.buffer.clear();
                break Some(Ok(array));
            }

            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok(n)) => this.buffer.push(n),
                Some(Err(cause)) => break Some(Err(cause)),
                None => {
                    let array = ArrayExt::from(this.buffer.as_slice());
                    this.buffer.clear();
                    break Some(Ok(array));
                }
            }
        })
    }
}

#[pin_project]
/// Struct for the [`ArrayTryStream::resize`] method.
pub struct Resize<S, T: af::HasAfEnum> {
    #[pin]
    source: Fuse<S>,
    buffer: Option<ArrayExt<T>>,
    block_size: usize,
}

impl<S: Stream, T: af::HasAfEnum> Resize<S, T> {
    fn new(source: S, block_size: usize) -> Self {
        Self {
            source: source.fuse(),
            buffer: None,
            block_size,
        }
    }
}

impl<T: af::HasAfEnum + Default, E, S: Stream<Item = Result<ArrayExt<T>, E>> + Unpin> Stream
    for Resize<S, T>
{
    type Item = Result<ArrayExt<T>, E>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            if let Some(buffer) = this.buffer.as_mut() {
                if buffer.len() == *this.block_size {
                    let mut new_buffer = None;
                    mem::swap(this.buffer, &mut new_buffer);
                    break new_buffer.map(Ok);
                } else if buffer.len() > *this.block_size {
                    let (block, buffer) = buffer.split(*this.block_size);
                    *this.buffer = Some(buffer);
                    break Some(Ok(block));
                }
            }

            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok(block)) => {
                    if let Some(buffer) = this.buffer {
                        *this.buffer = Some(ArrayExt::concatenate(buffer, &block));
                    } else {
                        *this.buffer = Some(block)
                    }
                }
                Some(Err(cause)) => break Some(Err(cause)),
                None if this.buffer.is_none() => break None,
                None => {
                    let mut new_buffer = None;
                    mem::swap(this.buffer, &mut new_buffer);
                    break new_buffer.map(Ok);
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let x = ArrayExt::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let x = af::moddims(&*x, af::Dim4::new(&[4, 2, 1, 1]));
        let actual = ArrayExt::<i32>::from(af::sum(&x, 0)).to_vec();
        assert_eq!(actual, vec![10, 26]);
    }
}
