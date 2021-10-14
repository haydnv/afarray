use std::mem;
use std::ops::{Add, Mul};
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt, TryStreamExt};
use pin_project::pin_project;

use crate::{ArrayExt, ArrayInstance, ArrayInstanceReduce, HasArrayExt};

pub trait ArrayTryStream<'a, T, E>
where
    T: af::HasAfEnum<AggregateOutType = T, ProductOutType = T>
        + HasArrayExt
        + Add<Output = T>
        + Mul<Output = T>
        + Copy
        + Default
        + Send
        + 'a,
    E: Send + 'a,
    Self: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    ArrayExt<T>: ArrayInstanceReduce<Sum = T, Product = T> + From<af::Array<T>>,
{
    fn reduce_product(
        self,
        block_size: usize,
        stride: u64,
    ) -> Box<dyn Stream<Item = Result<ArrayExt<T>, E>> + Send + Unpin + 'a> {
        reduce(
            self,
            block_size,
            stride,
            T::one(),
            Mul::mul,
            |block| af::product(&block, 1).into(),
            |block| block.product(),
        )
    }

    fn reduce_sum(
        self,
        block_size: usize,
        stride: u64,
    ) -> Box<dyn Stream<Item = Result<ArrayExt<T>, E>> + Send + Unpin + 'a> {
        reduce(
            self,
            block_size,
            stride,
            T::zero(),
            Add::add,
            |block| af::sum(&block, 1).into(),
            |block| block.sum(),
        )
    }

    fn resize(self, block_size: usize) -> Resize<Self, T> {
        Resize::new(self, block_size)
    }
}

fn reduce<'a, T, E, S, RV, RM, RB>(
    blocks: S,
    block_size: usize,
    stride: u64,
    base: T,
    reduce_value: RV,
    reduce_multi: RM,
    reduce_block: RB,
) -> Box<dyn Stream<Item = Result<ArrayExt<T>, E>> + Send + Unpin + 'a>
where
    T: af::HasAfEnum<AggregateOutType = T, ProductOutType = T>
        + Add<Output = T>
        + Mul<Output = T>
        + HasArrayExt
        + Copy
        + Default
        + Send
        + 'a,
    E: Send + 'a,
    S: ArrayTryStream<'a, T, E>,
    RV: Fn(T, T) -> T + Send + 'a,
    RM: Fn(af::Array<T>) -> ArrayExt<T> + Send + 'a,
    RB: Fn(ArrayExt<T>) -> T + Send + 'a,
    ArrayExt<T>: ArrayInstanceReduce<Sum = T, Product = T>,
{
    if stride < (block_size / 2) as u64 {
        if block_size as u64 % stride == 0 {
            let per_block = block_size as u64 / stride;
            debug_assert_eq!(per_block % stride, 0);
            let reduced = reduce_small(blocks, per_block, stride, reduce_multi);
            Box::new(reduced.resize(block_size))
        } else {
            let reduce_block_size = block_size - (block_size % stride as usize);
            let blocks = blocks.resize(reduce_block_size);
            let per_block = reduce_block_size as u64 / stride;
            debug_assert_eq!(per_block % stride, 0);
            let reduced = reduce_small(blocks, per_block, stride, reduce_multi);
            Box::new(reduced.resize(block_size))
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
    T: af::HasAfEnum<AggregateOutType = T, ProductOutType = T>
        + HasArrayExt
        + Add<Output = T>
        + Mul<Output = T>
        + Copy
        + Default
        + Send
        + 'a,
    E: Send + 'a,
    S: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    ArrayExt<T>: ArrayInstanceReduce<Sum = T, Product = T>,
{
}

fn reduce_small<T, E, S, R>(
    blocks: S,
    per_block: u64,
    stride: u64,
    reduce: R,
) -> impl Stream<Item = Result<ArrayExt<T>, E>>
where
    T: af::HasAfEnum,
    S: Stream<Item = Result<ArrayExt<T>, E>>,
    R: FnMut(af::Array<T>) -> ArrayExt<T>,
{
    let shape = af::Dim4::new(&[per_block, stride, 0, 0]);

    blocks
        .map_ok(move |block| af::moddims(&block, shape.clone()))
        .map_ok(reduce)
        .map_ok(ArrayExt::from)
}

#[pin_project]
struct ReduceLarge<S, T: af::HasAfEnum, RB, RV> {
    #[pin]
    source: Fuse<S>,
    reduced: T,
    reduced_size: u64,
    reduce_block: RB,
    reduce_value: RV,
    base: T,
    stride: u64,
}

impl<S: Stream, T: af::HasAfEnum + HasArrayExt, RB, RV> ReduceLarge<S, T, RB, RV> {
    fn new(source: S, stride: u64, base: T, reduce_block: RB, reduce_value: RV) -> Self {
        Self {
            source: source.fuse(),
            reduced: T::zero(),
            reduced_size: 0,
            base,
            reduce_block,
            reduce_value,
            stride,
        }
    }
}

impl<T, E, S, RB, RV> Stream for ReduceLarge<S, T, RB, RV>
where
    T: af::HasAfEnum + Copy + Default,
    S: Stream<Item = Result<ArrayExt<T>, E>>,
    RB: Fn(ArrayExt<T>) -> T,
    RV: Fn(T, T) -> T,
{
    type Item = Result<T, E>;

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
pub struct Aggregate<S, T: af::HasAfEnum> {
    #[pin]
    source: Fuse<S>,
    buffer: Vec<T>,
    block_size: usize,
}

impl<S: Stream, T: af::HasAfEnum> Aggregate<S, T> {
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
pub struct Resize<S, T: af::HasAfEnum> {
    #[pin]
    source: Fuse<S>,
    buffer: ArrayExt<T>,
    block_size: usize,
}

impl<S: Stream, T: af::HasAfEnum> Resize<S, T> {
    fn new(source: S, block_size: usize) -> Self {
        Self {
            source: source.fuse(),
            buffer: ArrayExt::from(vec![]),
            block_size,
        }
    }
}

impl<T: af::HasAfEnum + Default, E, S: Stream<Item = Result<ArrayExt<T>, E>> + Unpin> Stream
    for Resize<S, T>
{
    type Item = Result<ArrayExt<T>, E>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        fn clear_buffer<T: af::HasAfEnum>(buffer: &mut ArrayExt<T>) -> ArrayExt<T> {
            let mut block = ArrayExt::from(vec![]);
            mem::swap(buffer, &mut block);
            block
        }

        let mut this = self.project();

        Poll::Ready(loop {
            if this.buffer.len() == *this.block_size {
                break Some(Ok(clear_buffer(this.buffer)));
            } else if this.buffer.len() > *this.block_size {
                let (block, buffer) = this.buffer.split(*this.block_size);
                *this.buffer = buffer;
                break Some(Ok(block));
            }

            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok(block)) => {
                    *this.buffer = ArrayExt::concatenate(&this.buffer, &block);
                }
                Some(Err(cause)) => break Some(Err(cause)),
                None if this.buffer.is_empty() => break None,
                None => break Some(Ok(clear_buffer(this.buffer))),
            }
        })
    }
}
