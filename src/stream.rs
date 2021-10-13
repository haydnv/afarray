use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{self, Fuse, Stream, StreamExt, TryStreamExt};
use pin_project::pin_project;

use crate::{ArrayExt, ArrayInstance, ArrayInstanceReduce};

pub trait ArrayTryStream<'a, T, E>
where
    T: af::HasAfEnum<AggregateOutType = T> + Default + Send + 'a,
    T::BaseType: af::Fromf64,
    E: Send + 'a,
    ArrayExt<T>: ArrayInstanceReduce<Sum = T>,
    Self: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
{
    fn reduce_sum(
        self,
        block_size: usize,
        stride: u64,
    ) -> Box<dyn Stream<Item = Result<ArrayExt<T::AggregateOutType>, E>> + Send + Unpin + 'a> {
        if stride < (block_size / 2) as u64 {
            if block_size as u64 % stride == 0 {
                let per_block = block_size as u64 / stride;
                debug_assert_eq!(per_block % stride, 0);
                let reduced = reduce_small(self, per_block, stride);
                Box::new(reduced.resize(block_size))
            } else {
                let reduce_block_size = block_size - (block_size % stride as usize);
                let blocks = self.resize(reduce_block_size);
                let per_block = reduce_block_size as u64 / stride;
                debug_assert_eq!(per_block % stride, 0);
                let reduced = reduce_small(blocks, per_block, stride);
                Box::new(reduced.resize(block_size))
            }
        } else if stride < block_size as u64 {
            let reduced = reduce_medium(self.resize(stride as usize));
            Box::new(Aggregate::new(reduced, block_size))
        } else if stride == block_size as u64 {
            let reduced = reduce_medium(self);
            Box::new(Aggregate::new(reduced, block_size))
        } else {
            let reduced = reduce_large(self, stride);
            Box::new(Aggregate::new(reduced, block_size))
        }
    }

    fn resize(self, block_size: usize) -> Resize<Self, T> {
        Resize::new(self, block_size)
    }
}

impl<'a, T, E, S> ArrayTryStream<'a, T, E> for S
where
    T: af::HasAfEnum<AggregateOutType = T> + Default + Send + 'a,
    T::BaseType: af::Fromf64,
    E: Send + 'a,
    S: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
    ArrayExt<T>: ArrayInstanceReduce<Sum = T>,
{
}

fn reduce_small<T: af::HasAfEnum, E, S: Stream<Item = Result<ArrayExt<T>, E>>>(
    blocks: S,
    per_block: u64,
    stride: u64,
) -> impl Stream<Item = Result<ArrayExt<T::AggregateOutType>, E>> {
    let shape = af::Dim4::new(&[per_block, stride, 0, 0]);
    blocks
        .map_ok(move |block| af::moddims(&block, shape.clone()))
        .map_ok(|block| af::sum(&block, 1))
        .map_ok(ArrayExt::from)
}

fn reduce_medium<T: af::HasAfEnum, E, S: Stream<Item = Result<ArrayExt<T>, E>>>(
    blocks: S,
) -> impl Stream<Item = Result<T, E>>
where
    T: af::HasAfEnum<AggregateOutType = T>,
    T::BaseType: af::Fromf64,
    S: Stream<Item = Result<ArrayExt<T>, E>>,
    ArrayExt<T>: ArrayInstanceReduce<Sum = T>,
{
    blocks.map_ok(|block| ArrayInstanceReduce::sum(&block))
}

fn reduce_large<T: af::HasAfEnum, E, S: Stream<Item = Result<ArrayExt<T>, E>>>(
    _blocks: S,
    _stride: u64,
) -> impl Stream<Item = Result<T::AggregateOutType, E>> {
    stream::empty() // TODO
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

fn clear_buffer<T: af::HasAfEnum>(buffer: &mut ArrayExt<T>) -> ArrayExt<T> {
    let mut block = ArrayExt::from(vec![]);
    mem::swap(buffer, &mut block);
    block
}
