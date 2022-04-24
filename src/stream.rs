use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;

use crate::{ArrayExt, ArrayInstance, HasArrayExt};

/// Methods for handling a [`Stream`] of [`ArrayExt`]s.
pub trait ArrayTryStream<'a, T, E>
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    Self: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
{
    /// Change the block size of a stream of [`ArrayExt`]s.
    fn resize(self, block_size: usize) -> Resize<Self, T> {
        Resize::new(self, block_size)
    }
}

impl<'a, T, E, S> ArrayTryStream<'a, T, E> for S
where
    T: af::HasAfEnum + HasArrayExt + Copy + Default + Send + 'a,
    E: Send + 'a,
    S: Stream<Item = Result<ArrayExt<T>, E>> + Sized + Send + Unpin + 'a,
{
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
    pub(crate) fn new(source: S, block_size: usize) -> Self {
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
