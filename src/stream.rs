use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::stream::{Fuse, Stream};
use futures::{ready, StreamExt};
use pin_project::pin_project;

use crate::{ArrayExt, ArrayInstance};

#[pin_project]
pub struct ResizeArrays<S, T: af::HasAfEnum> {
    #[pin]
    source: Fuse<S>,
    block_size: usize,
    buffer: ArrayExt<T>,
}

impl<S: Stream, T: af::HasAfEnum> ResizeArrays<S, T> {
    pub fn new(source: S, block_size: usize) -> Self {
        Self {
            source: source.fuse(),
            block_size,
            buffer: ArrayExt::from(vec![]),
        }
    }
}

impl<T: af::HasAfEnum + Default, E, S: Stream<Item = Result<ArrayExt<T>, E>> + Unpin> Stream
    for ResizeArrays<S, T>
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
