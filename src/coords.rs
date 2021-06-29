use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt, TryStream, TryStreamExt};

use super::{coord_bounds, ArrayExt, ArrayInstance};

/// An n-dimensional coordinate.
pub type Coord = Vec<u64>;

/// One-dimensional array indices corresponding to n-dimensional coordinates.
pub type Offsets = ArrayExt<u64>;

/// A hardware-accelerated set of n-dimensional coordinates, all with the same dimension.
pub struct Coords(af::Array<u64>);

impl Coords {
    /// Constructs a new `Coords` from an [`ArrayExt`] of offsets with respect to the given shape.
    pub fn from_offsets(offsets: Offsets, shape: &[u64]) -> Self {
        let ndim = shape.len() as u64;
        let coord_bounds = coord_bounds(shape);

        let af_coord_bounds: af::Array<u64> =
            af::Array::new(&coord_bounds, af::Dim4::new(&[1, ndim, 1, 1]));
        let af_shape: af::Array<u64> =
            af::Array::new(&shape, af::Dim4::new(&[1, ndim, 1, 1]));

        let offsets = af::div(offsets.af(), &af_coord_bounds, true);
        let coords = af::modulo(&offsets, &af_shape, true);
        Self(af::transpose(&coords, false))
    }

    /// Constructs a new `Coords` from a [`Stream`] of [`Coord`]s.
    ///
    /// Panics: if any [`Coord`] has a length other than `ndim`.
    pub async fn from_stream<S: Stream<Item = Coord> + Unpin>(
        mut source: S,
        ndim: usize,
        size_hint: Option<usize>,
    ) -> Self {
        let mut num_coords = 0;
        let mut buffer = if let Some(size) = size_hint {
            Vec::with_capacity(size)
        } else {
            Vec::new()
        };

        while let Some(coord) = source.next().await {
            assert_eq!(coord.len(), ndim);
            buffer.extend(coord);
            num_coords += 1;
        }

        Self(af::Array::new(
            &buffer,
            af::Dim4::new(&[ndim as u64, num_coords, 1, 1]),
        ))
    }

    /// Constructs a new `Coords` from a [`TryStream`] of `Coord`s.
    ///
    /// Panics: if any `Coord` has a length other than `ndim`.
    pub async fn try_from_stream<E, S: TryStream<Ok = Coord, Error = E> + Unpin>(
        mut source: S,
        ndim: usize,
        size_hint: Option<usize>,
    ) -> Result<Self, E> {
        let mut num_coords = 0;
        let mut buffer = if let Some(size) = size_hint {
            Vec::with_capacity(size)
        } else {
            Vec::new()
        };

        while let Some(coord) = source.try_next().await? {
            assert_eq!(coord.len(), ndim);
            buffer.extend(coord);
            num_coords += 1;
        }

        Ok(Self(af::Array::new(
            &buffer,
            af::Dim4::new(&[ndim as u64, num_coords, 1, 1]),
        )))
    }

    /// Return a list of [`Coord`]s from these `Coords`.
    ///
    /// Panics: if the given number of dimensions does not fit the set of coordinates.
    pub fn to_vec(&self, ndim: usize) -> Vec<Vec<u64>> {
        assert_eq!(self.0.elements() % ndim, 0);
        let mut to_vec = vec![0u64; self.0.elements()];
        self.0.host(&mut to_vec);
        to_vec.chunks(ndim).map(|coord| coord.to_vec()).collect()
    }

    /// Convert these `Coords` into a list of [`Coord`]s.
    ///
    /// Panics: if the given number of dimensions does not fit the set of coordinates.
    pub fn into_vec(self, ndim: usize) -> Vec<Vec<u64>> {
        self.to_vec(ndim)
    }
}

pub struct CoordBlocks<S> {
    source: Fuse<S>,
    ndim: usize,
    block_size: usize,
    buffer: Vec<u64>,
}

impl<E, S: Stream<Item = Result<Coord, E>>> CoordBlocks<S> {
    pub fn new(source: S, ndim: usize, block_size: usize) -> Self {
        Self {
            source: source.fuse(),
            ndim,
            block_size,
            buffer: Vec::with_capacity(ndim * block_size),
        }
    }

    fn consume_buffer(&mut self) -> Coords {
        let ndim = self.ndim as u64;
        let num_coords = (self.buffer.len() / self.ndim) as u64;
        let dims = af::Dim4::new(&[ndim, num_coords, 1, 1]);
        self.buffer.clear();
        Coords(af::Array::new(&self.buffer, dims))
    }
}

impl<E, S: Stream<Item = Result<Coord, E>> + Unpin> Stream for CoordBlocks<S> {
    type Item = Result<Coords, E>;

    fn poll_next(mut self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(loop {
            match ready!(Pin::new(&mut self.source).poll_next(cxt)) {
                Some(Ok(coord)) => {
                    assert_eq!(coord.len(), self.ndim);
                    self.buffer.extend(coord);

                    if self.buffer.len() == self.block_size {
                        break Some(Ok(self.consume_buffer()));
                    }
                }
                Some(Err(cause)) => break Some(Err(cause)),
                None if self.buffer.is_empty() => break None,
                None => break Some(Ok(self.consume_buffer())),
            }
        })
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

    #[test]
    fn test_to_coords() {
        let offsets = ArrayExt::range(0, 5);
        let coords = Coords::from_offsets(offsets, &[5, 2]);
        assert_eq!(
            coords.into_vec(2),
            vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1], vec![2, 0],]
        )
    }
}
