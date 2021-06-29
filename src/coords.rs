use std::iter::IntoIterator;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt, TryStream, TryStreamExt};

use super::{coord_bounds, dim4, ArrayExt, ArrayInstance};

/// An n-dimensional coordinate.
pub type Coord = Vec<u64>;

/// One-dimensional array indices corresponding to n-dimensional coordinates.
pub type Offsets = ArrayExt<u64>;

/// A hardware-accelerated set of n-dimensional coordinates, all with the same dimension.
#[derive(Clone)]
pub struct Coords {
    array: af::Array<u64>,
    ndim: usize,
}

impl Coords {
    /// Constructs a new `Coords` from an iterator of [`Coord`]s.
    ///
    /// Panics: if any [`Coord`] is not of length `ndim`.
    pub fn from_iter<I: IntoIterator<Item = Coord>>(iter: I, ndim: usize) -> Self {
        let buffer: Vec<u64> = iter
            .into_iter()
            .inspect(|coord| assert_eq!(coord.len(), ndim))
            .flatten()
            .collect();

        let num_coords = buffer.len() / ndim;
        let dims = af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]);
        let array = af::Array::new(&buffer, dims);
        Self { array, ndim }
    }

    /// Constructs a new `Coords` from an [`ArrayExt`] of offsets with respect to the given shape.
    pub fn from_offsets(offsets: Offsets, shape: &[u64]) -> Self {
        let ndim = shape.len() as u64;
        let coord_bounds = coord_bounds(shape);

        let dims = af::Dim4::new(&[1, ndim, 1, 1]);
        let af_coord_bounds: af::Array<u64> = af::Array::new(&coord_bounds, dims);
        let af_shape: af::Array<u64> = af::Array::new(&shape, dims);

        let offsets = af::div(offsets.af(), &af_coord_bounds, true);
        let coords = af::modulo(&offsets, &af_shape, true);
        let array = af::transpose(&coords, false);

        Self {
            array,
            ndim: shape.len(),
        }
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

        let array = af::Array::new(&buffer, af::Dim4::new(&[ndim as u64, num_coords, 1, 1]));

        Self { array, ndim }
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

        let array = af::Array::new(&buffer, af::Dim4::new(&[ndim as u64, num_coords, 1, 1]));

        Ok(Self { array, ndim })
    }

    /// Return the number of coordinates stored in these `Coords`.
    pub fn len(&self) -> usize {
        self.array.elements() as usize / self.ndim
    }

    /// Return the number of dimensions of these `Coords`.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Return these `Coords` as [`Offsets`] with respect to the given shape.
    ///
    /// Panics: if `shape.len()` does not equal `self.ndim()`
    pub fn to_offsets(&self, shape: &[u64]) -> ArrayExt<u64> {
        let ndim = shape.len();
        assert_eq!(self.ndim, ndim);

        let coord_bounds = coord_bounds(shape);
        let af_coord_bounds: af::Array<u64> = af::Array::new(&coord_bounds, dim4(ndim));

        let offsets = af::mul(&self.array, &af_coord_bounds, true);
        let offsets = af::sum(&offsets, 0).into();
        af::moddims(&offsets, dim4(offsets.elements())).into()
    }

    /// Return a list of [`Coord`]s from these `Coords`.
    ///
    /// Panics: if the given number of dimensions does not fit the set of coordinates.
    pub fn to_vec(&self) -> Vec<Vec<u64>> {
        assert_eq!(self.array.elements() % self.ndim, 0);
        let mut to_vec = vec![0u64; self.array.elements()];
        self.array.host(&mut to_vec);
        to_vec
            .chunks(self.ndim)
            .map(|coord| coord.to_vec())
            .collect()
    }

    /// Convert these `Coords` into a list of [`Coord`]s.
    ///
    /// Panics: if the given number of dimensions does not fit the set of coordinates.
    pub fn into_vec(self) -> Vec<Vec<u64>> {
        self.to_vec()
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
        assert_eq!(self.buffer.len() % self.ndim, 0);

        let ndim = self.ndim as u64;
        let num_coords = (self.buffer.len() / self.ndim) as u64;
        let dims = af::Dim4::new(&[ndim, num_coords, 1, 1]);
        let coords = Coords {
            array: af::Array::new(&self.buffer, dims),
            ndim: self.ndim,
        };
        self.buffer.clear();
        coords
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

                    if self.buffer.len() == (self.block_size * self.ndim) {
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
    fn test_to_coords() {
        let offsets = ArrayExt::range(0, 5);
        let coords = Coords::from_offsets(offsets, &[5, 2]);
        assert_eq!(
            coords.into_vec(),
            vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1], vec![2, 0],]
        )
    }
}
