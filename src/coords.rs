use std::collections::HashMap;
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
    /// Constructs `Coords` with the given `size` full of zeros (origin points) for the given shape.
    pub fn empty(shape: &[u64], size: usize) -> Self {
        let ndim = shape.len();
        let dims = af::Dim4::new(&[ndim as u64, size as u64, 1, 1]);
        let array = af::constant(0u64, dims);
        Self { array, ndim }
    }

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
    /// Panics: if any [`Coord`] has a length other than `ndim`
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
    /// Panics: if any [`Coord`] has a length other than `ndim`
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

    /// Borrow these `Coords` as an `af::Array<u64>`.
    #[inline]
    pub fn af(&self) -> &af::Array<u64> {
        &self.array
    }

    /// Borrow these `Coords` mutably as an `af::Array<u64>`.
    #[inline]
    pub fn af_mut(&mut self) -> &mut af::Array<u64> {
        &mut self.array
    }

    /// Return the number of coordinates stored in these `Coords`.
    pub fn len(&self) -> usize {
        self.array.elements() as usize / self.ndim
    }

    /// Return the number of dimensions of these `Coords`.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Return a copy of these `Coords` without the specified axis.
    pub fn contract_dim(&self, axis: usize) -> Self {
        let mut index: Vec<u64> = (0..self.ndim as u64).collect();
        index.remove(axis);
        let index = af::Array::new(&index, dim4(index.len()));

        let seq4gen = af::Seq::new(0., (self.len() - 1) as f64, 1.);

        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
        indexer.set_index(&seq4gen, 1, Some(true));
        self.get(indexer)
    }

    /// Transpose these `Coords` according to the given `permutation`.
    ///
    /// If no permutation is given, the coordinate axes will be inverted.
    pub fn transpose(&self, permutation: Option<&[u64]>) -> Coords {
        if let Some(permutation) = permutation {
            let index = af::Array::new(permutation, dim4(permutation.len()));
            let seq4gen = af::Seq::new(0., (self.len() - 1) as f64, 1.);

            let mut indexer = af::Indexer::default();
            indexer.set_index(&index, 0, None);
            indexer.set_index(&seq4gen, 1, Some(true));
            self.get(indexer)
        } else {
            let array = af::transpose(&self.array, false);
            let ndim = self.ndim;
            Self { array, ndim }
        }
    }

    /// Invert the given broadcast of these `Coords`.
    ///
    /// Panics: if `source_shape` and `broadcast` are not the same length.
    pub fn unbroadcast(&self, source_shape: &[u64], broadcast: &[bool]) -> Coords {
        assert_eq!(self.ndim(), broadcast.len());

        let offset = self.ndim() - source_shape.len();
        let mut coords = Self::empty(source_shape, self.len());
        if source_shape.is_empty() || broadcast.iter().all(|b| *b) {
            return coords;
        }

        let axes: Vec<u64> = broadcast
            .iter()
            .enumerate()
            .filter_map(|(x, b)| if *b { None } else { Some(x as u64) })
            .collect();

        let index = af::Array::new(&axes, dim4(axes.len()));
        let seq4gen = af::Seq::new(0., (self.len() - 1) as f64, 1.);

        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
        indexer.set_index(&seq4gen, 1, Some(true));
        let unbroadcasted = self.get(indexer);

        let axes: Vec<u64> = broadcast
            .iter()
            .enumerate()
            .filter_map(|(x, b)| if *b { None } else { Some((x - offset) as u64) })
            .collect();

        let index = af::Array::new(&axes, dim4(axes.len()));
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
        indexer.set_index(&seq4gen, 1, Some(true));
        coords.set(&indexer, unbroadcasted);

        coords
    }

    /// Transform the coordinate basis of these `Coords` from a slice to a source tensor.
    ///
    /// Panics: if `source_shape.len() - self.ndim()` does not match `elided.len()`
    pub fn unslice(
        &self,
        source_shape: &[u64],
        elided: &HashMap<usize, u64>,
        offset: &HashMap<usize, u64>,
    ) -> Self {
        let ndim = source_shape.len();
        let mut axes = Vec::with_capacity(self.ndim);
        let mut unsliced = vec![0; source_shape.len()];
        let mut offsets = vec![0; source_shape.len()];
        for x in 0..ndim {
            if let Some(elide) = elided.get(&x) {
                unsliced[x] = *elide;
            } else {
                axes.push(x as u64);
                offsets[x] = *offset.get(&x).unwrap_or(&0);
            }
        }
        assert_eq!(axes.len(), self.ndim);

        let axes = af::Array::new(&axes, dim4(axes.len()));
        let seq4gen = af::Seq::new(0., (self.len() - 1) as f64, 1.);
        let mut indexer = af::Indexer::default();
        indexer.set_index(&axes, 0, None);
        indexer.set_index(&seq4gen, 1, Some(true));

        let unsliced = af::Array::new(&unsliced, dim4(ndim));
        let dims = af::Dim4::new(&[1, self.len() as u64, 1, 1]);
        let mut unsliced = af::tile(&unsliced, dims);
        af::assign_gen(&mut unsliced, &indexer, self.af());

        let offsets = af::Array::new(&offsets, dim4(ndim));
        let offsets = af::tile(&offsets, af::Dim4::new(&[1, self.len() as u64, 1, 1]));

        Self {
            array: unsliced + offsets,
            ndim,
        }
    }

    /// Construct a new `Coords` from the selected indices.
    ///
    /// Panics: if any index is out of bounds
    pub fn get(&self, indexer: af::Indexer) -> Self {
        let array = af::index_gen(self.af(), indexer);
        let ndim = array.dims()[0] as usize;
        let num_coords = array.elements() / ndim;
        let dims = af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]);
        let array = af::moddims(&array, dims);
        Self { array, ndim }
    }

    /// Update these `Coords` by writing the given `value` at the given `index`.
    ///
    /// Panics: if any index is out of bounds
    pub fn set(&mut self, indexer: &af::Indexer, value: Self) {
        af::assign_gen(self.af_mut(), indexer, value.af());
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
    /// Panics: if the given number of dimensions does not fit the set of coordinates
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

/// A [`Stream`] of [`Coords`], as constructed from an input stream of [`Coord`]s.
pub struct CoordBlocks<S> {
    source: Fuse<S>,
    ndim: usize,
    block_size: usize,
    buffer: Vec<u64>,
}

impl<E, S: Stream<Item = Result<Coord, E>>> CoordBlocks<S> {
    /// Construct a new `CoordBlocks`.
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

    #[test]
    fn test_get_and_set() {
        let source = Coords::from_iter(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]], 3);

        let indices = af::Array::new(&[1, 2], dim4(2));
        let seq4gen = af::Seq::new(0., 2., 1.);
        let mut indexer = af::Indexer::default();
        indexer.set_index(&indices, 0, None);
        indexer.set_index(&seq4gen, 1, Some(true));

        let value = source.get(indexer);

        assert_eq!(value.ndim(), 2);
        assert_eq!(value.to_vec(), vec![vec![1, 2], vec![4, 5], vec![7, 8]]);

        let mut dest = Coords::empty(&[10, 15, 20], 3);

        let indices = af::Array::new(&[0, 2], dim4(2));
        let mut indexer = af::Indexer::default();
        indexer.set_index(&indices, 0, None);
        indexer.set_index(&seq4gen, 1, Some(true));
        dest.set(&indexer, value);

        assert_eq!(dest.to_vec(), vec![[1, 0, 2], [4, 0, 5], [7, 0, 8],])
    }

    #[test]
    fn test_unbroadcast() {
        let coords = Coords::from_iter(vec![vec![8, 15, 2, 1, 10, 3], vec![9, 16, 3, 4, 11, 6]], 6);
        let actual = coords.unbroadcast(&[5, 1, 1, 10], &[true, true, false, true, true, false]);
        assert_eq!(actual.to_vec(), vec![vec![2, 0, 0, 3], vec![3, 0, 0, 6]]);
    }
}
