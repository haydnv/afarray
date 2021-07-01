use std::collections::HashMap;
use std::iter::IntoIterator;
use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, FusedStream, Stream, StreamExt, TryStream, TryStreamExt};
use pin_project::pin_project;

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

    /// Returns true if the number of coordinates in these `Coords` is zero.
    pub fn is_empty(&self) -> bool {
        self.array.elements() == 0
    }

    /// Return the number of coordinates stored in these `Coords`.
    pub fn len(&self) -> usize {
        self.array.elements() as usize / self.ndim
    }

    /// Return the number of dimensions of these `Coords`.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    fn first(&self) -> Coord {
        let dim0 = af::Seq::new(0., (self.ndim - 1) as f32, 1.);
        let dim1 = af::Seq::new(0., 0., 1.);
        let slice = af::index(self.af(), &[dim0, dim1]);
        let mut first = vec![0; self.ndim];
        slice.host(&mut first);
        first
    }

    fn last(&self) -> Coord {
        let i = (self.len() - 1) as f32;
        let dim0 = af::Seq::new(0., (self.ndim - 1) as f32, 1.);
        let dim1 = af::Seq::new(i, i, 1.);
        let slice = af::index(self.af(), &[dim0, dim1]);
        let mut first = vec![0; self.ndim];
        slice.host(&mut first);
        first
    }

    fn append(&self, other: &Coords) -> Self {
        assert_eq!(self.ndim, other.ndim);

        let array = af::join(0, self.af(), other.af());
        Self {
            array,
            ndim: self.ndim,
        }
    }

    fn split(&self, at: usize) -> (Self, Self) {
        assert!(at > 0);
        assert!(at < self.len());

        let left = af::Seq::new(0., (at - 1) as f32, 1.);
        let right = af::Seq::new(at as f32, (self.len() - 1) as f32, 1.);

        let left = af::index(self.af(), &[af::Seq::default(), left]);
        let right = af::index(self.af(), &[af::Seq::default(), right]);
        (
            Self {
                array: left,
                ndim: self.ndim,
            },
            Self {
                array: right,
                ndim: self.ndim,
            },
        )
    }

    fn split_lt(&self, lt: Coord, shape: &[u64]) -> (Self, Self) {
        assert_eq!(lt.len(), self.ndim);
        assert_eq!(shape.len(), self.ndim);

        let coord_bounds = coord_bounds(shape);
        let pivot = coord_to_offset(&lt, &coord_bounds);
        let pivot = af::Array::new(&[pivot], dim4(1));
        let offsets = self.to_offsets(shape);
        let left = af::lt(offsets.af(), &pivot, true);
        let (pivot, _) = af::sum_all(&left);
        self.split(pivot as usize)
    }

    fn sorted(&self) -> Self {
        let array = af::sort(self.af(), 1, true);
        Self {
            array,
            ndim: self.ndim,
        }
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

impl<E, S: Stream<Item = Result<Coord, E>> + Unpin> FusedStream for CoordBlocks<S> {
    fn is_terminated(&self) -> bool {
        self.source.is_terminated() && self.buffer.is_empty()
    }
}

/// Stream for merging two sorted [`CoordBlocks`] streams.
///
/// The behavior of `CoordMerge` is undefined if the input streams are not sorted.
#[pin_project]
pub struct CoordMerge<L, R> {
    #[pin]
    left: CoordBlocks<L>,

    #[pin]
    right: CoordBlocks<R>,

    pending_left: Option<Coords>,
    pending_right: Option<Coords>,
    buffer: Option<Coords>,
    block_size: usize,
    shape: Vec<u64>,
}

impl<L, R> CoordMerge<L, R> {
    /// Construct a new `CoordMerge` stream.
    ///
    /// Panics: if the dimensions of `left`, `right`, and `shape` do not match,
    /// or if block_size is zero.
    pub fn new(
        left: CoordBlocks<L>,
        right: CoordBlocks<R>,
        shape: Vec<u64>,
        block_size: usize,
    ) -> Self {
        assert!(block_size > 0);
        assert_eq!(left.ndim, right.ndim);
        assert_eq!(left.ndim, shape.len());

        Self {
            left,
            right,

            shape,
            block_size,

            pending_left: None,
            pending_right: None,
            buffer: None,
        }
    }
}

impl<E, L, R> Stream for CoordMerge<L, R>
where
    L: Stream<Item = Result<Coord, E>> + Unpin,
    R: Stream<Item = Result<Coord, E>> + Unpin,
{
    type Item = Result<Coords, E>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        if this.pending_left.is_none() && !this.left.is_terminated() {
            match ready!(this.left.poll_next(cxt)) {
                Some(Ok(coords)) => *this.pending_left = Some(coords),
                Some(Err(cause)) => return Poll::Ready(Some(Err(cause))),
                None => {}
            }
        }

        if this.pending_right.is_none() && !this.right.is_terminated() {
            match ready!(this.right.poll_next(cxt)) {
                Some(Ok(coords)) => *this.pending_right = Some(coords),
                Some(Err(cause)) => return Poll::Ready(Some(Err(cause))),
                None => {}
            }
        }

        Poll::Ready(loop {
            match (&mut *this.pending_left, &mut *this.pending_right) {
                (Some(l), Some(r)) if l.last() < r.first() => {
                    let mut l = None;
                    mem::swap(this.pending_left, &mut l);
                    create_or_append(this.buffer, l.unwrap());
                }
                (Some(l), Some(r)) if r.last() < l.first() => {
                    let mut r = None;
                    mem::swap(this.pending_right, &mut r);
                    create_or_append(this.buffer, r.unwrap());
                }
                (Some(l), Some(r)) if l.first() < r.first() => {
                    let (l, l_pending) = l.split_lt(r.first(), this.shape);
                    *this.pending_left = Some(l_pending);
                    create_or_append(this.buffer, l);
                }
                (Some(l), Some(r)) if r.first() < l.first() => {
                    let (r, r_pending) = r.split_lt(l.first(), this.shape);
                    *this.pending_right = Some(r_pending);
                    create_or_append(this.buffer, r);
                }
                (Some(l), Some(r)) => {
                    assert_eq!(l.first(), r.first());
                    let first = Coords::from_iter(vec![l.first()], l.ndim());
                    create_or_append(this.buffer, first);

                    if l.len() > 1 {
                        let (_, l) = l.split(1);
                        *this.pending_left = Some(l);
                    } else {
                        *this.pending_left = None;
                    }

                    if r.len() > 1 {
                        let (_, r) = r.split(1);
                        *this.pending_right = Some(r);
                    } else {
                        *this.pending_right = None;
                    }
                }
                (Some(_), None) => {
                    let mut new_l = None;
                    mem::swap(this.pending_left, &mut new_l);
                    create_or_append(this.buffer, new_l.unwrap());
                }
                (_, Some(_)) => {
                    let mut new_r = None;
                    mem::swap(this.pending_right, &mut new_r);
                    create_or_append(this.buffer, new_r.unwrap());
                }
                (None, None) if this.buffer.is_some() => {
                    let coords = this.buffer.as_ref().unwrap().sorted();
                    *this.buffer = None;
                    break Some(Ok(coords));
                }
                (None, None) => break None,
            }

            if let Some(buffer) = this.buffer {
                if buffer.len() == *this.block_size {
                    let mut coords = None;
                    mem::swap(&mut coords, this.buffer);
                    break Some(Ok(coords.unwrap().sorted()));
                } else if buffer.len() > *this.block_size {
                    let coords = buffer.sorted();
                    let (coords, buffer) = coords.split(*this.block_size);
                    *this.buffer = Some(buffer);
                    break Some(Ok(coords));
                }
            }
        })
    }
}

fn create_or_append(coords: &mut Option<Coords>, to_append: Coords) {
    *coords = match coords {
        Some(coords) => Some(coords.append(&to_append)),
        None if to_append.is_empty() => None,
        None => Some(to_append),
    };
}

#[inline]
pub fn coord_to_offset(coord: &[u64], coord_bounds: &[u64]) -> u64 {
    coord_bounds
        .iter()
        .zip(coord.iter())
        .map(|(d, x)| d * x)
        .sum()
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
    fn tests_for_merge_coord_blocks() {
        let coord_vec = vec![vec![0, 0, 0], vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]];
        let coords = Coords::from_iter(coord_vec.to_vec(), 3);

        assert_eq!(coords.first(), coord_vec[0]);
        assert_eq!(&coords.last(), coord_vec.last().unwrap());

        let (l, r) = coords.split(1);
        assert_eq!(l.to_vec(), &coord_vec[..1]);
        assert_eq!(r.to_vec(), &coord_vec[1..]);

        let (l, r) = coords.split_lt(vec![0, 1, 0], &[2, 2, 2]);
        assert_eq!(l.to_vec(), &coord_vec[..2]);
        assert_eq!(r.to_vec(), &coord_vec[2..]);
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
