use std::collections::HashMap;
use std::iter::IntoIterator;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayfire as af;
use futures::ready;
use futures::stream::{Fuse, FusedStream, Stream, StreamExt, TryStream, TryStreamExt};
use pin_project::pin_project;

use super::{coord_bounds, dim4, ArrayExt};

/// An n-dimensional coordinate.
pub type Coord = Vec<u64>;

/// One-dimensional array indices corresponding to n-dimensional coordinates.
pub type Offsets = ArrayExt<u64>;

/// A hardware-accelerated set of n-dimensional coordinates, all with the same dimension.
///
/// TODO: separate out a `CoordBasis` struct
#[derive(Clone)]
pub struct Coords {
    array: af::Array<u64>,
    ndim: usize,
}

impl Coords {
    /// Constructs `Coords` with the given `size` full of zeros (origin points) for the given shape.
    ///
    /// Panics: if shape is empty
    pub fn empty(shape: &[u64], size: usize) -> Self {
        assert!(!shape.is_empty());

        let ndim = shape.len();
        let dims = af::Dim4::new(&[ndim as u64, size as u64, 1, 1]);
        let array = af::constant(0u64, dims);
        Self { array, ndim }
    }

    /// Constructs a new `Coords` from an iterator of [`Coord`]s.
    ///
    /// Panics: if any [`Coord`] is not of length `ndim`, or if `ndim` is zero.
    pub fn from_iter<I: IntoIterator<Item = Coord>>(iter: I, ndim: usize) -> Self {
        assert!(ndim > 0);

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
    ///
    /// Panics: if `shape` is empty
    pub fn from_offsets(offsets: Offsets, shape: &[u64]) -> Self {
        assert!(!shape.is_empty());

        let ndim = shape.len() as u64;
        let coord_bounds = coord_bounds(shape);

        let dims = af::Dim4::new(&[1, ndim, 1, 1]);
        let af_coord_bounds: af::Array<u64> = af::Array::new(&coord_bounds, dims);
        let af_shape: af::Array<u64> = af::Array::new(&shape, dims);

        let offsets = af::div(offsets.deref(), &af_coord_bounds, true);
        let coords = af::modulo(&offsets, &af_shape, true);
        let array = af::transpose(&coords, false);

        Self {
            array,
            ndim: shape.len(),
        }
    }

    /// Constructs a new `Coords` from a [`Stream`] of [`Coord`]s.
    ///
    /// Panics: if any [`Coord`] has a length other than `ndim`, or if `ndim` is zero
    pub async fn from_stream<S: Stream<Item = Coord> + Unpin>(
        mut source: S,
        ndim: usize,
        size_hint: Option<usize>,
    ) -> Self {
        assert!(ndim > 0);

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

    /// Return `true` if the number of coordinates in these `Coords` is zero.
    pub fn is_empty(&self) -> bool {
        self.array.elements() == 0
    }

    /// Return `true` if these `Coords` are in sorted order with respect to the given `shape`.
    pub fn is_sorted(&self, shape: &[u64]) -> bool {
        self.to_offsets(shape).is_sorted()
    }

    /// Return the number of coordinates stored in these `Coords`.
    pub fn len(&self) -> usize {
        self.dims()[1] as usize
    }

    /// Return the number of dimensions of these `Coords`.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    fn last(&self) -> Coord {
        let i = (self.len() - 1) as f32;
        let dim0 = af::Seq::new(0., (self.ndim - 1) as f32, 1.);
        let dim1 = af::Seq::new(i, i, 1.);
        let slice = af::index(self, &[dim0, dim1]);
        let mut first = vec![0; self.ndim];
        slice.host(&mut first);
        first
    }

    fn append(&self, other: &Coords) -> Self {
        assert_eq!(self.ndim, other.ndim);

        let array = af::join(1, self, other);
        Self {
            array,
            ndim: self.ndim,
        }
    }

    fn split(&self, at: usize) -> (Self, Self) {
        assert!(at < self.len());

        if at == 0 {
            let shape: Vec<u64> = std::iter::repeat(0).take(self.ndim()).collect();
            return (Coords::empty(&shape, 0), self.clone());
        }

        let left = af::Seq::new(0., (at - 1) as f32, 1.);
        let right = af::Seq::new(at as f32, (self.len() - 1) as f32, 1.);

        let left = af::index(self, &[af::Seq::default(), left]);
        let right = af::index(self, &[af::Seq::default(), right]);

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

    fn split_lt(&self, lt: &[u64], shape: &[u64]) -> (Self, Self) {
        assert_eq!(lt.len(), self.ndim);
        assert_eq!(shape.len(), self.ndim);

        let coord_bounds = coord_bounds(shape);
        let pivot = coord_to_offset(lt, &coord_bounds);
        let pivot = af::Array::new(&[pivot], dim4(1));
        let offsets = self.to_offsets(shape);
        let left = af::lt(offsets.deref(), &pivot, true);
        let (pivot, _) = af::sum_all(&left);
        self.split(pivot as usize)
    }

    fn sorted(&self) -> Self {
        let array = af::sort(self, 2, true);
        Self {
            array,
            ndim: self.ndim,
        }
    }

    fn unique(&self, shape: &[u64]) -> Self {
        let offsets = self.to_offsets(shape);
        let offsets = af::set_unique(offsets.deref(), true);
        Self::from_offsets(offsets.into(), shape)
    }

    /// Return a copy of these `Coords` without the specified axis.
    ///
    /// Panics: if there is no dimension at `axis`
    pub fn contract_dim(&self, axis: usize) -> Self {
        assert!(axis < self.ndim);

        let mut index: Vec<usize> = (0..self.ndim).collect();
        index.remove(axis);

        self.get(&index)
    }

    /// Return a copy of these `Coords` with a new dimension at the given axis.
    ///
    /// Panics: if `axis` is greater than `self.ndim()`
    pub fn expand_dim(&self, axis: usize) -> Self {
        assert!(axis <= self.ndim);

        let ndim = self.ndim + 1;
        let dims = af::Dim4::new(&[ndim as u64, self.dims()[1], 1, 1]);
        let mut expanded = af::constant(0, dims);

        let index: Vec<u64> = (0..self.ndim())
            .map(|x| if x < axis { x } else { x + 1 })
            .map(|x| x as u64)
            .collect();

        index_set(&mut expanded, &index, self);

        Self {
            array: expanded,
            ndim: self.ndim + 1,
        }
    }

    /// Return these `Coords` as flipped around `axis` with respect to the given `shape`.
    ///
    /// E.g. flipping axis 1 of coordinate `[0, 1, 2]` in shape `[5, 5, 5]` produces `[0, 4, 2]`.
    ///
    /// Panics: if `self.ndim() != shape.len()`
    pub fn flip(self, shape: &[u64], axis: usize) -> Self {
        assert_eq!(self.ndim, shape.len());

        let mut mask = vec![0i64; self.ndim()];
        mask[axis] = (shape[axis] - 1) as i64;
        let mask = af::Array::new(&mask, af::Dim4::new(&[self.ndim() as u64, 1, 1, 1]));

        let coords: af::Array<i64> = self.array.cast();
        let flipped = af::sub(&mask, &coords, true);

        Self {
            array: af::abs(&flipped).cast(),
            ndim: self.ndim,
        }
    }

    /// Transform the coordinate basis of these `Coords` from a source tensor to a slice.
    pub fn slice(
        &self,
        shape: &[u64],
        elided: &HashMap<usize, u64>,
        offset: &HashMap<usize, u64>,
    ) -> Self {
        let ndim = shape.len();
        let mut offsets = Vec::with_capacity(ndim);
        let mut index = Vec::with_capacity(ndim);
        for x in 0..self.ndim {
            if elided.contains_key(&x) {
                continue;
            }

            let offset = offset.get(&x).unwrap_or(&0);
            offsets.push(*offset);
            index.push(x);
        }

        let offsets = af::Array::new(&offsets, dim4(offsets.len()));
        let array = af::sub(self.get(&index).deref(), &offsets, true);
        Self { array, ndim }
    }

    /// Transpose these `Coords` according to the given `permutation`.
    ///
    /// If no permutation is given, the coordinate axes will be inverted.
    pub fn transpose<P: AsRef<[usize]>>(&self, permutation: Option<P>) -> Coords {
        if let Some(permutation) = permutation {
            self.get(permutation.as_ref())
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

        let axes: Vec<usize> = broadcast
            .iter()
            .enumerate()
            .filter_map(|(x, b)| if *b { None } else { Some(x) })
            .collect();

        let unbroadcasted = self.get(&axes);

        let axes: Vec<usize> = broadcast
            .iter()
            .enumerate()
            .filter_map(|(x, b)| if *b { None } else { Some(x - offset) })
            .collect();

        coords.set(&axes, &unbroadcasted);

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

        let unsliced = af::Array::new(&unsliced, dim4(ndim));
        let tile_dims = af::Dim4::new(&[1, self.len() as u64, 1, 1]);
        let mut unsliced = af::tile(&unsliced, tile_dims);
        index_set(&mut unsliced, &axes, self);

        let offsets = af::Array::new(&offsets, dim4(ndim));
        let offsets = af::tile(&offsets, tile_dims);

        Self {
            array: unsliced + offsets,
            ndim,
        }
    }

    /// Construct a new `Coords` from the selected indices.
    ///
    /// Panics: if any index is out of bounds
    pub fn get(&self, axes: &[usize]) -> Self {
        let axes: Vec<u64> = axes
            .iter()
            .map(|x| {
                assert!(x < &self.ndim);
                *x as u64
            })
            .collect();

        let array = index_get(self, &axes);
        Self {
            array,
            ndim: axes.len(),
        }
    }

    /// Update these `Coords` by writing the given `value` at the given `index`.
    ///
    /// Panics: if any index is out of bounds, or if `value.len()` does not match `self.len()`
    pub fn set(&mut self, axes: &[usize], value: &Self) {
        let axes: Vec<u64> = axes
            .iter()
            .map(|x| {
                assert!(x < &self.ndim);
                *x as u64
            })
            .collect();

        index_set(self, &axes, value)
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
    pub fn to_vec(&self) -> Vec<Coord> {
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
    pub fn into_vec(self) -> Vec<Coord> {
        self.to_vec()
    }
}

impl Deref for Coords {
    type Target = af::Array<u64>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl DerefMut for Coords {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
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
    ///
    /// Panics: if `ndim == 0`
    pub fn new(source: S, ndim: usize, block_size: usize) -> Self {
        assert!(ndim > 0);

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
    left: Fuse<L>,

    #[pin]
    right: Fuse<R>,

    pending_left: Option<Coords>,
    pending_right: Option<Coords>,
    buffer: Option<Coords>,
    block_size: usize,
    shape: Vec<u64>,
}

impl<L: Stream, R: Stream> CoordMerge<L, R> {
    /// Construct a new `CoordMerge` stream.
    ///
    /// Panics: if the dimensions of `left`, `right`, and `shape` do not match,
    /// or if block_size is zero.
    pub fn new(left: L, right: R, shape: Vec<u64>, block_size: usize) -> Self {
        assert!(block_size > 0);

        Self {
            left: left.fuse(),
            right: right.fuse(),

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
    L: Stream<Item = Result<Coords, E>> + Unpin,
    R: Stream<Item = Result<Coords, E>> + Unpin,
{
    type Item = Result<Coords, E>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            if this.pending_left.is_none() && !this.left.is_terminated() {
                match ready!(this.left.as_mut().poll_next(cxt)) {
                    Some(Ok(coords)) => {
                        assert_eq!(coords.ndim(), this.shape.len());
                        *this.pending_left = Some(coords)
                    }
                    Some(Err(cause)) => return Poll::Ready(Some(Err(cause))),
                    None => {}
                }
            }

            if this.pending_right.is_none() && !this.right.is_terminated() {
                match ready!(this.right.as_mut().poll_next(cxt)) {
                    Some(Ok(coords)) => {
                        assert_eq!(coords.ndim(), this.shape.len());
                        *this.pending_right = Some(coords)
                    }
                    Some(Err(cause)) => return Poll::Ready(Some(Err(cause))),
                    None => {}
                }
            }

            match (&mut *this.pending_left, &mut *this.pending_right) {
                (Some(l), Some(r)) if l.last() < r.last() => {
                    let (r, r_pending) = r.split_lt(&l.last(), this.shape);
                    *this.pending_right = Some(r_pending);
                    create_or_append(this.buffer, r);

                    let mut l = None;
                    mem::swap(this.pending_left, &mut l);
                    create_or_append(this.buffer, l.unwrap());
                }
                (Some(l), Some(r)) if r.last() < l.last() => {
                    let (l, l_pending) = l.split_lt(&r.last(), this.shape);
                    *this.pending_left = Some(l_pending);
                    create_or_append(this.buffer, l);

                    let mut r = None;
                    mem::swap(this.pending_right, &mut r);
                    create_or_append(this.buffer, r.unwrap());
                }
                (Some(l), Some(r)) => {
                    assert_eq!(l.last(), r.last());

                    let mut l = None;
                    mem::swap(this.pending_left, &mut l);
                    create_or_append(this.buffer, l.unwrap());

                    let mut r = None;
                    mem::swap(this.pending_right, &mut r);
                    create_or_append(this.buffer, r.unwrap());
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

/// Return only the unique coordinates from a sorted stream of `Coords`.
///
/// Behavior is undefined if the input stream is not sorted.
#[pin_project]
pub struct CoordUnique<S> {
    #[pin]
    source: Fuse<S>,
    buffer: Option<Coords>,
    shape: Vec<u64>,
    block_size: usize,
}

impl<S: Stream> CoordUnique<S> {
    /// Construct a new `CoordUnique` stream from a sorted stream of `Coords`.
    pub fn new(source: S, shape: Vec<u64>, block_size: usize) -> Self {
        Self {
            source: source.fuse(),
            buffer: None,
            shape,
            block_size,
        }
    }
}

impl<E, S: Stream<Item = Result<Coords, E>>> Stream for CoordUnique<S> {
    type Item = Result<Coords, E>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok(block)) => {
                    let buffer = if let Some(buffer) = this.buffer {
                        buffer.append(&block).unique(this.shape)
                    } else {
                        block.unique(this.shape)
                    };

                    *this.buffer = Some(buffer);
                }
                Some(Err(cause)) => break Some(Err(cause)),
                None if this.buffer.is_some() => {
                    let mut buffer = None;
                    mem::swap(this.buffer, &mut buffer);
                    break buffer.map(Ok);
                }
                None => break None,
            }

            if let Some(buffer) = this.buffer {
                if buffer.len() > *this.block_size {
                    let (block, buffer) = buffer.split(*this.block_size);
                    *this.buffer = Some(buffer);
                    break Some(Ok(block));
                }
            }
        })
    }
}

#[inline]
fn create_or_append(coords: &mut Option<Coords>, to_append: Coords) {
    *coords = match coords {
        Some(coords) => Some(coords.append(&to_append)),
        None if to_append.is_empty() => None,
        None => Some(to_append),
    };
}

#[inline]
/// Convert a coordinate to a linear offset.
pub fn coord_to_offset(coord: &[u64], coord_bounds: &[u64]) -> u64 {
    coord_bounds
        .iter()
        .zip(coord.iter())
        .map(|(d, x)| d * x)
        .sum()
}

fn index_get(subject: &af::Array<u64>, index: &[u64]) -> af::Array<u64> {
    let len = subject.dims()[1];
    let index = af::Array::new(index, dim4(index.len()));
    let seq4gen = af::Seq::new(0., (len - 1) as f32, 1.);
    let mut indexer = af::Indexer::default();
    indexer.set_index(&index, 0, None);
    indexer.set_index(&seq4gen, 1, Some(true));

    af::index_gen(subject, indexer)
}

fn index_set(subject: &mut af::Array<u64>, index: &[u64], value: &af::Array<u64>) {
    debug_assert!(value.dims()[0] == index.len() as u64);
    debug_assert!(value.dims()[1] == subject.dims()[1]);

    let len = subject.dims()[1];
    let index = af::Array::new(index, dim4(index.len()));
    if len == 1 {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, Some(false));
        af::assign_gen(subject, &indexer, value);
    } else {
        let seq4gen = af::Seq::new(0., (len - 1) as f32, 1.);
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
        indexer.set_index(&seq4gen, 1, Some(true));

        af::assign_gen(subject, &indexer, value);
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
    fn test_merge_helpers() {
        let coord_vec = vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![1, 0, 0],
            vec![1, 1, 1],
        ];
        let coords = Coords::from_iter(coord_vec.to_vec(), 3);

        assert_eq!(&coords.last(), coord_vec.last().unwrap());

        let (l, r) = coords.split(1);
        assert_eq!(l.to_vec(), &coord_vec[..1]);
        assert_eq!(r.to_vec(), &coord_vec[1..]);

        let (l, r) = coords.split_lt(&[0, 1, 0], &[2, 2, 2]);
        assert_eq!(l.to_vec(), &coord_vec[..2]);
        assert_eq!(r.to_vec(), &coord_vec[2..]);

        let joined = l.append(&r);
        assert_eq!(joined.to_vec(), coords.to_vec());

        assert_eq!(coords.to_vec(), coords.sorted().to_vec());
    }

    #[test]
    fn test_unique_helpers() {
        let coord_vec = vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![1, 0, 0],
        ];

        let coords = Coords::from_iter(coord_vec.to_vec(), 3);

        let expected = vec![vec![0, 0, 0], vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]];
        assert_eq!(coords.unique(&[2, 2, 2]).to_vec(), expected);
    }

    #[test]
    fn test_get_and_set() {
        let source = Coords::from_iter(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]], 3);

        let value = source.get(&[1, 2]);

        assert_eq!(value.ndim(), 2);
        assert_eq!(value.to_vec(), vec![vec![1, 2], vec![4, 5], vec![7, 8]]);

        let mut dest = Coords::empty(&[10, 15, 20], 3);
        dest.set(&[0, 2], &value);

        assert_eq!(dest.to_vec(), vec![[1, 0, 2], [4, 0, 5], [7, 0, 8],])
    }

    #[test]
    fn test_unbroadcast() {
        let coords = Coords::from_iter(vec![vec![8, 15, 2, 1, 10, 3], vec![9, 16, 3, 4, 11, 6]], 6);
        let actual = coords.unbroadcast(&[5, 1, 1, 10], &[true, true, false, true, true, false]);
        assert_eq!(actual.to_vec(), vec![vec![2, 0, 0, 3], vec![3, 0, 0, 6]]);
    }
}
