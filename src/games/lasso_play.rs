use std::{collections::HashSet, hash::Hash};

use crate::games::{
    looping_play::LoopingPlay,
    play::{InfinitePlay, Play, VisitedPlay},
    play_sequence::PlaySequence,
};

/// A lasso-shaped infinite play.
///
/// A lasso consists of a finite stem followed by a non-empty cycle repeated
/// forever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LassoPlay<S> {
    /// Index at which the cycle begins.
    index: usize,

    /// Concatenation of stem and cycle.
    positions: Box<[S]>,
}

impl<S> LassoPlay<S> {
    /// Creates a lasso from a stem and a non-empty cycle.
    #[must_use]
    #[inline]
    pub fn new(stem: Vec<S>, cycle: Vec<S>) -> Self {
        assert!(!cycle.is_empty(), "lasso cycle must be non-empty");

        let index = stem.len();
        let positions = stem
            .into_iter()
            .chain(cycle)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { index, positions }
    }

    /// Creates a lasso from stored positions and the cycle start index.
    ///
    /// Panics if the cycle would be empty.
    #[must_use]
    #[inline]
    pub fn from_boxed_slice(index: usize, positions: Box<[S]>) -> Self {
        assert!(index < positions.len(), "lasso cycle must be non-empty");
        Self { index, positions }
    }

    /// Returns the cycle start index.
    #[must_use]
    #[inline]
    pub fn cycle_index(&self) -> usize {
        self.index
    }

    /// Returns the finite stem as a slice.
    #[must_use]
    #[inline]
    pub fn stem(&self) -> &[S] {
        &self.positions[..self.index]
    }

    /// Returns the finite cycle as a slice.
    #[must_use]
    #[inline]
    pub fn cycle(&self) -> &[S] {
        &self.positions[self.index..]
    }

    /// Returns the stored stem and cycle as one finite slice.
    #[must_use]
    #[inline]
    pub fn stored_positions(&self) -> &[S] {
        &self.positions
    }

    /// Clones the stem into a finite play.
    #[must_use]
    #[inline]
    pub fn stem_play(&self) -> PlaySequence<S>
    where
        S: Clone,
    {
        PlaySequence::new(self.stem().to_vec().into_boxed_slice())
    }

    /// Clones the cycle into a looping play.
    #[must_use]
    #[inline]
    pub fn cycle_play(&self) -> LoopingPlay<S>
    where
        S: Clone,
    {
        LoopingPlay::new(self.cycle().to_vec().into_boxed_slice())
    }
}

impl<S> LassoPlay<S>
where
    S: Eq + Hash + Copy,
{
    /// Returns the positions visited infinitely often.
    ///
    /// For a lasso, these are exactly the distinct positions in the cycle.
    #[inline]
    pub fn infinitely_often(&self) -> LassoVisited<'_, S> {
        LassoVisited::new(self.cycle())
    }
}

impl<S> Play for LassoPlay<S>
where
    S: Eq + Hash + Copy,
{
    type Position = S;

    type Positions<'a>
        = LassoPositions<'a, S>
    where
        Self: 'a;

    #[inline]
    fn positions(&self) -> Self::Positions<'_> {
        LassoPositions::new(self.index, &self.positions)
    }
}

impl<S> VisitedPlay for LassoPlay<S>
where
    S: Eq + Hash + Copy,
{
    type Visited<'a>
        = LassoVisited<'a, S>
    where
        Self: 'a;

    #[inline]
    fn visited(&self) -> Self::Visited<'_> {
        LassoVisited::new(&self.positions)
    }
}

impl<S> InfinitePlay for LassoPlay<S> where S: Eq + Hash + Copy {}

/// Iterator over the infinite sequence induced by a lasso.
#[derive(Clone, Debug)]
pub struct LassoPositions<'a, S> {
    index: usize,
    positions: &'a [S],
    offset: usize,
}

impl<'a, S> LassoPositions<'a, S> {
    #[must_use]
    #[inline]
    pub fn new(index: usize, positions: &'a [S]) -> Self {
        debug_assert!(index < positions.len(), "lasso cycle must be non-empty");

        Self {
            index,
            positions,
            offset: 0,
        }
    }
}

impl<S> Iterator for LassoPositions<'_, S>
where
    S: Copy,
{
    type Item = S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let position = if self.offset < self.positions.len() {
            self.positions[self.offset]
        } else {
            let cycle_len = self.positions.len() - self.index;
            let cycle_offset = (self.offset - self.index) % cycle_len;
            self.positions[self.index + cycle_offset]
        };

        self.offset += 1;
        Some(position)
    }
}

/// Iterator over the distinct positions in a finite slice.
///
/// Used both for all visited positions of a lasso and for the positions
/// visited infinitely often in its cycle.
#[derive(Clone, Debug)]
pub struct LassoVisited<'a, S>
where
    S: Eq + Hash + Copy,
{
    positions: &'a [S],
    index: usize,
    seen: HashSet<S>,
}

impl<'a, S> LassoVisited<'a, S>
where
    S: Eq + Hash + Copy,
{
    #[must_use]
    #[inline]
    pub fn new(positions: &'a [S]) -> Self {
        Self {
            positions,
            index: 0,
            seen: HashSet::new(),
        }
    }
}

impl<S> Iterator for LassoVisited<'_, S>
where
    S: Eq + Hash + Copy,
{
    type Item = S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.positions.len() {
            let position = self.positions[self.index];
            self.index += 1;

            if self.seen.insert(position) {
                return Some(position);
            }
        }

        None
    }
}
