use std::{collections::HashSet, hash::Hash, iter::Copied, slice};

use crate::games::play::{FinitePlay, Play, VisitedPlay};

/// A finite play stored as a non-empty sequence of positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaySequence<S> {
    positions: Box<[S]>,
}

impl<S> PlaySequence<S> {
    /// Creates a finite play from a non-empty boxed slice.
    #[must_use]
    #[inline]
    pub fn new(positions: Box<[S]>) -> Self {
        assert!(!positions.is_empty(), "finite play must be non-empty");
        Self { positions }
    }

    /// Creates a finite play from a non-empty vector.
    #[must_use]
    #[inline]
    pub fn from_vec(positions: Vec<S>) -> Self {
        Self::new(positions.into_boxed_slice())
    }

    /// Returns the underlying positions.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.positions
    }

    /// Consumes the play and returns its positions.
    #[must_use]
    #[inline]
    pub fn into_positions(self) -> Box<[S]> {
        self.positions
    }
}

impl<S> Play for PlaySequence<S>
where
    S: Eq + Hash + Copy,
{
    type Position = S;

    type Positions<'a>
        = Copied<slice::Iter<'a, S>>
    where
        Self: 'a;

    #[inline]
    fn positions(&self) -> Self::Positions<'_> {
        self.positions.iter().copied()
    }
}

impl<S> VisitedPlay for PlaySequence<S>
where
    S: Eq + Hash + Copy,
{
    type Visited<'a>
        = PlaySequenceVisited<'a, S>
    where
        Self: 'a;

    #[inline]
    fn visited(&self) -> Self::Visited<'_> {
        PlaySequenceVisited::new(&self.positions)
    }
}

impl<S> FinitePlay for PlaySequence<S>
where
    S: Eq + Hash + Copy,
{
    #[inline]
    fn len(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    fn first(&self) -> Option<Self::Position> {
        self.positions.first().copied()
    }

    #[inline]
    fn last(&self) -> Option<Self::Position> {
        self.positions.last().copied()
    }
}

/// Iterator over the distinct positions occurring in a finite play.
#[derive(Clone, Debug)]
pub struct PlaySequenceVisited<'a, S>
where
    S: Eq + Hash + Copy,
{
    positions: &'a [S],
    index: usize,
    seen: HashSet<S>,
}

impl<'a, S> PlaySequenceVisited<'a, S>
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

impl<S> Iterator for PlaySequenceVisited<'_, S>
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
