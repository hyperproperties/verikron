use std::{collections::HashSet, hash::Hash, iter::Copied};

use crate::games::play::{FinitePlay, Play};

/// A finite play stored as a non-empty sequence of positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaySequence<S> {
    positions: Box<[S]>,
}

impl<S> PlaySequence<S> {
    /// Creates a finite play from a non-empty slice.
    #[must_use]
    #[inline]
    pub fn new(positions: Box<[S]>) -> Self {
        assert!(!positions.is_empty(), "finite play must be non-empty");
        Self { positions }
    }

    /// Creates a finite play from a vector.
    #[must_use]
    #[inline]
    pub fn from_vec(positions: Vec<S>) -> Self {
        Self::new(positions.into_boxed_slice())
    }

    /// Returns the underlying positions.
    #[must_use]
    #[inline]
    pub fn positions(&self) -> &[S] {
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

    type Sequence<'a>
        = Copied<std::slice::Iter<'a, S>>
    where
        Self: 'a;

    type Visited<'a>
        = PlaySequenceVisited<'a, S>
    where
        Self: 'a;

    #[inline]
    fn sequence(&self) -> Self::Sequence<'_> {
        self.positions.iter().copied()
    }

    #[inline]
    fn visited(&self) -> Self::Visited<'_> {
        PlaySequenceVisited::new(&self.positions)
    }
}

impl<S> FinitePlay for PlaySequence<S>
where
    S: Eq + Hash + Copy,
{
    type FinitelyOften<'a>
        = Copied<std::slice::Iter<'a, S>>
    where
        Self: 'a;

    #[inline]
    fn finitely_often(&self) -> Self::FinitelyOften<'_> {
        self.sequence()
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
