use std::{collections::HashSet, hash::Hash};

use crate::games::play::{InfinitePlay, Play};

/// An infinite play obtained by repeating a non-empty finite block forever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopingPlay<S> {
    positions: Box<[S]>,
}

impl<S> LoopingPlay<S> {
    /// Creates a looping play from a non-empty block.
    #[must_use]
    #[inline]
    pub fn new(positions: Box<[S]>) -> Self {
        assert!(!positions.is_empty(), "looping play must be non-empty");
        Self { positions }
    }

    /// Creates a looping play from a vector.
    #[must_use]
    #[inline]
    pub fn from_vec(positions: Vec<S>) -> Self {
        Self::new(positions.into_boxed_slice())
    }

    /// Returns the repeated block.
    #[must_use]
    #[inline]
    pub fn positions(&self) -> &[S] {
        &self.positions
    }
}

impl<S> Play for LoopingPlay<S>
where
    S: Eq + Hash + Copy,
{
    type Position = S;

    type Sequence<'a>
        = LoopingPositions<'a, S>
    where
        Self: 'a;

    type Visited<'a>
        = LoopingVisited<'a, S>
    where
        Self: 'a;

    #[inline]
    fn sequence(&self) -> Self::Sequence<'_> {
        LoopingPositions::new(&self.positions)
    }

    #[inline]
    fn visited(&self) -> Self::Visited<'_> {
        LoopingVisited::new(&self.positions)
    }
}

impl<S> InfinitePlay for LoopingPlay<S>
where
    S: Eq + Hash + Copy,
{
    type InfinitelyOften<'a>
        = LoopingVisited<'a, S>
    where
        Self: 'a;

    #[inline]
    fn infinitely_often(&self) -> Self::InfinitelyOften<'_> {
        self.visited()
    }
}

/// Iterator over a looping play.
#[derive(Clone, Debug)]
pub struct LoopingPositions<'a, S> {
    positions: &'a [S],
    index: usize,
}

impl<'a, S> LoopingPositions<'a, S> {
    /// Creates an iterator over the repeated block.
    #[must_use]
    #[inline]
    pub fn new(positions: &'a [S]) -> Self {
        debug_assert!(!positions.is_empty(), "looping play must be non-empty");

        Self {
            positions,
            index: 0,
        }
    }
}

impl<S> Iterator for LoopingPositions<'_, S>
where
    S: Copy,
{
    type Item = S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let position = self.positions[self.index % self.positions.len()];
        self.index += 1;
        Some(position)
    }
}

/// Iterator over the distinct positions occurring in the looping block.
#[derive(Clone, Debug)]
pub struct LoopingVisited<'a, S>
where
    S: Eq + Hash + Copy,
{
    positions: &'a [S],
    index: usize,
    seen: HashSet<S>,
}

impl<'a, S> LoopingVisited<'a, S>
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

impl<S> Iterator for LoopingVisited<'_, S>
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
