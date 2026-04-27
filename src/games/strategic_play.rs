use std::{collections::HashSet, fmt::Debug, hash::Hash};

use crate::games::{
    arena::Arena,
    play::{Play, VisitedPlay},
    strategy::Strategy,
};

#[derive(Debug)]
pub struct StrategicPlay<'a, A, S>
where
    A: Arena,
    S: Strategy<Arena = A>,
{
    strategy: &'a S,
    memory: S::Memory,
    current: Option<A::Position>,
}

impl<'a, A, S> Clone for StrategicPlay<'a, A, S>
where
    A: Arena,
    S: Strategy<Arena = A>,
    S::Memory: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy,
            memory: self.memory.clone(),
            current: self.current,
        }
    }
}

impl<'a, A, S> StrategicPlay<'a, A, S>
where
    A: Arena,
    S: Strategy<Arena = A>,
{
    #[must_use]
    #[inline]
    pub fn new(strategy: &'a S, start: A::Position) -> Self {
        Self {
            strategy,
            memory: strategy.initial_memory(),
            current: Some(start),
        }
    }
}

impl<A, S> Play for StrategicPlay<'_, A, S>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
    S: Strategy<Arena = A>,
    S::Memory: Clone + Debug,
{
    type Position = A::Position;

    type Positions<'a>
        = StrategicPlay<'a, A, S>
    where
        Self: 'a;

    #[inline]
    fn positions(&self) -> Self::Positions<'_> {
        self.clone()
    }
}

impl<A, S> VisitedPlay for StrategicPlay<'_, A, S>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
    S: Strategy<Arena = A>,
    S::Memory: Clone + Debug,
{
    type Visited<'a>
        = VisitedStrategicPlay<'a, A, S>
    where
        Self: 'a;

    #[inline]
    fn visited(&self) -> Self::Visited<'_> {
        VisitedStrategicPlay::new(self.clone())
    }
}

impl<A, S> Iterator for StrategicPlay<'_, A, S>
where
    A: Arena,
    S: Strategy<Arena = A>,
{
    type Item = A::Position;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;
        let next = self.strategy.choice(&self.memory, current);

        if let Some(successor) = next {
            self.memory = self.strategy.update(&self.memory, successor);
            self.current = Some(successor);
        } else {
            self.current = None;
        }

        Some(current)
    }
}

/// Iterator over the distinct positions visited by a strategic play.
///
/// Note: if the strategic play is infinite and keeps revisiting already-seen
/// positions forever, this iterator may not terminate while searching for a new
/// unseen position.
#[derive(Clone, Debug)]
pub struct VisitedStrategicPlay<'a, A, S>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
    S: Strategy<Arena = A>,
    S::Memory: Clone + Debug,
{
    play: StrategicPlay<'a, A, S>,
    seen: HashSet<A::Position>,
}

impl<'a, A, S> VisitedStrategicPlay<'a, A, S>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
    S: Strategy<Arena = A>,
    S::Memory: Clone + Debug,
{
    #[must_use]
    #[inline]
    pub fn new(play: StrategicPlay<'a, A, S>) -> Self {
        Self {
            play,
            seen: HashSet::new(),
        }
    }
}

impl<A, S> Iterator for VisitedStrategicPlay<'_, A, S>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
    S: Strategy<Arena = A>,
    S::Memory: Clone + Debug,
{
    type Item = A::Position;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        for position in self.play.by_ref() {
            if self.seen.insert(position) {
                return Some(position);
            }
        }

        None
    }
}
