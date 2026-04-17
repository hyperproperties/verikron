use std::hash::Hash;

use crate::{
    automata::acceptors::{Acceptor, StateSummary},
    lattices::set::Set,
};

/// Streett pair `(E, F)`.
///
/// A pair is satisfied iff either no state from `trigger` is visited
/// infinitely often, or some state from `guarantee` is visited infinitely
/// often.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreettPair<S: Eq + Hash> {
    trigger: Set<S>,
    guarantee: Set<S>,
}

impl<S> StreettPair<S>
where
    S: Eq + Hash,
{
    /// Creates a Streett pair `(trigger, guarantee)`.
    #[must_use]
    #[inline]
    pub fn new(trigger: Set<S>, guarantee: Set<S>) -> Self {
        Self { trigger, guarantee }
    }

    /// Returns the trigger set.
    #[must_use]
    #[inline]
    pub fn trigger(&self) -> &Set<S> {
        &self.trigger
    }

    /// Returns the guarantee set.
    #[must_use]
    #[inline]
    pub fn guarantee(&self) -> &Set<S> {
        &self.guarantee
    }

    /// Returns whether this pair is satisfied by `states`.
    #[must_use]
    #[inline]
    pub fn accepts(&self, states: &Set<S>) -> bool {
        self.trigger.is_disjoint(states) || !self.guarantee.is_disjoint(states)
    }
}

/// Streett acceptance condition.
///
/// A run is accepting iff it is infinite and every Streett pair is satisfied
/// by the set of states visited infinitely often.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Streett<S: Eq + Hash> {
    pairs: Vec<StreettPair<S>>,
}

impl<S> Streett<S>
where
    S: Eq + Hash,
{
    /// Creates a Streett condition with the given pairs.
    #[must_use]
    #[inline]
    pub fn new(pairs: Vec<StreettPair<S>>) -> Self {
        Self { pairs }
    }

    /// Returns the Streett pairs.
    #[must_use]
    #[inline]
    pub fn pairs(&self) -> &[StreettPair<S>] {
        &self.pairs
    }
}

impl<S> From<Vec<StreettPair<S>>> for Streett<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(pairs: Vec<StreettPair<S>>) -> Self {
        Self::new(pairs)
    }
}

impl<S> Acceptor for Streett<S>
where
    S: Eq + Hash,
{
    type Summary = StateSummary<S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        match summary {
            StateSummary::Finite { .. } => false,
            StateSummary::Infinite { states } => self.pairs.iter().all(|pair| pair.accepts(states)),
        }
    }
}
