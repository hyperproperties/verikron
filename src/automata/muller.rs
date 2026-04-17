use std::hash::Hash;

use crate::{
    automata::acceptors::{Acceptor, StateSummary},
    lattices::set::Set,
};

/// Muller acceptance condition.
///
/// A run is accepting iff it is infinite and the set of states visited
/// infinitely often is one of the accepting families.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Muller<S: Eq + Hash> {
    families: Vec<Set<S>>,
}

impl<S> Muller<S>
where
    S: Eq + Hash,
{
    /// Creates a Muller condition with the given accepting families.
    #[must_use]
    #[inline]
    pub fn new(families: Vec<Set<S>>) -> Self {
        Self { families }
    }

    /// Returns the accepting families.
    #[must_use]
    #[inline]
    pub fn families(&self) -> &[Set<S>] {
        &self.families
    }
}

impl<S> From<Vec<Set<S>>> for Muller<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(families: Vec<Set<S>>) -> Self {
        Self::new(families)
    }
}

impl<S> Acceptor for Muller<S>
where
    S: Eq + Hash,
{
    type Summary = StateSummary<S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        match summary {
            StateSummary::Finite { .. } => false,
            StateSummary::Infinite { states } => {
                self.families.iter().any(|family| family == states)
            }
        }
    }
}
