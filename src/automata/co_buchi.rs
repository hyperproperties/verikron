use std::hash::Hash;

use crate::{
    automata::acceptors::{Acceptor, StateSummary},
    lattices::set::Set,
};

/// Co-Büchi acceptance condition.
///
/// A run is accepting iff it is infinite and visits no rejecting state
/// infinitely often.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoBuchi<S: Eq + Hash> {
    rejecting: Set<S>,
}

impl<S> CoBuchi<S>
where
    S: Eq + Hash,
{
    /// Creates a co-Büchi condition with the given rejecting states.
    #[must_use]
    #[inline]
    pub fn new(rejecting: Set<S>) -> Self {
        Self { rejecting }
    }

    /// Returns the rejecting states.
    #[must_use]
    #[inline]
    pub fn rejecting(&self) -> &Set<S> {
        &self.rejecting
    }
}

impl<S> From<Set<S>> for CoBuchi<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(rejecting: Set<S>) -> Self {
        Self::new(rejecting)
    }
}

impl<S> Acceptor for CoBuchi<S>
where
    S: Eq + Hash,
{
    type Summary = StateSummary<S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        match summary {
            StateSummary::Finite { .. } => false,
            StateSummary::Infinite { states } => self.rejecting.is_disjoint(states),
        }
    }
}
