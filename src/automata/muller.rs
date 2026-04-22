use std::{collections::HashSet, hash::Hash};

use crate::{
    automata::{
        acceptors::{Acceptor, OmegaAcceptor},
        infinite_summary::InfiniteStateSummary,
    },
    lattices::set::Set,
};

/// A summary that is sufficient to evaluate Muller acceptance.
pub trait MullerSummary {
    type State: Eq + Hash;

    fn infinitely_often_equals(&self, family: &Set<Self::State>) -> bool;
}

impl<T> MullerSummary for T
where
    T: InfiniteStateSummary,
{
    type State = T::State;

    #[inline]
    fn infinitely_often_equals(&self, family: &Set<Self::State>) -> bool {
        let inf: HashSet<_> = self.infinitely_often().into_iter().collect();

        family.iter().all(|state| inf.contains(state))
            && inf.iter().all(|state| family.contains(*state))
    }
}

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
    type Summary = dyn MullerSummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        self.families
            .iter()
            .any(|family| summary.infinitely_often_equals(family))
    }
}

impl<S> OmegaAcceptor for Muller<S> where S: Eq + Hash {}
