use std::{collections::HashSet, hash::Hash};

use crate::{
    automata::{
        acceptors::{Acceptor, OmegaAcceptor},
        infinite_summary::InfiniteStateSummary,
    },
    lattices::set::Set,
};

/// Rabin pair `(F, I)`.
///
/// A pair is satisfied iff no state from `forbidden` is visited infinitely
/// often and some state from `required` is visited infinitely often.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RabinPair<S: Eq + Hash> {
    forbidden: Set<S>,
    required: Set<S>,
}

impl<S> RabinPair<S>
where
    S: Eq + Hash,
{
    /// Creates a Rabin pair `(forbidden, required)`.
    #[must_use]
    #[inline]
    pub fn new(forbidden: Set<S>, required: Set<S>) -> Self {
        Self {
            forbidden,
            required,
        }
    }

    /// Returns the forbidden set.
    #[must_use]
    #[inline]
    pub fn forbidden(&self) -> &Set<S> {
        &self.forbidden
    }

    /// Returns the required set.
    #[must_use]
    #[inline]
    pub fn required(&self) -> &Set<S> {
        &self.required
    }

    /// Returns whether this pair is satisfied by `states`.
    #[must_use]
    #[inline]
    pub fn accepts(&self, states: &Set<S>) -> bool {
        self.forbidden.is_disjoint(states) && !self.required.is_disjoint(states)
    }
}

/// A summary that is sufficient to evaluate Rabin acceptance.
pub trait RabinSummary {
    type State: Eq + Hash;

    fn satisfies_rabin_pair(&self, pair: &RabinPair<Self::State>) -> bool;
}

impl<T> RabinSummary for T
where
    T: InfiniteStateSummary,
{
    type State = T::State;

    #[inline]
    fn satisfies_rabin_pair(&self, pair: &RabinPair<Self::State>) -> bool {
        let inf: HashSet<_> = self.infinitely_often().into_iter().collect();

        pair.forbidden().iter().all(|state| !inf.contains(state))
            && pair.required().iter().any(|state| inf.contains(state))
    }
}

/// Rabin acceptance condition.
///
/// A run is accepting iff it is infinite and some Rabin pair is satisfied by
/// the set of states visited infinitely often.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rabin<S: Eq + Hash> {
    pairs: Vec<RabinPair<S>>,
}

impl<S> Rabin<S>
where
    S: Eq + Hash,
{
    /// Creates a Rabin condition with the given pairs.
    #[must_use]
    #[inline]
    pub fn new(pairs: Vec<RabinPair<S>>) -> Self {
        Self { pairs }
    }

    /// Returns the Rabin pairs.
    #[must_use]
    #[inline]
    pub fn pairs(&self) -> &[RabinPair<S>] {
        &self.pairs
    }

    /// Consumes `self` and returns the Rabin pairs.
    #[must_use]
    #[inline]
    pub fn into_pairs(self) -> Vec<RabinPair<S>> {
        self.pairs
    }
}

impl<S> From<Vec<RabinPair<S>>> for Rabin<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(pairs: Vec<RabinPair<S>>) -> Self {
        Self::new(pairs)
    }
}

impl<S> Acceptor for Rabin<S>
where
    S: Eq + Hash,
{
    type Summary = dyn RabinSummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        self.pairs
            .iter()
            .any(|pair| summary.satisfies_rabin_pair(pair))
    }
}

impl<S> OmegaAcceptor for Rabin<S> where S: Eq + Hash {}
