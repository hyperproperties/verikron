use std::hash::Hash;

use crate::{
    automata::{
        acceptors::{Acceptor, OmegaAcceptor},
        infinite_summary::InfiniteStateSummary,
    },
    lattices::set::Set,
};

/// A summary that is sufficient to evaluate Büchi acceptance.
pub trait BuchiSummary {
    type State: Eq + Hash;

    fn visits_accepting_infinitely_often(&self, accepting: &Set<Self::State>) -> bool;
}

impl<T> BuchiSummary for T
where
    T: InfiniteStateSummary,
{
    type State = T::State;

    #[inline]
    fn visits_accepting_infinitely_often(&self, accepting: &Set<Self::State>) -> bool {
        self.infinitely_often()
            .into_iter()
            .any(|state| accepting.contains(state))
    }
}

/// Büchi acceptance condition.
///
/// A run is accepting iff it is infinite and visits at least one accepting
/// state infinitely often.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buchi<S: Eq + Hash> {
    accepting: Set<S>,
}

impl<S> Buchi<S>
where
    S: Eq + Hash,
{
    /// Creates a Büchi condition with the given accepting states.
    #[must_use]
    #[inline]
    pub fn new(accepting: Set<S>) -> Self {
        Self { accepting }
    }

    /// Returns the accepting states.
    #[must_use]
    #[inline]
    pub fn accepting(&self) -> &Set<S> {
        &self.accepting
    }
}

impl<S> From<Set<S>> for Buchi<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(accepting: Set<S>) -> Self {
        Self::new(accepting)
    }
}

impl<S> Acceptor for Buchi<S>
where
    S: Eq + Hash,
{
    type Summary = dyn BuchiSummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        summary.visits_accepting_infinitely_often(&self.accepting)
    }
}

impl<S> OmegaAcceptor for Buchi<S> where S: Eq + Hash {}
