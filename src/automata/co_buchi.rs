use std::hash::Hash;

use crate::{
    automata::{
        acceptors::{Acceptor, OmegaAcceptor},
        infinite_summary::InfiniteStateSummary,
    },
    lattices::set::Set,
};

/// A summary that is sufficient to evaluate co-Büchi acceptance.
pub trait CoBuchiSummary {
    type State: Eq + Hash;

    fn visits_no_rejecting_state_infinitely_often(&self, rejecting: &Set<Self::State>) -> bool;
}

impl<T> CoBuchiSummary for T
where
    T: InfiniteStateSummary,
{
    type State = T::State;

    #[inline]
    fn visits_no_rejecting_state_infinitely_often(&self, rejecting: &Set<Self::State>) -> bool {
        self.infinitely_often()
            .into_iter()
            .all(|state| !rejecting.contains(state))
    }
}

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
    type Summary = dyn CoBuchiSummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        summary.visits_no_rejecting_state_infinitely_often(&self.rejecting)
    }
}

impl<S> OmegaAcceptor for CoBuchi<S> where S: Eq + Hash {}
