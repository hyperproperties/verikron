use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::automata::{
    acceptors::{Acceptor, OmegaAcceptor},
    infinite_summary::InfiniteStateSummary,
};

/// Parity acceptance convention.
///
/// The convention chooses whether the minimum or maximum priority seen
/// infinitely often is relevant, and whether acceptance requires that
/// priority to be even or odd.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityConvention {
    MinEven,
    MinOdd,
    MaxEven,
    MaxOdd,
}

impl ParityConvention {
    /// Returns whether this convention uses the minimum infinitely-often
    /// priority rather than the maximum.
    #[must_use]
    #[inline]
    pub fn uses_min(self) -> bool {
        matches!(self, Self::MinEven | Self::MinOdd)
    }

    /// Returns whether `priority` is accepting under this convention.
    #[must_use]
    #[inline]
    pub fn accepts_priority(self, priority: usize) -> bool {
        match self {
            Self::MinEven | Self::MaxEven => priority.is_multiple_of(2),
            Self::MinOdd | Self::MaxOdd => !priority.is_multiple_of(2),
        }
    }
}

/// A summary that is sufficient to evaluate parity acceptance.
pub trait ParitySummary {
    type State: Eq + Hash;

    /// Returns the extremal priority seen infinitely often under `convention`.
    ///
    /// Returns `None` if no state is seen infinitely often or if some
    /// infinitely-often state has no assigned priority.
    fn extremal_priority(
        &self,
        priorities: &FxHashMap<Self::State, usize>,
        convention: ParityConvention,
    ) -> Option<usize>;
}

impl<T> ParitySummary for T
where
    T: InfiniteStateSummary,
{
    type State = T::State;

    #[inline]
    fn extremal_priority(
        &self,
        priorities: &FxHashMap<Self::State, usize>,
        convention: ParityConvention,
    ) -> Option<usize> {
        let mut inf = self
            .infinitely_often()
            .into_iter()
            .map(|state| priorities.get(state).copied());

        let first = inf.next()??;

        if convention.uses_min() {
            inf.try_fold(first, |best, priority| Some(best.min(priority?)))
        } else {
            inf.try_fold(first, |best, priority| Some(best.max(priority?)))
        }
    }
}

/// Parity acceptance condition.
///
/// A run is accepting iff it is infinite and the extremal priority among the
/// states visited infinitely often is accepting under the chosen convention.
///
/// If some infinitely-often state has no assigned priority, the run is
/// rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parity<S: Eq + Hash> {
    priorities: FxHashMap<S, usize>,
    convention: ParityConvention,
}

impl<S> Parity<S>
where
    S: Eq + Hash,
{
    /// Creates a parity condition from a priority map and convention.
    #[must_use]
    #[inline]
    pub fn new(priorities: FxHashMap<S, usize>, convention: ParityConvention) -> Self {
        Self {
            priorities,
            convention,
        }
    }

    /// Returns the priority map.
    #[must_use]
    #[inline]
    pub fn priorities(&self) -> &FxHashMap<S, usize> {
        &self.priorities
    }

    /// Returns the parity convention.
    #[must_use]
    #[inline]
    pub fn convention(&self) -> ParityConvention {
        self.convention
    }
}

impl<S> Acceptor for Parity<S>
where
    S: Eq + Hash,
{
    type Summary = dyn ParitySummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        summary
            .extremal_priority(&self.priorities, self.convention)
            .is_some_and(|priority| self.convention.accepts_priority(priority))
    }
}

impl<S> OmegaAcceptor for Parity<S> where S: Eq + Hash {}
