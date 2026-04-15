use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::{
    automata::{
        acceptors::{Acceptor, StateSummary},
        automaton::{Automaton, IoLabel},
    },
    graphs::{
        backward::Backward,
        forward::Forward,
        graph::{Directed, EdgeOf, Graph, VertexOf}, labeled::LabeledEdges,
    },
    lattices::set::Set,
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

    /// Consumes `self` and returns the priority map and convention.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (FxHashMap<S, usize>, ParityConvention) {
        (self.priorities, self.convention)
    }

    /// Returns the extremal priority of `states` under this convention.
    ///
    /// Returns `None` if `states` is empty or if some state has no priority.
    #[must_use]
    fn extremal_priority(&self, states: &Set<S>) -> Option<usize> {
        let mut priorities = states
            .iter()
            .map(|state| self.priorities.get(state).copied());

        let first = priorities.next()??;
        let extremal = if self.convention.uses_min() {
            priorities.try_fold(first, |best, priority| Some(best.min(priority?)))
        } else {
            priorities.try_fold(first, |best, priority| Some(best.max(priority?)))
        };

        extremal
    }
}

impl<S> Acceptor for Parity<S>
where
    S: Eq + Hash,
{
    type Summary = StateSummary<S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        match summary {
            StateSummary::Finite { .. } => false,
            StateSummary::Infinite { states } => self
                .extremal_priority(states)
                .is_some_and(|priority| self.convention.accepts_priority(priority)),
        }
    }
}

impl<G> Automaton<G, Parity<VertexOf<G>>>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    VertexOf<G>: Eq + Hash,
{
    /// Creates an automaton with parity acceptance.
    #[must_use]
    #[inline]
    pub fn with_parity(
        initial: VertexOf<G>,
        graph: G,
        priorities: FxHashMap<VertexOf<G>, usize>,
        convention: ParityConvention,
    ) -> Self {
        Self::new(initial, graph, Parity::new(priorities, convention))
    }
}