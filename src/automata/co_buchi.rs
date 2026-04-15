use std::hash::Hash;

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

    /// Consumes `self` and returns the rejecting states.
    #[must_use]
    #[inline]
    pub fn into_rejecting(self) -> Set<S> {
        self.rejecting
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

impl<G> Automaton<G, CoBuchi<VertexOf<G>>>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    VertexOf<G>: Eq + Hash,
{
    /// Creates an automaton with co-Büchi acceptance.
    #[must_use]
    #[inline]
    pub fn with_co_buchi(initial: VertexOf<G>, graph: G, rejecting: Set<VertexOf<G>>) -> Self {
        Self::new(initial, graph, CoBuchi::new(rejecting))
    }
}
