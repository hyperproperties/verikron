use std::hash::Hash;

use crate::{
    automata::{
        acceptors::{Acceptor, StateSummary},
        automaton::{Automaton, IoLabel},
    },
    graphs::{
        backward::Backward,
        forward::Forward,
        graph::{Directed, EdgeOf, Graph, VertexOf},
        labeled::LabeledEdges,
    },
    lattices::set::Set,
};

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

    /// Consumes `self` and returns the accepting states.
    #[must_use]
    #[inline]
    pub fn into_accepting(self) -> Set<S> {
        self.accepting
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
    type Summary = StateSummary<S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        match summary {
            StateSummary::Finite { .. } => false,
            StateSummary::Infinite { states } => !self.accepting.is_disjoint(states),
        }
    }
}

impl<G> Automaton<G, Buchi<VertexOf<G>>>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    VertexOf<G>: Eq + Hash,
{
    /// Creates an automaton with Büchi acceptance.
    #[must_use]
    #[inline]
    pub fn with_buchi(initial: VertexOf<G>, graph: G, accepting: Set<VertexOf<G>>) -> Self {
        Self::new(initial, graph, Buchi::new(accepting))
    }
}
