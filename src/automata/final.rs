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

/// Final-state acceptance condition.
///
/// A run is accepting iff it is finite and its terminal state is accepting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Final<S: Eq + Hash> {
    accepting: Set<S>,
}

impl<S> Final<S>
where
    S: Eq + Hash,
{
    /// Creates a final-state condition with the given accepting states.
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

impl<S> From<Set<S>> for Final<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(accepting: Set<S>) -> Self {
        Self::new(accepting)
    }
}

impl<S> Acceptor for Final<S>
where
    S: Eq + Hash,
{
    type Summary = StateSummary<S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        match summary {
            StateSummary::Finite { terminal } => self.accepting.contains(terminal),
            StateSummary::Infinite { .. } => false,
        }
    }
}

impl<G> Automaton<G, Final<VertexOf<G>>>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    VertexOf<G>: Eq + Hash,
{
    /// Creates an automaton with final-state acceptance.
    #[must_use]
    #[inline]
    pub fn with_final(initial: VertexOf<G>, graph: G, accepting: Set<VertexOf<G>>) -> Self {
        Self::new(initial, graph, Final::new(accepting))
    }
}
