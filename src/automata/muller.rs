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

    /// Consumes `self` and returns the accepting families.
    #[must_use]
    #[inline]
    pub fn into_families(self) -> Vec<Set<S>> {
        self.families
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
    type Summary = StateSummary<S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        match summary {
            StateSummary::Finite { .. } => false,
            StateSummary::Infinite { states } => {
                self.families.iter().any(|family| family == states)
            }
        }
    }
}

impl<G> Automaton<G, Muller<VertexOf<G>>>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    VertexOf<G>: Eq + Hash,
{
    /// Creates an automaton with Muller acceptance.
    #[must_use]
    #[inline]
    pub fn with_muller(initial: VertexOf<G>, graph: G, families: Vec<Set<VertexOf<G>>>) -> Self {
        Self::new(initial, graph, Muller::new(families))
    }
}
