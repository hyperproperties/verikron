use std::{fmt::Debug, hash::Hash};

use crate::{
    automata::{
        acceptors::Acceptor,
        automaton::{Automaton, IoLabel},
        trace::Summary,
    },
    graphs::{
        backward::Backward, edges::Edges, forward::Forward, graph::ReadGraph,
        labeled_edges::ReadLabeledEdges,
    },
    lattices::set::Set,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Final<S: Eq + Hash> {
    accepting: Set<S>,
}

impl<S: Eq + Hash> Final<S> {
    #[inline]
    pub fn new(accepting: Set<S>) -> Self {
        Self { accepting }
    }

    #[inline]
    pub fn accepting(&self) -> &Set<S> {
        &self.accepting
    }

    #[inline]
    pub fn into_accepting(self) -> Set<S> {
        self.accepting
    }
}

impl<S: Eq + Hash> From<Set<S>> for Final<S> {
    #[inline]
    fn from(accepting: Set<S>) -> Self {
        Self::new(accepting)
    }
}

impl<S: Eq + Hash> Acceptor<S> for Final<S> {
    #[inline]
    fn accepts(&self, summary: &Summary<S>) -> bool {
        summary
            .terminal()
            .is_some_and(|q| self.accepting.contains(q))
    }
}

impl<G> Automaton<G, Final<<G as ReadGraph>::Vertex>>
where
    G: ReadGraph
        + Forward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>,
    <G as ReadGraph>::Vertex: Eq + Hash + Debug,
    <G as ReadGraph>::Edges: ReadLabeledEdges<Vertex = <G as ReadGraph>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_final(
        initial: <G as ReadGraph>::Vertex,
        graph: G,
        accepting: Set<<G as ReadGraph>::Vertex>,
    ) -> Self {
        Self::new(initial, graph, Final::new(accepting))
    }
}
