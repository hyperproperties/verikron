use std::{fmt::Debug, hash::Hash};

use crate::{
    automata::{
        acceptors::Acceptor,
        automaton::{Automaton, IoLabel},
        trace::Summary,
    },
    graphs::{
        backward::Backward,
        forward::Forward,
        graph::{EdgeType, Edges, Graph, VertexType},
        labeled_edges::ReadLabeledEdges,
    },
    lattices::set::Set,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buchi<S: Eq + Hash> {
    accepting: Set<S>,
}

impl<S: Eq + Hash> Buchi<S> {
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

impl<S: Eq + Hash> From<Set<S>> for Buchi<S> {
    #[inline]
    fn from(accepting: Set<S>) -> Self {
        Self::new(accepting)
    }
}

impl<S: Eq + Hash> Acceptor<S> for Buchi<S> {
    #[inline]
    fn accepts(&self, summary: &Summary<S>) -> bool {
        summary
            .infinite_states()
            .is_some_and(|states| !self.accepting.is_disjoint(states))
    }
}

impl<G> Automaton<G, Buchi<<G as VertexType>::Vertex>>
where
    G: Graph + Forward + Backward,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_buchi(
        initial: <G as VertexType>::Vertex,
        graph: G,
        accepting: Set<<G as VertexType>::Vertex>,
    ) -> Self {
        Self::new(initial, graph, Buchi::new(accepting))
    }
}
