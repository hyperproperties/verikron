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
        graph::{EdgeType, Graph, VertexType},
        labeled_edges::LabeledEdges,
    },
    lattices::set::Set,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoBuchi<S: Eq + Hash> {
    rejecting: Set<S>,
}

impl<S: Eq + Hash> CoBuchi<S> {
    #[inline]
    pub fn new(rejecting: Set<S>) -> Self {
        Self { rejecting }
    }

    #[inline]
    pub fn rejecting(&self) -> &Set<S> {
        &self.rejecting
    }

    #[inline]
    pub fn into_rejecting(self) -> Set<S> {
        self.rejecting
    }
}

impl<S: Eq + Hash> From<Set<S>> for CoBuchi<S> {
    #[inline]
    fn from(rejecting: Set<S>) -> Self {
        Self::new(rejecting)
    }
}

impl<S: Eq + Hash> Acceptor<S> for CoBuchi<S> {
    #[inline]
    fn accepts(&self, summary: &Summary<S>) -> bool {
        summary
            .infinite_states()
            .is_some_and(|states| self.rejecting.is_disjoint(states))
    }
}

impl<G> Automaton<G, CoBuchi<<G as VertexType>::Vertex>>
where
    G: Graph + Forward + Backward,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>,
    <G as Graph>::Edges: LabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_co_buchi(
        initial: <G as VertexType>::Vertex,
        graph: G,
        rejecting: Set<<G as VertexType>::Vertex>,
    ) -> Self {
        Self::new(initial, graph, CoBuchi::new(rejecting))
    }
}
