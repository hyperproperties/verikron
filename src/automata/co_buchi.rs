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

impl<G> Automaton<G, CoBuchi<<G as ReadGraph>::Vertex>>
where
    G: ReadGraph
        + Forward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>,
    <G as ReadGraph>::Vertex: Eq + Hash + Debug,
    <G as ReadGraph>::Edges: ReadLabeledEdges<Vertex = <G as ReadGraph>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_co_buchi(
        initial: <G as ReadGraph>::Vertex,
        graph: G,
        rejecting: Set<<G as ReadGraph>::Vertex>,
    ) -> Self {
        Self::new(initial, graph, CoBuchi::new(rejecting))
    }
}
