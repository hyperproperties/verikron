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
        graph::{Edges, Graph},
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

impl<G> Automaton<G, Buchi<<G as Graph>::Vertex>>
where
    G: Graph
        + Forward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>,
    <G as Graph>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as Graph>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_buchi(
        initial: <G as Graph>::Vertex,
        graph: G,
        accepting: Set<<G as Graph>::Vertex>,
    ) -> Self {
        Self::new(initial, graph, Buchi::new(accepting))
    }
}
