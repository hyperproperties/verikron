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
pub struct Muller<S: Eq + Hash> {
    families: Vec<Set<S>>,
}

impl<S: Eq + Hash> Muller<S> {
    #[inline]
    pub fn new(families: Vec<Set<S>>) -> Self {
        Self { families }
    }

    #[inline]
    pub fn families(&self) -> &[Set<S>] {
        &self.families
    }

    #[inline]
    pub fn into_families(self) -> Vec<Set<S>> {
        self.families
    }
}

impl<S: Eq + Hash> From<Vec<Set<S>>> for Muller<S> {
    #[inline]
    fn from(families: Vec<Set<S>>) -> Self {
        Self::new(families)
    }
}

impl<S: Eq + Hash> Acceptor<S> for Muller<S> {
    #[inline]
    fn accepts(&self, summary: &Summary<S>) -> bool {
        summary
            .infinite_states()
            .is_some_and(|states| self.families.iter().any(|family| family == states))
    }
}

impl<G> Automaton<G, Muller<<G as Graph>::Vertex>>
where
    G: Graph
        + Forward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>,
    <G as Graph>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as Graph>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_muller(
        initial: <G as Graph>::Vertex,
        graph: G,
        families: Vec<Set<<G as Graph>::Vertex>>,
    ) -> Self {
        Self::new(initial, graph, Muller::new(families))
    }
}
