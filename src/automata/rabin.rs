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
pub struct RabinPair<S: Eq + Hash> {
    forbidden: Set<S>,
    required: Set<S>,
}

impl<S: Eq + Hash> RabinPair<S> {
    #[inline]
    pub fn new(forbidden: Set<S>, required: Set<S>) -> Self {
        Self {
            forbidden,
            required,
        }
    }

    #[inline]
    pub fn forbidden(&self) -> &Set<S> {
        &self.forbidden
    }

    #[inline]
    pub fn required(&self) -> &Set<S> {
        &self.required
    }

    #[inline]
    pub fn into_parts(self) -> (Set<S>, Set<S>) {
        (self.forbidden, self.required)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rabin<S: Eq + Hash> {
    pairs: Vec<RabinPair<S>>,
}

impl<S: Eq + Hash> Rabin<S> {
    #[inline]
    pub fn new(pairs: Vec<RabinPair<S>>) -> Self {
        Self { pairs }
    }

    #[inline]
    pub fn pairs(&self) -> &[RabinPair<S>] {
        &self.pairs
    }

    #[inline]
    pub fn into_pairs(self) -> Vec<RabinPair<S>> {
        self.pairs
    }
}

impl<S: Eq + Hash> From<Vec<RabinPair<S>>> for Rabin<S> {
    #[inline]
    fn from(pairs: Vec<RabinPair<S>>) -> Self {
        Self::new(pairs)
    }
}

impl<S: Eq + Hash> Acceptor<S> for Rabin<S> {
    #[inline]
    fn accepts(&self, summary: &Summary<S>) -> bool {
        summary.infinite_states().is_some_and(|states| {
            self.pairs.iter().any(|pair| {
                pair.forbidden.is_disjoint(states) && !pair.required.is_disjoint(states)
            })
        })
    }
}

impl<G> Automaton<G, Rabin<<G as VertexType>::Vertex>>
where
    G: Graph + Forward + Backward,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_rabin(
        initial: <G as VertexType>::Vertex,
        graph: G,
        pairs: Vec<RabinPair<<G as VertexType>::Vertex>>,
    ) -> Self {
        Self::new(initial, graph, Rabin::new(pairs))
    }
}
