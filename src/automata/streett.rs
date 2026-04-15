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
pub struct StreettPair<S: Eq + Hash> {
    trigger: Set<S>,
    guarantee: Set<S>,
}

impl<S: Eq + Hash> StreettPair<S> {
    #[inline]
    pub fn new(trigger: Set<S>, guarantee: Set<S>) -> Self {
        Self { trigger, guarantee }
    }

    #[inline]
    pub fn trigger(&self) -> &Set<S> {
        &self.trigger
    }

    #[inline]
    pub fn guarantee(&self) -> &Set<S> {
        &self.guarantee
    }

    #[inline]
    pub fn into_parts(self) -> (Set<S>, Set<S>) {
        (self.trigger, self.guarantee)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Streett<S: Eq + Hash> {
    pairs: Vec<StreettPair<S>>,
}

impl<S: Eq + Hash> Streett<S> {
    #[inline]
    pub fn new(pairs: Vec<StreettPair<S>>) -> Self {
        Self { pairs }
    }

    #[inline]
    pub fn pairs(&self) -> &[StreettPair<S>] {
        &self.pairs
    }

    #[inline]
    pub fn into_pairs(self) -> Vec<StreettPair<S>> {
        self.pairs
    }
}

impl<S: Eq + Hash> From<Vec<StreettPair<S>>> for Streett<S> {
    #[inline]
    fn from(pairs: Vec<StreettPair<S>>) -> Self {
        Self::new(pairs)
    }
}

impl<S: Eq + Hash> Acceptor<S> for Streett<S> {
    #[inline]
    fn accepts(&self, summary: &Summary<S>) -> bool {
        summary.infinite_states().is_some_and(|states| {
            self.pairs
                .iter()
                .all(|pair| pair.trigger.is_disjoint(states) || !pair.guarantee.is_disjoint(states))
        })
    }
}

impl<G> Automaton<G, Streett<<G as Graph>::Vertex>>
where
    G: Graph
        + Forward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>,
    <G as Graph>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as Graph>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_streett(
        initial: <G as Graph>::Vertex,
        graph: G,
        pairs: Vec<StreettPair<<G as Graph>::Vertex>>,
    ) -> Self {
        Self::new(initial, graph, Streett::new(pairs))
    }
}
