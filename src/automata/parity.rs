use std::{fmt::Debug, hash::Hash};

use rustc_hash::FxHashMap;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityConvention {
    MinEven,
    MinOdd,
    MaxEven,
    MaxOdd,
}

impl ParityConvention {
    #[inline]
    fn uses_min(self) -> bool {
        matches!(self, Self::MinEven | Self::MinOdd)
    }

    #[inline]
    fn accepts_priority(self, priority: usize) -> bool {
        match self {
            Self::MinEven | Self::MaxEven => priority % 2 == 0,
            Self::MinOdd | Self::MaxOdd => priority % 2 == 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parity<S: Eq + Hash> {
    priorities: FxHashMap<S, usize>,
    convention: ParityConvention,
}

impl<S: Eq + Hash> Parity<S> {
    #[inline]
    pub fn new(priorities: FxHashMap<S, usize>, convention: ParityConvention) -> Self {
        Self {
            priorities,
            convention,
        }
    }

    #[inline]
    pub fn priorities(&self) -> &FxHashMap<S, usize> {
        &self.priorities
    }

    #[inline]
    pub fn convention(&self) -> ParityConvention {
        self.convention
    }

    fn extremal_priority(&self, states: &Set<S>) -> Option<usize> {
        let mut iter = states.iter();
        let first = iter.next()?;
        let mut best = *self.priorities.get(first)?;

        for state in iter {
            let p = *self.priorities.get(state)?;
            if self.convention.uses_min() {
                best = best.min(p);
            } else {
                best = best.max(p);
            }
        }

        Some(best)
    }
}

impl<S: Eq + Hash> Acceptor<S> for Parity<S> {
    #[inline]
    fn accepts(&self, summary: &Summary<S>) -> bool {
        summary.infinite_states().is_some_and(|states| {
            self.extremal_priority(states)
                .is_some_and(|p| self.convention.accepts_priority(p))
        })
    }
}

impl<G> Automaton<G, Parity<<G as VertexType>::Vertex>>
where
    G: Graph + Forward + Backward,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
{
    #[inline]
    pub fn with_parity(
        initial: <G as VertexType>::Vertex,
        graph: G,
        priorities: FxHashMap<<G as VertexType>::Vertex, usize>,
        convention: ParityConvention,
    ) -> Self {
        Self::new(initial, graph, Parity::new(priorities, convention))
    }
}
