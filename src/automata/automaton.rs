use std::{fmt::Debug, hash::Hash};

use symbol_table::Symbol;

use crate::{
    automata::{acceptors::Acceptor, trace::Summary},
    graphs::{
        backward::Backward,
        forward::Forward,
        graph::{Edges, Graph, Vertices},
        labeled_edges::ReadLabeledEdges,
    },
    lattices::set::Set,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoLabel {
    pub input: Symbol,
    pub output: Symbol,
}

#[derive(Clone, Debug)]
pub struct Automaton<G, A>
where
    G: Graph
        + Forward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>,
    <G as Graph>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as Graph>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as Graph>::Vertex>,
{
    initial: <G as Graph>::Vertex,
    graph: G,
    acceptor: A,
}

impl<G, A> Automaton<G, A>
where
    G: Graph
        + Forward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as Graph>::Vertex, Edge = <<G as Graph>::Edges as Edges>::Edge>,
    <G as Graph>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: ReadLabeledEdges<Vertex = <G as Graph>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as Graph>::Vertex>,
{
    #[inline]
    pub fn new(initial: <G as Graph>::Vertex, graph: G, acceptor: A) -> Self {
        assert!(graph.vertex_store().contains(&initial));
        Self {
            initial,
            graph,
            acceptor,
        }
    }

    #[inline]
    pub fn initial(&self) -> &<G as Graph>::Vertex {
        &self.initial
    }

    #[inline]
    pub fn graph(&self) -> &G {
        &self.graph
    }

    #[inline]
    pub fn acceptor(&self) -> &A {
        &self.acceptor
    }

    #[inline]
    pub fn accepts(&self, summary: &Summary<<G as Graph>::Vertex>) -> bool {
        self.acceptor.accepts(summary)
    }

    #[inline]
    pub fn label(&self, edge: <<G as Graph>::Edges as Edges>::Edge) -> Option<&IoLabel> {
        self.graph.edge_store().label(edge)
    }

    #[inline]
    pub fn successors(&self, vertex: <G as Graph>::Vertex) -> <G as Forward>::Successors<'_> {
        self.graph.successors(vertex)
    }

    #[inline]
    pub fn predecessors(
        &self,
        vertex: <G as Graph>::Vertex,
    ) -> <G as Backward>::Predecessors<'_> {
        self.graph.predecessors(vertex)
    }

    pub fn labeled_successors(
        &self,
        vertex: <G as Graph>::Vertex,
    ) -> impl Iterator<Item = (<G as Graph>::Vertex, IoLabel, <G as Graph>::Vertex)> + '_
    {
        self.successors(vertex)
            .filter_map(|(from, edge, to)| self.label(edge).copied().map(|label| (from, label, to)))
    }

    pub fn labeled_predecessors(
        &self,
        vertex: <G as Graph>::Vertex,
    ) -> impl Iterator<Item = (<G as Graph>::Vertex, IoLabel, <G as Graph>::Vertex)> + '_
    {
        self.predecessors(vertex)
            .filter_map(|(from, edge, to)| self.label(edge).copied().map(|label| (from, label, to)))
    }

    pub fn input_alphabet(&self) -> Set<Symbol> {
        self.graph
            .edge_store()
            .labeled_edges()
            .map(|(_, _, label, _)| label.input)
            .collect()
    }

    pub fn output_alphabet(&self) -> Set<Symbol> {
        self.graph
            .edge_store()
            .labeled_edges()
            .map(|(_, _, label, _)| label.output)
            .collect()
    }
}
