use std::{fmt::Debug, hash::Hash};

use symbol_table::Symbol;

use crate::{
    automata::{acceptors::Acceptor, trace::Summary},
    graphs::{
        backward::Backward,
        forward::Forward,
        graph::{Edges, ReadGraph, ReadVertices},
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
    G: ReadGraph
        + Forward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>,
    <G as ReadGraph>::Vertex: Eq + Hash + Debug,
    <G as ReadGraph>::Edges: ReadLabeledEdges<Vertex = <G as ReadGraph>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as ReadGraph>::Vertex>,
{
    initial: <G as ReadGraph>::Vertex,
    graph: G,
    acceptor: A,
}

impl<G, A> Automaton<G, A>
where
    G: ReadGraph
        + Forward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>
        + Backward<Vertex = <G as ReadGraph>::Vertex, Edge = <<G as ReadGraph>::Edges as Edges>::Edge>,
    <G as ReadGraph>::Vertex: Eq + Hash + Debug,
    <G as ReadGraph>::Edges: ReadLabeledEdges<Vertex = <G as ReadGraph>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as ReadGraph>::Vertex>,
{
    #[inline]
    pub fn new(initial: <G as ReadGraph>::Vertex, graph: G, acceptor: A) -> Self {
        assert!(graph.vertex_store().contains(&initial));
        Self {
            initial,
            graph,
            acceptor,
        }
    }

    #[inline]
    pub fn initial(&self) -> &<G as ReadGraph>::Vertex {
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
    pub fn accepts(&self, summary: &Summary<<G as ReadGraph>::Vertex>) -> bool {
        self.acceptor.accepts(summary)
    }

    #[inline]
    pub fn label(&self, edge: <<G as ReadGraph>::Edges as Edges>::Edge) -> Option<&IoLabel> {
        self.graph.edge_store().label(edge)
    }

    #[inline]
    pub fn successors(&self, vertex: <G as ReadGraph>::Vertex) -> <G as Forward>::Successors<'_> {
        self.graph.successors(vertex)
    }

    #[inline]
    pub fn predecessors(
        &self,
        vertex: <G as ReadGraph>::Vertex,
    ) -> <G as Backward>::Predecessors<'_> {
        self.graph.predecessors(vertex)
    }

    pub fn labeled_successors(
        &self,
        vertex: <G as ReadGraph>::Vertex,
    ) -> impl Iterator<Item = (<G as ReadGraph>::Vertex, IoLabel, <G as ReadGraph>::Vertex)> + '_
    {
        self.successors(vertex)
            .filter_map(|(from, edge, to)| self.label(edge).copied().map(|label| (from, label, to)))
    }

    pub fn labeled_predecessors(
        &self,
        vertex: <G as ReadGraph>::Vertex,
    ) -> impl Iterator<Item = (<G as ReadGraph>::Vertex, IoLabel, <G as ReadGraph>::Vertex)> + '_
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
