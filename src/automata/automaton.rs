use std::{fmt::Debug, hash::Hash};

use symbol_table::Symbol;

use crate::{
    automata::{acceptors::Acceptor, trace::Summary},
    graphs::{
        backward::Backward,
        forward::Forward,
        graph::{EdgeType, FiniteEdges, FiniteVertices, Graph, VertexType},
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
    G: Graph + Forward + Backward,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>
        + ReadLabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as VertexType>::Vertex>,
{
    initial: <G as VertexType>::Vertex,
    graph: G,
    acceptor: A,
}

impl<G, A> Automaton<G, A>
where
    G: Graph + Forward + Backward,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>
        + ReadLabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as VertexType>::Vertex>,
{
    /// Creates an automaton without checking whether `initial` belongs to the graph.
    ///
    /// This constructor works for both finite and infinite graphs.
    #[inline]
    pub fn new(initial: <G as VertexType>::Vertex, graph: G, acceptor: A) -> Self {
        Self {
            initial,
            graph,
            acceptor,
        }
    }

    #[inline]
    pub fn initial(&self) -> &<G as VertexType>::Vertex {
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
    pub fn accepts(&self, summary: &Summary<<G as VertexType>::Vertex>) -> bool {
        self.acceptor.accepts(summary)
    }

    #[inline]
    pub fn label(&self, edge: <G as EdgeType>::Edge) -> Option<&IoLabel> {
        self.graph.edge_store().label(edge)
    }

    #[inline]
    pub fn successors(&self, vertex: <G as VertexType>::Vertex) -> <G as Forward>::Successors<'_> {
        self.graph.successors(vertex)
    }

    #[inline]
    pub fn predecessors(
        &self,
        vertex: <G as VertexType>::Vertex,
    ) -> <G as Backward>::Predecessors<'_> {
        self.graph.predecessors(vertex)
    }

    #[inline]
    pub fn labeled_successors(
        &self,
        vertex: <G as VertexType>::Vertex,
    ) -> impl Iterator<
        Item = (
            <G as VertexType>::Vertex,
            IoLabel,
            <G as VertexType>::Vertex,
        ),
    > + '_ {
        self.successors(vertex)
            .filter_map(|(from, edge, to)| self.label(edge).copied().map(|label| (from, label, to)))
    }

    #[inline]
    pub fn labeled_predecessors(
        &self,
        vertex: <G as VertexType>::Vertex,
    ) -> impl Iterator<
        Item = (
            <G as VertexType>::Vertex,
            IoLabel,
            <G as VertexType>::Vertex,
        ),
    > + '_ {
        self.predecessors(vertex)
            .filter_map(|(from, edge, to)| self.label(edge).copied().map(|label| (from, label, to)))
    }
}

impl<G, A> Automaton<G, A>
where
    G: Graph + Forward + Backward,
    G::Vertices: FiniteVertices<Vertex = <G as VertexType>::Vertex>,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>
        + ReadLabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as VertexType>::Vertex>,
{
    /// Creates an automaton and checks that `initial` belongs to the graph.
    #[inline]
    pub fn new_checked(initial: <G as VertexType>::Vertex, graph: G, acceptor: A) -> Self {
        assert!(graph.vertex_store().contains(&initial));
        Self::new(initial, graph, acceptor)
    }

    /// Creates an automaton if `initial` belongs to the graph.
    #[inline]
    pub fn try_new(initial: <G as VertexType>::Vertex, graph: G, acceptor: A) -> Option<Self> {
        graph
            .vertex_store()
            .contains(&initial)
            .then(|| Self::new(initial, graph, acceptor))
    }
}

impl<G, A> Automaton<G, A>
where
    G: Graph + Forward + Backward,
    G::Edges: FiniteEdges<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>,
    <G as VertexType>::Vertex: Eq + Hash + Debug,
    <G as Graph>::Edges: EdgeType<Vertex = <G as VertexType>::Vertex, Edge = <G as EdgeType>::Edge>
        + ReadLabeledEdges<Vertex = <G as VertexType>::Vertex, Label = IoLabel>,
    A: Acceptor<<G as VertexType>::Vertex>,
{
    /// Returns the set of input symbols appearing on edges.
    #[inline]
    pub fn input_alphabet(&self) -> Set<Symbol> {
        self.graph
            .edge_store()
            .labeled_edges()
            .map(|(_, _, label, _)| label.input)
            .collect()
    }

    /// Returns the set of output symbols appearing on edges.
    #[inline]
    pub fn output_alphabet(&self) -> Set<Symbol> {
        self.graph
            .edge_store()
            .labeled_edges()
            .map(|(_, _, label, _)| label.output)
            .collect()
    }
}
