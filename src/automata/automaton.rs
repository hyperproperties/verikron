use symbol_table::Symbol;

use crate::{
    automata::acceptors::Acceptor,
    graphs::{
        backward::Backward,
        forward::Forward,
        graph::{Directed, EdgeOf, FiniteEdges, FiniteVertices, Graph, VertexOf},
        labeled::LabeledEdges,
    },
    lattices::set::Set,
};

/// Input/output label carried by an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoLabel {
    /// Consumed input symbol.
    pub input: Symbol,
    /// Produced output symbol.
    pub output: Symbol,
}

impl IoLabel {
    /// Creates a new input/output label.
    #[must_use]
    #[inline]
    pub const fn new(input: Symbol, output: Symbol) -> Self {
        Self { input, output }
    }
}

/// Automaton over a labeled graph with an external acceptance condition.
///
/// The graph stores the transition structure and edge labels.
/// The acceptor decides acceptance from its own summary type.
#[derive(Clone)]
pub struct Automaton<G, A>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    A: Acceptor,
{
    initial: VertexOf<G>,
    graph: G,
    acceptor: A,
}

impl<G, A> Automaton<G, A>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    A: Acceptor,
{
    /// Creates an automaton without checking that `initial` belongs to `graph`.
    ///
    /// This works for both finite and infinite graphs.
    #[must_use]
    pub fn new(initial: VertexOf<G>, graph: G, acceptor: A) -> Self {
        Self {
            initial,
            graph,
            acceptor,
        }
    }

    /// Returns the initial vertex.
    #[must_use]
    #[inline]
    pub fn initial(&self) -> VertexOf<G> {
        self.initial
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// Returns the acceptance condition.
    #[must_use]
    #[inline]
    pub fn acceptor(&self) -> &A {
        &self.acceptor
    }

    /// Consumes the automaton and returns `(initial, graph, acceptor)`.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (VertexOf<G>, G, A) {
        (self.initial, self.graph, self.acceptor)
    }

    /// Delegates acceptance of `summary` to the underlying acceptor.
    #[must_use]
    #[inline]
    pub fn accepts(&self, summary: &A::Summary) -> bool {
        self.acceptor.accept(summary)
    }

    /// Returns the endpoints of `edge`.
    #[must_use]
    #[inline]
    pub fn endpoints(&self, edge: EdgeOf<G>) -> (VertexOf<G>, VertexOf<G>) {
        (self.graph.source(edge), self.graph.target(edge))
    }

    /// Returns the label of `edge`, if present.
    #[must_use]
    #[inline]
    pub fn label(&self, edge: EdgeOf<G>) -> Option<&IoLabel> {
        self.graph.edge_store().label(edge)
    }

    /// Returns the labeled edge `(from, label, to)`, if `edge` is labeled.
    #[must_use]
    #[inline]
    pub fn labeled_edge(&self, edge: EdgeOf<G>) -> Option<(VertexOf<G>, IoLabel, VertexOf<G>)> {
        let (from, to) = self.endpoints(edge);
        self.label(edge).copied().map(|label| (from, label, to))
    }

    /// Returns the outgoing edges of `vertex`.
    #[inline]
    pub fn successors(&self, vertex: VertexOf<G>) -> <G as Forward>::Successors<'_> {
        self.graph.successors(vertex)
    }

    /// Returns the incoming edges of `vertex`.
    #[inline]
    pub fn predecessors(&self, vertex: VertexOf<G>) -> <G as Backward>::Predecessors<'_> {
        self.graph.predecessors(vertex)
    }

    /// Returns labeled outgoing edges of `vertex`.
    ///
    /// Unlabeled edges are skipped.
    #[inline]
    pub fn labeled_successors(
        &self,
        vertex: VertexOf<G>,
    ) -> impl Iterator<Item = (VertexOf<G>, IoLabel, VertexOf<G>)> + '_ {
        self.successors(vertex)
            .filter_map(|(from, edge, to)| self.label(edge).copied().map(|label| (from, label, to)))
    }

    /// Returns labeled incoming edges of `vertex`.
    ///
    /// Unlabeled edges are skipped.
    #[inline]
    pub fn labeled_predecessors(
        &self,
        vertex: VertexOf<G>,
    ) -> impl Iterator<Item = (VertexOf<G>, IoLabel, VertexOf<G>)> + '_ {
        self.predecessors(vertex)
            .filter_map(|(from, edge, to)| self.label(edge).copied().map(|label| (from, label, to)))
    }
}

impl<G, A> Automaton<G, A>
where
    G: Graph + Forward + Backward + Directed,
    G::Vertices: FiniteVertices<Vertex = VertexOf<G>>,
    G::Edges: LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    A: Acceptor,
{
    /// Creates an automaton and checks that `initial` belongs to `graph`.
    ///
    /// Panics if `initial` is not a vertex of `graph`.
    #[must_use]
    pub fn new_checked(initial: VertexOf<G>, graph: G, acceptor: A) -> Self {
        assert!(
            graph.vertex_store().contains(&initial),
            "initial vertex must belong to the graph",
        );
        Self::new(initial, graph, acceptor)
    }

    /// Creates an automaton if `initial` belongs to `graph`.
    #[must_use]
    pub fn try_new(initial: VertexOf<G>, graph: G, acceptor: A) -> Option<Self> {
        graph
            .vertex_store()
            .contains(&initial)
            .then(|| Self::new(initial, graph, acceptor))
    }

    /// Returns whether `vertex` belongs to the graph.
    #[must_use]
    #[inline]
    pub fn contains_vertex(&self, vertex: VertexOf<G>) -> bool {
        self.graph.vertex_store().contains(&vertex)
    }

    /// Returns whether the initial vertex belongs to the graph.
    #[must_use]
    #[inline]
    pub fn has_valid_initial(&self) -> bool {
        self.contains_vertex(self.initial)
    }
}

impl<G, A> Automaton<G, A>
where
    G: Graph + Forward + Backward + Directed,
    G::Edges: FiniteEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>>
        + LabeledEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>, Label = IoLabel>,
    A: Acceptor,
{
    /// Returns the set of labels appearing on labeled edges.
    #[must_use]
    pub fn alphabet(&self) -> Set<IoLabel> {
        self.graph
            .edge_store()
            .labeled_edges()
            .map(|(_, _, label, _)| *label)
            .collect()
    }

    /// Returns the set of input symbols appearing on labeled edges.
    #[must_use]
    pub fn input_alphabet(&self) -> Set<Symbol> {
        self.graph
            .edge_store()
            .labeled_edges()
            .map(|(_, _, label, _)| label.input)
            .collect()
    }

    /// Returns the set of output symbols appearing on labeled edges.
    #[must_use]
    pub fn output_alphabet(&self) -> Set<Symbol> {
        self.graph
            .edge_store()
            .labeled_edges()
            .map(|(_, _, label, _)| label.output)
            .collect()
    }
}
