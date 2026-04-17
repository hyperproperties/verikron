use symbol_table::Symbol;

use crate::{
    automata::acceptors::Acceptor,
    graphs::{
        attributed::AttributedGraph,
        backward::Backward,
        forward::Forward,
        graph::{Directed, Graph},
        properties::Properties,
        structure::{EdgeOf, EdgeType, FiniteVertices, Structure, VertexOf, VertexType},
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

/// Automaton over a directed transition structure with edge labels stored as
/// edge properties in an attributed graph.
///
/// The underlying graph provides the transition structure.
/// The edge-property store provides [`IoLabel`] values.
/// The alphabet is declared explicitly, so it is available even when the
/// transition structure is infinite.
/// The acceptor decides acceptance from its own summary type.
#[derive(Clone, PartialEq, Eq)]
pub struct Automaton<G, VP, EP, A>
where
    G: Structure + Forward + Backward + Directed,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
    initial: VertexOf<G>,
    graph: AttributedGraph<G, VP, EP>,
    alphabet: Set<IoLabel>,
    acceptor: A,
}

impl<G, VP, EP, A> Automaton<G, VP, EP, A>
where
    G: Structure + Forward + Backward + Directed,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
    /// Creates an automaton without checking that `initial` belongs to `graph`.
    ///
    /// This works for both finite and infinite structures.
    #[must_use]
    #[inline]
    pub fn new(
        initial: VertexOf<G>,
        graph: AttributedGraph<G, VP, EP>,
        alphabet: Set<IoLabel>,
        acceptor: A,
    ) -> Self {
        Self {
            initial,
            graph,
            alphabet,
            acceptor,
        }
    }

    /// Returns the initial vertex.
    #[must_use]
    #[inline]
    pub fn initial(&self) -> VertexOf<G> {
        self.initial
    }

    /// Returns the declared label alphabet.
    #[must_use]
    #[inline]
    pub fn alphabet(&self) -> &Set<IoLabel> {
        &self.alphabet
    }

    /// Returns the declared input alphabet.
    #[must_use]
    #[inline]
    pub fn input_alphabet(&self) -> Set<Symbol> {
        self.alphabet.iter().map(|label| label.input).collect()
    }

    /// Returns the declared output alphabet.
    #[must_use]
    #[inline]
    pub fn output_alphabet(&self) -> Set<Symbol> {
        self.alphabet.iter().map(|label| label.output).collect()
    }

    /// Delegates acceptance of `summary` to the underlying acceptor.
    #[must_use]
    #[inline]
    pub fn accepts(&self, summary: &A::Summary) -> bool {
        self.acceptor.accept(summary)
    }

    /// Returns the label of `edge`, if present.
    #[must_use]
    #[inline]
    pub fn label(&self, edge: EdgeOf<G>) -> Option<&IoLabel> {
        self.graph.edge_properties().property(edge)
    }
}

impl<G, VP, EP, A> Automaton<G, VP, EP, A>
where
    G: Structure + Forward + Backward + Directed,
    G::Vertices: FiniteVertices<Vertex = VertexOf<G>>,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
    /// Creates an automaton and checks that `initial` belongs to `graph`.
    ///
    /// Panics if `initial` is not a vertex of `graph`.
    #[must_use]
    #[inline]
    pub fn new_checked(
        initial: VertexOf<G>,
        graph: AttributedGraph<G, VP, EP>,
        alphabet: Set<IoLabel>,
        acceptor: A,
    ) -> Self {
        assert!(
            graph.vertex_store().contains(&initial),
            "initial vertex must belong to the graph",
        );
        Self::new(initial, graph, alphabet, acceptor)
    }

    /// Creates an automaton if `initial` belongs to `graph`.
    #[must_use]
    #[inline]
    pub fn try_checked(
        initial: VertexOf<G>,
        graph: AttributedGraph<G, VP, EP>,
        alphabet: Set<IoLabel>,
        acceptor: A,
    ) -> Option<Self> {
        graph
            .vertex_store()
            .contains(&initial)
            .then(|| Self::new(initial, graph, alphabet, acceptor))
    }

    /// Returns whether `vertex` belongs to the transition structure.
    #[must_use]
    #[inline]
    pub fn contains_vertex(&self, vertex: VertexOf<G>) -> bool {
        self.graph.vertex_store().contains(&vertex)
    }

    /// Returns whether the initial vertex belongs to the transition structure.
    #[must_use]
    #[inline]
    pub fn has_valid_initial(&self) -> bool {
        self.contains_vertex(self.initial)
    }
}

impl<G, VP, EP, A> VertexType for Automaton<G, VP, EP, A>
where
    G: Structure + Forward + Backward + Directed,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
    type Vertex = VertexOf<G>;
}

impl<G, VP, EP, A> EdgeType for Automaton<G, VP, EP, A>
where
    G: Structure + Forward + Backward + Directed,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
    type Edge = EdgeOf<G>;
}

impl<G, VP, EP, A> Structure for Automaton<G, VP, EP, A>
where
    G: Structure + Forward + Backward + Directed,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
    type Vertices = G::Vertices;
    type Edges = G::Edges;

    #[inline]
    fn vertex_store(&self) -> &Self::Vertices {
        self.graph.vertex_store()
    }

    #[inline]
    fn edge_store(&self) -> &Self::Edges {
        self.graph.edge_store()
    }
}

impl<G, VP, EP, A> Graph for Automaton<G, VP, EP, A>
where
    G: Graph + Forward + Backward + Directed,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
}

impl<G, VP, EP, A> Directed for Automaton<G, VP, EP, A>
where
    G: Structure + Forward + Backward + Directed,
    EP: Properties<Key = EdgeOf<G>, Property = IoLabel>,
    A: Acceptor,
{
    type Outgoing<'a>
        = <G as Directed>::Outgoing<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a,
        A: 'a;

    type Ingoing<'a>
        = <G as Directed>::Ingoing<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a,
        A: 'a;

    type Connections<'a>
        = <G as Directed>::Connections<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a,
        A: 'a;

    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.graph().source(edge)
    }

    #[inline]
    fn destination(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.graph().destination(edge)
    }

    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        self.graph.graph().outgoing(source)
    }

    #[inline]
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        self.graph.graph().ingoing(destination)
    }

    #[inline]
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        self.graph.graph().connections(from, to)
    }
}
