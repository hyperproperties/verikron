use crate::graphs::{
    graph::{FiniteGraph, Graph},
    structure::{EdgeOf, EdgeType, FiniteEdges, FiniteVertices, Structure},
};

/// Hyperedge identifier type of `H`.
pub type HyperedgeOf<H> = EdgeOf<H>;

/// Undirected hypergraph interface based on local exploration.
///
/// Suitable for finite, infinite, or implicit hypergraphs.
pub trait UndirectedHypergraph: Graph {
    /// Iterator over the member vertices of a hyperedge.
    type Members<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over the hyperedges incident to a vertex.
    type Incident<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns the member vertices of `hyperedge`.
    fn members(&self, hyperedge: Self::Edge) -> Self::Members<'_>;

    /// Returns the hyperedges incident to `vertex`.
    fn incident(&self, vertex: Self::Vertex) -> Self::Incident<'_>;
}

/// Finite undirected hypergraph.
///
/// Extends [`UndirectedHypergraph`] with convenience queries.
pub trait FiniteUndirectedHypergraph: UndirectedHypergraph + FiniteGraph
where
    <Self as Structure>::Vertices: FiniteVertices<Vertex = Self::Vertex>,
    <Self as Structure>::Edges: FiniteEdges<Vertex = Self::Vertex, Edge = Self::Edge>,
{
    /// Returns the number of members of `hyperedge`.
    fn cardinality(&self, hyperedge: Self::Edge) -> usize {
        self.members(hyperedge).count()
    }

    /// Returns the number of incident hyperedges of `vertex`.
    fn degree(&self, vertex: Self::Vertex) -> usize {
        self.incident(vertex).count()
    }

    /// Returns whether `vertex` is a member of `hyperedge`.
    fn contains_member(&self, hyperedge: Self::Edge, vertex: Self::Vertex) -> bool {
        self.members(hyperedge).any(|u| u == vertex)
    }
}

impl<T> FiniteUndirectedHypergraph for T
where
    T: UndirectedHypergraph + FiniteGraph,
    <T as Structure>::Vertices: FiniteVertices<Vertex = T::Vertex>,
    <T as Structure>::Edges: FiniteEdges<Vertex = T::Vertex, Edge = T::Edge>,
{
}

/// Undirected hyperedge insertion.
pub trait InsertUndirectedHyperedge: EdgeType {
    /// Inserts a hyperedge from its member vertices.
    fn insert_hyperedge<I>(&mut self, members: I) -> Option<Self::Edge>
    where
        I: IntoIterator<Item = Self::Vertex>;
}

/// Directed hypergraph interface based on local exploration.
///
/// Suitable for finite, infinite, or implicit hypergraphs.
pub trait DirectedHypergraph: Graph {
    /// Iterator over the tail vertices of a hyperedge.
    type Tail<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over the head vertices of a hyperedge.
    type Head<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over hyperedges whose tail contains `vertex`.
    type Outgoing<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Iterator over hyperedges whose head contains `vertex`.
    type Ingoing<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns the tail vertices of `hyperedge`.
    fn tail(&self, hyperedge: Self::Edge) -> Self::Tail<'_>;

    /// Returns the head vertices of `hyperedge`.
    fn head(&self, hyperedge: Self::Edge) -> Self::Head<'_>;

    /// Returns the hyperedges whose tail contains `vertex`.
    fn outgoing(&self, vertex: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns the hyperedges whose head contains `vertex`.
    fn ingoing(&self, vertex: Self::Vertex) -> Self::Ingoing<'_>;
}

/// Finite directed hypergraph.
///
/// Extends [`DirectedHypergraph`] with convenience queries.
pub trait FiniteDirectedHypergraph: DirectedHypergraph + FiniteGraph
where
    <Self as Structure>::Vertices: FiniteVertices<Vertex = Self::Vertex>,
    <Self as Structure>::Edges: FiniteEdges<Vertex = Self::Vertex, Edge = Self::Edge>,
{
    /// Returns whether `vertex` is in the tail of `hyperedge`.
    fn in_tail(&self, hyperedge: Self::Edge, vertex: Self::Vertex) -> bool {
        self.tail(hyperedge).any(|u| u == vertex)
    }

    /// Returns whether `vertex` is in the head of `hyperedge`.
    fn in_head(&self, hyperedge: Self::Edge, vertex: Self::Vertex) -> bool {
        self.head(hyperedge).any(|u| u == vertex)
    }

    /// Returns the size of the tail of `hyperedge`.
    fn tail_cardinality(&self, hyperedge: Self::Edge) -> usize {
        self.tail(hyperedge).count()
    }

    /// Returns the size of the head of `hyperedge`.
    fn head_cardinality(&self, hyperedge: Self::Edge) -> usize {
        self.head(hyperedge).count()
    }

    /// Returns the number of outgoing hyperedges of `vertex`.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex).count()
    }

    /// Returns the number of ingoing hyperedges of `vertex`.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.ingoing(vertex).count()
    }
}

impl<T> FiniteDirectedHypergraph for T
where
    T: DirectedHypergraph + FiniteGraph,
    <T as Structure>::Vertices: FiniteVertices<Vertex = T::Vertex>,
    <T as Structure>::Edges: FiniteEdges<Vertex = T::Vertex, Edge = T::Edge>,
{
}

/// Directed hyperedge insertion.
pub trait InsertDirectedHyperedge: EdgeType {
    /// Inserts a directed hyperedge from tail and head vertices.
    fn insert_hyperedge<Tail, Head>(&mut self, tail: Tail, head: Head) -> Option<Self::Edge>
    where
        Tail: IntoIterator<Item = Self::Vertex>,
        Head: IntoIterator<Item = Self::Vertex>;
}

/// Undirected hyperedge described by its member vertices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HyperedgeMembers<V> {
    /// Member vertices.
    pub members: Vec<V>,
}

impl<V> HyperedgeMembers<V> {
    /// Creates an undirected hyperedge from its member vertices.
    #[must_use]
    #[inline]
    pub fn new<I>(members: I) -> Self
    where
        I: IntoIterator<Item = V>,
    {
        Self {
            members: members.into_iter().collect(),
        }
    }
}

/// Directed hyperedge described by its tail and head vertices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DirectedHyperedge<V> {
    /// Tail vertices.
    pub tail: Vec<V>,

    /// Head vertices.
    pub head: Vec<V>,
}

impl<V> DirectedHyperedge<V> {
    /// Creates a directed hyperedge from tail and head vertices.
    #[must_use]
    #[inline]
    pub fn new<T, H>(tail: T, head: H) -> Self
    where
        T: IntoIterator<Item = V>,
        H: IntoIterator<Item = V>,
    {
        Self {
            tail: tail.into_iter().collect(),
            head: head.into_iter().collect(),
        }
    }
}

/// Hypergraph constructible from owned undirected hyperedges.
pub trait FromHyperedges: Sized + Graph {
    /// Creates a hypergraph from owned hyperedges.
    fn from_hyperedges<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = HyperedgeMembers<Self::Vertex>>;
}

/// Directed hypergraph constructible from owned directed hyperedges.
pub trait FromDirectedHyperedges: Sized + Graph {
    /// Creates a directed hypergraph from owned directed hyperedges.
    fn from_directed_hyperedges<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = DirectedHyperedge<Self::Vertex>>;
}
