use crate::graphs::{
    arc::Arc,
    graph::{FiniteGraph, Graph},
    members::Members,
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

    /// Iterator over the hyperedges containing both `u` and `v`.
    type Connections<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns the member vertices of `hyperedge`.
    fn members(&self, hyperedge: Self::Edge) -> Self::Members<'_>;

    /// Returns the owned member collection of `hyperedge`.
    #[must_use]
    fn hyperedge(&self, hyperedge: Self::Edge) -> Members<Vec<Self::Vertex>>
    where
        Self: Sized,
    {
        Members::new(self.members(hyperedge).collect())
    }

    /// Returns the hyperedges incident to `vertex`.
    fn incident(&self, vertex: Self::Vertex) -> Self::Incident<'_>;

    /// Returns the hyperedges containing both `u` and `v`.
    fn connections(&self, u: Self::Vertex, v: Self::Vertex) -> Self::Connections<'_>;
}

/// Finite undirected hypergraph.
///
/// Extends [`UndirectedHypergraph`] with finite-style convenience queries.
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

    /// Returns whether there exists some hyperedge containing both `u` and `v`.
    fn is_connected(&self, u: Self::Vertex, v: Self::Vertex) -> bool {
        self.connections(u, v).next().is_some()
    }

    /// Returns whether `hyperedge` contains both `u` and `v`.
    fn has_edge(&self, u: Self::Vertex, hyperedge: Self::Edge, v: Self::Vertex) -> bool {
        self.connections(u, v).any(|e| e == hyperedge)
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
    fn insert_hyperedge<S>(&mut self, hyperedge: Members<S>) -> Option<Self::Edge>
    where
        S: IntoIterator<Item = Self::Vertex>;
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
    type Incoming<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Iterator over hyperedges whose tail contains `source`
    /// and whose head contains `destination`.
    type Connections<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns the tail vertices of `hyperedge`.
    fn tail(&self, hyperedge: Self::Edge) -> Self::Tail<'_>;

    /// Returns the head vertices of `hyperedge`.
    fn head(&self, hyperedge: Self::Edge) -> Self::Head<'_>;

    /// Returns the owned directed incidence of `hyperedge`.
    #[must_use]
    fn arc(&self, hyperedge: Self::Edge) -> Arc<Vec<Self::Vertex>>
    where
        Self: Sized,
    {
        Arc::new(
            self.tail(hyperedge).collect(),
            self.head(hyperedge).collect(),
        )
    }

    /// Returns the hyperedges whose tail contains `vertex`.
    fn outgoing(&self, vertex: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns the hyperedges whose head contains `vertex`.
    fn incoming(&self, vertex: Self::Vertex) -> Self::Incoming<'_>;

    /// Returns the hyperedges whose tail contains `source`
    /// and whose head contains `destination`.
    fn connections(&self, source: Self::Vertex, destination: Self::Vertex)
    -> Self::Connections<'_>;
}

/// Finite directed hypergraph.
///
/// Extends [`DirectedHypergraph`] with finite-style convenience queries.
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

    /// Returns the number of incoming hyperedges of `vertex`.
    fn incoming_degree(&self, vertex: Self::Vertex) -> usize {
        self.incoming(vertex).count()
    }

    /// Returns whether there exists some hyperedge whose tail contains `source`
    /// and whose head contains `destination`.
    fn is_connected(&self, source: Self::Vertex, destination: Self::Vertex) -> bool {
        self.connections(source, destination).next().is_some()
    }

    /// Returns whether `hyperedge` connects `source` to `destination`.
    fn has_edge(
        &self,
        source: Self::Vertex,
        hyperedge: Self::Edge,
        destination: Self::Vertex,
    ) -> bool {
        self.connections(source, destination)
            .any(|e| e == hyperedge)
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
    /// Inserts a directed hyperedge from tail and head vertex collections.
    fn insert_hyperedge<S>(&mut self, hyperedge: Arc<S>) -> Option<Self::Edge>
    where
        S: IntoIterator<Item = Self::Vertex>;
}

/// Undirected hypergraph constructible from owned member collections.
pub trait FromMembers: Sized + Graph {
    /// Creates an undirected hypergraph from owned member collections.
    fn from_members<I, S>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = Members<S>>,
        S: IntoIterator<Item = Self::Vertex>;
}

/// Directed hypergraph constructible from owned directed incidence.
pub trait FromHyperarcs: Sized + Graph {
    /// Creates a directed hypergraph from owned tail/head collections.
    fn from_hyperarcs<I, S>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = Arc<S>>,
        S: IntoIterator<Item = Self::Vertex>;
}
