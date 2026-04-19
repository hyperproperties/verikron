use crate::graphs::structure::{FiniteEdges, FiniteStructure, FiniteVertices, Structure};

/// Marker trait for edge-vertex structures.
///
/// Ordinary graphs and hypergraphs both implement this trait.
/// More specific incidence behavior is provided by traits such as
/// [`Directed`], [`Undirected`], [`DirectedHypergraph`], and
/// [`UndirectedHypergraph`].
pub trait Graph: Structure {}

/// Directed graph interface based on local exploration.
///
/// Suitable for infinite or implicit graphs.
pub trait Directed: Graph {
    /// Iterator over outgoing edges from a source vertex.
    type Outgoing<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Iterator over incoming edges to a destination vertex.
    type Ingoing<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Iterator over edges from `from` to `to`.
    type Connections<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns the source vertex of `edge`.
    fn source(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns the destination vertex of `edge`.
    fn destination(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns the endpoints of `edge`.
    fn endpoints(&self, edge: Self::Edge) -> Endpoints<Self::Vertex> {
        Endpoints::new(self.source(edge), self.destination(edge))
    }

    /// Returns all outgoing edges from `source`.
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns all incoming edges to `destination`.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_>;

    /// Returns all edges from `from` to `to`.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_>;
}

/// Finite directed graph.
///
/// Extends [`Directed`] with finite-style convenience queries.
pub trait FiniteDirected: Directed + FiniteGraph
where
    <Self as Structure>::Vertices: FiniteVertices<Vertex = Self::Vertex>,
    <Self as Structure>::Edges: FiniteEdges<Vertex = Self::Vertex, Edge = Self::Edge>,
{
    /// Returns the number of outgoing edges from `vertex`.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex).count()
    }

    /// Returns the number of incoming edges to `vertex`.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.ingoing(vertex).count()
    }

    /// Returns the number of loop edges at `vertex`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize;

    /// Returns true if `from` and `to` is connected by some edge.
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.connections(from, to).next().is_some()
    }

    /// Returns whether `edge` is an edge from `from` to `to`.
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.connections(from, to).any(|(_, e, _)| e == edge)
    }
}

/// Directed graph with indexable local adjacency.
///
/// This trait provides random access to the outgoing and ingoing neighbor lists
/// of each vertex, without exposing edge identifiers.
///
/// It is suitable for algorithms that need efficient local traversal state,
/// such as iterative DFS, SCC decomposition, reachability, and reverse search.
///
/// The graph may have infinitely many edges globally, as long as each vertex
/// has finitely many outgoing and ingoing neighbors.
pub trait IndexedDirected: Graph
where
    <Self as Structure>::Vertices: FiniteVertices<Vertex = Self::Vertex>,
{
    /// Returns the number of outgoing neighbors of `vertex`.
    fn outgoing_count(&self, vertex: Self::Vertex) -> usize;

    /// Returns the `index`th outgoing neighbor of `vertex`.
    ///
    /// Requires `index < outgoing_count(vertex)`.
    fn outgoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex>;

    /// Returns the number of ingoing neighbors of `vertex`.
    fn ingoing_count(&self, vertex: Self::Vertex) -> usize;

    /// Returns the `index`th ingoing neighbor of `vertex`.
    ///
    /// Requires `index < ingoing_count(vertex)`.
    fn ingoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex>;

    /// Returns whether `vertex` has a self-loop.
    fn has_loop(&self, vertex: Self::Vertex) -> bool {
        (0..self.outgoing_count(vertex)).any(|i| self.outgoing_at(vertex, i).unwrap() == vertex)
    }

    /// Returns whether `from` has `to` as an outgoing neighbor.
    fn connects_to(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        (0..self.outgoing_count(from)).any(|i| self.outgoing_at(from, i).unwrap() == to)
    }

    /// Returns the outgoing neighbors of `vertex`.
    fn outgoing_neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        (0..self.outgoing_count(vertex)).map(move |i| self.outgoing_at(vertex, i).unwrap())
    }

    /// Returns the ingoing neighbors of `vertex`.
    fn ingoing_neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        (0..self.ingoing_count(vertex)).map(move |i| self.ingoing_at(vertex, i).unwrap())
    }
}

/// Undirected graph interface based on local exploration.
///
/// Suitable for infinite or implicit graphs.
pub trait Undirected: Graph {
    /// Iterator over edges incident to a vertex.
    type Incident<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Iterator over edges between two vertices.
    type Connections<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns the endpoints of `edge`.
    fn endpoints(&self, edge: Self::Edge) -> (Self::Vertex, Self::Vertex);

    /// Returns all edges incident to `vertex`.
    fn incident(&self, vertex: Self::Vertex) -> Self::Incident<'_>;

    /// Returns all edges between `u` and `v`.
    fn connections(&self, u: Self::Vertex, v: Self::Vertex) -> Self::Connections<'_>;
}

/// Finite undirected graph.
///
/// Extends [`Undirected`] with finite-style convenience queries.
pub trait FiniteUndirected: Undirected + FiniteGraph
where
    <Self as Structure>::Vertices: FiniteVertices<Vertex = Self::Vertex>,
    <Self as Structure>::Edges: FiniteEdges<Vertex = Self::Vertex, Edge = Self::Edge>,
{
    /// Returns the degree of `vertex`.
    ///
    /// By convention, each loop contributes `2`.
    fn degree(&self, vertex: Self::Vertex) -> usize {
        self.incident(vertex)
            .map(|(u, _, v)| if u == v { 2 } else { 1 })
            .sum()
    }

    /// Returns the number of loop edges at `vertex`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.incident(vertex)
            .filter(|(u, _, v)| *u == vertex && *v == vertex)
            .count()
    }

    /// Returns whether there exists an edge between `u` and `v`.
    fn is_connected(&self, u: Self::Vertex, v: Self::Vertex) -> bool {
        self.connections(u, v).next().is_some()
    }

    /// Returns whether `edge` is an edge between `u` and `v`.
    fn has_edge(&self, u: Self::Vertex, edge: Self::Edge, v: Self::Vertex) -> bool {
        self.connections(u, v).any(|(_, e, _)| e == edge)
    }
}

/// Undirected graph with indexable local adjacency.
///
/// This trait provides random access to the neighbor list of each vertex,
/// without exposing edge identifiers.
///
/// It is suitable for algorithms that need efficient local traversal state,
/// such as iterative DFS, connectivity, and local neighborhood queries.
///
/// The graph may have infinitely many edges globally, as long as each vertex
/// has finitely many incident neighbors.
pub trait IndexedUndirected: Graph
where
    <Self as Structure>::Vertices: FiniteVertices<Vertex = Self::Vertex>,
{
    /// Returns the number of neighbors of `vertex`.
    fn incident_count(&self, vertex: Self::Vertex) -> usize;

    /// Returns the `index`th neighbor of `vertex`.
    ///
    /// Requires `index < incident_count(vertex)`.
    fn incident_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex>;

    /// Returns whether `vertex` has a loop.
    fn has_loop(&self, vertex: Self::Vertex) -> bool {
        (0..self.incident_count(vertex)).any(|i| self.incident_at(vertex, i).unwrap() == vertex)
    }

    /// Returns whether `u` and `v` are adjacent.
    fn is_adjacent(&self, u: Self::Vertex, v: Self::Vertex) -> bool {
        (0..self.incident_count(u)).any(|i| self.incident_at(u, i).unwrap() == v)
    }

    /// Returns the neighbors of `vertex`.
    fn neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        (0..self.incident_count(vertex)).map(move |i| self.incident_at(vertex, i).unwrap())
    }
}

/// Finite graph.
///
/// This is just a finite structure viewed as a graph.
pub trait FiniteGraph: Graph + FiniteStructure
where
    <Self as Structure>::Vertices: FiniteVertices,
    <Self as Structure>::Edges: FiniteEdges,
{
}

impl<T> FiniteGraph for T
where
    T: Graph + FiniteStructure,
    <T as Structure>::Vertices: FiniteVertices,
    <T as Structure>::Edges: FiniteEdges,
{
}

/// Endpoints of an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Endpoints<V> {
    /// Source or first endpoint.
    pub from: V,
    /// Target or second endpoint.
    pub to: V,
}

impl<V> Endpoints<V> {
    /// Creates a new pair of endpoints.
    #[inline]
    pub const fn new(from: V, to: V) -> Self {
        Self { from, to }
    }
}

/// Graph constructible from owned edge endpoints.
pub trait FromEndpoints: Sized + Graph {
    /// Creates a graph from owned edge endpoints.
    fn from_endpoints<I>(edges: I) -> Self
    where
        I: IntoIterator<Item = Endpoints<Self::Vertex>>;
}
