use crate::graphs::{
    arc::Arc,
    endpoints::Endpoints,
    structure::{FiniteEdges, FiniteStructure, FiniteVertices, Structure},
};

/// Marker trait for edge-vertex structures.
///
/// Ordinary graphs and hypergraphs both implement this trait.
/// More specific incidence behavior is provided by traits such as
/// [`Directed`], [`Undirected`], [`DirectedHypergraph`], and
/// [`UndirectedHypergraph`].
pub trait Graph: Structure {}

/// Directed graph interface based on local exploration.
///
/// Suitable for finite, infinite, or implicit graphs.
pub trait Directed: Graph {
    /// Iterator over outgoing edges from a source vertex.
    type Outgoing<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Iterator over incoming edges to a destination vertex.
    type Incoming<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Iterator over edges from `source` to `destination`.
    type Connections<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns the source vertex of `edge`.
    fn source(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns the destination vertex of `edge`.
    fn destination(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns the directed endpoints of `edge`.
    #[must_use]
    fn arc(&self, edge: Self::Edge) -> Arc<Self::Vertex> {
        Arc::new(self.source(edge), self.destination(edge))
    }

    /// Returns all outgoing edges from `source`.
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns all incoming edges to `destination`.
    fn incoming(&self, destination: Self::Vertex) -> Self::Incoming<'_>;

    /// Returns all edges from `source` to `destination`.
    fn connections(&self, source: Self::Vertex, destination: Self::Vertex)
    -> Self::Connections<'_>;
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
    fn incoming_degree(&self, vertex: Self::Vertex) -> usize {
        self.incoming(vertex).count()
    }

    /// Returns the number of loop edges at `vertex`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize;

    /// Returns whether there exists some edge from `source` to `destination`.
    fn is_connected(&self, source: Self::Vertex, destination: Self::Vertex) -> bool {
        self.connections(source, destination).next().is_some()
    }

    /// Returns whether `edge` is an edge from `source` to `destination`.
    fn has_edge(&self, source: Self::Vertex, edge: Self::Edge, destination: Self::Vertex) -> bool {
        self.connections(source, destination).any(|e| e == edge)
    }
}

/// Directed graph with indexable local adjacency.
///
/// This trait provides random access to the outgoing and incoming neighbor
/// lists of each vertex, without exposing edge identifiers.
///
/// It is suitable for algorithms that need efficient local traversal state,
/// such as iterative DFS, SCC decomposition, reachability, and reverse search.
///
/// The graph need not have a finite global vertex set. It is enough that each
/// individual vertex has finitely many outgoing and incoming neighbors.
pub trait IndexedDirected: Graph {
    /// Returns the number of outgoing neighbors of `vertex`.
    fn outgoing_count(&self, vertex: Self::Vertex) -> usize;

    /// Returns the `index`th outgoing neighbor of `vertex`.
    ///
    /// Requires `index < outgoing_count(vertex)`.
    fn outgoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex>;

    /// Returns the number of incoming neighbors of `vertex`.
    fn incoming_count(&self, vertex: Self::Vertex) -> usize;

    /// Returns the `index`th incoming neighbor of `vertex`.
    ///
    /// Requires `index < incoming_count(vertex)`.
    fn incoming_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex>;

    /// Returns whether `vertex` has a self-loop.
    fn has_loop(&self, vertex: Self::Vertex) -> bool {
        (0..self.outgoing_count(vertex)).any(|i| {
            self.outgoing_at(vertex, i)
                .expect("outgoing_at must be defined for indices below outgoing_count")
                == vertex
        })
    }

    /// Returns whether `source` has `destination` as an outgoing neighbor.
    fn connects_to(&self, source: Self::Vertex, destination: Self::Vertex) -> bool {
        (0..self.outgoing_count(source)).any(|i| {
            self.outgoing_at(source, i)
                .expect("outgoing_at must be defined for indices below outgoing_count")
                == destination
        })
    }

    /// Returns the outgoing neighbors of `vertex`.
    fn outgoing_neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        (0..self.outgoing_count(vertex)).map(move |i| {
            self.outgoing_at(vertex, i)
                .expect("outgoing_at must be defined for indices below outgoing_count")
        })
    }

    /// Returns the incoming neighbors of `vertex`.
    fn incoming_neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        (0..self.incoming_count(vertex)).map(move |i| {
            self.incoming_at(vertex, i)
                .expect("incoming_at must be defined for indices below incoming_count")
        })
    }
}

/// Undirected graph interface based on local exploration.
///
/// Suitable for finite, infinite, or implicit graphs.
pub trait Undirected: Graph {
    /// Iterator over edges incident to a vertex.
    type Incident<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Iterator over edges between two vertices.
    type Connections<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns the endpoints of `edge`.
    fn endpoints(&self, edge: Self::Edge) -> Endpoints<Self::Vertex>;

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
            .map(|e| {
                let endpoints = self.endpoints(e);
                if endpoints.u == endpoints.v { 2 } else { 1 }
            })
            .sum()
    }

    /// Returns the number of loop edges at `vertex`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.incident(vertex)
            .filter(|e| {
                let endpoints = self.endpoints(*e);
                endpoints.u == vertex && endpoints.v == vertex
            })
            .count()
    }

    /// Returns whether there exists an edge between `u` and `v`.
    fn is_connected(&self, u: Self::Vertex, v: Self::Vertex) -> bool {
        self.connections(u, v).next().is_some()
    }

    /// Returns whether `edge` is an edge between `u` and `v`.
    fn has_edge(&self, u: Self::Vertex, edge: Self::Edge, v: Self::Vertex) -> bool {
        self.connections(u, v).any(|e| e == edge)
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
/// The graph need not have a finite global vertex set. It is enough that each
/// individual vertex has finitely many incident neighbors.
pub trait IndexedUndirected: Graph {
    /// Returns the number of neighbors of `vertex`.
    fn incident_count(&self, vertex: Self::Vertex) -> usize;

    /// Returns the `index`th neighbor of `vertex`.
    ///
    /// Requires `index < incident_count(vertex)`.
    fn incident_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex>;

    /// Returns whether `vertex` has a loop.
    fn has_loop(&self, vertex: Self::Vertex) -> bool {
        (0..self.incident_count(vertex)).any(|i| {
            self.incident_at(vertex, i)
                .expect("incident_at must be defined for indices below incident_count")
                == vertex
        })
    }

    /// Returns whether `u` and `v` are adjacent.
    fn is_adjacent(&self, u: Self::Vertex, v: Self::Vertex) -> bool {
        (0..self.incident_count(u)).any(|i| {
            self.incident_at(u, i)
                .expect("incident_at must be defined for indices below incident_count")
                == v
        })
    }

    /// Returns the neighbors of `vertex`.
    fn neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        (0..self.incident_count(vertex)).map(move |i| {
            self.incident_at(vertex, i)
                .expect("incident_at must be defined for indices below incident_count")
        })
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
