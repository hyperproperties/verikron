/// Common vertex identifier type.
pub trait VertexType {
    /// Type used to identify vertices.
    type Vertex: Eq + Copy;
}

/// Finite graph with global access to its vertices.
pub trait Vertices: VertexType {
    /// Iterator over all vertices.
    type Vertices<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Returns an iterator over all vertices.
    fn vertices(&self) -> Self::Vertices<'_>;

    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize {
        self.vertices().count()
    }

    /// Returns true when the graph contains `vertex`.
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        self.vertices().any(|u| &u == vertex)
    }
}

/// Vertex insertion.
pub trait InsertVertex: VertexType {
    /// Inserts a new vertex and returns its identifier on success.
    fn insert_vertex(&mut self) -> Option<Self::Vertex>;
}

/// Vertex removal.
pub trait RemoveVertex: VertexType {
    /// Removes `vertex` and returns whether it existed.
    fn remove_vertex(&mut self, vertex: Self::Vertex) -> bool;
}

/// Finite mutable vertex store.
pub trait VerticesMut: Vertices + InsertVertex + RemoveVertex {}
impl<T> VerticesMut for T where T: Vertices + InsertVertex + RemoveVertex {}

/// Common edge identifier type.
pub trait EdgeType: VertexType {
    /// Type used to identify edges.
    type Edge: Eq + Copy;
}

/// Finite graph with global access to its edges.
pub trait Edges: EdgeType {
    /// Iterator over all edges as `(source, edge, destination)`.
    type Edges<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns an iterator over all edges.
    fn edges(&self) -> Self::Edges<'_>;

    /// Returns the number of edges.
    fn edge_count(&self) -> usize {
        self.edges().count()
    }
}

/// Edge insertion.
pub trait InsertEdge: EdgeType {
    /// Inserts an edge between the given endpoints.
    fn insert_edge(&mut self, endpoints: (Self::Vertex, Self::Vertex)) -> Option<Self::Edge>;
}

/// Edge removal.
pub trait RemoveEdge: EdgeType {
    /// Removes `edge` and returns its endpoints on success.
    fn remove_edge(&mut self, edge: Self::Edge) -> Option<(Self::Vertex, Self::Vertex)>;
}

/// Finite mutable edge store.
pub trait EdgesMut: Edges + InsertEdge + RemoveEdge {}
impl<T> EdgesMut for T where T: Edges + InsertEdge + RemoveEdge {}

/// Core abstraction for directed graphs.
///
/// The trait exposes structural queries such as sources, targets,
/// outgoing edges, incoming edges, and direct connections.
pub trait Directed: EdgeType {
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
    fn target(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns all outgoing edges from `source`.
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns the number of outgoing edges from `vertex`.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex).count()
    }

    /// Returns all incoming edges to `destination`.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_>;

    /// Returns the number of incoming edges to `vertex`.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.ingoing(vertex).count()
    }

    /// Returns the number of loop edges at `vertex`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize;

    /// Returns true when there exists an edge from `from` to `to`.
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.connections(from, to).next().is_some()
    }

    /// Returns true when `edge` is an edge from `from` to `to`.
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.connections(from, to).any(|(_, e, _)| e == edge)
    }

    /// Returns all edges from `from` to `to`.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_>;
}

/// Core abstraction for undirected graphs.
///
/// The trait exposes structural queries such as endpoints,
/// incident edges, degrees, loops, and direct connections.
pub trait Undirected: EdgeType {
    /// Iterator over edges incident to a vertex.
    type Incident<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Iterator over edges between two vertices.
    type Connections<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns the endpoints of `edge`.
    ///
    /// For a loop edge, both endpoints are the same vertex.
    fn endpoints(&self, edge: Self::Edge) -> (Self::Vertex, Self::Vertex);

    /// Returns all edges incident to `vertex`.
    fn incident(&self, vertex: Self::Vertex) -> Self::Incident<'_>;

    /// Returns the degree of `vertex`.
    ///
    /// By convention, each loop contributes `2` to the degree.
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

    /// Returns true when there exists an edge between `u` and `v`.
    fn is_connected(&self, u: Self::Vertex, v: Self::Vertex) -> bool {
        self.connections(u, v).next().is_some()
    }

    /// Returns true when `edge` is an edge between `u` and `v`.
    fn has_edge(&self, u: Self::Vertex, edge: Self::Edge, v: Self::Vertex) -> bool {
        self.connections(u, v).any(|(_, e, _)| e == edge)
    }

    /// Returns all edges between `u` and `v`.
    ///
    /// For undirected graphs, `(u, v)` and `(v, u)` refer to the same pair.
    fn connections(&self, u: Self::Vertex, v: Self::Vertex) -> Self::Connections<'_>;
}

/// High-level abstraction for a finite graph.
///
/// A graph is composed from a vertex store and an edge store
/// that share the same vertex identifier type.
pub trait Graph: VertexType {
    /// Vertex storage component.
    type Vertices: Vertices<Vertex = Self::Vertex>;

    /// Edge storage component.
    type Edges: Edges<Vertex = Self::Vertex>;

    /// Returns the edge store.
    fn edge_store(&self) -> &Self::Edges;

    /// Returns the vertex store.
    fn vertex_store(&self) -> &Self::Vertices;

    /// Returns a default size measure: vertices plus edges.
    fn size(&self) -> usize {
        self.vertex_store().vertex_count() + self.edge_store().edge_count()
    }

    /// Returns true when the graph is empty.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

/// Graph-like structure with local exploration but without assumed
/// global enumeration of all vertices or all edges.
///
/// This is intended for infinite or implicitly defined graphs such as
/// transition systems, symbolic graphs, or lazily generated state spaces.
pub trait InfiniteGraph: Directed {}
