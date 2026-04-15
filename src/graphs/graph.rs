/// Finite graph with global access to its vertices.
pub trait Vertices {
    /// Type used to identify vertices.
    type Vertex: Eq + Copy;

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
pub trait InsertVertex: Vertices {
    /// Inserts a new vertex and returns its identifier on success.
    fn insert_vertex(&mut self) -> Option<Self::Vertex>;
}

/// Vertex removal.
pub trait RemoveVertex: Vertices {
    /// Removes `vertex` and returns whether it existed.
    fn remove_vertex(&mut self, vertex: Self::Vertex) -> bool;
}

/// Finite mutable vertex store.
pub trait VerticesMut: Vertices + InsertVertex + RemoveVertex {}
impl<T> VerticesMut for T where T: Vertices + InsertVertex + RemoveVertex {}

/// Finite graph with global access to its edges.
pub trait Edges {
    /// Type used to identify vertices.
    type Vertex: Eq + Copy;

    /// Type used to identify edges.
    type Edge: Eq + Copy;

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
pub trait InsertEdge: Edges {
    /// Inserts an edge between the given endpoints.
    fn insert_edge(&mut self, endpoints: (Self::Vertex, Self::Vertex)) -> Option<Self::Edge>;
}

/// Edge removal.
pub trait RemoveEdge: Edges {
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
pub trait Directed: Edges {
    /// Returns the source vertex of `edge`.
    fn source(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns the destination vertex of `edge`.
    fn target(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns all outgoing edges from `source`.
    fn outgoing(&self, source: Self::Vertex) -> Self::Edges<'_>;

    /// Returns the number of outgoing edges from `vertex`.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex).count()
    }

    /// Returns all incoming edges to `destination`.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Edges<'_>;

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
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Edges<'_>;
}

/// High-level abstraction for a finite graph.
///
/// A graph is composed from a vertex store and an edge store
/// that share the same vertex identifier type.
pub trait Graph {
    /// Common vertex type.
    type Vertex: Eq + Copy;

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

/// Marker trait for infinite or implicitly defined graphs.
///
/// Such graphs may support local exploration without supporting
/// global enumeration of all vertices or all edges.
pub trait InfiniteGraph {}