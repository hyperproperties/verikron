/// Common vertex identifier type used by vertex-related traits.
///
/// Types implementing graph traits are expected to use [`Vertices::Vertex`]
/// to identify vertices.
pub trait Vertices {
    /// Type used to identify vertices.
    ///
    /// Typically a small copyable value such as `usize`.
    type Vertex: Eq + Copy;
}

/// A graph that supports read-only access to its vertices.
///
/// This trait provides an abstract view of the vertex set of a graph.
/// Vertices are identified by [`Vertices::Vertex`] values, which can be
/// enumerated via [`ReadVertices::vertices`].
pub trait ReadVertices: Vertices {
    /// Iterator over all vertices in the graph.
    ///
    /// The order of vertices depends on the implementation, but should be
    /// stable for a given graph instance unless the graph is mutated.
    type Vertices<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Returns an iterator over all vertices in the graph.
    fn vertices(&self) -> Self::Vertices<'_>;

    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize {
        self.vertices().count()
    }

    /// Returns true if the vertices includes the vertex.
    fn contains(&self, v: &Self::Vertex) -> bool {
        self.vertices().any(|u| &u == v)
    }
}

/// A graph that supports both querying and mutating vertices.
///
/// This is a convenience alias for types that implement:
/// - [`ReadVertices`] for read-only access,
/// - [`InsertVertex`] for inserting vertices, and
/// - [`RemoveVertex`] for removing vertices.
///
/// Implementors are expected to use a consistent [`Vertices::Vertex`] type
/// across all four traits.
pub trait VerticesMut: Vertices + ReadVertices + InsertVertex + RemoveVertex {}

/// Blanket implementation of [`VerticesMut`] for any type that provides the
/// required capabilities.
///
/// Any type that implements [`Vertices`], [`ReadVertices`], [`InsertVertex`],
/// and [`RemoveVertex`] automatically implements [`VerticesMut`].
impl<T> VerticesMut for T where T: Vertices + ReadVertices + InsertVertex + RemoveVertex {}

/// A graph that supports insertion of vertices.
///
/// This trait describes the ability to add new vertices to a graph. The
/// associated vertex type is inherited from [`Vertices`].
pub trait InsertVertex: Vertices {
    /// Inserts a new vertex into the graph.
    ///
    /// On success, returns `Some(v)` where `v` is the identifier of the
    /// newly inserted vertex. If the vertex cannot be inserted (for example
    /// due to capacity limits), returns `None`.
    ///
    /// Implementations may choose whether or not to recycle identifiers of
    /// previously removed vertices.
    fn insert_vertex(&mut self) -> Option<Self::Vertex>;
}

/// A graph that supports removal of vertices.
///
/// This trait describes the ability to remove existing vertices from a graph.
/// The associated vertex type is inherited from [`Vertices`].
pub trait RemoveVertex: Vertices {
    /// Removes an existing vertex from the graph.
    ///
    /// The `vertex` parameter is the identifier of the vertex to remove.
    ///
    /// On success, returns `Some(())`. If no such vertex exists in the graph,
    /// returns `None`.
    ///
    /// Implementations are free to decide how incident edges are handled:
    /// they may be removed automatically, or removal may be allowed only
    /// when the vertex has no incident edges. This behavior should be
    /// documented by each implementation.
    ///
    /// After successful removal, the given vertex identifier must not refer
    /// to a valid vertex in the graph. Implementations may choose whether
    /// to recycle identifiers for subsequently inserted vertices.
    fn remove_vertex(&mut self, vertex: Self::Vertex) -> bool;
}

/// Common vertex and edge identifier types used by edge-related traits.
///
/// Types implementing graph traits are expected to use:
/// - [`Edges::Vertex`] to identify vertices, and
/// - [`Edges::Edge`] to identify edges.
pub trait Edges {
    /// Type used to identify vertices.
    ///
    /// Typically a small copyable value such as `usize`.
    type Vertex: Eq + Copy;

    /// Type used to identify edges.
    ///
    /// Usually a small copyable value such as `usize`.
    /// This makes it possible for a pair of vertices to be connected
    /// by multiple distinct edges. Each directed edge can be viewed as
    /// a triple with source, edge, and destination.
    type Edge: Eq + Copy;
}

/// A graph that supports read-only access to its edges.
///
/// This trait provides an abstract view of a (possibly directed or undirected)
/// graph. Each edge is identified by an [`Edges::Edge`] value and connects two
/// vertices identified by [`Edges::Vertex`]. Edges can be enumerated via
/// [`ReadEdges::edges`], which yields triples of the form
/// `(source, edge, destination)`.
///
/// The precise interpretation of direction, the handling of parallel edges,
/// and whether self-loops are permitted is left to the implementation.
pub trait ReadEdges: Edges {
    /// Iterator over all edges in the graph.
    ///
    /// Each item is a triple with source, edge, and destination.
    /// The order of edges depends on the implementation, but should be stable
    /// for a given graph instance unless the graph is mutated.
    type Edges<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns an iterator over all edges in the graph.
    ///
    /// Each item is a triple with source, edge, and destination.
    fn edges(&self) -> Self::Edges<'_>;

    /// Returns the number of edges.
    fn edge_count(&self) -> usize {
        self.edges().count()
    }
}

/// A graph that supports both querying and mutating edges.
///
/// This is a convenience alias for types that implement:
/// - [`ReadEdges`] for read-only access,
/// - [`InsertEdge`] for inserting edges, and
/// - [`RemoveEdge`] for removing edges.
///
/// Implementors are expected to use a consistent [`Edges::Vertex`] and
/// [`Edges::Edge`] type across all four traits.
pub trait EdgesMut: Edges + ReadEdges + InsertEdge + RemoveEdge {}

/// Blanket implementation of [`EdgesMut`] for any type that provides the
/// required capabilities.
///
/// Any type that implements [`Edges`], [`ReadEdges`], [`InsertEdge`], and
/// [`RemoveEdge`] automatically implements [`EdgesMut`].
impl<T> EdgesMut for T where T: Edges + ReadEdges + InsertEdge + RemoveEdge {}

/// A graph that supports insertion of edges.
///
/// This trait describes the ability to add new edges to a graph. The associated
/// [`Edges::Vertex`] and [`Edges::Edge`] types are inherited from [`Edges`].
pub trait InsertEdge: Edges {
    /// Inserts a new edge (directed or undirected) into the graph.
    ///
    /// The `endpoints` parameter is a pair `(u, v)` specifying the vertices
    /// that the edge connects:
    /// - For a directed graph, this is `(source, destination)`.
    /// - For an undirected graph, the order of the vertices is ignored.
    ///
    /// On success, returns `Some(edge)` identifying the inserted edge.
    /// If the edge cannot be inserted, returns `None`.
    ///
    /// The exact behavior when:
    /// - inserting multiple parallel edges between the same pair of vertices, or
    /// - inserting an edge whose endpoints are not already present in the graph
    ///
    /// is left to the implementation.
    fn insert_edge(&mut self, endpoints: (Self::Vertex, Self::Vertex)) -> Option<Self::Edge>;
}

/// A graph that supports removal of edges.
///
/// This trait describes the ability to remove existing edges from a graph. The
/// associated [`Edges::Vertex`] and [`Edges::Edge`] types are inherited from
/// [`Edges`].
pub trait RemoveEdge: Edges {
    /// Removes an existing edge from the graph.
    ///
    /// The `edge` parameter is the identifier of the edge to remove.
    ///
    /// On success, returns `Some((u, v))` containing the endpoints of the
    /// removed edge, or `None` if no such edge exists in the graph.
    ///
    /// For directed graphs, the returned pair is `(source, destination)`.
    /// For undirected graphs, the order of the returned vertices is
    /// implementation-defined (and may or may not match the order used
    /// when the edge was inserted).
    ///
    /// After successful removal, the given edge identifier must not refer
    /// to a valid edge in the graph. Implementations may choose whether
    /// to recycle identifiers for subsequently inserted edges.
    fn remove_edge(&mut self, edge: Self::Edge) -> Option<(Self::Vertex, Self::Vertex)>;
}

/// Core abstraction for finite directed graphs.
///
/// Directed describes a directed graph in which vertices and edges are
/// identified by small copyable values.
/// Vertices represent nodes in the graph, and edges represent directed
/// connections from a source vertex to a destination vertex.
///
/// The trait focuses on structural access to vertices, edges, and incidences.
/// Algorithms such as traversal, reachability, or shortest path computation
/// are expected to be written as generic helper functions or extension traits
/// built on top of this interface.
///
/// Iteration uses associated iterator types, so implementations can avoid
/// extra allocation and avoid dynamic dispatch.
pub trait Directed: ReadEdges {
    /// Source vertex of an edge.
    fn source(&self, edge: Self::Edge) -> Self::Vertex;

    /// Destination vertex of an edge.
    fn target(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns all outgoing edges from the given source vertex.
    ///
    /// Each item is a triple with source, edge_id, and destination,
    /// and the source component is equal to the given source vertex.
    fn outgoing(&self, source: Self::Vertex) -> Self::Edges<'_>;

    /// Returns the number of edges with the given source vertex.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex).count()
    }

    /// Returns all incoming edges to the given destination vertex.
    ///
    /// Each item is a triple with source, edge_id, and destination,
    /// and the destination component is equal to the given destination vertex.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Edges<'_>;

    /// Returns the number of edges with the given destination vertex.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.ingoing(vertex).count()
    }

    /// Returns the number of edges with the same source and destination vertex.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize;

    /// Returns true when there exists at least one edge whose source is from,
    /// and whose destination is to.
    ///
    /// This checks for a single step edge only,
    /// it does not perform a reachability query through longer paths.
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.connections(from, to).next().is_some()
    }

    /// Returns true when edge is a directed edge whose source is from,
    /// and whose destination is to.
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.connections(from, to).any(|(_, e, _)| e == edge)
    }

    /// Returns an iterator over all edges whose source is from,
    /// and whose destination is to.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Edges<'_>;
}

/// High level abstraction for a finite graph.
///
/// Graph composes two separate components a vertex store and an edge store.
/// The Vertices type is responsible for describing the vertex set and any
/// vertex related data or operations.
/// The Edges type is responsible for describing the edge set and structural
/// properties such as connectivity and degrees.
///
/// This trait does not prescribe how vertices and edges are stored.
/// It only requires that each component implements the corresponding
/// `Vertices` or `Edges` trait.
/// Concrete graph types can use a single structure for both roles or use
/// separate structures for vertex data and edge data.
pub trait ReadGraph {
    /// Common vertex type for both the vertex store and the edge store.
    type Vertex: Eq + Copy;

    /// Vertex storage component.
    type Vertices: ReadVertices<Vertex = Self::Vertex>;

    /// Edge storage component.
    type Edges: ReadEdges<Vertex = Self::Vertex>;

    /// Returns a shared reference to the edge storage.
    fn edge_store(&self) -> &Self::Edges;

    /// Returns a shared reference to the vertex storage.
    fn vertex_store(&self) -> &Self::Vertices;

    /// Size measure for the whole graph.
    ///
    /// Implementations usually define this as the sum of the number of
    /// vertices and the number of edges but other consistent measures
    /// are also allowed.
    fn size(&self) -> usize {
        self.vertex_store().vertex_count() + self.edge_store().edge_count()
    }

    /// Returns true when the graph is empty.
    ///
    /// By default this is the case when the size method returns zero.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}
