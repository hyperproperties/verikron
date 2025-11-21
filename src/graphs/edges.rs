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
