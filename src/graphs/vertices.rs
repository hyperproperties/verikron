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
