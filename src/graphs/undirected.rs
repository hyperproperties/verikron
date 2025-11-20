use crate::graphs::edges::Edges;

/// Core abstraction for finite undirected graphs.
///
/// `Undirected` is the analogue of `Directed` for undirected graphs:
/// vertices and edges are identified by small copyable values,
/// vertices represent nodes in the graph, and edges represent
/// unordered connections between two (not necessarily distinct)
/// vertices.
///
/// The trait focuses on structural access to vertices, edges, and incidences.
/// Algorithms such as traversal, connectivity, or shortest path computation
/// are expected to be written as generic helper functions or extension traits
/// built on top of this interface.
///
/// Iteration uses associated iterator types, so implementations can avoid
/// extra allocation and avoid dynamic dispatch.
pub trait Undirected: Edges {
    /// Endpoints of an undirected edge.
    ///
    /// The returned pair `(a, b)` are the two vertices incident to `edge`.
    /// For a loop edge, both components are equal.
    fn endpoints(&self, edge: Self::Edge) -> (Self::Vertex, Self::Vertex);

    /// Returns the degree of the given vertex.
    ///
    /// The degree is the number of edges incident to the vertex.
    /// Loop edges contribute 2 to the degree, in accordance with
    /// standard graph-theoretic conventions.
    /// In otherwords, it is the sum of the ingoing and outgoing degrees.
    fn degree(&self, vertex: Self::Vertex) -> usize;

    /// Returns the number of loop edges incident to the given vertex.
    ///
    /// A loop edge has the same vertex as both of its endpoints.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize;

    /// Returns true when there exists at least one edge between `a` and `b`.
    ///
    /// This checks for a single-step edge only,
    /// it does not perform a reachability query through longer paths.
    fn is_connected(&self, a: Self::Vertex, b: Self::Vertex) -> bool {
        self.connections(a, b).next().is_some()
    }

    /// Returns true when `edge` is an undirected edge whose endpoints
    /// are `a` and `b` in some order.
    fn has_edge(&self, a: Self::Vertex, edge: Self::Edge, b: Self::Vertex) -> bool {
        self.connections(a, b).any(|(_, e, _)| e == edge)
    }

    /// Returns an iterator over all edges whose endpoints are `a` and `b`
    /// (in either order).
    ///
    /// Each item is a triple with `(u, edge, v)`, where `u` and `v` are
    /// the endpoints of the edge. For every item, the unordered pair
    /// `{u, v}` is equal to the unordered pair `{a, b}`.
    ///
    /// For `a == b`, this returns all loop edges at `a`.
    fn connections(&self, a: Self::Vertex, b: Self::Vertex) -> Self::Edges<'_>;
}
