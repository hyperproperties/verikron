use crate::graphs::edges::ReadEdges;

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
