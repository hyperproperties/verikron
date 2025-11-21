use crate::graphs::{edges::ReadEdges, vertices::ReadVertices};

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
pub trait Graph {
    /// Vertex storage component.
    type Vertices: ReadVertices;

    /// Edge storage component.
    type Edges: ReadEdges;

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
