pub trait Edges {
    /// Type used to identify vertices.
    ///
    /// Typically a small copyable value such as usize.
    type Vertex: Eq + Copy;

    /// Type used to identify edges.
    ///
    /// Usually a small copyable value such as usize.
    /// This makes it possible for a pair of vertices to be connected
    /// by multiple distinct edges.
    /// Each directed edge can be viewed as a triple with source, edge_id, and destination.
    type Edge: Eq + Copy;

    /// Iterator over all edges in the graph.
    ///
    /// Each item is a triple with source, edge_id, and destination.
    /// The order of edges depends on the implementation, but should be stable
    /// for a given graph instance unless the graph is mutated.
    type Edges<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns an iterator over all edges in the graph.
    ///
    /// Each item is a triple with source, edge_id, and destination.
    fn edges(&self) -> Self::Edges<'_>;

    /// Returns the number of edges.
    fn edge_count(&self) -> usize {
        self.edges().count()
    }
}
