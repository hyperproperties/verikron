pub trait Vertices {
    /// Type used to identify vertices.
    type Vertex;

    /// Iterator over all vertices in the graph.
    ///
    /// The order of vertices depends on the implementation, but should be stable
    /// for a given graph instance unless the graph is mutated.
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
