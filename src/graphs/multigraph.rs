/// A finite directed multigraph abstraction.
///
/// `DiMultiGraph` describes a directed graph in which vertices and edges
/// are identified by small copyable values.
/// Two vertices may be connected by multiple distinct edges
/// in that case each edge is distinguished only by its edge identifier
///
/// The trait is intentionally minimal and structural
/// it focuses on access to vertices edges and incidences.
/// Algorithms such as reachability and shortest path computation can be
/// provided as helper functions or extension traits built on top of this
/// interface.
///
/// Iteration uses associated iterator types so implementations can avoid
/// extra allocation and avoid dynamic dispatch.
pub trait DiMultiGraph {
    /// Type used to identify vertices.
    ///
    /// Typically a small copyable value such as `usize`.
    type Vertex: Copy + Eq;

    /// Type used to identify edges.
    ///
    /// Usually a small copyable value such as `usize`.
    /// This makes it possible for a pair of vertices to be connected
    /// by multiple distinct edges.
    /// Each directed edge can be viewed as a triple: (to, edge, from).
    type Edge: Copy + Eq;

    /// Iterator over all vertices in the graph.
    ///
    /// The order of vertices is implementation defined but should be stable
    /// for a given graph instance unless the graph is mutated.
    type Vertices<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over all edges in the graph.
    ///
    /// Each item is a triple source edge_id destination.
    /// The order of edges is implementation defined but should be stable
    /// for a given graph instance unless the graph is mutated.
    type Edges<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Number of vertices.
    fn vertex_count(&self) -> usize;

    /// Number of edges.
    fn edge_count(&self) -> usize;

    /// Size measure for the whole graph.
    fn size(&self) -> usize {
        self.vertex_count() + self.edge_count()
    }

    /// Returns true when the graph has no vertices.
    fn is_empty(&self) -> bool {
        self.vertex_count() == 0
    }

    /// Returns an iterator over all vertices in the graph.
    fn vertices(&self) -> Self::Vertices<'_>;

    /// Returns an iterator over all edges in the graph.
    fn edges(&self) -> Self::Edges<'_>;

    /// Returns all outgoing edges from the given source vertex
    ///
    /// Each item is a triple source edge_id destination
    /// and the source component is equal to the given source vertex
    fn from(&self, source: Self::Vertex) -> Self::Edges<'_>;

    /// Returns the number of edges with the given source vertex
    fn from_degree(&self, source: Self::Vertex) -> usize {
        self.from(source).count()
    }

    /// Returns all incoming edges to the given destination vertex
    ///
    /// Each item is a triple source edge_id destination
    /// and the destination component is equal to the given destination vertex
    fn to(&self, destination: Self::Vertex) -> Self::Edges<'_>;

    /// Returns the number of edges with the given destination vertex
    fn to_degree(&self, destination: Self::Vertex) -> usize {
        self.to(destination).count()
    }

    /// Returns true when there exists at least one edge whose source is from
    /// and whose destination is to
    ///
    /// This checks for a single step edge only
    /// it does not perform a reachability query through longer paths
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.from(from).any(|(_, _, dst)| dst == to)
    }

    /// Returns true when edge is a directed edge whose source is from
    /// and whose destination is to
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.from(from).any(|(_, e, dst)| e == edge && dst == to)
    }
}

/// Returns an iterator over all edges whose source is from
/// and whose destination is to
///
/// The iterator yields triples of source edge and destination
/// It is a generic helper for any type that implements DiMultiGraph
pub fn connecting<G>(
    g: &G,
    from: G::Vertex,
    to: G::Vertex,
) -> impl Iterator<Item = (G::Vertex, G::Edge, G::Vertex)> + '_
where
    G: DiMultiGraph,
{
    g.from(from)
        .filter(move |&(_, _, destination)| destination == to)
}