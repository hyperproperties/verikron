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
pub trait Directed {
    /// Type used to identify vertices.
    ///
    /// Typically a small copyable value such as usize.
    type Vertex: Copy + Eq;

    /// Type used to identify edges.
    ///
    /// Usually a small copyable value such as usize.
    /// This makes it possible for a pair of vertices to be connected
    /// by multiple distinct edges.
    /// Each directed edge can be viewed as a triple with source, edge_id, and destination.
    type Edge: Copy + Eq;

    /// Iterator over all vertices in the graph.
    ///
    /// The order of vertices depends on the implementation, but should be stable
    /// for a given graph instance unless the graph is mutated.
    type Vertices<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over all edges in the graph.
    ///
    /// Each item is a triple with source, edge_id, and destination.
    /// The order of edges depends on the implementation, but should be stable
    /// for a given graph instance unless the graph is mutated.
    type Edges<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Number of vertices.
    fn vertex_count(&self) -> usize;

    /// Number of edges.
    fn edge_count(&self) -> usize;

    /// Size measure for the whole graph.
    ///
    /// Equal to vertex_count plus edge_count.
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
    ///
    /// Each item is a triple with source, edge_id, and destination.
    fn edges(&self) -> Self::Edges<'_>;

    /// Source vertex of an edge.
    fn source(&self, edge: Self::Edge) -> Self::Vertex;

    /// Destination vertex of an edge.
    fn target(&self, edge: Self::Edge) -> Self::Vertex;

    /// Returns all outgoing edges from the given source vertex.
    ///
    /// Each item is a triple with source, edge_id, and destination,
    /// and the source component is equal to the given source vertex.
    fn from(&self, source: Self::Vertex) -> Self::Edges<'_>;

    /// Returns the number of edges with the given source vertex.
    fn from_degree(&self, source: Self::Vertex) -> usize {
        self.from(source).count()
    }

    /// Returns all incoming edges to the given destination vertex.
    ///
    /// Each item is a triple with source, edge_id, and destination,
    /// and the destination component is equal to the given destination vertex.
    fn to(&self, destination: Self::Vertex) -> Self::Edges<'_>;

    /// Returns the number of edges with the given destination vertex.
    fn to_degree(&self, destination: Self::Vertex) -> usize {
        self.to(destination).count()
    }

    /// Returns true when there exists at least one edge whose source is from,
    /// and whose destination is to.
    ///
    /// This checks for a single step edge only,
    /// it does not perform a reachability query through longer paths.
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.from(from).any(|(_, _, dst)| dst == to)
    }

    /// Returns true when edge is a directed edge whose source is from,
    /// and whose destination is to.
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.from(from).any(|(_, e, dst)| e == edge && dst == to)
    }
}

/// Returns an iterator over all edges whose source is from,
/// and whose destination is to.
///
/// The iterator yields triples with source, edge, and destination.
/// It is a generic helper for any type that implements Directed.
pub fn connecting<G>(
    g: &G,
    from: G::Vertex,
    to: G::Vertex,
) -> impl Iterator<Item = (G::Vertex, G::Edge, G::Vertex)> + '_
where
    G: Directed,
{
    g.from(from)
        .filter(move |&(_, _, destination)| destination == to)
}

/// A finite directed multigraph abstraction.
///
/// DiMultiGraph describes a directed graph in which vertices and edges
/// are identified by small copyable values.
/// Two vertices may be connected by multiple distinct edges,
/// in that case each edge is distinguished only by its edge identifier.
///
/// The trait does not add new methods beyond Directed.
/// It serves as a marker for graphs that allow parallel edges.
pub trait DiMultiGraph: Directed { }

/// A finite directed graph abstraction with no parallel edges.
///
/// DiGraph extends Directed with an edge type that carries no information,
/// the associated edge type is the unit type.
/// Implementors are expected to maintain the invariant that for any ordered
/// pair of vertices there is at most one directed edge from the first vertex
/// to the second vertex.
///
/// In other words edges are determined entirely by their pair of endpoint
/// vertices, and the edge identifier is present only for type level symmetry
/// and has no runtime cost.
///
/// This trait does not add new methods beyond Directed.
/// It documents the simple graph invariant and can be used to select
/// algorithms that rely on the absence of parallel edges.
pub trait DiGraph: Directed<Edge = ()> { }