/// Common vertex identifier type.
pub trait VertexType {
    /// Type used to identify vertices.
    type Vertex: Eq + Copy;
}

/// Vertex store with global enumeration.
///
/// The iterator may be finite or infinite.
pub trait Vertices: VertexType {
    /// Iterator over all vertices.
    type Vertices<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Returns an iterator over all vertices.
    fn vertices(&self) -> Self::Vertices<'_>;
}

/// Finite vertex store.
pub trait FiniteVertices: Vertices {
    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize {
        self.vertices().count()
    }

    /// Returns whether `vertex` exists.
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        self.vertices().any(|u| &u == vertex)
    }
}

/// Vertex insertion.
pub trait InsertVertex: VertexType {
    /// Inserts a new vertex and returns its identifier on success.
    fn insert_vertex(&mut self) -> Option<Self::Vertex>;
}

/// Vertex removal.
pub trait RemoveVertex: VertexType {
    /// Removes `vertex` and returns whether it existed.
    fn remove_vertex(&mut self, vertex: Self::Vertex) -> bool;
}

/// Common edge identifier type.
pub trait EdgeType: VertexType {
    /// Type used to identify edges.
    type Edge: Eq + Copy;
}

/// Edge store with global enumeration.
///
/// The iterator may be finite or infinite.
pub trait Edges: EdgeType {
    /// Iterator over all edge identifiers.
    type Edges<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns an iterator over all edge identifiers.
    fn edges(&self) -> Self::Edges<'_>;
}

/// Finite edge store.
pub trait FiniteEdges: Edges {
    /// Returns the number of edges.
    fn edge_count(&self) -> usize {
        self.edges().count()
    }

    /// Returns true if the edge is in this collection.
    fn contains_edge(&self, edge: &Self::Edge) -> bool {
        self.edges().any(|e| &e == edge)
    }
}

/// Edge insertion.
pub trait InsertEdge: EdgeType {
    /// Inserts an edge between the given endpoints.
    fn insert_edge(&mut self, from: Self::Vertex, to: Self::Vertex) -> Option<Self::Edge>;
}

/// Edge removal.
pub trait RemoveEdge: EdgeType {
    /// Returns true if the `edge` was removed.
    fn remove_edge(&mut self, edge: Self::Edge) -> bool;
}

pub type VertexOf<S> = <S as VertexType>::Vertex;
pub type EdgeOf<S> = <S as EdgeType>::Edge;
pub type VerticesOf<S> = <S as Structure>::Vertices;
pub type EdgesOf<S> = <S as Structure>::Edges;

/// Structure composed from a vertex store and an edge store.
///
/// This is the common base for graphs, hypergraphs, and similar objects.
/// It only says that the structure has vertices and edges; it says nothing
/// about how edges are interpreted.
pub trait Structure: VertexType + EdgeType {
    /// Vertex storage component.
    type Vertices: Vertices<Vertex = Self::Vertex>;

    /// Edge storage component.
    type Edges: Edges<Vertex = Self::Vertex, Edge = Self::Edge>;

    /// Returns the edge store.
    fn edge_store(&self) -> &Self::Edges;

    /// Returns the vertex store.
    fn vertex_store(&self) -> &Self::Vertices;
}

/// Finite structure composed from a finite vertex store and a finite edge store.
pub trait FiniteStructure: Structure
where
    Self::Vertices: FiniteVertices<Vertex = Self::Vertex>,
    Self::Edges: FiniteEdges<Vertex = Self::Vertex, Edge = Self::Edge>,
{
    /// Returns a default size measure: vertices plus edges.
    fn size(&self) -> usize {
        self.vertex_store().vertex_count() + self.edge_store().edge_count()
    }

    /// Returns whether the structure is empty.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

impl<T> FiniteStructure for T
where
    T: Structure,
    T::Vertices: FiniteVertices<Vertex = T::Vertex>,
    T::Edges: FiniteEdges<Vertex = T::Vertex, Edge = T::Edge>,
{
}
