use crate::graphs::graph::{FiniteVertices, VertexType};

/// Common vertex and hyperedge identifier types.
pub trait HyperedgeType: VertexType {
    /// Type used to identify hyperedges.
    type Hyperedge: Eq + Copy;
}

/// Finite hyperedge store with global access to all hyperedges.
pub trait Hyperedges: HyperedgeType {
    /// Iterator over all hyperedges.
    type Hyperedges<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    /// Returns an iterator over all hyperedges.
    fn hyperedges(&self) -> Self::Hyperedges<'_>;

    /// Returns the number of hyperedges.
    fn hyperedge_count(&self) -> usize {
        self.hyperedges().count()
    }
}

/// Hyperedge removal.
pub trait RemoveHyperedge: HyperedgeType {
    /// Removes `hyperedge` and returns whether it existed.
    fn remove_hyperedge(&mut self, hyperedge: Self::Hyperedge) -> bool;
}

/// Finite hypergraph composed from a vertex store and a hyperedge store.
pub trait Hypergraph: VertexType {
    /// Common hyperedge type.
    type Hyperedge: Eq + Copy;

    /// Vertex storage component.
    type Vertices: FiniteVertices<Vertex = Self::Vertex>;

    /// Hyperedge storage component.
    type Hyperedges: Hyperedges<Vertex = Self::Vertex, Hyperedge = Self::Hyperedge>;

    /// Returns the vertex store.
    fn vertex_store(&self) -> &Self::Vertices;

    /// Returns the hyperedge store.
    fn hyperedge_store(&self) -> &Self::Hyperedges;

    /// Returns a default size measure: vertices plus hyperedges.
    fn size(&self) -> usize {
        self.vertex_store().vertex_count() + self.hyperedge_store().hyperedge_count()
    }

    /// Returns whether the hypergraph is empty.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

/// Undirected hyperedge insertion.
pub trait InsertUndirectedHyperedge: HyperedgeType {
    /// Inserts a hyperedge from its member vertices.
    fn insert_hyperedge<I>(&mut self, members: I) -> Option<Self::Hyperedge>
    where
        I: IntoIterator<Item = Self::Vertex>;
}

/// Undirected hypergraph interface without assumed global enumeration.
pub trait InfiniteUndirectedHypergraph: HyperedgeType {
    /// Iterator over the member vertices of a hyperedge.
    type Members<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over the hyperedges incident to a vertex.
    type Incident<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    /// Returns the member vertices of `hyperedge`.
    fn members(&self, hyperedge: Self::Hyperedge) -> Self::Members<'_>;

    /// Returns the hyperedges incident to `vertex`.
    fn incident(&self, vertex: Self::Vertex) -> Self::Incident<'_>;
}

/// Finite undirected hypergraph.
///
/// Extends [`InfiniteUndirectedHypergraph`] with convenience queries.
pub trait UndirectedHypergraph: InfiniteUndirectedHypergraph {
    /// Returns the number of members of `hyperedge`.
    fn cardinality(&self, hyperedge: Self::Hyperedge) -> usize {
        self.members(hyperedge).count()
    }

    /// Returns the number of incident hyperedges of `vertex`.
    fn degree(&self, vertex: Self::Vertex) -> usize {
        self.incident(vertex).count()
    }

    /// Returns whether `hyperedge` contains `vertex`.
    fn contains(&self, hyperedge: Self::Hyperedge, vertex: Self::Vertex) -> bool {
        self.members(hyperedge).any(|u| u == vertex)
    }
}

/// Mutable undirected hyperedge store.
pub trait UndirectedHypergraphMut:
    Hyperedges + InsertUndirectedHyperedge + RemoveHyperedge
{
}
impl<T> UndirectedHypergraphMut for T where
    T: Hyperedges + InsertUndirectedHyperedge + RemoveHyperedge
{
}

/// Directed hyperedge insertion.
pub trait InsertDirectedHyperedge: HyperedgeType {
    /// Inserts a directed hyperedge from tail and head vertices.
    fn insert_hyperedge<T, H>(&mut self, tail: T, head: H) -> Option<Self::Hyperedge>
    where
        T: IntoIterator<Item = Self::Vertex>,
        H: IntoIterator<Item = Self::Vertex>;
}

/// Directed hypergraph interface without assumed global enumeration.
pub trait InfiniteDirectedHypergraph: HyperedgeType {
    /// Iterator over the tail vertices of a hyperedge.
    type Tail<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over the head vertices of a hyperedge.
    type Head<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over hyperedges whose tail contains `vertex`.
    type Outgoing<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    /// Iterator over hyperedges whose head contains `vertex`.
    type Ingoing<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    /// Returns the tail vertices of `hyperedge`.
    fn tail(&self, hyperedge: Self::Hyperedge) -> Self::Tail<'_>;

    /// Returns the head vertices of `hyperedge`.
    fn head(&self, hyperedge: Self::Hyperedge) -> Self::Head<'_>;

    /// Returns all hyperedges whose tail contains `vertex`.
    fn outgoing(&self, vertex: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns all hyperedges whose head contains `vertex`.
    fn ingoing(&self, vertex: Self::Vertex) -> Self::Ingoing<'_>;
}

/// Finite directed hypergraph.
///
/// Extends [`InfiniteDirectedHypergraph`] with convenience queries.
pub trait DirectedHypergraph: InfiniteDirectedHypergraph {
    /// Returns whether `vertex` is in the tail of `hyperedge`.
    fn in_tail(&self, hyperedge: Self::Hyperedge, vertex: Self::Vertex) -> bool {
        self.tail(hyperedge).any(|u| u == vertex)
    }

    /// Returns whether `vertex` is in the head of `hyperedge`.
    fn in_head(&self, hyperedge: Self::Hyperedge, vertex: Self::Vertex) -> bool {
        self.head(hyperedge).any(|u| u == vertex)
    }

    /// Returns the size of the tail of `hyperedge`.
    fn tail_cardinality(&self, hyperedge: Self::Hyperedge) -> usize {
        self.tail(hyperedge).count()
    }

    /// Returns the size of the head of `hyperedge`.
    fn head_cardinality(&self, hyperedge: Self::Hyperedge) -> usize {
        self.head(hyperedge).count()
    }

    /// Returns the number of outgoing hyperedges of `vertex`.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex).count()
    }

    /// Returns the number of ingoing hyperedges of `vertex`.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.ingoing(vertex).count()
    }
}

/// Mutable directed hyperedge store.
pub trait DirectedHypergraphMut: Hyperedges + InsertDirectedHyperedge + RemoveHyperedge {}
impl<T> DirectedHypergraphMut for T where T: Hyperedges + InsertDirectedHyperedge + RemoveHyperedge {}

/// Undirected hyperedge described by its member vertices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HyperedgeMembers<V> {
    /// Vertices contained in the hyperedge.
    pub members: Vec<V>,
}

impl<V> HyperedgeMembers<V> {
    /// Creates a hyperedge from its member vertices.
    #[inline]
    pub fn new<I>(members: I) -> Self
    where
        I: IntoIterator<Item = V>,
    {
        Self {
            members: members.into_iter().collect(),
        }
    }
}

/// Directed hyperedge described by its tail and head vertices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DirectedHyperedge<V> {
    /// Tail vertices.
    pub tail: Vec<V>,

    /// Head vertices.
    pub head: Vec<V>,
}

impl<V> DirectedHyperedge<V> {
    /// Creates a directed hyperedge from tail and head vertices.
    #[inline]
    pub fn new<T, H>(tail: T, head: H) -> Self
    where
        T: IntoIterator<Item = V>,
        H: IntoIterator<Item = V>,
    {
        Self {
            tail: tail.into_iter().collect(),
            head: head.into_iter().collect(),
        }
    }
}

/// Undirected hypergraph constructible from owned hyperedges.
pub trait FromHyperedges: Sized + VertexType {
    /// Creates a hypergraph from owned hyperedges.
    fn from_hyperedges<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = HyperedgeMembers<Self::Vertex>>;
}

/// Directed hypergraph constructible from owned directed hyperedges.
pub trait FromDirectedHyperedges: Sized + VertexType {
    /// Creates a directed hypergraph from owned directed hyperedges.
    fn from_directed_hyperedges<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = DirectedHyperedge<Self::Vertex>>;
}