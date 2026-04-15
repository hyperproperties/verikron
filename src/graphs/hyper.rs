use crate::graphs::graph::Vertices;

/// Finite hypergraph with global access to its hyperedges.
pub trait Hyperedges {
    /// Type used to identify vertices.
    type Vertex: Eq + Copy;

    /// Type used to identify hyperedges.
    type Hyperedge: Eq + Copy;

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
/// Hyperedge insertion.
///
/// This is the common insertion trait for hypergraphs whose hyperedges
/// are described by a collection of member vertices.
pub trait InsertHyperedge: Hyperedges {
    /// Inserts a hyperedge and returns its identifier on success.
    fn insert_hyperedge<I>(&mut self, members: I) -> Option<Self::Hyperedge>
    where
        I: IntoIterator<Item = Self::Vertex>;
}

/// Hyperedge removal.
pub trait RemoveHyperedge: Hyperedges {
    /// Removes `hyperedge` and returns whether it existed.
    fn remove_hyperedge(&mut self, hyperedge: Self::Hyperedge) -> bool;
}

/// Mutable hyperedge store.
pub trait HyperedgesMut: Hyperedges + InsertHyperedge + RemoveHyperedge {}
impl<T> HyperedgesMut for T where T: Hyperedges + InsertHyperedge + RemoveHyperedge {}

/// High-level abstraction for a finite hypergraph.
///
/// A hypergraph is composed from a vertex store and a hyperedge store
/// that share the same vertex identifier type.
pub trait Hypergraph {
    /// Common vertex type.
    type Vertex: Eq + Copy;

    /// Common hyperedge type.
    type Hyperedge: Eq + Copy;

    /// Vertex storage component.
    type Vertices: Vertices<Vertex = Self::Vertex>;

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

    /// Returns true when the hypergraph is empty.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

/// Core abstraction for undirected hypergraphs.
///
/// Each hyperedge is incident to zero or more vertices.
pub trait UndirectedHypergraph: Hyperedges {
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

    /// Returns the number of members of `hyperedge`.
    fn cardinality(&self, hyperedge: Self::Hyperedge) -> usize {
        self.members(hyperedge).count()
    }

    /// Returns the number of incident hyperedges of `vertex`.
    fn degree(&self, vertex: Self::Vertex) -> usize {
        self.incident(vertex).count()
    }

    /// Returns true when `hyperedge` contains `vertex`.
    fn contains(&self, hyperedge: Self::Hyperedge, vertex: Self::Vertex) -> bool {
        self.members(hyperedge).any(|u| u == vertex)
    }
}

/// Core abstraction for directed hypergraphs.
///
/// Each hyperedge has a tail set and a head set.
pub trait DirectedHypergraph: Hyperedges {
    /// Iterator over the tail vertices of a hyperedge.
    type Tail<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over the head vertices of a hyperedge.
    type Head<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over hyperedges whose tail contains the given vertex.
    type Outgoing<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    /// Iterator over hyperedges whose head contains the given vertex.
    type Ingoing<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    /// Returns the tail vertices of `hyperedge`.
    fn tail(&self, hyperedge: Self::Hyperedge) -> Self::Tail<'_>;

    /// Returns the head vertices of `hyperedge`.
    fn head(&self, hyperedge: Self::Hyperedge) -> Self::Head<'_>;

    /// Returns true if the vertex is in the edge's tail.
    fn in_tail(&self, e: Self::Hyperedge, v: Self::Vertex) -> bool {
        self.tail(e).any(|u| u == v)
    }

    /// Returns true if the vertex is in the edge's head.
    fn in_head(&self, e: Self::Hyperedge, v: Self::Vertex) -> bool {
        self.head(e).any(|u| u == v)
    }

    /// Returns all hyperedges whose tail contains `vertex`.
    fn outgoing(&self, vertex: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns all hyperedges whose head contains `vertex`.
    fn ingoing(&self, vertex: Self::Vertex) -> Self::Ingoing<'_>;

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

/// Insertion interface for directed hypergraphs.
pub trait InsertDirectedHyperedge: Hyperedges {
    /// Inserts a directed hyperedge with the given tail and head vertices.
    fn insert_hyperedge<T, H>(&mut self, tail: T, head: H) -> Option<Self::Hyperedge>
    where
        T: IntoIterator<Item = Self::Vertex>,
        H: IntoIterator<Item = Self::Vertex>;
}