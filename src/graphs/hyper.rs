use crate::graphs::graph::ReadVertices;

/// Common identifier types used by hypergraph traits.
///
/// Hyperedges defines the basic vertex and hyperedge identifier types
/// shared by the hypergraph interfaces in this module.
///
/// Vertices are identified by small copyable values.
/// Hyperedges are likewise identified by small copyable values.
///
/// Concrete storage backends are free to choose the actual identifier types,
/// but in most implementations these will be compact values such as `usize`.
pub trait Hyperedges {
    /// Identifier type for vertices.
    type Vertex: Eq + Copy;

    /// Identifier type for hyperedges.
    type Hyperedge: Eq + Copy;
}

/// Read-only access to hyperedge identifiers.
///
/// ReadHyperedges provides iteration over all hyperedges in a hypergraph,
/// without committing to a particular structural interpretation
/// such as undirected membership or directed tail/head incidence.
///
/// Structural queries are provided by higher-level traits such as
/// `UndirectedHypergraph` and `DirectedHypergraph`.
///
/// Iteration uses an associated iterator type, so implementations can avoid
/// extra allocation and avoid dynamic dispatch.
pub trait ReadHyperedges: Hyperedges {
    /// Iterator over all hyperedge identifiers.
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

/// High-level abstraction for finite hypergraphs.
///
/// ReadHypergraph bundles together a vertex store and a hyperedge store
/// that use a common vertex identifier type.
///
/// This mirrors the structure of the ordinary graph traits:
/// vertices and hyperedges are stored separately, while algorithms and
/// higher-level views can be written generically over the combined interface.
///
/// The trait does not impose whether the hypergraph is directed or undirected.
/// That distinction is handled by `UndirectedHypergraph` and `DirectedHypergraph`.
pub trait ReadHypergraph {
    /// Common vertex identifier type.
    type Vertex: Eq + Copy;

    /// Common hyperedge identifier type.
    type Hyperedge: Eq + Copy;

    /// Vertex storage component.
    type Vertices: ReadVertices<Vertex = Self::Vertex>;

    /// Hyperedge storage component.
    type Hyperedges: ReadHyperedges<Vertex = Self::Vertex, Hyperedge = Self::Hyperedge>;

    /// Access to the vertex store.
    fn vertex_store(&self) -> &Self::Vertices;

    /// Access to the hyperedge store.
    fn hyperedge_store(&self) -> &Self::Hyperedges;

    /// Returns the total size of the hypergraph,
    /// measured as number of vertices plus number of hyperedges.
    fn size(&self) -> usize {
        self.vertex_store().vertex_count() + self.hyperedge_store().hyperedge_count()
    }

    /// Returns true when the hypergraph has no vertices and no hyperedges.
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

/// Core abstraction for finite undirected hypergraphs.
///
/// An undirected hypergraph consists of vertices and hyperedges,
/// where each hyperedge is incident to zero or more vertices.
///
/// Unlike an ordinary undirected graph edge, a hyperedge may connect
/// any number of vertices rather than exactly two.
///
/// The trait focuses on structural access to hyperedges and their incidences.
/// Algorithms such as traversal, connectedness, clustering,
/// or combinational analysis are expected to be written as generic helper
/// functions or extension traits built on top of this interface.
///
/// Iteration uses associated iterator types, so implementations can avoid
/// extra allocation and avoid dynamic dispatch.
pub trait UndirectedHypergraph: ReadHyperedges {
    /// Iterator over the member vertices of a hyperedge.
    type Members<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Iterator over the hyperedges incident to a vertex.
    type Incident<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    /// Returns all member vertices of the given hyperedge.
    fn members(&self, e: Self::Hyperedge) -> Self::Members<'_>;

    /// Returns all hyperedges incident to the given vertex.
    fn incident(&self, v: Self::Vertex) -> Self::Incident<'_>;

    /// Returns the number of member vertices of a hyperedge.
    fn cardinality(&self, e: Self::Hyperedge) -> usize {
        self.members(e).count()
    }

    /// Returns the number of incident hyperedges of a vertex.
    fn degree(&self, v: Self::Vertex) -> usize {
        self.incident(v).count()
    }

    /// Returns true when the given hyperedge contains the given vertex.
    fn contains(&self, e: Self::Hyperedge, v: Self::Vertex) -> bool {
        self.members(e).any(|u| u == v)
    }
}

/// Core abstraction for finite directed hypergraphs.
///
/// A directed hypergraph consists of vertices and hyperedges,
/// where each hyperedge has a tail set and a head set.
///
/// This generalizes an ordinary directed edge:
/// instead of connecting one source vertex to one destination vertex,
/// a directed hyperedge connects a set of tail vertices to a set of head vertices.
///
/// The trait focuses on structural access to hyperedges and their incidences.
/// Algorithms such as propagation, reachability, dependency analysis,
/// or flow-like procedures are expected to be written as generic helper
/// functions or extension traits built on top of this interface.
///
/// Iteration uses associated iterator types, so implementations can avoid
/// extra allocation and avoid dynamic dispatch.
pub trait DirectedHypergraph: ReadHyperedges {
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

    /// Returns the tail vertices of the given hyperedge.
    fn tail(&self, e: Self::Hyperedge) -> Self::Tail<'_>;

    /// Returns the head vertices of the given hyperedge.
    fn head(&self, e: Self::Hyperedge) -> Self::Head<'_>;

    /// Returns all hyperedges whose tail contains the given vertex.
    fn outgoing(&self, v: Self::Vertex) -> Self::Outgoing<'_>;

    /// Returns all hyperedges whose head contains the given vertex.
    fn ingoing(&self, v: Self::Vertex) -> Self::Ingoing<'_>;

    /// Returns the number of tail vertices of a hyperedge.
    fn tail_cardinality(&self, e: Self::Hyperedge) -> usize {
        self.tail(e).count()
    }

    /// Returns the number of head vertices of a hyperedge.
    fn head_cardinality(&self, e: Self::Hyperedge) -> usize {
        self.head(e).count()
    }

    /// Returns the number of outgoing hyperedges of a vertex,
    /// meaning hyperedges whose tail contains that vertex.
    fn outgoing_degree(&self, v: Self::Vertex) -> usize {
        self.outgoing(v).count()
    }

    /// Returns the number of ingoing hyperedges of a vertex,
    /// meaning hyperedges whose head contains that vertex.
    fn ingoing_degree(&self, v: Self::Vertex) -> usize {
        self.ingoing(v).count()
    }
}

/// Insertion interface for undirected hypergraphs.
///
/// The inserted hyperedge is defined by its member vertices.
///
/// Implementations may impose additional restrictions,
/// such as rejecting invalid vertices or duplicate hyperedges.
pub trait InsertUndirectedHyperedge: Hyperedges {
    /// Inserts a hyperedge with the given member vertices.
    ///
    /// Returns the identifier of the inserted hyperedge on success,
    /// or `None` if insertion fails.
    fn insert_hyperedge<I>(&mut self, members: I) -> Option<Self::Hyperedge>
    where
        I: IntoIterator<Item = Self::Vertex>;
}

/// Insertion interface for directed hypergraphs.
///
/// The inserted hyperedge is defined by a tail set and a head set.
///
/// Implementations may impose additional restrictions,
/// such as rejecting invalid vertices or duplicate hyperedges.
pub trait InsertDirectedHyperedge: Hyperedges {
    /// Inserts a directed hyperedge with the given tail and head vertices.
    ///
    /// Returns the identifier of the inserted hyperedge on success,
    /// or `None` if insertion fails.
    fn insert_hyperedge<T, H>(&mut self, tail: T, head: H) -> Option<Self::Hyperedge>
    where
        T: IntoIterator<Item = Self::Vertex>,
        H: IntoIterator<Item = Self::Vertex>;
}

/// Removal interface for hypergraphs.
///
/// Hyperedge removal deletes an existing hyperedge from the structure.
/// The exact treatment of identifiers after removal is implementation-defined.
/// Some backends may preserve stable identifiers,
/// while others may compact storage and invalidate old ones.
pub trait RemoveHyperedge: Hyperedges {
    /// Removes the given hyperedge.
    ///
    /// Returns true if a hyperedge was removed,
    /// and false if the identifier did not refer to an existing hyperedge.
    fn remove_hyperedge(&mut self, e: Self::Hyperedge) -> bool;
}
