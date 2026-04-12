use crate::graphs::{
    colored::{ColoredVertices, InsertColoredVertex, ReadColoredVertices},
    vertices::RemoveVertex,
};

/// Canonical side type used by bipartite-vertex traits.
///
/// Every vertex of a bipartite graph belongs to exactly one of two sides:
/// the left side or the right side.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Side {
    Left = 0,
    Right = 1,
}

impl Side {
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

    #[inline]
    pub const fn as_bit(self) -> bool {
        matches!(self, Self::Right)
    }

    #[inline]
    pub const fn from_bit(bit: bool) -> Self {
        if bit { Self::Right } else { Self::Left }
    }

    #[inline]
    pub fn from_bit_ref(bit: bool) -> &'static Self {
        if bit { &Self::Right } else { &Self::Left }
    }
}

impl From<bool> for Side {
    #[inline]
    fn from(bit: bool) -> Self {
        Self::from_bit(bit)
    }
}

impl From<Side> for bool {
    #[inline]
    fn from(side: Side) -> Self {
        side.as_bit()
    }
}

/// A graph whose vertices are partitioned into two sides.
///
/// This trait specializes [`ColoredVertices`] to the canonical bipartite
/// coloring [`Side`]. Every existing vertex is expected to belong to exactly
/// one of the two sides.
pub trait BipartiteVertices: ColoredVertices<Color = Side> {}

/// Blanket implementation of [`BipartiteVertices`] for any type whose vertex
/// color type is [`Side`].
impl<T> BipartiteVertices for T where T: ColoredVertices<Color = Side> {}

/// A bipartite graph that supports read-only access to vertex sides.
///
/// This trait extends [`ReadColoredVertices`] with bipartite-specific
/// terminology and convenience methods. The underlying coloring is expected
/// to be total for all existing vertices.
pub trait ReadBipartiteVertices: BipartiteVertices + ReadColoredVertices {
    /// Returns the side of the given vertex.
    ///
    /// The `vertex` parameter is the identifier of the vertex whose side
    /// should be returned.
    ///
    /// On success, returns `Some(side)` where `side` is the side containing
    /// the given vertex. If no such vertex exists in the graph, returns
    /// `None`.
    fn side_of(&self, vertex: Self::Vertex) -> Option<Side> {
        self.vertex_color(vertex).copied()
    }

    /// Returns true if the given vertex belongs to the left side.
    ///
    /// Returns `false` if the vertex does not exist.
    fn is_left(&self, vertex: Self::Vertex) -> bool {
        self.side_of(vertex) == Some(Side::Left)
    }

    /// Returns true if the given vertex belongs to the right side.
    ///
    /// Returns `false` if the vertex does not exist.
    fn is_right(&self, vertex: Self::Vertex) -> bool {
        self.side_of(vertex) == Some(Side::Right)
    }

    /// Returns true if both vertices exist and belong to the same side.
    ///
    /// If either vertex does not exist, returns `None`.
    fn same_side(&self, u: Self::Vertex, v: Self::Vertex) -> Option<bool> {
        Some(self.side_of(u)? == self.side_of(v)?)
    }

    /// Returns true if both vertices exist and belong to opposite sides.
    ///
    /// If either vertex does not exist, returns `None`.
    fn opposite_sides(&self, u: Self::Vertex, v: Self::Vertex) -> Option<bool> {
        Some(self.side_of(u)? != self.side_of(v)?)
    }

    /// Returns the number of vertices on the left side.
    fn left_vertex_count(&self) -> usize {
        self.vertices().filter(|&v| self.is_left(v)).count()
    }

    /// Returns the number of vertices on the right side.
    fn right_vertex_count(&self) -> usize {
        self.vertices().filter(|&v| self.is_right(v)).count()
    }
}

/// Blanket implementation of [`ReadBipartiteVertices`] for any type that
/// provides the required capabilities.
impl<T> ReadBipartiteVertices for T where T: BipartiteVertices + ReadColoredVertices {}

/// A bipartite graph that supports insertion of vertices on a chosen side.
///
/// This trait extends [`InsertColoredVertex`] with bipartite-specific
/// insertion operations.
pub trait InsertBipartiteVertex: BipartiteVertices + InsertColoredVertex {
    /// Inserts a new vertex on the given side.
    ///
    /// On success, returns `Some(v)` where `v` is the identifier of the
    /// newly inserted vertex. If the vertex cannot be inserted, returns
    /// `None`.
    fn insert_vertex_on(&mut self, side: Side) -> Option<Self::Vertex> {
        self.insert_colored_vertex(side)
    }

    /// Inserts a new vertex on the left side.
    ///
    /// On success, returns `Some(v)` where `v` is the identifier of the
    /// newly inserted vertex. If the vertex cannot be inserted, returns
    /// `None`.
    fn insert_left_vertex(&mut self) -> Option<Self::Vertex> {
        self.insert_vertex_on(Side::Left)
    }

    /// Inserts a new vertex on the right side.
    ///
    /// On success, returns `Some(v)` where `v` is the identifier of the
    /// newly inserted vertex. If the vertex cannot be inserted, returns
    /// `None`.
    fn insert_right_vertex(&mut self) -> Option<Self::Vertex> {
        self.insert_vertex_on(Side::Right)
    }
}

/// Blanket implementation of [`InsertBipartiteVertex`] for any type that
/// provides the required capabilities.
impl<T> InsertBipartiteVertex for T where T: BipartiteVertices + InsertColoredVertex {}

/// A bipartite graph that supports changing the side of existing vertices.
///
/// Unlike generic recoloring, repartitioning must preserve the bipartite
/// invariant. Implementations are expected to reject changes that would make
/// the graph non-bipartite.
pub trait RepartitionVertex: BipartiteVertices {
    /// Moves an existing vertex to the given side.
    ///
    /// The `vertex` parameter is the identifier of the vertex to move. The
    /// `side` parameter is the side to which the vertex should be moved.
    ///
    /// On success, returns `Ok(old_side)` where `old_side` is the previous
    /// side of the vertex.
    ///
    /// If no such vertex exists or if moving the vertex would violate
    /// the bipartite invariant, returns Error.
    fn repartition_vertex(&mut self, vertex: Self::Vertex, side: Side) -> Result<Side, ()>;
}

/// A bipartite graph that supports both querying and mutating its partition.
///
/// This is a convenience alias for types that implement:
/// - [`BipartiteVertices`] for the canonical side type,
/// - [`ReadBipartiteVertices`] for read-only access to vertex sides,
/// - [`InsertBipartiteVertex`] for inserting vertices on either side,
/// - [`RepartitionVertex`] for changing sides while preserving the bipartite
///   invariant, and
/// - [`RemoveVertex`] for removing vertices.
///
/// Removing a vertex also removes its side membership. No separate
/// side-removal operation is provided, since every existing vertex belongs
/// to exactly one side.
pub trait BipartiteVerticesMut:
    BipartiteVertices + ReadBipartiteVertices + InsertBipartiteVertex + RepartitionVertex + RemoveVertex
{
}

/// Blanket implementation of [`BipartiteVerticesMut`] for any type that
/// provides the required capabilities.
impl<T> BipartiteVerticesMut for T where
    T: BipartiteVertices
        + ReadBipartiteVertices
        + InsertBipartiteVertex
        + RepartitionVertex
        + RemoveVertex
{
}
