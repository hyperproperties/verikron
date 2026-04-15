use crate::graphs::{
    colored::{ColoredGraph, ColoredVertices, InsertColoredVertex},
    graph::{RemoveVertex, VertexOf, Vertices},
};

/// Side of a bipartite partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Side {
    Left = 0,
    Right = 1,
}

impl Side {
    /// Returns the opposite side.
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

impl From<bool> for Side {
    #[inline]
    fn from(bit: bool) -> Self {
        if bit { Self::Right } else { Self::Left }
    }
}

impl From<Side> for bool {
    #[inline]
    fn from(side: Side) -> Self {
        matches!(side, Side::Right)
    }
}

/// Vertex coloring specialized to the canonical bipartite side type.
pub trait BipartiteVertices: ColoredVertices<Color = Side> {
    /// Returns the side of `vertex`, or `None` if it does not exist.
    #[inline]
    fn side_of(&self, vertex: Self::Vertex) -> Option<Side> {
        self.vertex_color(vertex)
    }

    /// Returns whether `vertex` is on the left side.
    #[inline]
    fn is_left(&self, vertex: Self::Vertex) -> bool {
        self.side_of(vertex) == Some(Side::Left)
    }

    /// Returns whether `vertex` is on the right side.
    #[inline]
    fn is_right(&self, vertex: Self::Vertex) -> bool {
        self.side_of(vertex) == Some(Side::Right)
    }

    /// Returns whether `u` and `v` are on the same side.
    ///
    /// Returns `None` if either vertex does not exist.
    #[inline]
    fn same_side(&self, u: Self::Vertex, v: Self::Vertex) -> Option<bool> {
        Some(self.side_of(u)? == self.side_of(v)?)
    }

    /// Returns whether `u` and `v` are on opposite sides.
    ///
    /// Returns `None` if either vertex does not exist.
    #[inline]
    fn opposite_sides(&self, u: Self::Vertex, v: Self::Vertex) -> Option<bool> {
        Some(self.side_of(u)? != self.side_of(v)?)
    }
}

impl<T> BipartiteVertices for T where T: ColoredVertices<Color = Side> {}

/// Bipartite vertex insertion.
pub trait InsertBipartiteVertex: BipartiteVertices + InsertColoredVertex {
    /// Inserts a vertex on `side`.
    #[inline]
    fn insert_vertex_on(&mut self, side: Side) -> Option<Self::Vertex> {
        self.insert_colored_vertex(side)
    }

    /// Inserts a vertex on the left side.
    #[inline]
    fn insert_left_vertex(&mut self) -> Option<Self::Vertex> {
        self.insert_vertex_on(Side::Left)
    }

    /// Inserts a vertex on the right side.
    #[inline]
    fn insert_right_vertex(&mut self) -> Option<Self::Vertex> {
        self.insert_vertex_on(Side::Right)
    }
}

impl<T> InsertBipartiteVertex for T where T: BipartiteVertices + InsertColoredVertex {}

/// Side-changing operation that must preserve bipartiteness.
pub trait RepartitionVertex: BipartiteVertices {
    /// Moves `vertex` to `side`.
    ///
    /// Returns the old side on success, or `Err(())` if the vertex does not
    /// exist or the change would violate bipartiteness.
    fn repartition_vertex(&mut self, vertex: Self::Vertex, side: Side) -> Result<Side, ()>;
}

/// Mutable bipartite vertex store.
pub trait BipartiteVerticesMut:
    BipartiteVertices + InsertBipartiteVertex + RepartitionVertex + RemoveVertex
{
}

impl<T> BipartiteVerticesMut for T where
    T: BipartiteVertices + InsertBipartiteVertex + RepartitionVertex + RemoveVertex
{
}

/// Bipartite graph interface.
pub trait BipartiteGraph: ColoredGraph<Color = Side>
where
    Self::Vertices: BipartiteVertices<Vertex = VertexOf<Self>>,
{
    /// Returns the side of `vertex`, or `None` if it does not exist.
    #[inline]
    fn side_of(&self, vertex: VertexOf<Self>) -> Option<Side> {
        self.vertex_store().vertex_color(vertex)
    }

    /// Returns whether `vertex` is on the left side.
    #[inline]
    fn is_left(&self, vertex: VertexOf<Self>) -> bool {
        self.side_of(vertex) == Some(Side::Left)
    }

    /// Returns whether `vertex` is on the right side.
    #[inline]
    fn is_right(&self, vertex: VertexOf<Self>) -> bool {
        self.side_of(vertex) == Some(Side::Right)
    }

    /// Returns whether `u` and `v` are on the same side.
    ///
    /// Returns `None` if either vertex does not exist.
    #[inline]
    fn same_side(&self, u: VertexOf<Self>, v: VertexOf<Self>) -> Option<bool> {
        Some(self.side_of(u)? == self.side_of(v)?)
    }

    /// Returns whether `u` and `v` are on opposite sides.
    ///
    /// Returns `None` if either vertex does not exist.
    #[inline]
    fn opposite_sides(&self, u: VertexOf<Self>, v: VertexOf<Self>) -> Option<bool> {
        Some(self.side_of(u)? != self.side_of(v)?)
    }

    /// Returns the number of left-side vertices.
    #[inline]
    fn left_vertex_count(&self) -> usize {
        self.vertex_store()
            .vertices()
            .filter(|&v| self.is_left(v))
            .count()
    }

    /// Returns the number of right-side vertices.
    #[inline]
    fn right_vertex_count(&self) -> usize {
        self.vertex_store()
            .vertices()
            .filter(|&v| self.is_right(v))
            .count()
    }
}

impl<T> BipartiteGraph for T
where
    T: ColoredGraph<Color = Side>,
    T::Vertices: BipartiteVertices<Vertex = VertexOf<T>>,
{
}
