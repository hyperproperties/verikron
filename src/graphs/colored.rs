use crate::graphs::graph::{Graph, RemoveVertex, VertexType, Vertices};

/// Vertex color information.
pub trait VertexColor: VertexType {
    /// Type used to represent vertex colors.
    type Color;
}

/// Finite graph with global access to vertex colors.
pub trait ColoredVertices: Vertices + VertexColor {
    /// Returns the color of `vertex`, or `None` if it does not exist.
    fn vertex_color(&self, vertex: Self::Vertex) -> Option<Self::Color>;
}

/// Colored vertex insertion.
pub trait InsertColoredVertex: VertexColor {
    /// Inserts a new vertex with `color` and returns its identifier on success.
    fn insert_colored_vertex(&mut self, color: Self::Color) -> Option<Self::Vertex>;
}

/// Colored vertex recoloring.
pub trait RecolorVertex: VertexColor {
    /// Replaces the color of `vertex`.
    ///
    /// Returns the previous color on success, or `None` if `vertex`
    /// does not exist.
    fn recolor_vertex(&mut self, vertex: Self::Vertex, color: Self::Color) -> Option<Self::Color>;
}

/// Finite mutable colored vertex store.
///
/// Removing a vertex also removes its color.
pub trait ColoredVerticesMut:
    ColoredVertices + InsertColoredVertex + RecolorVertex + RemoveVertex
{
}

impl<T> ColoredVerticesMut for T where
    T: ColoredVertices + InsertColoredVertex + RecolorVertex + RemoveVertex
{
}

/// Graph whose vertex store is colored.
pub trait ColoredGraph: Graph + VertexColor
where
    Self::Vertices: ColoredVertices<Vertex = <Self as VertexType>::Vertex, Color = <Self as VertexColor>::Color>,
{
}

impl<T> ColoredGraph for T
where
    T: Graph + VertexColor,
    T::Vertices:
        ColoredVertices<Vertex = <T as VertexType>::Vertex, Color = <T as VertexColor>::Color>,
{
}
