use crate::graphs::{
    graph::{Endpoints, Graph},
    structure::{FiniteVertices, RemoveVertex, Structure, VertexOf, VertexType},
};

/// Vertex color information.
pub trait VertexColor: VertexType {
    /// Type used to represent vertex colors.
    type Color;
}

/// Finite vertex store with global access to vertex colors.
pub trait ColoredVertices: FiniteVertices + VertexColor {
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

/// Structure whose vertex store is colored.
///
/// This is the common colored abstraction shared by graphs, hypergraphs,
/// and similar structures.
pub trait ColoredStructure: Structure + VertexColor
where
    Self::Vertices: ColoredVertices<Vertex = VertexOf<Self>, Color = Self::Color>,
{
    /// Returns the color of `vertex`, or `None` if it does not exist.
    #[must_use]
    #[inline]
    fn vertex_color(&self, vertex: VertexOf<Self>) -> Option<Self::Color> {
        self.vertex_store().vertex_color(vertex)
    }

    /// Returns whether `vertex` has color `color`.
    ///
    /// Returns `false` if `vertex` does not exist.
    #[must_use]
    #[inline]
    fn has_color(&self, vertex: VertexOf<Self>, color: Self::Color) -> bool
    where
        Self::Color: Eq,
    {
        self.vertex_color(vertex) == Some(color)
    }
}

impl<T> ColoredStructure for T
where
    T: Structure + VertexColor,
    T::Vertices: ColoredVertices<Vertex = VertexOf<T>, Color = T::Color>,
{
}

/// Graph whose vertex store is colored.
pub trait ColoredGraph: Graph + VertexColor
where
    Self::Vertices: ColoredVertices<Vertex = VertexOf<Self>, Color = Self::Color>,
{
    /// Returns the color of `vertex`, or `None` if it does not exist.
    #[must_use]
    #[inline]
    fn vertex_color(&self, vertex: VertexOf<Self>) -> Option<Self::Color> {
        self.vertex_store().vertex_color(vertex)
    }

    /// Returns whether `vertex` has color `color`.
    ///
    /// Returns `false` if `vertex` does not exist.
    #[must_use]
    #[inline]
    fn has_color(&self, vertex: VertexOf<Self>, color: Self::Color) -> bool
    where
        Self::Color: Eq,
    {
        self.vertex_color(vertex) == Some(color)
    }
}

impl<T> ColoredGraph for T
where
    T: Graph + VertexColor,
    T::Vertices: ColoredVertices<Vertex = VertexOf<T>, Color = T::Color>,
{
}

/// Graph constructible from edge endpoints together with one color per vertex.
pub trait FromColoredEndpoints: VertexType + VertexColor + Sized {
    /// Builds a graph from owned edge endpoints and vertex colors.
    ///
    /// The color iterator supplies one color per vertex id.
    fn from_endpoints_and_colors<E, C>(edges: E, colors: C) -> Self
    where
        E: IntoIterator<Item = Endpoints<Self::Vertex>>,
        C: IntoIterator<Item = Self::Color>;
}
