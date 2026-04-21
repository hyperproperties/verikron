use crate::graphs::{
    arc::Arc,
    endpoints::Endpoints,
    graph::Graph,
    members::Members,
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
///
/// This is the graph-specific colored abstraction.
pub trait ColoredGraph: Graph + VertexColor
where
    Self::Vertices: ColoredVertices<Vertex = VertexOf<Self>, Color = Self::Color>,
{
}

impl<T> ColoredGraph for T
where
    T: Graph + VertexColor,
    T::Vertices: ColoredVertices<Vertex = VertexOf<T>, Color = T::Color>,
{
}

/// Directed graph constructible from owned arcs together with one color per vertex.
pub trait FromColoredArcs: Sized + Graph + VertexColor {
    /// Builds a directed graph from owned arcs and vertex colors.
    ///
    /// The color iterator supplies one color per vertex id.
    fn from_arcs_and_colors<A, C>(arcs: A, colors: C) -> Self
    where
        A: IntoIterator<Item = Arc<VertexOf<Self>>>,
        C: IntoIterator<Item = Self::Color>;
}

/// Undirected graph constructible from owned endpoint pairs together with one
/// color per vertex.
pub trait FromColoredEndpoints: Sized + Graph + VertexColor {
    /// Builds an undirected graph from owned endpoint pairs and vertex colors.
    ///
    /// The color iterator supplies one color per vertex id.
    fn from_endpoints_and_colors<E, C>(edges: E, colors: C) -> Self
    where
        E: IntoIterator<Item = Endpoints<VertexOf<Self>>>,
        C: IntoIterator<Item = Self::Color>;
}

/// Undirected hypergraph constructible from owned member collections together
/// with one color per vertex.
pub trait FromColoredMembers: Sized + Graph + VertexColor {
    /// Builds an undirected hypergraph from owned member collections and
    /// vertex colors.
    ///
    /// The color iterator supplies one color per vertex id.
    fn from_members_and_colors<H, S, C>(hyperedges: H, colors: C) -> Self
    where
        H: IntoIterator<Item = Members<S>>,
        S: IntoIterator<Item = VertexOf<Self>>,
        C: IntoIterator<Item = Self::Color>;
}

/// Directed hypergraph constructible from owned hyperarcs together with one
/// color per vertex.
pub trait FromColoredHyperarcs: Sized + Graph + VertexColor {
    /// Builds a directed hypergraph from owned hyperarcs and vertex colors.
    ///
    /// The color iterator supplies one color per vertex id.
    fn from_hyperarcs_and_colors<H, S, C>(hyperedges: H, colors: C) -> Self
    where
        H: IntoIterator<Item = Arc<S>>,
        S: IntoIterator<Item = VertexOf<Self>>,
        C: IntoIterator<Item = Self::Color>;
}
