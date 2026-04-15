use crate::graphs::graph::{Vertices, RemoveVertex};

/// A graph whose vertices are each associated with a color.
///
/// This trait extends [`Vertices`] with an associated color type. The color
/// type is used by the related colored-vertex traits to read, insert, and
/// update vertex colors.
///
/// Types implementing graph traits are expected to use [`Vertices::Vertex`]
/// to identify vertices, and [`ColoredVertices::Color`] to represent the
/// color assigned to each existing vertex.
pub trait ColoredVertices: Vertices {
    /// Type used to represent vertex colors.
    ///
    /// This may be any type suitable for the application, such as an enum,
    /// integer, or label type.
    type Color;
}

/// A graph that supports read-only access to vertex colors.
///
/// This trait provides an abstract view of a total coloring of the vertex set.
/// Every existing vertex is expected to have a color. Since a queried vertex
/// identifier may be invalid, [`ReadColoredVertices::vertex_color`] returns
/// an [`Option`].
pub trait ReadColoredVertices: ColoredVertices + Vertices {
    /// Returns the color of the given vertex.
    ///
    /// The `vertex` parameter is the identifier of the vertex whose color
    /// should be returned.
    ///
    /// On success, returns `Some(&color)` where `color` is the color assigned
    /// to the given vertex. If no such vertex exists in the graph, returns
    /// `None`.
    fn vertex_color(&self, vertex: Self::Vertex) -> Option<&Self::Color>;
}

/// A graph that supports insertion of colored vertices.
///
/// This trait describes the ability to add new vertices together with their
/// color. Since coloring is total, newly inserted vertices are created with
/// an associated color rather than being inserted uncolored.
pub trait InsertColoredVertex: ColoredVertices {
    /// Inserts a new vertex with the given color into the graph.
    ///
    /// The `color` parameter specifies the color assigned to the newly
    /// inserted vertex.
    ///
    /// On success, returns `Some(v)` where `v` is the identifier of the
    /// newly inserted vertex. If the vertex cannot be inserted (for example
    /// due to capacity limits), returns `None`.
    ///
    /// Implementations may choose whether or not to recycle identifiers of
    /// previously removed vertices.
    fn insert_colored_vertex(&mut self, color: Self::Color) -> Option<Self::Vertex>;
}

/// A graph that supports changing the color of existing vertices.
///
/// This trait describes the ability to update the color assigned to a vertex.
/// Since coloring is total, recoloring replaces the previous color of an
/// existing vertex.
pub trait RecolorVertex: ColoredVertices {
    /// Replaces the color of an existing vertex.
    ///
    /// The `vertex` parameter is the identifier of the vertex to recolor.
    /// The `color` parameter is the new color to assign to that vertex.
    ///
    /// On success, returns `Some(old_color)` where `old_color` is the
    /// previous color of the vertex. If no such vertex exists in the graph,
    /// returns `None`.
    fn recolor_vertex(&mut self, vertex: Self::Vertex, color: Self::Color) -> Option<Self::Color>;
}

/// A graph that supports both querying and mutating colored vertices.
///
/// This is a convenience alias for types that implement:
/// - [`ColoredVertices`] for the associated color type,
/// - [`ReadColoredVertices`] for read-only access to vertex colors,
/// - [`InsertColoredVertex`] for inserting colored vertices,
/// - [`RecolorVertex`] for updating vertex colors, and
/// - [`RemoveVertex`] for removing vertices.
///
/// Removing a vertex also removes its color. No separate color-removal
/// operation is provided, since coloring is total for all existing vertices.
///
/// Implementors are expected to use a consistent [`Vertices::Vertex`] type
/// across all included traits.
pub trait ColoredVerticesMut:
    ColoredVertices + ReadColoredVertices + InsertColoredVertex + RecolorVertex + RemoveVertex
{
}

/// Blanket implementation of [`ColoredVerticesMut`] for any type that provides
/// the required capabilities.
///
/// Any type that implements [`ColoredVertices`], [`ReadColoredVertices`],
/// [`InsertColoredVertex`], [`RecolorVertex`], and [`RemoveVertex`]
/// automatically implements [`ColoredVerticesMut`].
impl<T> ColoredVerticesMut for T where
    T: ColoredVertices + ReadColoredVertices + InsertColoredVertex + RecolorVertex + RemoveVertex
{
}
