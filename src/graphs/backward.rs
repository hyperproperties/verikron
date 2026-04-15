use crate::graphs::graph::Directed;

/// Backward exploration for directed graphs.
///
/// This is a thin convenience layer over [`Directed`] that exposes incoming
/// edges under predecessor terminology.
pub trait Backward: Directed {
    /// Iterator over predecessor edges of a vertex.
    ///
    /// Each item is `(source, edge, destination)`, where `destination`
    /// equals the queried vertex.
    type Predecessors<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns all predecessor edges of `vertex`.
    fn predecessors(&self, vertex: Self::Vertex) -> Self::Predecessors<'_>;
}

impl<T> Backward for T
where
    T: Directed,
{
    type Predecessors<'a>
        = <T as Directed>::Ingoing<'a>
    where
        Self: 'a;

    #[inline]
    fn predecessors(&self, vertex: Self::Vertex) -> Self::Predecessors<'_> {
        self.ingoing(vertex)
    }
}
