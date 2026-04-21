use crate::graphs::graph::Directed;

/// Forward exploration for directed graphs.
///
/// This is a thin convenience layer over [`Directed`] that exposes outgoing
/// edges under successor terminology.
pub trait Forward: Directed {
    /// Iterator over successor edges of a vertex.
    ///
    /// Each returned edge has source equal to the queried vertex.
    type Successors<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns all successor edges of `vertex`.
    fn successors(&self, vertex: Self::Vertex) -> Self::Successors<'_>;
}

impl<T> Forward for T
where
    T: Directed,
{
    type Successors<'a>
        = <T as Directed>::Outgoing<'a>
    where
        Self: 'a;

    #[inline]
    fn successors(&self, vertex: Self::Vertex) -> Self::Successors<'_> {
        self.outgoing(vertex)
    }
}
