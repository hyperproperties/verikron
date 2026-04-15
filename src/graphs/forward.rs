use crate::graphs::graph::Directed;

/// Successor view of a directed graph.
pub trait Forward: Directed {
    /// Iterator over successors as `(from, edge, to)`.
    type Successors<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    /// Returns all successors of `vertex`.
    fn successors(&self, vertex: Self::Vertex) -> Self::Successors<'_>;
}

impl<T> Forward for T
where
    T: Directed,
{
    type Successors<'a>
        = T::Outgoing<'a>
    where
        Self: 'a;

    #[inline]
    fn successors(&self, vertex: Self::Vertex) -> Self::Successors<'_> {
        self.outgoing(vertex)
    }
}