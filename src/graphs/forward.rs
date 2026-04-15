use crate::graphs::graph::Directed;

pub trait Forward: Directed {
    type Successors<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

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

    fn successors(&self, vertex: Self::Vertex) -> Self::Successors<'_> {
        self.outgoing(vertex)
    }
}
