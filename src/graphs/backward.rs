use crate::graphs::graph::Directed;

pub trait Backward: Directed {
    type Predecessors<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    fn predecessors(&self, vertex: Self::Vertex) -> Self::Predecessors<'_>;
}

impl<T> Backward for T
where
    T: Directed,
{
    type Predecessors<'a>
        = T::Ingoing<'a>
    where
        Self: 'a;

    fn predecessors(&self, vertex: Self::Vertex) -> Self::Predecessors<'_> {
        self.ingoing(vertex)
    }
}
