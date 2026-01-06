use crate::graphs::{directed::Directed, edges::ReadEdges};

pub trait Backward: Directed {
    type Successors<'a>: Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)>
    where
        Self: 'a;

    fn predecessors(&self, vertex: Self::Vertex) -> Self::Successors<'_>;
}

impl<T> Backward for T
where
    T: Directed,
{
    type Successors<'a>
        = <T as ReadEdges>::Edges<'a>
    where
        Self: 'a;

    fn predecessors(&self, vertex: Self::Vertex) -> Self::Successors<'_> {
        self.ingoing(vertex)
    }
}
