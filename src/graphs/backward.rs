use crate::graphs::{graph::Directed, hyper::DirectedHypergraph};

/// Backward exploration for directed graphs.
///
/// This is a thin convenience layer over [`Directed`] that exposes incoming
/// edges under predecessor terminology.
pub trait Backward: Directed {
    /// Iterator over predecessor edges of a vertex.
    ///
    /// Each returned edge has destination equal to the queried vertex.
    type Predecessors<'a>: Iterator<Item = Self::Edge>
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
        = <T as Directed>::Incoming<'a>
    where
        Self: 'a;

    #[inline]
    fn predecessors(&self, vertex: Self::Vertex) -> Self::Predecessors<'_> {
        self.incoming(vertex)
    }
}

/// Backward exploration for directed hypergraphs.
///
/// This is a thin convenience layer over [`DirectedHypergraph`] that exposes
/// incoming hyperedges under predecessor terminology.
pub trait HyperBackward: DirectedHypergraph {
    /// Iterator over predecessor hyperedges of a vertex.
    ///
    /// Each item is a hyperedge whose head contains the queried vertex.
    type Predecessors<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Returns all predecessor hyperedges of `vertex`.
    fn predecessors(&self, vertex: Self::Vertex) -> Self::Predecessors<'_>;
}

impl<T> HyperBackward for T
where
    T: DirectedHypergraph,
{
    type Predecessors<'a>
        = <T as DirectedHypergraph>::Incoming<'a>
    where
        Self: 'a;

    #[inline]
    fn predecessors(&self, vertex: Self::Vertex) -> Self::Predecessors<'_> {
        self.incoming(vertex)
    }
}
