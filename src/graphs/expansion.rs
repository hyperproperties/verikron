use crate::graphs::{backward::Backward, forward::Forward, structure::VertexOf};

/// Local expansion relation for a search.
///
/// This abstracts over the underlying structure being searched.
pub trait Expansion {
    /// State yielded by the search.
    type State;

    /// Iterator over successor states.
    type Successors<'a>: Iterator<Item = Self::State>
    where
        Self: 'a;

    /// Returns the successor states of `state`.
    fn successors(&self, state: Self::State) -> Self::Successors<'_>;
}

/// Forward graph expansion yielding successor vertices.
#[derive(Debug, Clone, Copy)]
pub struct ForwardExpansion<'g, G> {
    graph: &'g G,
}

impl<'g, G> ForwardExpansion<'g, G> {
    /// Creates a forward expansion wrapper.
    #[must_use]
    #[inline]
    pub const fn new(graph: &'g G) -> Self {
        Self { graph }
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub const fn graph(&self) -> &'g G {
        self.graph
    }
}

/// Backward graph expansion yielding predecessor vertices.
#[derive(Debug, Clone, Copy)]
pub struct BackwardExpansion<'g, G> {
    graph: &'g G,
}

impl<'g, G> BackwardExpansion<'g, G> {
    /// Creates a backward expansion wrapper.
    #[must_use]
    #[inline]
    pub const fn new(graph: &'g G) -> Self {
        Self { graph }
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub const fn graph(&self) -> &'g G {
        self.graph
    }
}

/// Iterator over successor vertices of a graph vertex.
#[derive(Debug, Clone)]
pub struct SuccessorVertices<'g, G>
where
    G: Forward,
{
    graph: &'g G,
    edges: G::Successors<'g>,
}

impl<'g, G> Iterator for SuccessorVertices<'g, G>
where
    G: Forward,
{
    type Item = VertexOf<G>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|edge| self.graph.destination(edge))
    }
}

/// Iterator over predecessor vertices of a graph vertex.
#[derive(Debug, Clone)]
pub struct PredecessorVertices<'g, G>
where
    G: Backward,
{
    graph: &'g G,
    edges: G::Predecessors<'g>,
}

impl<'g, G> Iterator for PredecessorVertices<'g, G>
where
    G: Backward,
{
    type Item = VertexOf<G>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|edge| self.graph.source(edge))
    }
}

impl<'g, G> Expansion for ForwardExpansion<'g, G>
where
    G: Forward,
{
    type State = VertexOf<G>;

    type Successors<'a>
        = SuccessorVertices<'a, G>
    where
        Self: 'a;

    #[inline]
    fn successors(&self, state: Self::State) -> Self::Successors<'_> {
        SuccessorVertices {
            graph: self.graph,
            edges: self.graph.successors(state),
        }
    }
}

impl<'g, G> Expansion for BackwardExpansion<'g, G>
where
    G: Backward,
{
    type State = VertexOf<G>;

    type Successors<'a>
        = PredecessorVertices<'a, G>
    where
        Self: 'a;

    #[inline]
    fn successors(&self, state: Self::State) -> Self::Successors<'_> {
        PredecessorVertices {
            graph: self.graph,
            edges: self.graph.predecessors(state),
        }
    }
}
