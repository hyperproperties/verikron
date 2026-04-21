use crate::graphs::{
    backward::{Backward, HyperBackward},
    forward::{Forward, HyperForward},
    structure::VertexOf,
};

pub type ExpansionStateOf<X> = <X as Expansion>::State;

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

/// Forward directed-hypergraph expansion yielding head vertices.
///
/// From a vertex `v`, this follows every successor hyperedge whose tail
/// contains `v`, and then yields all vertices in the head of those hyperedges.
#[derive(Debug, Clone, Copy)]
pub struct HyperForwardExpansion<'h, H> {
    hypergraph: &'h H,
}

impl<'h, H> HyperForwardExpansion<'h, H> {
    /// Creates a forward hypergraph expansion wrapper.
    #[must_use]
    #[inline]
    pub const fn new(hypergraph: &'h H) -> Self {
        Self { hypergraph }
    }

    /// Returns the underlying hypergraph.
    #[must_use]
    #[inline]
    pub const fn hypergraph(&self) -> &'h H {
        self.hypergraph
    }
}

impl<'h, H> Expansion for HyperForwardExpansion<'h, H>
where
    H: HyperForward,
{
    type State = VertexOf<H>;

    type Successors<'a>
        = HyperSuccessorVertices<'a, H>
    where
        Self: 'a;

    #[inline]
    fn successors(&self, state: Self::State) -> Self::Successors<'_> {
        HyperSuccessorVertices {
            hypergraph: self.hypergraph,
            hyperedges: self.hypergraph.successors(state),
            members: None,
        }
    }
}

/// Iterator over successor vertices of a hypergraph vertex.
///
/// This flattens
/// `successor hyperedges -> head vertices`.
#[derive(Debug, Clone)]
pub struct HyperSuccessorVertices<'h, H>
where
    H: HyperForward,
{
    hypergraph: &'h H,
    hyperedges: H::Successors<'h>,
    members: Option<H::Head<'h>>,
}

impl<'h, H> Iterator for HyperSuccessorVertices<'h, H>
where
    H: HyperForward,
{
    type Item = VertexOf<H>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(members) = &mut self.members {
                if let Some(vertex) = members.next() {
                    return Some(vertex);
                }
            }

            let hyperedge = self.hyperedges.next()?;
            self.members = Some(self.hypergraph.head(hyperedge));
        }
    }
}

/// Backward directed-hypergraph expansion yielding tail vertices.
///
/// From a vertex `v`, this follows every predecessor hyperedge whose head
/// contains `v`, and then yields all vertices in the tail of those hyperedges.
#[derive(Debug, Clone, Copy)]
pub struct HyperBackwardExpansion<'h, H> {
    hypergraph: &'h H,
}

impl<'h, H> HyperBackwardExpansion<'h, H> {
    /// Creates a backward hypergraph expansion wrapper.
    #[must_use]
    #[inline]
    pub const fn new(hypergraph: &'h H) -> Self {
        Self { hypergraph }
    }

    /// Returns the underlying hypergraph.
    #[must_use]
    #[inline]
    pub const fn hypergraph(&self) -> &'h H {
        self.hypergraph
    }
}

impl<'h, H> Expansion for HyperBackwardExpansion<'h, H>
where
    H: HyperBackward,
{
    type State = VertexOf<H>;

    type Successors<'a>
        = HyperPredecessorVertices<'a, H>
    where
        Self: 'a;

    #[inline]
    fn successors(&self, state: Self::State) -> Self::Successors<'_> {
        HyperPredecessorVertices {
            hypergraph: self.hypergraph,
            hyperedges: self.hypergraph.predecessors(state),
            members: None,
        }
    }
}

/// Iterator over predecessor vertices of a hypergraph vertex.
///
/// This flattens
/// `predecessor hyperedges -> tail vertices`.
#[derive(Debug, Clone)]
pub struct HyperPredecessorVertices<'h, H>
where
    H: HyperBackward,
{
    hypergraph: &'h H,
    hyperedges: H::Predecessors<'h>,
    members: Option<H::Tail<'h>>,
}

impl<'h, H> Iterator for HyperPredecessorVertices<'h, H>
where
    H: HyperBackward,
{
    type Item = VertexOf<H>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(members) = &mut self.members {
                if let Some(vertex) = members.next() {
                    return Some(vertex);
                }
            }

            let hyperedge = self.hyperedges.next()?;
            self.members = Some(self.hypergraph.tail(hyperedge));
        }
    }
}
