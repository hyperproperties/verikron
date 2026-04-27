use std::hash::Hash;

use crate::graphs::{
    backward::Backward,
    expansion::{
        BackwardExpansion, Expansion, ForwardExpansion, HyperBackwardExpansion,
        HyperForwardExpansion,
    },
    forward::Forward,
    frontier::SearchFrontier,
    queue_frontier::QueueFrontier,
    search::{Discovery, Search},
    stack_frontier::StackFrontier,
    structure::{VertexOf, VertexType},
    visited::Visited,
};

/// Sequential search over a forward graph.
pub type SequentialForwardSearch<'g, G, V, F> = SequentialSearch<ForwardExpansion<'g, G>, V, F>;

/// Sequential search over a backward graph.
pub type SequentialBackwardSearch<'g, G, V, F> = SequentialSearch<BackwardExpansion<'g, G>, V, F>;

/// Sequential search over a forward hypergraph.
pub type SequentialHyperForwardSearch<'g, G, V, F> =
    SequentialSearch<HyperForwardExpansion<'g, G>, V, F>;

/// Sequential search over a backward hypergraph.
pub type SequentialHyperBackwardSearch<'g, G, V, F> =
    SequentialSearch<HyperBackwardExpansion<'g, G>, V, F>;

/// Generic sequential search over an expansion relation.
///
/// The traversal order is determined by the frontier:
/// queue-based frontiers yield BFS, stack-based frontiers yield DFS.
#[derive(Debug, Clone)]
pub struct SequentialSearch<X, V, F>
where
    X: Expansion,
    X::Vertex: Eq + Hash + Copy,
    V: Visited<X::Vertex>,
    F: SearchFrontier<Discovery<X::Vertex>>,
{
    expansion: X,
    visited: V,
    frontier: F,
}

impl<X, V, F> SequentialSearch<X, V, F>
where
    X: Expansion,
    X::Vertex: Eq + Hash + Copy,
    V: Visited<X::Vertex>,
    F: SearchFrontier<Discovery<X::Vertex>>,
{
    /// Creates a search from an expansion relation, an explicit visited
    /// structure, and initial states.
    #[must_use]
    pub fn with_expansion_and_visited(
        expansion: X,
        initials: impl IntoIterator<Item = X::Vertex>,
        mut visited: V,
        mut frontier: F,
    ) -> Self {
        for state in initials {
            if visited.visit(state) {
                frontier.push(Discovery::root(state));
            }
        }

        Self {
            expansion,
            visited,
            frontier,
        }
    }

    /// Returns the underlying expansion relation.
    #[must_use]
    #[inline]
    pub fn expansion(&self) -> &X {
        &self.expansion
    }

    /// Returns whether the search frontier is empty.
    #[must_use]
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.frontier.is_empty()
    }

    /// Borrows the visited structure while the search is still running.
    #[must_use]
    #[inline]
    pub fn visited(&self) -> &V {
        &self.visited
    }

    /// Consumes the search and returns the visited structure.
    #[must_use]
    #[inline]
    pub fn into_visited(self) -> V {
        self.visited
    }
}

impl<X, V, F> SequentialSearch<X, V, F>
where
    X: Expansion,
    X::Vertex: Eq + Hash + Copy,
    V: Visited<X::Vertex> + Default,
    F: SearchFrontier<Discovery<X::Vertex>>,
{
    /// Creates a search from an expansion relation and initial states.
    #[must_use]
    #[inline]
    pub fn with_expansion(
        expansion: X,
        initials: impl IntoIterator<Item = X::Vertex>,
        frontier: F,
    ) -> Self {
        Self::with_expansion_and_visited(expansion, initials, V::default(), frontier)
    }
}

impl<X, V, F> VertexType for SequentialSearch<X, V, F>
where
    X: Expansion,
    X::Vertex: Eq + Hash + Copy,
    V: Visited<X::Vertex>,
    F: SearchFrontier<Discovery<X::Vertex>>,
{
    type Vertex = X::Vertex;
}

impl<X, V, F> Search for SequentialSearch<X, V, F>
where
    X: Expansion,
    X::Vertex: Eq + Hash + Copy,
    V: Visited<X::Vertex>,
    F: SearchFrontier<Discovery<X::Vertex>>,
{
    #[inline]
    fn discover(&mut self) -> Option<Discovery<Self::Vertex>> {
        let discovery = self.frontier.pop()?;
        let vertex = discovery.vertex();

        for successor in self.expansion.successors(vertex) {
            if self.visited.visit(successor) {
                self.frontier.push(Discovery::child(vertex, successor));
            }
        }

        Some(discovery)
    }
}

impl<'g, G, V, F> SequentialSearch<ForwardExpansion<'g, G>, V, F>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<Discovery<VertexOf<G>>>,
{
    /// Creates a search from a graph, a frontier, an explicit visited
    /// structure, and initial vertices.
    #[must_use]
    #[inline]
    pub fn with_frontier_and_visited(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        visited: V,
        frontier: F,
    ) -> Self {
        Self::with_expansion_and_visited(ForwardExpansion::new(graph), initials, visited, frontier)
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &'g G {
        self.expansion.graph()
    }
}

impl<'g, G, V, F> SequentialSearch<ForwardExpansion<'g, G>, V, F>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
    F: SearchFrontier<Discovery<VertexOf<G>>>,
{
    /// Creates a search from a graph, a frontier, and initial vertices.
    #[must_use]
    #[inline]
    pub fn with_frontier(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        frontier: F,
    ) -> Self {
        Self::with_frontier_and_visited(graph, initials, V::default(), frontier)
    }
}

impl<'g, G, V> SequentialSearch<ForwardExpansion<'g, G>, V, StackFrontier<Discovery<VertexOf<G>>>>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a depth-first search.
    #[must_use]
    #[inline]
    pub fn dfs(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_frontier(graph, initials, StackFrontier::new())
    }
}

impl<'g, G, V> SequentialSearch<ForwardExpansion<'g, G>, V, QueueFrontier<Discovery<VertexOf<G>>>>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a breadth-first search.
    #[must_use]
    #[inline]
    pub fn bfs(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_frontier(graph, initials, QueueFrontier::new())
    }
}

impl<'g, G, V, F> SequentialSearch<BackwardExpansion<'g, G>, V, F>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<Discovery<VertexOf<G>>>,
{
    /// Creates a backward search from a graph, a frontier, an explicit visited
    /// structure, and initial vertices.
    #[must_use]
    #[inline]
    pub fn with_frontier_and_visited_backward(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        visited: V,
        frontier: F,
    ) -> Self {
        Self::with_expansion_and_visited(BackwardExpansion::new(graph), initials, visited, frontier)
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &'g G {
        self.expansion.graph()
    }
}

impl<'g, G, V, F> SequentialSearch<BackwardExpansion<'g, G>, V, F>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
    F: SearchFrontier<Discovery<VertexOf<G>>>,
{
    /// Creates a backward search from a graph, a frontier, and initial vertices.
    #[must_use]
    #[inline]
    pub fn with_frontier_backward(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        frontier: F,
    ) -> Self {
        Self::with_frontier_and_visited_backward(graph, initials, V::default(), frontier)
    }
}

impl<'g, G, V> SequentialSearch<BackwardExpansion<'g, G>, V, StackFrontier<Discovery<VertexOf<G>>>>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a depth-first backward search.
    #[must_use]
    #[inline]
    pub fn backward_dfs(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_frontier_backward(graph, initials, StackFrontier::new())
    }
}

impl<'g, G, V> SequentialSearch<BackwardExpansion<'g, G>, V, QueueFrontier<Discovery<VertexOf<G>>>>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a breadth-first backward search.
    #[must_use]
    #[inline]
    pub fn backward_bfs(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_frontier_backward(graph, initials, QueueFrontier::new())
    }
}
