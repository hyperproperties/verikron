use std::hash::Hash;

use crate::graphs::{
    backward::Backward,
    expansion::{BackwardExpansion, Expansion, ForwardExpansion},
    forward::Forward,
    frontier::{QueueFrontier, SearchFrontier, StackFrontier},
    reachability::LinearReachability,
    search::VisitedSearch,
    structure::VertexOf,
    visited::Visited,
    worklist::ExhaustiveWorklist,
};

/// Sequential search over a forward graph.
pub type SequentialGraphSearch<'g, G, V, F> = SequentialSearch<ForwardExpansion<'g, G>, V, F>;
/// Sequential search over a backward graph.
pub type SequentialBackwardSearch<'g, G, V, F> = SequentialSearch<BackwardExpansion<'g, G>, V, F>;

/// Generic sequential search over an expansion relation.
///
/// The traversal order is determined by the frontier:
/// queue-based frontiers yield BFS, stack-based frontiers yield DFS.
#[derive(Debug, Clone)]
pub struct SequentialSearch<X, V, F>
where
    X: Expansion,
    X::State: Eq + Hash + Copy,
    V: Visited<X::State>,
    F: SearchFrontier<X::State>,
{
    expansion: X,
    visited: V,
    frontier: F,
}

impl<X, V, F> SequentialSearch<X, V, F>
where
    X: Expansion,
    X::State: Eq + Hash + Copy,
    V: Visited<X::State>,
    F: SearchFrontier<X::State>,
{
    /// Creates a search from an expansion relation, an explicit visited
    /// structure, and initial states.
    #[must_use]
    pub fn with_expansion_and_visited(
        expansion: X,
        initials: impl IntoIterator<Item = X::State>,
        mut visited: V,
        mut frontier: F,
    ) -> Self {
        for state in initials {
            if visited.visit(state) {
                frontier.push(state);
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
    X::State: Eq + Hash + Copy,
    V: Visited<X::State> + Default,
    F: SearchFrontier<X::State>,
{
    /// Creates a search from an expansion relation and initial states.
    #[must_use]
    #[inline]
    pub fn with_expansion(
        expansion: X,
        initials: impl IntoIterator<Item = X::State>,
        frontier: F,
    ) -> Self {
        Self::with_expansion_and_visited(expansion, initials, V::default(), frontier)
    }
}

impl<X, V, F> Iterator for SequentialSearch<X, V, F>
where
    X: Expansion,
    X::State: Eq + Hash + Copy,
    V: Visited<X::State>,
    F: SearchFrontier<X::State>,
{
    type Item = X::State;

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.frontier.pop()?;

        for successor in self.expansion.successors(state) {
            if self.visited.visit(successor) {
                self.frontier.push(successor);
            }
        }

        Some(state)
    }
}

impl<X, V, F> VisitedSearch for SequentialSearch<X, V, F>
where
    X: Expansion,
    X::State: Eq + Hash + Copy,
    V: Visited<X::State>,
    F: SearchFrontier<X::State>,
{
    type Visited = V;

    #[inline]
    fn into_visited(self) -> Self::Visited {
        self.visited
    }

    #[inline]
    fn visited(&self) -> &Self::Visited {
        &self.visited
    }
}

impl<X, V, F> LinearReachability for SequentialSearch<X, V, F>
where
    X: Expansion,
    X::State: Eq + Hash + Copy,
    V: Visited<X::State>,
    F: SearchFrontier<X::State>,
{
}

impl<X, V, F> ExhaustiveWorklist for SequentialSearch<X, V, F>
where
    X: Expansion,
    X::State: Eq + Hash + Copy,
    V: Visited<X::State>,
    F: SearchFrontier<X::State>,
{
}

impl<'g, G, V, F> SequentialSearch<ForwardExpansion<'g, G>, V, F>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<VertexOf<G>>,
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
    F: SearchFrontier<VertexOf<G>>,
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

impl<'g, G, V> SequentialSearch<ForwardExpansion<'g, G>, V, StackFrontier<VertexOf<G>>>
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

impl<'g, G, V> SequentialSearch<ForwardExpansion<'g, G>, V, QueueFrontier<VertexOf<G>>>
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
    F: SearchFrontier<VertexOf<G>>,
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
    F: SearchFrontier<VertexOf<G>>,
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

impl<'g, G, V> SequentialSearch<BackwardExpansion<'g, G>, V, StackFrontier<VertexOf<G>>>
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

impl<'g, G, V> SequentialSearch<BackwardExpansion<'g, G>, V, QueueFrontier<VertexOf<G>>>
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
