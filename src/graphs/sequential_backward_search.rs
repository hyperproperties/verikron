use std::hash::Hash;

use crate::graphs::{
    backward::Backward,
    frontier::{QueueFrontier, SearchFrontier, StackFrontier},
    search::VisitedSearch,
    structure::VertexOf,
    visited::Visited,
};

/// Sequential backward search over a graph.
///
/// The search follows predecessors of visited vertices.
/// The frontier determines the traversal order, for example:
/// - breadth-first with [`QueueFrontier`],
/// - depth-first with [`StackFrontier`].
#[derive(Debug, Clone)]
pub struct SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<VertexOf<G>>,
{
    graph: &'g G,
    visited: V,
    frontier: F,
}

impl<'g, G, V, F> SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<VertexOf<G>>,
{
    /// Creates a search from a frontier, an explicit visited structure,
    /// and initial vertices.
    #[must_use]
    pub fn with_frontier_and_visited(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        mut visited: V,
        mut frontier: F,
    ) -> Self {
        for vertex in initials {
            if visited.visit(vertex) {
                frontier.push(vertex);
            }
        }

        Self {
            graph,
            visited,
            frontier,
        }
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &'g G {
        self.graph
    }

    /// Returns whether the frontier is empty.
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

impl<'g, G, V, F> SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
    F: SearchFrontier<VertexOf<G>>,
{
    /// Creates a search from a frontier and initial vertices.
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

impl<'g, G, V> SequentialBackwardSearch<'g, G, V, StackFrontier<VertexOf<G>>>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a depth-first backward search.
    #[must_use]
    #[inline]
    pub fn dfs(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_frontier(graph, initials, StackFrontier::new())
    }
}

impl<'g, G, V> SequentialBackwardSearch<'g, G, V, QueueFrontier<VertexOf<G>>>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a breadth-first backward search.
    #[must_use]
    #[inline]
    pub fn bfs(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_frontier(graph, initials, QueueFrontier::new())
    }
}

impl<'g, G, V, F> Iterator for SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<VertexOf<G>>,
{
    type Item = VertexOf<G>;

    fn next(&mut self) -> Option<Self::Item> {
        let vertex = self.frontier.pop()?;

        for edge in self.graph.predecessors(vertex) {
            let predecessor = self.graph.source(edge);
            if self.visited.visit(predecessor) {
                self.frontier.push(predecessor);
            }
        }

        Some(vertex)
    }
}

impl<'g, G, V, F> VisitedSearch for SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<VertexOf<G>>,
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;

    use crate::graphs::{
        arc::{Arc, FromArcs},
        backward::Backward,
        csr::CSR,
        graph::Directed,
        structure::FiniteVertices,
        worklist::Worklist,
    };
    use crate::lattices::set::Set;
    use proptest::prelude::*;

    fn reference_backward_reachable(graph: &CSR, initials: &[usize]) -> Set<usize> {
        let mut visited = Set::default();
        let mut queue = VecDeque::new();

        for &start in initials {
            if start < graph.vertex_count() && visited.visit(start) {
                queue.push_back(start);
            }
        }

        while let Some(vertex) = queue.pop_front() {
            for edge in graph.predecessors(vertex) {
                let predecessor = graph.source(edge);
                if visited.visit(predecessor) {
                    queue.push_back(predecessor);
                }
            }
        }

        visited
    }

    fn arbitrary_instance() -> impl Strategy<Value = (Vec<(usize, usize)>, Vec<usize>)> {
        prop::collection::vec((0usize..16, 0usize..16), 0..64).prop_flat_map(|edges| {
            let vertex_count = edges
                .iter()
                .map(|&(from, to)| from.max(to))
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);

            let initials = if vertex_count == 0 {
                Just(Vec::new()).boxed()
            } else {
                prop::collection::vec(0usize..vertex_count, 0..16).boxed()
            };

            (Just(edges), initials)
        })
    }

    #[test]
    fn empty_graph_without_initials_is_empty() {
        let graph = CSR::from_arcs(std::iter::empty::<Arc<usize>>());
        let mut search = SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(
            &graph,
            std::iter::empty(),
        );

        assert!(search.next().is_none());
        assert!(search.into_visited().is_empty());
    }

    #[test]
    fn backward_bfs_on_line_visits_all_predecessors() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 3)]);
        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [3]);

        let seen: Vec<_> = search.by_ref().collect();

        assert_eq!(seen, vec![3, 2, 1, 0]);
        assert_eq!(search.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn backward_bfs_on_branching_graph_reaches_all_predecessors() {
        let graph = CSR::from_arcs([Arc::new(0, 2), Arc::new(1, 2), Arc::new(2, 3)]);
        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [3]);

        let mut seen: Vec<_> = search.by_ref().collect();
        seen.sort_unstable();

        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(search.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn multiple_initials_reach_the_union() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(2, 3), Arc::new(3, 4)]);
        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [1, 4]);

        let mut seen: Vec<_> = search.by_ref().collect();
        seen.sort_unstable();

        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        assert_eq!(search.into_visited(), [0, 1, 2, 3, 4].into_iter().collect());
    }

    #[test]
    fn cycles_terminate() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 1)]);
        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [2]);

        let mut seen: Vec<_> = search.by_ref().collect();
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(search.into_visited(), [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn backward_dfs_and_bfs_reach_the_same_set() {
        let graph = CSR::from_arcs([
            Arc::new(0, 2),
            Arc::new(1, 2),
            Arc::new(2, 3),
            Arc::new(3, 4),
        ]);

        let bfs: Set<usize> =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [4])
                .worklist();

        let dfs: Set<usize> =
            SequentialBackwardSearch::<_, Set<usize>, StackFrontier<_>>::dfs(&graph, [4])
                .worklist();

        assert_eq!(bfs, dfs);
        assert_eq!(bfs, [0, 1, 2, 3, 4].into_iter().collect());
    }

    #[test]
    fn duplicate_initials_are_ignored() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2)]);
        let visited: Set<usize> =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [2, 2, 2])
                .worklist();

        assert_eq!(visited, [0, 1, 2].into_iter().collect());
    }

    proptest! {
        #[test]
        fn prop_worklist_matches_reference((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter().copied().map(|(from, to)| Arc::new(from, to))
            );

            let actual: Set<usize> =
                SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(
                    &graph,
                    initials.iter().copied(),
                )
                .worklist();

            let expected = reference_backward_reachable(&graph, &initials);

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn prop_worklist_contains_initials((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter().copied().map(|(from, to)| Arc::new(from, to))
            );

            let visited: Set<usize> =
                SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(
                    &graph,
                    initials.iter().copied(),
                )
                .worklist();

            for &initial in &initials {
                prop_assert!(visited.contains(&initial));
            }
        }

        #[test]
        fn prop_worklist_is_idempotent((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter().copied().map(|(from, to)| Arc::new(from, to))
            );

            let first: Set<usize> =
                SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(
                    &graph,
                    initials.iter().copied(),
                )
                .worklist();

            let second: Set<usize> =
                SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(
                    &graph,
                    first.iter().copied(),
                )
                .worklist();

            prop_assert_eq!(first, second);
        }
    }
}
