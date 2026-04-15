use std::hash::Hash;

use crate::graphs::{
    backward::Backward,
    frontier::{QueueFrontier, SearchFrontier, StackFrontier},
    sequential_forward_search::{BFS, DFS},
    visited::Visited,
    worklist::Worklist,
};

/// Sequential backward search over a graph.
///
/// The search follows predecessors of visited vertices. The frontier decides
/// the traversal order, for example BFS with [`QueueFrontier`] or DFS with
/// [`StackFrontier`].
pub struct SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
    F: SearchFrontier<G::Vertex>,
{
    graph: &'g G,
    visited: V,
    frontier: F,
}

impl<'g, G, V, F> SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
    F: SearchFrontier<G::Vertex>,
{
    /// Builds a search from a frontier and initial vertices.
    pub fn with_frontier(
        graph: &'g G,
        initials: impl IntoIterator<Item = G::Vertex>,
        mut frontier: F,
    ) -> Self
    where
        V: Default,
    {
        let mut visited = V::default();

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

    /// Returns the visited set.
    #[inline]
    pub fn into_visited(self) -> V {
        self.visited
    }
}

impl<'g, G, V> SequentialBackwardSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
{
    /// Creates a depth-first backward search.
    #[inline]
    pub fn dfs(graph: &'g G, initial: G::Vertex) -> Self {
        Self::with_frontier(graph, [initial], StackFrontier::new())
    }
}

impl<'g, G, V> SequentialBackwardSearch<'g, G, V, QueueFrontier<G::Vertex>>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
{
    /// Creates a breadth-first backward search.
    #[inline]
    pub fn bfs(graph: &'g G, initials: impl IntoIterator<Item = G::Vertex>) -> Self {
        Self::with_frontier(graph, initials, QueueFrontier::new())
    }
}

impl<'g, G, V, F> Iterator for SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
    F: SearchFrontier<G::Vertex>,
{
    type Item = G::Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        let vertex = self.frontier.pop()?;

        for (predecessor, _, _) in self.graph.predecessors(vertex) {
            if self.visited.visit(predecessor) {
                self.frontier.push(predecessor);
            }
        }

        Some(vertex)
    }
}

impl<'g, G, V, F> Worklist<G::Vertex, V> for SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
    F: SearchFrontier<G::Vertex>,
{
    fn worklist(mut self) -> V {
        while self.next().is_some() {}
        self.into_visited()
    }
}

impl<'g, G, V> DFS<G::Vertex> for SequentialBackwardSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

impl<'g, G, V> BFS<G::Vertex> for SequentialBackwardSearch<'g, G, V, QueueFrontier<G::Vertex>>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;

    use crate::graphs::{
        csr::CSR,
        graph::{Endpoints, FiniteVertices, FromEndpoints},
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
            for (predecessor, _, _) in graph.predecessors(vertex) {
                if visited.visit(predecessor) {
                    queue.push_back(predecessor);
                }
            }
        }

        visited
    }

    fn arbitrary_instance() -> impl Strategy<Value = (Vec<Endpoints<usize>>, Vec<usize>)> {
        prop::collection::vec((0usize..16, 0usize..16), 0..64).prop_flat_map(|raw_edges| {
            let edges: Vec<_> = raw_edges
                .into_iter()
                .map(|(from, to)| Endpoints::new(from, to))
                .collect();

            let vertex_count = edges
                .iter()
                .map(|edge| edge.from.max(edge.to))
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
        let graph = CSR::from_endpoints(std::iter::empty::<Endpoints<usize>>());
        let mut search = SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(
            &graph,
            std::iter::empty(),
        );

        assert!(search.next().is_none());
        assert!(search.into_visited().is_empty());
    }

    #[test]
    fn backward_bfs_on_line_visits_all_predecessors() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);
        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [3]);

        let seen: Vec<_> = search.by_ref().collect();

        assert_eq!(seen, vec![3, 2, 1, 0]);
        assert_eq!(search.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn backward_bfs_on_branching_graph_reaches_all_predecessors() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 2),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);
        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [3]);

        let mut seen: Vec<_> = search.by_ref().collect();
        seen.sort_unstable();

        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(search.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn multiple_initials_reach_the_union() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(2, 3),
            Endpoints::new(3, 4),
        ]);
        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [1, 4]);

        let mut seen: Vec<_> = search.by_ref().collect();
        seen.sort_unstable();

        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        assert_eq!(search.into_visited(), [0, 1, 2, 3, 4].into_iter().collect());
    }

    #[test]
    fn cycles_terminate() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 1),
        ]);
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
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 2),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
            Endpoints::new(3, 4),
        ]);

        let bfs: Set<usize> =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [4])
                .worklist();

        let dfs: Set<usize> =
            SequentialBackwardSearch::<_, Set<usize>, StackFrontier<_>>::dfs(&graph, 4).worklist();

        assert_eq!(bfs, dfs);
        assert_eq!(bfs, [0, 1, 2, 3, 4].into_iter().collect());
    }

    #[test]
    fn duplicate_initials_are_ignored() {
        let graph = CSR::from_endpoints([Endpoints::new(0, 1), Endpoints::new(1, 2)]);
        let visited: Set<usize> =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&graph, [2, 2, 2])
                .worklist();

        assert_eq!(visited, [0, 1, 2].into_iter().collect());
    }

    proptest! {
        #[test]
        fn prop_worklist_matches_reference((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(edges.clone());

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
            let graph = CSR::from_endpoints(edges);

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
            let graph = CSR::from_endpoints(edges);

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
