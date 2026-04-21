use std::hash::Hash;

use crate::graphs::{
    forward::Forward,
    frontier::{QueueFrontier, SearchFrontier, StackFrontier},
    search::VisitedSearch,
    structure::VertexOf,
    visited::Visited,
};

/// Sequential search over a forward graph.
///
/// The traversal order is determined by the frontier:
/// queue-based frontiers yield BFS, stack-based frontiers yield DFS.
#[derive(Debug, Clone)]
pub struct SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<VertexOf<G>>,
{
    graph: &'g G,
    visited: V,
    frontier: F,
}

impl<'g, G, V, F> SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
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
        seed_frontier(&mut visited, &mut frontier, initials);

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

impl<'g, G, V, F> SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
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

impl<'g, G, V> SequentialGraphSearch<'g, G, V, StackFrontier<VertexOf<G>>>
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

impl<'g, G, V> SequentialGraphSearch<'g, G, V, QueueFrontier<VertexOf<G>>>
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

impl<'g, G, V, F> Iterator for SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
    F: SearchFrontier<VertexOf<G>>,
{
    type Item = VertexOf<G>;

    fn next(&mut self) -> Option<Self::Item> {
        let vertex = self.frontier.pop()?;

        for (_, _, successor) in self.graph.successors(vertex) {
            if self.visited.visit(successor) {
                self.frontier.push(successor);
            }
        }

        Some(vertex)
    }
}

impl<'g, G, V, F> VisitedSearch for SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
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

#[inline]
fn seed_frontier<T, V, F>(visited: &mut V, frontier: &mut F, initials: impl IntoIterator<Item = T>)
where
    T: Copy,
    V: Visited<T>,
    F: SearchFrontier<T>,
{
    for vertex in initials {
        if visited.visit(vertex) {
            frontier.push(vertex);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        csr::CSR,
        graph::{Endpoints, FromEndpoints},
        structure::FiniteVertices,
        worklist::Worklist,
    };
    use crate::lattices::{bit_vector::BitVector, set::Set};

    use proptest::prelude::*;
    use std::collections::VecDeque;

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

    fn reference_reachable(graph: &CSR, initials: &[usize]) -> Set<usize> {
        let mut visited = Set::default();
        let mut queue = VecDeque::new();

        for &source in initials {
            if source < graph.vertex_count() && visited.visit(source) {
                queue.push_back(source);
            }
        }

        while let Some(from) = queue.pop_front() {
            for (_, _, to) in graph.successors(from) {
                if visited.visit(to) {
                    queue.push_back(to);
                }
            }
        }

        visited
    }

    #[test]
    fn stack_frontier_is_lifo() {
        let mut frontier = StackFrontier::new();

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), None);
    }

    #[test]
    fn queue_frontier_is_fifo() {
        let mut frontier = QueueFrontier::new();

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), None);
    }

    #[test]
    fn dfs_on_line_visits_all_vertices() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&graph, [0]);

        let order: Vec<_> = dfs.by_ref().collect();

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(dfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn dfs_on_cycle_terminates() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 1),
        ]);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&graph, [0]);

        let mut seen: Vec<_> = dfs.by_ref().collect();
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(dfs.into_visited(), [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn bfs_on_empty_graph_is_empty() {
        let graph = CSR::from_endpoints(std::iter::empty::<Endpoints<usize>>());

        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&graph, std::iter::empty());

        assert!(bfs.next().is_none());
        assert!(bfs.into_visited().is_empty());
    }

    #[test]
    fn bfs_on_line_visits_vertices_in_layer_order() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);

        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&graph, [0]);

        let order: Vec<_> = bfs.by_ref().collect();

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn bfs_respects_layers() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 3),
            Endpoints::new(2, 3),
        ]);

        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&graph, [0]);

        let order: Vec<_> = bfs.by_ref().collect();

        let position = |vertex| order.iter().position(|&v| v == vertex).unwrap();

        assert_eq!(position(0), 0);
        assert!(position(1) > 0);
        assert!(position(2) > 0);
        assert!(position(3) > position(1));
        assert!(position(3) > position(2));

        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn bfs_on_disconnected_graph_depends_on_initials() {
        let graph = CSR::from_endpoints([Endpoints::new(0, 1), Endpoints::new(2, 3)]);

        let left: Set<usize> =
            SequentialGraphSearch::<CSR, Set<usize>, QueueFrontier<usize>>::bfs(&graph, [0])
                .worklist();

        let both: Set<usize> =
            SequentialGraphSearch::<CSR, Set<usize>, QueueFrontier<usize>>::bfs(&graph, [0, 2])
                .worklist();

        assert_eq!(left, [0, 1].into_iter().collect());
        assert_eq!(both, [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn bfs_and_dfs_reach_the_same_set() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 3),
            Endpoints::new(2, 4),
        ]);

        let bfs: Set<usize> =
            SequentialGraphSearch::<CSR, Set<usize>, QueueFrontier<usize>>::bfs(&graph, [0])
                .worklist();

        let dfs: Set<usize> =
            SequentialGraphSearch::<CSR, Set<usize>, StackFrontier<usize>>::dfs(&graph, [0])
                .worklist();

        assert_eq!(bfs, dfs);
        assert_eq!(bfs, [0, 1, 2, 3, 4].into_iter().collect());
    }

    proptest! {
        #[test]
        fn prop_bfs_bitvector_and_set_visited_agree((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(edges);

            let set_visited: Set<usize> =
                SequentialGraphSearch::<CSR, Set<usize>, QueueFrontier<usize>>::bfs(
                    &graph,
                    initials.iter().copied(),
                )
                .worklist();

            let bit_visited: BitVector =
                SequentialGraphSearch::<CSR, BitVector, QueueFrontier<usize>>::bfs(
                    &graph,
                    initials.iter().copied(),
                )
                .worklist();

            let mut from_bits = Set::default();
            for vertex in 0..graph.vertex_count() {
                if bit_visited.is_visited(&vertex) {
                    from_bits.visit(vertex);
                }
            }

            prop_assert_eq!(set_visited, from_bits);
        }

        #[test]
        fn prop_bfs_every_initial_is_visited((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(edges);

            let reachable: Set<usize> =
                SequentialGraphSearch::<CSR, Set<usize>, QueueFrontier<usize>>::bfs(
                    &graph,
                    initials.iter().copied(),
                )
                .worklist();

            for &initial in &initials {
                prop_assert!(reachable.contains(&initial));
            }
        }

        #[test]
        fn prop_bfs_matches_reference((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(edges.clone());

            let actual: Set<usize> =
                SequentialGraphSearch::<CSR, Set<usize>, QueueFrontier<usize>>::bfs(
                    &graph,
                    initials.iter().copied(),
                )
                .worklist();

            let expected = reference_reachable(&graph, &initials);

            prop_assert_eq!(actual, expected);
        }
    }
}
