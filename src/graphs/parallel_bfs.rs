use std::hash::Hash;

use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};

use crate::graphs::{
    forward::Forward,
    frontier::{IncrementalFrontier, LayeredFrontier},
    search::VisitedSearch,
    structure::VertexOf,
    visited::Visited,
};

/// Default minimum layer size for parallel expansion.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 1024;

/// Parallel breadth-first search over a forward graph.
///
/// The search keeps exactly one BFS layer in memory and expands it either
/// sequentially or in parallel depending on the configured threshold.
///
/// Initial vertices are deduplicated through the visited structure.
/// The caller is responsible for supplying valid initial vertices for `graph`.
pub struct ParallelGraphBFS<'g, G, V>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
{
    graph: &'g G,
    visited: V,
    frontier: LayeredFrontier<VertexOf<G>>,
    scratch: Vec<VertexOf<G>>,
    index: usize,
    parallel_threshold: usize,
}

impl<'g, G, V> ParallelGraphBFS<'g, G, V>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
{
    /// Creates a BFS with a custom visited structure and the default
    /// parallel threshold.
    #[must_use]
    #[inline]
    pub fn with_visited(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        visited: V,
    ) -> Self {
        Self::with_visited_and_parallel_threshold(
            graph,
            initials,
            visited,
            DEFAULT_PARALLEL_THRESHOLD,
        )
    }

    /// Creates a BFS with a custom visited structure and a custom
    /// parallel threshold.
    ///
    /// Layers smaller than `parallel_threshold` are expanded sequentially.
    /// Layers at least that large are expanded in parallel.
    ///
    /// A threshold of `0` means that every non-empty layer is expanded in
    /// parallel.
    #[must_use]
    pub fn with_visited_and_parallel_threshold(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        mut visited: V,
        parallel_threshold: usize,
    ) -> Self {
        let mut initial_layer = Vec::new();

        for vertex in initials {
            if visited.visit(vertex) {
                initial_layer.push(vertex);
            }
        }

        Self {
            graph,
            visited,
            frontier: LayeredFrontier::new(initial_layer),
            scratch: Vec::new(),
            index: 0,
            parallel_threshold,
        }
    }

    /// Returns the configured parallel threshold.
    #[must_use]
    #[inline]
    pub fn parallel_threshold(&self) -> usize {
        self.parallel_threshold
    }

    /// Sets the parallel threshold.
    #[inline]
    pub fn set_parallel_threshold(&mut self, parallel_threshold: usize) {
        self.parallel_threshold = parallel_threshold;
    }

    /// Returns the current BFS layer.
    #[must_use]
    #[inline]
    pub fn layer(&self) -> &[VertexOf<G>] {
        self.frontier.layer()
    }

    /// Returns whether the search has no more vertices to expand.
    #[must_use]
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.frontier.is_empty()
    }

    /// Returns the visited structure.
    #[must_use]
    #[inline]
    pub fn visited(&self) -> &V {
        &self.visited
    }

    /// Consumes the BFS and returns the visited structure.
    #[must_use]
    #[inline]
    pub fn into_visited(self) -> V {
        self.visited
    }

    /// Expands one layer sequentially and returns the previous layer.
    #[inline]
    pub fn sequential_step(&mut self) -> Option<Vec<VertexOf<G>>> {
        self.frontier.step(|current, next| {
            for &from in current {
                for (_, _, to) in self.graph.successors(from) {
                    if self.visited.visit(to) {
                        next.push(to);
                    }
                }
            }

            debug_assert!(next.iter().all(|vertex| self.visited.is_visited(vertex)));
        })
    }
}

impl<'g, G, V> ParallelGraphBFS<'g, G, V>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a BFS with the default visited structure and the default
    /// parallel threshold.
    #[must_use]
    #[inline]
    pub fn new(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_parallel_threshold(graph, initials, DEFAULT_PARALLEL_THRESHOLD)
    }

    /// Creates a BFS with the default visited structure and a custom
    /// parallel threshold.
    #[must_use]
    #[inline]
    pub fn with_parallel_threshold(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        parallel_threshold: usize,
    ) -> Self {
        Self::with_visited_and_parallel_threshold(graph, initials, V::default(), parallel_threshold)
    }
}

impl<'g, G, V> ParallelGraphBFS<'g, G, V>
where
    G: Forward + Sync,
    VertexOf<G>: Eq + Hash + Copy + Send + Sync,
    V: Visited<VertexOf<G>>,
{
    /// Expands one layer in parallel and returns the previous layer.
    #[inline]
    pub fn parallel_step(&mut self) -> Option<Vec<VertexOf<G>>> {
        self.frontier.step(|current, next| {
            self.scratch.clear();

            self.scratch.par_extend(
                current
                    .par_iter()
                    .flat_map_iter(|&from| self.graph.successors(from).map(|(_, _, to)| to)),
            );

            for to in self.scratch.drain(..) {
                if self.visited.visit(to) {
                    next.push(to);
                }
            }

            debug_assert!(next.iter().all(|vertex| self.visited.is_visited(vertex)));
        })
    }

    /// Expands one layer, choosing sequential or parallel execution
    /// automatically.
    #[inline]
    pub fn step(&mut self) -> Option<Vec<VertexOf<G>>> {
        if self.frontier.len() < self.parallel_threshold {
            self.sequential_step()
        } else {
            self.parallel_step()
        }
    }
}

impl<'g, G, V> Iterator for ParallelGraphBFS<'g, G, V>
where
    G: Forward + Sync,
    VertexOf<G>: Eq + Hash + Copy + Send + Sync,
    V: Visited<VertexOf<G>>,
{
    type Item = VertexOf<G>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.frontier.is_empty() {
                return None;
            }

            if self.index < self.frontier.len() {
                let vertex = self.frontier[self.index];
                self.index += 1;
                return Some(vertex);
            }

            self.step()?;
            self.index = 0;
        }
    }
}

impl<'g, G, V> VisitedSearch for ParallelGraphBFS<'g, G, V>
where
    G: Forward + Sync,
    VertexOf<G>: Eq + Hash + Copy + Send + Sync,
    V: Visited<VertexOf<G>>,
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

    use crate::graphs::{
        csr::CSR,
        graph::{Endpoints, FromEndpoints},
        structure::FiniteVertices,
        worklist::Worklist,
    };
    use crate::lattices::{bit_vector::BitVector, set::Set};

    use proptest::prelude::*;
    use std::collections::VecDeque;

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
    fn bfs_empty_graph_without_initials_is_empty() {
        let graph = CSR::from_endpoints(std::iter::empty::<Endpoints<usize>>());
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> =
            ParallelGraphBFS::new(&graph, std::iter::empty());

        assert!(bfs.next().is_none());
        assert!(bfs.visited.is_empty());
    }

    #[test]
    fn bfs_line_graph_visits_vertices_in_layer_order() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        let order: Vec<_> = bfs.by_ref().collect();

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(bfs.visited, [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn bfs_branching_graph_reaches_all_reachable_vertices() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 3),
            Endpoints::new(2, 3),
        ]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        let mut seen: Vec<_> = bfs.by_ref().collect();
        seen.sort_unstable();

        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(bfs.visited, [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn bfs_cycle_graph_terminates() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 1),
        ]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        let mut seen: Vec<_> = bfs.by_ref().collect();
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(bfs.visited, [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn worklist_on_disconnected_graph_depends_on_initials() {
        let graph = CSR::from_endpoints([Endpoints::new(0, 1), Endpoints::new(2, 3)]);

        let from_left: Set<usize> =
            ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, [0]).worklist();
        let from_right: Set<usize> =
            ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, [2]).worklist();
        let from_both: Set<usize> =
            ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, [0, 2]).worklist();

        assert_eq!(from_left, [0, 1].into_iter().collect());
        assert_eq!(from_right, [2, 3].into_iter().collect());
        assert_eq!(from_both, [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn worklist_handles_loops() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 0),
            Endpoints::new(0, 1),
            Endpoints::new(1, 1),
        ]);

        let reachable: Set<usize> =
            ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, [0]).worklist();

        assert_eq!(reachable, [0, 1].into_iter().collect());
    }

    #[test]
    fn sequential_and_parallel_step_agree() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 3),
            Endpoints::new(2, 3),
            Endpoints::new(3, 4),
            Endpoints::new(4, 5),
            Endpoints::new(5, 3),
        ]);

        let mut sequential: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);
        let mut parallel: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        loop {
            let left = sequential.sequential_step();
            let right = parallel.parallel_step();

            match (left, right) {
                (None, None) => break,
                (Some(mut a), Some(mut b)) => {
                    a.sort_unstable();
                    b.sort_unstable();
                    assert_eq!(a, b);
                }
                (a, b) => panic!("step mismatch: {:?} vs {:?}", a, b),
            }
        }

        assert_eq!(sequential.visited, parallel.visited);
    }

    proptest! {
        #[test]
        fn prop_worklist_matches_reference_bfs((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let actual: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            let expected = reference_reachable(&graph, &initials);

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn prop_bitvector_and_set_visited_agree((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let set_result: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            let bit_result: BitVector =
                ParallelGraphBFS::<CSR, BitVector>::new(&graph, initials.iter().copied()).worklist();

            let mut from_bits = Set::default();
            for vertex in 0..graph.vertex_count() {
                if bit_result.is_visited(&vertex) {
                    from_bits.visit(vertex);
                }
            }

            prop_assert_eq!(set_result, from_bits);
        }

        #[test]
        fn prop_worklist_is_idempotent((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let first: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            let second: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, first.iter().copied()).worklist();

            prop_assert_eq!(first, second);
        }

        #[test]
        fn prop_every_initial_is_reached((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let reachable: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            for &initial in &initials {
                prop_assert!(reachable.contains(&initial));
            }
        }
    }
}
