use std::hash::Hash;

use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};
use rustc_hash::FxHashSet;

use crate::graphs::{
    forward::Forward,
    frontier::{IncrementalFrontier, LayeredFrontier},
    visited::Visited,
    worklist::Worklist,
};

/// Default minimum layer size for parallel expansion.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 1024;

/// Parallel breadth-first search over a forward graph.
///
/// The search keeps one BFS layer in memory and expands it either
/// sequentially or in parallel depending on the configured threshold.
pub struct ParallelGraphBFS<'g, G, V>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
    graph: &'g G,
    visited: V,
    frontier: LayeredFrontier<G::Vertex>,
    buffer: Vec<G::Vertex>,
    index: usize,
    parallel_threshold: usize,
}

impl<'g, G, V> ParallelGraphBFS<'g, G, V>
where
    G: Forward + Sync,
    G::Vertex: Eq + Hash + Copy + Send + Sync,
    V: Visited<G::Vertex> + Default,
{
    /// Creates a BFS with the default parallel threshold.
    #[inline]
    pub fn new(graph: &'g G, initials: impl IntoIterator<Item = G::Vertex>) -> Self {
        Self::with_parallel_threshold(graph, initials, DEFAULT_PARALLEL_THRESHOLD)
    }

    /// Creates a BFS with a custom parallel threshold.
    ///
    /// Layers smaller than `parallel_threshold` are expanded sequentially.
    /// Larger layers are expanded in parallel.
    #[inline]
    pub fn with_parallel_threshold(
        graph: &'g G,
        initials: impl IntoIterator<Item = G::Vertex>,
        parallel_threshold: usize,
    ) -> Self {
        let mut visited = V::default();
        let mut initial_layer = Vec::new();

        for vertex in initials {
            if visited.visit(vertex) {
                initial_layer.push(vertex);
            }
        }

        let search = Self {
            graph,
            visited,
            frontier: LayeredFrontier::new(initial_layer),
            buffer: Vec::new(),
            index: 0,
            parallel_threshold,
        };

        debug_assert!(
            search
                .frontier
                .layer()
                .iter()
                .all(|v| search.visited.is_visited(v))
        );
        debug_assert!(has_no_duplicates(search.frontier.layer()));

        search
    }

    /// Returns the configured parallel threshold.
    #[inline]
    pub fn parallel_threshold(&self) -> usize {
        self.parallel_threshold
    }

    /// Sets the parallel threshold.
    #[inline]
    pub fn set_parallel_threshold(&mut self, parallel_threshold: usize) {
        self.parallel_threshold = parallel_threshold;
    }

    /// Returns the visited set.
    #[inline]
    pub fn into_visited(self) -> V {
        self.visited
    }

    /// Returns the current BFS layer.
    #[inline]
    pub fn layer(&self) -> &[G::Vertex] {
        self.frontier.layer()
    }

    /// Expands one layer sequentially and returns the previous layer.
    #[inline]
    pub fn sequential_step(&mut self) -> Option<Vec<G::Vertex>> {
        self.frontier.step(|current, next| {
            for &from in current {
                for (_, _, to) in self.graph.successors(from) {
                    if self.visited.visit(to) {
                        next.push(to);
                    }
                }
            }

            debug_assert!(next.iter().all(|v| self.visited.is_visited(v)));
            debug_assert!(has_no_duplicates(next));
        })
    }

    /// Expands one layer in parallel and returns the previous layer.
    #[inline]
    pub fn parallel_step(&mut self) -> Option<Vec<G::Vertex>> {
        self.frontier.step(|current, next| {
            self.buffer.clear();

            self.buffer.par_extend(
                current
                    .par_iter()
                    .flat_map_iter(|&from| self.graph.successors(from).map(|(_, _, to)| to)),
            );

            for to in self.buffer.drain(..) {
                if self.visited.visit(to) {
                    next.push(to);
                }
            }

            debug_assert!(next.iter().all(|v| self.visited.is_visited(v)));
            debug_assert!(has_no_duplicates(next));
        })
    }

    /// Expands one layer, choosing sequential or parallel execution automatically.
    #[inline]
    pub fn step(&mut self) -> Option<Vec<G::Vertex>> {
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
    G::Vertex: Eq + Hash + Copy + Send + Sync,
    V: Visited<G::Vertex> + Default,
{
    type Item = G::Vertex;

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

impl<'g, G, V> Worklist<G::Vertex, V> for ParallelGraphBFS<'g, G, V>
where
    G: Forward + Sync,
    G::Vertex: Eq + Hash + Copy + Send + Sync,
    V: Visited<G::Vertex> + Default,
{
    fn worklist(mut self) -> V {
        while self.next().is_some() {}
        self.into_visited()
    }
}

#[inline]
fn has_no_duplicates<T>(values: &[T]) -> bool
where
    T: Eq + Hash + Copy,
{
    let mut seen = FxHashSet::default();
    values.iter().copied().all(|value| seen.insert(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        csr::CSR,
        graph::{Endpoints, FiniteVertices, FromEndpoints},
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
        assert!(bfs.into_visited().is_empty());
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
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
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
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
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
        assert_eq!(bfs.into_visited(), [0, 1, 2].into_iter().collect());
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

        assert_eq!(sequential.into_visited(), parallel.into_visited());
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
