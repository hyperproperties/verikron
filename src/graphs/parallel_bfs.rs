use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};
use rustc_hash::FxHashSet;

use crate::graphs::frontier::IncrementalFrontier;
use crate::graphs::visited::Visited;
use crate::graphs::worklist::Worklist;
use crate::graphs::{forward::Forward, frontier::LayeredFrontier};
use std::hash::Hash;

// TODO: Should be a parameter on the search.
const PARALLEL_THRESHOLD: usize = 1024;

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
}

impl<'g, G, V> ParallelGraphBFS<'g, G, V>
where
    G: Forward + Sync,
    G::Vertex: Eq + Hash + Copy + Send + Sync,
    V: Visited<G::Vertex> + Default,
{
    pub fn new(graph: &'g G, initials: impl IntoIterator<Item = G::Vertex>) -> Self {
        let mut visited = V::default();
        let mut initial_frontier: Vec<G::Vertex> = Vec::default();

        for value in initials {
            if visited.visit(value) {
                initial_frontier.push(value);
            }
        }

        let frontier = LayeredFrontier::new(initial_frontier);

        // debug: no duplicates, all in visited
        debug_assert!(frontier.layer().iter().all(|v| visited.is_visited(v)));
        debug_assert!({
            let mut seen = FxHashSet::default();
            frontier.layer().iter().all(|v| seen.insert(*v))
        });

        Self {
            graph,
            visited,
            frontier,
            buffer: Vec::new(),
            index: 0,
        }
    }

    #[inline]
    pub fn into_visited(self) -> V {
        self.visited
    }

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

            // debug invariants: next has no duplicates, and is subset of visited
            debug_assert!(next.iter().all(|v| self.visited.is_visited(v)));
            debug_assert!({
                let mut seen = FxHashSet::default();
                next.iter().all(|v| seen.insert(*v))
            });
        })
    }

    #[inline]
    pub fn parallel_step(&mut self) -> Option<Vec<G::Vertex>> {
        self.frontier.step(|current, next| {
            // Large layer: use rayon to collect successors, then filter.
            self.buffer.clear();

            self.buffer.par_extend(
                current
                    .par_iter()
                    .flat_map_iter(|&from| self.graph.successors(from).map(|(_, _, to)| to)),
            );

            for successor in self.buffer.drain(..) {
                if self.visited.visit(successor) {
                    next.push(successor);
                }
            }

            // debug invariants: next has no duplicates, and is subset of visited
            debug_assert!(next.iter().all(|v| self.visited.is_visited(v)));
            debug_assert!({
                let mut seen = FxHashSet::default();
                next.iter().all(|v| seen.insert(*v))
            });
        })
    }
}

impl<'g, G, V> Iterator for ParallelGraphBFS<'g, G, V>
where
    G: Forward + Sync,
    G::Vertex: Eq + Hash + Copy + Send + Sync,
    V: Visited<G::Vertex>,
{
    type Item = G::Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // No more frontier at all → BFS finished.
            if self.frontier.is_empty() {
                return None;
            }

            // Current layer we’re streaming from.
            let layer = self.frontier.layer();

            // Still have vertices left in this layer? Yield the next one.
            if self.index < layer.len() {
                let v = layer[self.index];
                self.index += 1;
                return Some(v);
            }

            // Current layer exhausted: advance to the next layer.
            // This modifies the frontier’s internal layer.
            if layer.len() < PARALLEL_THRESHOLD {
                self.sequential_step();
            } else {
                self.parallel_step();
            }

            // Reset index to start streaming from the new layer.
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
