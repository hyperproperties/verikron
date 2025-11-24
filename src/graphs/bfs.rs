use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};
use rustc_hash::FxHashSet;

use crate::graphs::frontier::Frontier;
use crate::graphs::visited::Visited;
use crate::graphs::worklist::Worklist;
use crate::graphs::{forward::Forward, frontier::LayeredFrontier};
use std::hash::Hash;

const PARALLEL_THRESHOLD: usize = 1024;

pub trait BFS<T>: Iterator<Item = Vec<T>> {}

pub struct GraphBFS<'g, G, V>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
    graph: &'g G,
    visited: V,
    frontier: LayeredFrontier<G::Vertex>,
    buffer: Vec<G::Vertex>,
}

impl<'g, G, V> GraphBFS<'g, G, V>
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

impl<'g, G, V> Iterator for GraphBFS<'g, G, V>
where
    G: Forward + Sync,
    G::Vertex: Eq + Hash + Copy + Send + Sync,
    V: Visited<G::Vertex>,
{
    type Item = Vec<G::Vertex>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frontier.len() < PARALLEL_THRESHOLD {
            self.sequential_step()
        } else {
            self.parallel_step()
        }
    }
}

impl<'g, G, V> Worklist<G::Vertex, V> for GraphBFS<'g, G, V>
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

    use bit_vec::BitVec;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::VecDeque;

    use crate::graphs::csr::CSR;
    use crate::graphs::vertices::ReadVertices;
    use crate::graphs::worklist::Worklist;
    use crate::lattices::set::Set;

    #[test]
    fn bfs_empty_graph_no_initials() {
        let g = CSR::from(Vec::<(usize, usize)>::new());
        assert_eq!(g.vertex_count(), 0);

        // No initials.
        let mut bfs: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, std::iter::empty::<usize>());

        // No layers.
        assert!(bfs.next().is_none());

        // Visited stays empty.
        let visited = bfs.into_visited();
        assert!(visited.is_empty());
    }

    #[test]
    fn bfs_line_graph_layers() {
        // 0 -> 1 -> 2 -> 3
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 3)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 4);

        let initials = [0_usize];
        let mut bfs: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials);

        let mut layers = Vec::new();
        while let Some(layer) = bfs.next() {
            layers.push(layer);
        }

        assert_eq!(layers.len(), 4);
        assert_eq!(layers[0], vec![0]);
        assert_eq!(layers[1], vec![1]);
        assert_eq!(layers[2], vec![2]);
        assert_eq!(layers[3], vec![3]);

        let visited = bfs.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn bfs_branching_graph_layers_unordered() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let edges = vec![(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 4);

        let initials = [0_usize];
        let mut bfs: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials);

        let mut layers = Vec::new();
        while let Some(layer) = bfs.next() {
            layers.push(layer);
        }

        assert_eq!(layers.len(), 3);

        let mut l0 = layers[0].clone();
        let mut l1 = layers[1].clone();
        let mut l2 = layers[2].clone();

        l0.sort_unstable();
        l1.sort_unstable();
        l2.sort_unstable();

        assert_eq!(l0, vec![0]);
        assert_eq!(l1, vec![1, 2]); // order within layer not guaranteed
        assert_eq!(l2, vec![3]);

        let visited = bfs.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn bfs_cycle_graph_terminates_and_reaches_all() {
        // 0 -> 1 -> 2 -> 1 (cycle between 1 and 2)
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 1)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 3);

        let initials = [0_usize];
        let mut bfs: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials);

        let mut layers = Vec::new();
        while let Some(layer) = bfs.next() {
            assert!(!layer.is_empty());
            layers.push(layer);
        }

        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0], vec![0]);
        assert_eq!(layers[1], vec![1]);
        assert_eq!(layers[2], vec![2]);

        let visited = bfs.into_visited();
        let expected: Set<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn worklist_par_single_vertex_no_edges() {
        let g = CSR::from(Vec::<(usize, usize)>::new());
        assert_eq!(g.vertex_count(), 0);

        let initials: [usize; 0] = [];

        // Use GraphBFS as a Worklist "engine".
        let engine: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials.into_iter());

        let res: Set<usize> = engine.worklist();
        assert!(res.is_empty());
    }

    #[test]
    fn worklist_par_linear_chain() {
        // 0 -> 1 -> 2
        let edges = vec![(0_usize, 1_usize), (1, 2)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 3);

        let initials = [0_usize];

        let engine: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials.into_iter());
        let res: Set<usize> = engine.worklist();

        let expected: Set<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(res, expected);

        for v in initials {
            assert!(res.contains(&v));
        }
    }

    #[test]
    fn worklist_par_disconnected_components() {
        // Component A: 0 -> 1
        // Component B: 2 -> 3
        let edges = vec![(0_usize, 1_usize), (2, 3)];
        let g = CSR::from(edges);

        // Start in component B only.
        let initials_b = [2_usize];
        let engine: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials_b.into_iter());
        let res_b: Set<usize> = engine.worklist();
        let expected_b: Set<usize> = [2, 3].into_iter().collect();
        assert_eq!(res_b, expected_b);

        // Start in both components.
        let initials_all = [0_usize, 2_usize];
        let engine: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials_all.into_iter());
        let res_all: Set<usize> = engine.worklist();
        let expected_all: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(res_all, expected_all);
    }

    #[test]
    fn worklist_par_self_loops_and_branching() {
        // 0 -> 0 (loop), 0 -> 1, 1 -> 1 (loop)
        let edges = vec![(0_usize, 0_usize), (0, 1), (1, 1)];
        let g = CSR::from(edges);

        let initials = [0_usize];
        let engine: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials.into_iter());

        let res: Set<usize> = engine.worklist();

        let expected: Set<usize> = [0, 1].into_iter().collect();
        assert_eq!(res, expected);
    }

    #[test]
    fn worklist_par_line_graph_reachability() {
        // 0 -> 1 -> 2 -> 3
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 3)];
        let g = CSR::from(edges);

        let initials = [0_usize];
        let engine: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials.into_iter());

        let res: Set<usize> = engine.worklist();

        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(res, expected);
    }

    #[test]
    fn sequential_and_parallel_steps_produce_same_layers_and_visited() {
        // A graph with branching and a cycle.
        let edges = vec![
            (0_usize, 1_usize),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 3), // cycle 3 -> 4 -> 5 -> 3
        ];
        let g = CSR::from(edges);
        let initials = [0_usize];

        let mut bfs_seq: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials);
        let mut bfs_par: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials);

        loop {
            let l1 = bfs_seq.sequential_step();
            let l2 = bfs_par.parallel_step();

            match (l1, l2) {
                (None, None) => break,
                (Some(mut a), Some(mut b)) => {
                    a.sort_unstable();
                    b.sort_unstable();
                    assert_eq!(a, b);
                }
                (x, y) => panic!("sequential/parallel mismatch: {:?} vs {:?}", x, y),
            }
        }

        let visited_seq = bfs_seq.into_visited();
        let visited_par = bfs_par.into_visited();
        assert_eq!(visited_seq, visited_par);
    }

    // Random edge generator for small CSR graphs.
    prop_compose! {
        fn random_edge_list()
            (edges in prop::collection::vec((0u8..=15, 0u8..=15), 0..=64))
            -> Vec<(usize, usize)>
        {
            edges
                .into_iter()
                .map(|(from, to)| (from as usize, to as usize))
                .collect()
        }
    }

    proptest! {
        // GraphBFS layers must match shortest-path distances from initials.
        #[test]
        fn prop_bfs_layers_match_shortest_paths(
            edges in random_edge_list(),
            initials_raw in prop::collection::vec(0u8..=15, 0..=16),
        ) {
            let g = CSR::from(edges);
            let vcount = g.vertex_count();

            let initials: Vec<usize> = if vcount == 0 {
                Vec::new()
            } else {
                initials_raw.into_iter()
                    .map(|x| (x as usize) % vcount)
                    .collect()
            };

            // Run GraphBFS to obtain layers.
            let mut bfs: GraphBFS<CSR, Set<usize>> =
                GraphBFS::new(&g, initials.iter().copied());

            let mut layers = Vec::<Vec<usize>>::new();
            while let Some(layer) = bfs.next() {
                prop_assert!(!layer.is_empty());
                layers.push(layer);
            }

            let visited = bfs.into_visited();

            // Reference BFS distances (inlined, no helper).
            let dist = {
                let n = g.vertex_count();
                let mut dist = vec![None; n];
                let mut q = VecDeque::new();

                for &s in &initials {
                    if s < n && dist[s].is_none() {
                        dist[s] = Some(0);
                        q.push_back(s);
                    }
                }

                while let Some(u) = q.pop_front() {
                    let du = dist[u].unwrap();
                    for (_, _, v) in g.successors(u) {
                        if dist[v].is_none() {
                            dist[v] = Some(du + 1);
                            q.push_back(v);
                        }
                    }
                }

                dist
            };

            // Map vertex -> layer index.
            let mut layer_of = vec![None::<usize>; vcount];
            for (i, layer) in layers.iter().enumerate() {
                for &v in layer {
                    prop_assert!(layer_of[v].is_none(), "vertex {} appears in multiple layers", v);
                    layer_of[v] = Some(i);
                }
            }

            // Every visited vertex must have finite distance and layer == distance.
            for v in 0..vcount {
                if visited.is_visited(&v) {
                    let d = dist[v].expect("visited vertex must have finite distance");
                    let li = layer_of[v].expect("visited vertex must be in some layer");
                    prop_assert_eq!(d, li, "distance/layer mismatch for vertex {}", v);
                }
            }

            // Every vertex with finite distance must be visited.
            for v in 0..vcount {
                if dist[v].is_some() {
                    prop_assert!(visited.is_visited(&v),
                        "vertex {} has finite distance but not visited", v);
                }
            }
        }

        // Worklist (with Set visited) must always contain all initials.
        #[test]
        fn prop_worklist_contains_initials(
            edges in random_edge_list(),
            initials_raw in prop::collection::vec(0u8..=15, 0..=16),
        ) {
            let g = CSR::from(edges);
            let vcount = g.vertex_count();

            let initials: Vec<usize> = if vcount == 0 {
                Vec::new()
            } else {
                initials_raw.into_iter()
                    .map(|x| (x as usize) % vcount)
                    .collect()
            };

            let engine: GraphBFS<CSR, Set<usize>> =
                GraphBFS::new(&g, initials.iter().copied());

            let res: Set<usize> = engine.worklist();

            for v in initials {
                prop_assert!(res.contains(&v), "result must contain initial {}", v);
            }
        }

        // Worklist is idempotent: rerunning from the fixpoint set does not change it.
        #[test]
        fn prop_worklist_idempotent(
            edges in random_edge_list(),
            initials_raw in prop::collection::vec(0u8..=15, 0..=16),
        ) {
            let g = CSR::from(edges);
            let vcount = g.vertex_count();

            let initials: Vec<usize> = if vcount == 0 {
                Vec::new()
            } else {
                initials_raw.into_iter()
                    .map(|x| (x as usize) % vcount)
                    .collect()
            };

            let engine: GraphBFS<CSR, Set<usize>> =
                GraphBFS::new(&g, initials.iter().copied());

            let first: Set<usize> = engine.worklist();

            // Rerun with the first result as initials.
            let engine: GraphBFS<CSR, Set<usize>> =
                GraphBFS::new(&g, first.iter().copied());
            let second: Set<usize> =
                engine.worklist();

            prop_assert_eq!(first, second);
        }

        // BitVec-based visited must produce the same reachable set as Set-based visited.
        #[test]
        fn prop_bitvec_vs_set_reachability(
            edges in random_edge_list(),
            initials_raw in prop::collection::vec(0u8..=15, 0..=16),
        ) {
            let g = CSR::from(edges);
            let vcount = g.vertex_count();

            let initials: Vec<usize> = if vcount == 0 {
                Vec::new()
            } else {
                initials_raw.into_iter()
                    .map(|x| (x as usize) % vcount)
                    .collect()
            };

            let bfs_set: GraphBFS<CSR, Set<usize>> =
                GraphBFS::new(&g, initials.iter().copied());
            let set_res: Set<usize> = bfs_set.worklist();

            let bfs_bits: GraphBFS<CSR, BitVec> =
                GraphBFS::new(&g, initials.iter().copied());
            let bits_res: BitVec = bfs_bits.worklist();

            let mut from_bits = Set::default();
            for v in 0..vcount {
                if bits_res.is_visited(&v) {
                    from_bits.visit(v);
                }
            }

            prop_assert_eq!(set_res, from_bits);
        }
    }

    #[test]
    fn random_stress_worklist_matches_reference_bfs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x_4246_535F_5354_5245);

        for _case in 0..100 {
            // random small multigraph
            let edge_count = rng.random_range(0..=80usize);
            let mut edges = Vec::with_capacity(edge_count);

            for _ in 0..edge_count {
                let from: usize = rng.random_range(0..=10);
                let to: usize = rng.random_range(0..=10);
                edges.push((from, to));
            }

            let g = CSR::from(edges);
            let n = g.vertex_count();

            // random initials
            let init_len = rng.random_range(0..=5usize);
            let mut initials = Vec::with_capacity(init_len);
            for _ in 0..init_len {
                if n == 0 {
                    break;
                }
                initials.push(rng.random_range(0..n));
            }

            // Worklist via GraphBFS + Set.
            let engine: GraphBFS<CSR, Set<usize>> = GraphBFS::new(&g, initials.iter().copied());
            let res: Set<usize> = engine.worklist();

            // Reference BFS distances (inlined).
            let dist = {
                let mut dist = vec![None; n];
                let mut q = VecDeque::new();

                for &s in &initials {
                    if s < n && dist[s].is_none() {
                        dist[s] = Some(0);
                        q.push_back(s);
                    }
                }

                while let Some(u) = q.pop_front() {
                    let du = dist[u].unwrap();
                    for (_, _, v) in g.successors(u) {
                        if dist[v].is_none() {
                            dist[v] = Some(du + 1);
                            q.push_back(v);
                        }
                    }
                }

                dist
            };

            let mut ref_set = Set::default();
            for v in 0..n {
                if dist[v].is_some() {
                    ref_set.visit(v);
                }
            }

            assert_eq!(res, ref_set);
        }
    }
}
