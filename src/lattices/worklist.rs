use std::{collections::VecDeque, hash::Hash};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustc_hash::FxHashSet;

use crate::{
    graphs::{directed::Directed, edges::Edges, vertices::ReadVertices},
    lattices::set::Set,
};

#[inline]
pub fn identity_transfer<V, E>(_from: V, _edge: E, to: V) -> V {
    to
}

pub fn worklist_par<G, F>(
    graph: &G,
    initials: impl IntoIterator<Item = <G as Edges>::Vertex>,
    transfer: F,
) -> Set<<G as Edges>::Vertex>
where
    G: Directed + ReadVertices<Vertex = <G as Edges>::Vertex> + Sync,
    <G as Edges>::Vertex: Eq + Hash + Copy + Send + Sync,
    <G as Edges>::Edge: Send + Sync,
    F: Sync
        + Fn(<G as Edges>::Vertex, <G as Edges>::Edge, <G as Edges>::Vertex) -> <G as Edges>::Vertex,
{
    type V<G> = <G as Edges>::Vertex;
    type E<G> = <G as Edges>::Edge;

    let mut visited: Set<V<G>> = Set::default();

    // We'll treat the worklist as "frontiers" (levels) so we can batch work
    let mut frontier: VecDeque<V<G>> = VecDeque::new();

    // seed visited + first frontier
    for initial in initials {
        if !visited.contains(&initial) {
            visited.insert(initial);
            frontier.push_back(initial);
        }
    }

    // Basic invariants after seeding.
    debug_assert!(frontier.iter().all(|v| visited.contains(v)));
    debug_assert!({
        // Insertion into seen returns false (and fails the assertion) upon duplicate entries.
        let mut seen = FxHashSet::default();
        frontier.iter().all(|v| seen.insert(*v))
    });

    // Process layer by layer
    while !frontier.is_empty() {
        // Frontier must always be a subset of visited.
        debug_assert!(frontier.iter().all(|v| visited.contains(v)));

        // 1. Collect all outgoing edges from this frontier (sequential).
        let mut edges: Vec<(V<G>, E<G>, V<G>)> = Vec::new();
        for &current in &frontier {
            debug_assert!(visited.contains(&current));
            edges.extend(graph.outgoing(current));
        }

        // All edge sources should be visited.
        debug_assert!(edges.iter().all(|(from, _, _)| visited.contains(from)));

        // 2. Run "transfer" on all edges in parallel (expensive bit).
        let successors: Vec<V<G>> = edges
            .into_par_iter()
            .map(|(from, edge, to)| transfer(from, edge, to))
            .collect();

        // 3. Build next frontier sequentially, updating visited.
        frontier.clear();
        for successor in successors {
            if !visited.contains(&successor) {
                visited.insert(successor);
                frontier.push_back(successor);
            }
        }

        // New frontier must also be subset of visited, and unique.
        debug_assert!(frontier.iter().all(|v| visited.contains(v)));
        debug_assert!({
            // Insertion into seen returns false (and fails the assertion) upon duplicate entries.
            let mut seen = FxHashSet::default();
            frontier.iter().all(|v| seen.insert(*v))
        });
    }

    visited
}
#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::csr::CSR;
    use crate::graphs::{directed::Directed, edges::Edges, vertices::ReadVertices};
    use crate::lattices::set::Set;

    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::VecDeque;
    use std::hash::Hash;

    /// Sequential worklist / BFS-based reachable set, used as a reference.
    ///
    /// Same semantics as `worklist_par` with `transfer(from, edge, to) = to`.
    fn reachable_seq<G>(
        graph: &G,
        initials: impl IntoIterator<Item = <G as Edges>::Vertex>,
    ) -> Set<<G as Edges>::Vertex>
    where
        G: Directed + ReadVertices<Vertex = <G as Edges>::Vertex>,
        <G as Edges>::Vertex: Eq + Hash + Copy,
    {
        type V<G> = <G as Edges>::Vertex;

        let mut visited: Set<V<G>> = Set::default();
        let mut queue: VecDeque<V<G>> = VecDeque::new();

        // Seed
        for v in initials {
            if !visited.contains(&v) {
                visited.insert(v);
                queue.push_back(v);
            }
        }

        // BFS / worklist
        while let Some(current) = queue.pop_front() {
            for (_, _, to) in graph.outgoing(current) {
                if !visited.contains(&to) {
                    visited.insert(to);
                    queue.push_back(to);
                }
            }
        }

        visited
    }

    #[test]
    fn worklist_par_single_vertex_no_edges() {
        // Graph with 1 vertex and no edges.
        let g = CSR::from(Vec::<(usize, usize)>::new());
        assert_eq!(g.vertex_count(), 0);

        // No initials: result should be empty.
        let res = worklist_par(
            &g,
            std::iter::empty::<usize>(),
            identity_transfer::<usize, usize>,
        );
        assert!(res.is_empty());
    }

    #[test]
    fn worklist_par_linear_chain() {
        // 0 -> 1 -> 2
        let edges = vec![(0_usize, 1_usize), (1, 2)];
        let g = CSR::from(edges);

        assert_eq!(g.vertex_count(), 3);

        let initials = [0_usize];

        let res = worklist_par(&g, initials, identity_transfer::<usize, usize>);

        let expected: Set<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(res, expected);

        // Must at least contain all initials.
        for v in initials {
            assert!(res.contains(&v));
        }
    }

    #[test]
    fn worklist_par_cycle_graph() {
        // 0 -> 1 -> 2 -> 1 (cycle between 1 and 2)
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 1)];
        let g = CSR::from(edges);

        let initials = [0_usize];
        let res = worklist_par(&g, initials, identity_transfer::<usize, usize>);

        let expected: Set<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(res, expected);
    }

    #[test]
    fn worklist_par_disconnected_components() {
        // Component A: 0 -> 1
        // Component B: 2 -> 3
        let edges = vec![(0_usize, 1_usize), (2, 3)];
        let g = CSR::from(edges);

        // Start in component B only.
        let initials = [2_usize];
        let res = worklist_par(&g, initials, identity_transfer::<usize, usize>);
        let expected: Set<usize> = [2, 3].into_iter().collect();
        assert_eq!(res, expected);

        // Start in both components.
        let initials = [0_usize, 2_usize];
        let res2 = worklist_par(&g, initials, identity_transfer::<usize, usize>);
        let expected2: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(res2, expected2);
    }

    #[test]
    fn worklist_par_self_loops_and_branching() {
        // 0 -> 0 (loop), 0 -> 1, 1 -> 1 (loop)
        let edges = vec![(0_usize, 0_usize), (0, 1), (1, 1)];
        let g = CSR::from(edges);

        let initials = [0_usize];
        let res = worklist_par(&g, initials, identity_transfer::<usize, usize>);

        // Should terminate and include both 0 and 1.
        let expected: Set<usize> = [0, 1].into_iter().collect();
        assert_eq!(res, expected);
    }

    #[test]
    fn worklist_par_with_nontrivial_transfer() {
        // 0 -> 1, 1 -> 2, 2 -> 3 (line)
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 3)];
        let g = CSR::from(edges);

        let vcount = g.vertex_count();
        assert!(vcount >= 4);

        // Transfer "shifts" the target by +1 modulo vertex_count.
        let transfer = move |_from: usize, _e: usize, to: usize| (to + 1) % vcount;

        let initials = [0_usize];
        let res = worklist_par(&g, initials, transfer);

        // As we walk:
        //   0 -> 1   => succ = 2
        //   2 -> 3   => succ = 0
        // So closure is {0, 2}.
        let expected: Set<usize> = [0, 2].into_iter().collect();
        assert_eq!(res, expected);

        // Extra sanity:
        assert!(res.contains(&0));
        assert!(res.contains(&2));
        assert!(!res.contains(&1));
        assert!(!res.contains(&3));
    }

    #[test]
    fn worklist_par_matches_seq_on_random_graphs() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x574C5F52414E444F); // "WL_RANDO"

        for _case in 0..100 {
            // Random number of vertices and edges.
            let n_vertices: usize = rng.random_range(0..=10);
            let n_edges: usize = rng.random_range(0..=30);

            let mut edges = Vec::with_capacity(n_edges);

            if n_vertices > 0 {
                for _ in 0..n_edges {
                    let from = rng.random_range(0..n_vertices) as usize;
                    let to = rng.random_range(0..n_vertices) as usize;
                    edges.push((from, to));
                }
            }

            let g = CSR::from(edges);

            let vcount = g.vertex_count();

            // Random initials (possibly empty).
            let initials: Vec<usize> = if vcount == 0 {
                Vec::new()
            } else {
                let n_initials = rng.random_range(0..=vcount);
                (0..n_initials)
                    .map(|_| rng.random_range(0..vcount))
                    .collect()
            };

            // Parallel result
            let par_res = worklist_par(&g, initials.clone(), identity_transfer::<usize, usize>);

            // Sequential reference
            let seq_res = reachable_seq(&g, initials.clone());

            assert_eq!(
                par_res, seq_res,
                "parallel and sequential reachable sets differ on random graph"
            );
        }
    }

    // Generate a random edge list for a small CSR graph.
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
        // For any graph and any set of initials, worklist_par with identity transfer
        // must produce the same set as the sequential BFS-like reference.
        #[test]
        fn prop_worklist_par_matches_seq(
            edges in random_edge_list(),
            initials_raw in prop::collection::vec(0u8..=15, 0..=16),
        ) {
            let g = CSR::from(edges);

            let vcount = g.vertex_count();

            // Map raw initials into the vertex range of the graph.
            let initials: Vec<usize> = if vcount == 0 {
                Vec::new()
            } else {
                initials_raw.into_iter()
                    .map(|x| (x as usize) % vcount)
                    .collect()
            };

            // Parallel result
            let par_res = worklist_par(&g, initials.clone(), identity_transfer::<usize, usize>);

            // Sequential reference result
            let seq_res = reachable_seq(&g, initials.clone());

            prop_assert_eq!(par_res, seq_res);
        }

        // Starting from a set of initials, the result must always contain all initials.
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

            let res = worklist_par(&g, initials.clone(), identity_transfer::<usize, usize>);

            for v in initials {
                prop_assert!(res.contains(&v));
            }
        }

        // Idempotence property: running worklist_par twice with the result
        // as initials should not change the set.
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

            let first = worklist_par(&g, initials.clone(), identity_transfer::<usize, usize>);
            let second = worklist_par(
                &g,
                first.iter().copied(),
                identity_transfer::<usize, usize>
            );

            prop_assert_eq!(first, second);
        }
    }
}
