use crate::graphs::{
    forward::Forward,
    frontier::{QueueFrontier, SearchFrontier, StackFrontier},
    visited::Visited,
    worklist::Worklist,
};
use std::hash::Hash;

pub trait Search<T>: Iterator<Item = T> {}
pub trait DFS<T>: Search<T> {}
pub trait BFS<T>: Search<T> {}

pub struct SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
    F: SearchFrontier<G::Vertex>,
{
    graph: &'g G,
    visited: V,
    frontier: F,
}

impl<'g, G, V, F> SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
    F: SearchFrontier<G::Vertex>,
{
    /// Consume and return the visited structure (reachability set / lattice).
    pub fn into_visited(self) -> V {
        self.visited
    }
}

impl<'g, G, V> SequentialGraphSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
{
    pub fn dfs(graph: &'g G, initial: G::Vertex) -> Self {
        let mut visited = V::default();
        let mut frontier = StackFrontier::new();

        if visited.visit(initial) {
            frontier.push(initial);
        }

        Self {
            graph,
            visited,
            frontier,
        }
    }
}

impl<'g, G, V> SequentialGraphSearch<'g, G, V, QueueFrontier<G::Vertex>>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
{
    pub fn bfs(graph: &'g G, initials: impl IntoIterator<Item = G::Vertex>) -> Self {
        let mut visited = V::default();
        let mut frontier = QueueFrontier::new();

        for v in initials {
            if visited.visit(v) {
                frontier.push(v);
            }
        }

        Self {
            graph,
            visited,
            frontier,
        }
    }
}

impl<'g, G, V, F> Iterator for SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
    F: SearchFrontier<G::Vertex>,
{
    type Item = G::Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        // Standard worklist-style loop:
        // pop a vertex, push all newly discovered successors.
        while let Some(u) = self.frontier.pop() {
            // For robustness, you *could* re-check visited here, but by
            // construction we never push something twice.
            for (_, _, v) in self.graph.successors(u) {
                if self.visited.visit(v) {
                    self.frontier.push(v);
                }
            }
            return Some(u);
        }

        None
    }
}

impl<'g, G, V> Worklist<G::Vertex, V> for SequentialGraphSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
{
    fn worklist(mut self) -> V {
        while self.next().is_some() {}
        self.into_visited()
    }
}

impl<'g, G, V> Search<G::Vertex> for SequentialGraphSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

impl<'g, G, V> DFS<G::Vertex> for SequentialGraphSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

impl<'g, G, V> Search<G::Vertex> for SequentialGraphSearch<'g, G, V, QueueFrontier<G::Vertex>>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

impl<'g, G, V> BFS<G::Vertex> for SequentialGraphSearch<'g, G, V, QueueFrontier<G::Vertex>>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    use crate::graphs::csr::CSR;
    use crate::graphs::vertices::ReadVertices;
    use crate::graphs::worklist::Worklist;
    use crate::lattices::bit_vector::BitVector;
    use crate::lattices::set::Set;

    #[test]
    fn stack_frontier_behaves_like_stack() {
        let mut f = StackFrontier::new();
        assert!(f.is_empty());

        f.push(1);
        f.push(2);
        f.push(3);

        assert_eq!(f.pop(), Some(3));
        assert_eq!(f.pop(), Some(2));
        assert_eq!(f.pop(), Some(1));
        assert_eq!(f.pop(), None);
        assert!(f.is_empty());
    }

    #[test]
    fn queue_frontier_behaves_like_queue() {
        let mut f = QueueFrontier::new();
        assert!(f.is_empty());

        f.push(1);
        f.push(2);
        f.push(3);

        assert_eq!(f.pop(), Some(1));
        assert_eq!(f.pop(), Some(2));
        assert_eq!(f.pop(), Some(3));
        assert_eq!(f.pop(), None);
        assert!(f.is_empty());
    }

    #[test]
    fn dfs_line_graph_visits_all_vertices() {
        // 0 -> 1 -> 2 -> 3
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 3)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 4);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&g, 0);

        let mut order = Vec::new();
        while let Some(v) = dfs.next() {
            order.push(v);
        }

        // On a simple line, DFS order is deterministic: 0,1,2,3
        assert_eq!(order, vec![0, 1, 2, 3]);

        let visited = dfs.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn dfs_branching_graph_visits_all_reachable() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let edges = vec![(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let g = CSR::from(edges);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&g, 0);

        let mut seen = Vec::new();
        while let Some(v) = dfs.next() {
            seen.push(v);
        }

        // We don't constrain exact order here, only that all reachable are seen.
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, vec![0, 1, 2, 3]);

        let visited = dfs.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn dfs_cycle_graph_terminates_and_reaches_all() {
        // 0 -> 1 -> 2 -> 1 (cycle between 1 and 2)
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 1)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 3);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&g, 0);

        let mut seen = Vec::new();
        while let Some(v) = dfs.next() {
            seen.push(v);
        }

        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, vec![0, 1, 2]);

        let visited = dfs.into_visited();
        let expected: Set<usize> = [0, 1, 2].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn dfs_worklist_on_line_graph() {
        // 0 -> 1 -> 2 -> 3
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 3)];
        let g = CSR::from(edges);

        let dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&g, 0);

        let res: Set<usize> = dfs.worklist();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(res, expected);
    }

    #[test]
    fn bfs_empty_graph_no_initials() {
        let g = CSR::from(Vec::<(usize, usize)>::new());
        assert_eq!(g.vertex_count(), 0);

        let initials: [usize; 0] = [];
        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&g, initials);

        assert!(bfs.next().is_none());

        let visited = bfs.into_visited();
        assert!(visited.is_empty());
    }

    #[test]
    fn bfs_line_graph_order() {
        // 0 -> 1 -> 2 -> 3
        let edges = vec![(0_usize, 1_usize), (1, 2), (2, 3)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 4);

        let initials = [0_usize];
        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&g, initials);

        let mut order = Vec::new();
        while let Some(v) = bfs.next() {
            order.push(v);
        }

        assert_eq!(order, vec![0, 1, 2, 3]);

        let visited = bfs.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn bfs_branching_graph_levels() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let edges = vec![(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let g = CSR::from(edges);
        assert_eq!(g.vertex_count(), 4);

        let initials = [0_usize];
        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&g, initials);

        let mut order = Vec::new();
        while let Some(v) = bfs.next() {
            order.push(v);
        }

        // Layer expectations:
        //  dist(0) = 0
        //  dist(1) = dist(2) = 1
        //  dist(3) = 2
        let pos = |x| order.iter().position(|&v| v == x).unwrap();
        assert_eq!(pos(0), 0);
        let p1 = pos(1);
        let p2 = pos(2);
        let p3 = pos(3);
        assert!(p1 > 0 && p2 > 0);
        assert!(p3 > p1 && p3 > p2);

        let visited = bfs.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn bfs_disconnected_components() {
        // Component A: 0 -> 1
        // Component B: 2 -> 3
        let edges = vec![(0_usize, 1_usize), (2, 3)];
        let g = CSR::from(edges);

        // Start in component B only.
        let initials_b = [2_usize];
        let mut bfs_b: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&g, initials_b);
        let mut seen_b = Vec::new();
        while let Some(v) = bfs_b.next() {
            seen_b.push(v);
        }
        seen_b.sort_unstable();
        assert_eq!(seen_b, vec![2, 3]);

        let visited_b = bfs_b.into_visited();
        let expected_b: Set<usize> = [2, 3].into_iter().collect();
        assert_eq!(visited_b, expected_b);

        // Start in both components.
        let initials_all = [0_usize, 2_usize];
        let mut bfs_all: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&g, initials_all);
        let mut seen_all = Vec::new();
        while let Some(v) = bfs_all.next() {
            seen_all.push(v);
        }
        seen_all.sort_unstable();
        assert_eq!(seen_all, vec![0, 1, 2, 3]);

        let visited_all = bfs_all.into_visited();
        let expected_all: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited_all, expected_all);
    }

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
        #[test]
        fn prop_bfs_bitvector_vs_set(
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

            let mut bfs_set: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
                SequentialGraphSearch::bfs(&g, initials.iter().copied());
            while bfs_set.next().is_some() {}
            let set_res = bfs_set.into_visited();

            let mut bfs_bits: SequentialGraphSearch<CSR, BitVector, QueueFrontier<usize>> =
                SequentialGraphSearch::bfs(&g, initials.iter().copied());
            while bfs_bits.next().is_some() {}
            let bits_res = bfs_bits.into_visited();

            let mut from_bits = Set::default();
            for v in 0..vcount {
                if bits_res.is_visited(&v) {
                    from_bits.visit(v);
                }
            }

            prop_assert_eq!(set_res, from_bits);
        }
    }
}
