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

impl<'g, G, V, F> Worklist<G::Vertex, V> for SequentialGraphSearch<'g, G, V, F>
where
    G: Forward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex> + Default,
    F: SearchFrontier<G::Vertex>,
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

    use crate::graphs::{
        csr::CSR,
        graph::{Endpoints, FiniteVertices, FromEndpoints},
        worklist::Worklist,
    };
    use crate::lattices::{bit_vector::BitVector, set::Set};

    use proptest::prelude::*;

    #[test]
    fn stack_frontier_behaves_like_stack() {
        let mut frontier = StackFrontier::new();

        assert!(frontier.is_empty());

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), None);
        assert!(frontier.is_empty());
    }

    #[test]
    fn queue_frontier_behaves_like_queue() {
        let mut frontier = QueueFrontier::new();

        assert!(frontier.is_empty());

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), None);
        assert!(frontier.is_empty());
    }

    #[test]
    fn dfs_line_graph_visits_all_vertices() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&graph, 0);

        let order: Vec<_> = dfs.by_ref().collect();

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(dfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn dfs_branching_graph_visits_all_reachable_vertices() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 3),
            Endpoints::new(2, 3),
        ]);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&graph, 0);

        let mut seen: Vec<_> = dfs.by_ref().collect();
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(dfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn dfs_cycle_graph_terminates() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 1),
        ]);

        let mut dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&graph, 0);

        let mut seen: Vec<_> = dfs.by_ref().collect();
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(dfs.into_visited(), [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn dfs_worklist_returns_reachable_set() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);

        let dfs: SequentialGraphSearch<CSR, Set<usize>, StackFrontier<usize>> =
            SequentialGraphSearch::dfs(&graph, 0);

        let reachable: Set<usize> = dfs.worklist();

        assert_eq!(reachable, [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn bfs_empty_graph_without_initials_is_empty() {
        let graph = CSR::from_endpoints(std::iter::empty::<Endpoints<usize>>());

        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&graph, std::iter::empty());

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

        let mut bfs: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&graph, [0]);

        let order: Vec<_> = bfs.by_ref().collect();

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn bfs_branching_graph_respects_layers() {
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
    fn bfs_disconnected_graph_depends_on_initials() {
        let graph = CSR::from_endpoints([Endpoints::new(0, 1), Endpoints::new(2, 3)]);

        let mut from_left: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&graph, [0]);
        let mut seen_left: Vec<_> = from_left.by_ref().collect();
        seen_left.sort_unstable();

        assert_eq!(seen_left, vec![0, 1]);
        assert_eq!(from_left.into_visited(), [0, 1].into_iter().collect());

        let mut from_both: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
            SequentialGraphSearch::bfs(&graph, [0, 2]);
        let mut seen_both: Vec<_> = from_both.by_ref().collect();
        seen_both.sort_unstable();

        assert_eq!(seen_both, vec![0, 1, 2, 3]);
        assert_eq!(from_both.into_visited(), [0, 1, 2, 3].into_iter().collect());
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

    proptest! {
        #[test]
        fn prop_bfs_bitvector_and_set_visited_agree((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let mut bfs_set: SequentialGraphSearch<CSR, Set<usize>, QueueFrontier<usize>> =
                SequentialGraphSearch::bfs(&graph, initials.iter().copied());
            while bfs_set.next().is_some() {}
            let set_visited = bfs_set.into_visited();

            let mut bfs_bits: SequentialGraphSearch<CSR, BitVector, QueueFrontier<usize>> =
                SequentialGraphSearch::bfs(&graph, initials.iter().copied());
            while bfs_bits.next().is_some() {}
            let bit_visited = bfs_bits.into_visited();

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
            let graph = CSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

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
    }
}
