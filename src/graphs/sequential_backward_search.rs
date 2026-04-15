use crate::graphs::{
    backward::Backward,
    frontier::{QueueFrontier, SearchFrontier, StackFrontier},
    sequential_forward_search::{BFS, DFS, Search},
    visited::Visited,
    worklist::Worklist,
};
use std::hash::Hash;

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
    V: Visited<G::Vertex> + Default,
    F: SearchFrontier<G::Vertex>,
{
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

impl<'g, G, V> SequentialBackwardSearch<'g, G, V, QueueFrontier<G::Vertex>>
where
    G: Backward,
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

impl<'g, G, V, F> Iterator for SequentialBackwardSearch<'g, G, V, F>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
    F: SearchFrontier<G::Vertex>,
{
    type Item = G::Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        // Pop a vertex from the frontier and push all newly discovered predecessors.
        while let Some(u) = self.frontier.pop() {
            // Backward search: follow *incoming* edges and enqueue predecessors.
            for (pred, _edge, _this) in self.graph.predecessors(u) {
                if self.visited.visit(pred) {
                    self.frontier.push(pred);
                }
            }
            return Some(u);
        }
        None
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

impl<'g, G, V> Search<G::Vertex> for SequentialBackwardSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

impl<'g, G, V> DFS<G::Vertex> for SequentialBackwardSearch<'g, G, V, StackFrontier<G::Vertex>>
where
    G: Backward,
    G::Vertex: Eq + Hash + Copy,
    V: Visited<G::Vertex>,
{
}

impl<'g, G, V> Search<G::Vertex> for SequentialBackwardSearch<'g, G, V, QueueFrontier<G::Vertex>>
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

    use crate::graphs::csr::CSR;
    use crate::graphs::graph::FiniteVertices;
    use crate::lattices::set::Set;
    use proptest::prelude::*;

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

    #[test]
    fn backward_empty_graph_no_initials() {
        let g = CSR::from(Vec::<(usize, usize)>::new());
        assert_eq!(g.vertex_count(), 0);

        // No initials: we must not try to start from vertex 0 etc.
        let search = SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(
            &g,
            std::iter::empty::<usize>(),
        );

        let mut search = search;
        assert!(search.next().is_none());
        let visited = search.into_visited();
        assert!(visited.is_empty());
    }

    #[test]
    fn backward_line_graph() {
        // 0 -> 1 -> 2 -> 3
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let g = CSR::from(edges);

        // Backward from {3} should give {3,2,1,0}.
        let initials = [3usize];

        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&g, initials);

        for _v in search.by_ref() {}

        let visited = search.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    #[test]
    fn backward_branching_graph() {
        // 0 -> 2, 1 -> 2, 2 -> 3
        let edges = vec![(0, 2), (1, 2), (2, 3)];
        let g = CSR::from(edges);

        // Backward from {3} should give {3,2,0,1}.
        let initials = [3usize];

        let mut search =
            SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>::bfs(&g, initials);

        for _v in search.by_ref() {}

        let visited = search.into_visited();
        let expected: Set<usize> = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(visited, expected);
    }

    // ---------- property-based tests ----------

    proptest! {
        // Every initial vertex must end up in the visited set (backward BFS).
        #[test]
        fn prop_backward_contains_initials(
            edges in random_edge_list(),
            initials_raw in prop::collection::vec(0u8..=15, 0..=16),
        ) {
            let g = CSR::from(edges);
            let vcount = g.vertex_count();

            // If there are no vertices, there's nothing to test.
            if vcount == 0 {
                // Just ensure that constructing a "no initial" search does not panic.
                let search = SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>
                    ::bfs(&g, std::iter::empty::<usize>());
                let mut search = search;
                while search.next().is_some() {}
                let visited = search.into_visited();
                prop_assert!(visited.is_empty());
                return Ok(());
            }

            // Map initials_raw into the valid vertex range.
            let initials: Vec<usize> = initials_raw
                .into_iter()
                .map(|x| (x as usize) % vcount)
                .collect();

            // Here it's actually fine if initials is empty: then the visited
            // set should also be empty and the "contains initials" property
            // is vacuously true.
            let mut search =
                SequentialBackwardSearch::<_, Set<usize>, QueueFrontier<_>>
                    ::bfs(&g, initials.iter().copied());

            for _v in search.by_ref() {}

            let visited = search.into_visited();

            for v in initials {
                prop_assert!(visited.contains(&v));
            }
        }
    }
}
