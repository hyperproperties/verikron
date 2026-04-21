use std::{fmt::Debug, hash::Hash};

use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};

use crate::graphs::{
    backward::Backward,
    expansion::{BackwardExpansion, Expansion, ExpansionStateOf, ForwardExpansion},
    forward::Forward,
    frontier::{IncrementalFrontier, LayeredFrontier},
    reachability::LinearReachability,
    search::VisitedSearch,
    structure::VertexOf,
    visited::Visited,
    worklist::ExhaustiveWorklist,
};

/// Default minimum layer size for parallel expansion.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 1024;

/// Parallel layered search over an expansion relation.
///
/// This is the parallel counterpart to sequential search, specialized to
/// breadth-first / layer-by-layer exploration.
///
/// The current layer is expanded either sequentially or in parallel depending
/// on `parallel_threshold`.
#[derive(Debug, Clone)]
pub struct ParallelSearch<X, V>
where
    X: Expansion,
    ExpansionStateOf<X>: Eq + Hash + Copy + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
    expansion: X,
    visited: V,
    frontier: LayeredFrontier<ExpansionStateOf<X>>,
    scratch: Vec<ExpansionStateOf<X>>,
    index: usize,
    parallel_threshold: usize,
}

impl<X, V> ParallelSearch<X, V>
where
    X: Expansion,
    ExpansionStateOf<X>: Eq + Hash + Copy + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
    /// Creates a parallel search from an expansion relation, an explicit
    /// visited structure, initial states, and a parallel threshold.
    #[must_use]
    pub fn with_expansion_and_visited_and_parallel_threshold(
        expansion: X,
        initials: impl IntoIterator<Item = ExpansionStateOf<X>>,
        mut visited: V,
        parallel_threshold: usize,
    ) -> Self {
        let mut initial_layer = Vec::new();

        for state in initials {
            if visited.visit(state) {
                initial_layer.push(state);
            }
        }

        Self {
            expansion,
            visited,
            frontier: LayeredFrontier::new(initial_layer),
            scratch: Vec::new(),
            index: 0,
            parallel_threshold,
        }
    }

    /// Creates a parallel search from an expansion relation, an explicit
    /// visited structure, and initial states.
    #[must_use]
    #[inline]
    pub fn with_expansion_and_visited(
        expansion: X,
        initials: impl IntoIterator<Item = ExpansionStateOf<X>>,
        visited: V,
    ) -> Self {
        Self::with_expansion_and_visited_and_parallel_threshold(
            expansion,
            initials,
            visited,
            DEFAULT_PARALLEL_THRESHOLD,
        )
    }

    /// Returns the underlying expansion relation.
    #[must_use]
    #[inline]
    pub fn expansion(&self) -> &X {
        &self.expansion
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

    /// Returns the current search layer.
    #[must_use]
    #[inline]
    pub fn layer(&self) -> &[ExpansionStateOf<X>] {
        self.frontier.layer()
    }

    /// Returns whether the search has no more states to expand.
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

    /// Expands one layer sequentially and returns the previous layer.
    #[inline]
    pub fn sequential_step(&mut self) -> Option<Vec<ExpansionStateOf<X>>> {
        self.frontier.step(|current, next| {
            for &state in current {
                for successor in self.expansion.successors(state) {
                    if self.visited.visit(successor) {
                        next.push(successor);
                    }
                }
            }

            debug_assert!(next.iter().all(|state| self.visited.is_visited(state)));
        })
    }
}

impl<X, V> ParallelSearch<X, V>
where
    X: Expansion,
    ExpansionStateOf<X>: Eq + Hash + Copy + Debug,
    V: Visited<ExpansionStateOf<X>> + Default,
{
    /// Creates a parallel search from an expansion relation, initial states,
    /// and a parallel threshold.
    #[must_use]
    #[inline]
    pub fn with_expansion_and_parallel_threshold(
        expansion: X,
        initials: impl IntoIterator<Item = ExpansionStateOf<X>>,
        parallel_threshold: usize,
    ) -> Self {
        Self::with_expansion_and_visited_and_parallel_threshold(
            expansion,
            initials,
            V::default(),
            parallel_threshold,
        )
    }

    /// Creates a parallel search from an expansion relation and initial states.
    #[must_use]
    #[inline]
    pub fn with_expansion(
        expansion: X,
        initials: impl IntoIterator<Item = ExpansionStateOf<X>>,
    ) -> Self {
        Self::with_expansion_and_parallel_threshold(expansion, initials, DEFAULT_PARALLEL_THRESHOLD)
    }
}

impl<X, V> ParallelSearch<X, V>
where
    X: Expansion + Sync,
    ExpansionStateOf<X>: Eq + Hash + Copy + Send + Sync + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
    /// Expands one layer in parallel and returns the previous layer.
    #[inline]
    pub fn parallel_step(&mut self) -> Option<Vec<ExpansionStateOf<X>>> {
        self.frontier.step(|current, next| {
            self.scratch.clear();

            self.scratch.par_extend(
                current
                    .par_iter()
                    .flat_map_iter(|&state| self.expansion.successors(state)),
            );

            for successor in self.scratch.drain(..) {
                if self.visited.visit(successor) {
                    next.push(successor);
                }
            }

            debug_assert!(next.iter().all(|state| self.visited.is_visited(state)));
        })
    }

    /// Expands one layer, choosing sequential or parallel execution
    /// automatically.
    #[inline]
    pub fn step(&mut self) -> Option<Vec<ExpansionStateOf<X>>> {
        if self.frontier.len() < self.parallel_threshold {
            self.sequential_step()
        } else {
            self.parallel_step()
        }
    }
}

impl<X, V> Iterator for ParallelSearch<X, V>
where
    X: Expansion + Sync,
    ExpansionStateOf<X>: Eq + Hash + Copy + Send + Sync + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
    type Item = ExpansionStateOf<X>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.frontier.is_empty() {
                return None;
            }

            if self.index < self.frontier.len() {
                let state = self.frontier[self.index];
                self.index += 1;
                return Some(state);
            }

            self.step()?;
            self.index = 0;
        }
    }
}

impl<X, V> VisitedSearch for ParallelSearch<X, V>
where
    X: Expansion + Sync,
    ExpansionStateOf<X>: Eq + Hash + Copy + Send + Sync + Debug,
    V: Visited<ExpansionStateOf<X>>,
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

impl<X, V> LinearReachability for ParallelSearch<X, V>
where
    X: Expansion + Sync,
    ExpansionStateOf<X>: Eq + Hash + Copy + Send + Sync + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
}

impl<X, V> ExhaustiveWorklist for ParallelSearch<X, V>
where
    X: Expansion + Sync,
    ExpansionStateOf<X>: Eq + Hash + Copy + Send + Sync + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
}

/// Parallel breadth-first search over a forward graph.
pub type ParallelGraphBFS<'g, G, V> = ParallelSearch<ForwardExpansion<'g, G>, V>;

/// Parallel breadth-first backward search over a graph.
pub type ParallelBackwardBFS<'g, G, V> = ParallelSearch<BackwardExpansion<'g, G>, V>;

impl<'g, G, V> ParallelSearch<ForwardExpansion<'g, G>, V>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy + Debug,
    V: Visited<VertexOf<G>>,
{
    /// Creates a parallel BFS with a custom visited structure and the default
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

    /// Creates a parallel BFS with a custom visited structure and a custom
    /// parallel threshold.
    #[must_use]
    #[inline]
    pub fn with_visited_and_parallel_threshold(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        visited: V,
        parallel_threshold: usize,
    ) -> Self {
        Self::with_expansion_and_visited_and_parallel_threshold(
            ForwardExpansion::new(graph),
            initials,
            visited,
            parallel_threshold,
        )
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &'g G {
        self.expansion().graph()
    }
}

impl<'g, G, V> ParallelSearch<ForwardExpansion<'g, G>, V>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy + Debug,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a parallel BFS with the default visited structure and the
    /// default parallel threshold.
    #[must_use]
    #[inline]
    pub fn new(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_parallel_threshold(graph, initials, DEFAULT_PARALLEL_THRESHOLD)
    }

    /// Creates a parallel BFS with the default visited structure and a custom
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

impl<'g, G, V> ParallelSearch<BackwardExpansion<'g, G>, V>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy + Debug,
    V: Visited<VertexOf<G>>,
{
    /// Creates a parallel backward BFS with a custom visited structure and the
    /// default parallel threshold.
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

    /// Creates a parallel backward BFS with a custom visited structure and a
    /// custom parallel threshold.
    #[must_use]
    #[inline]
    pub fn with_visited_and_parallel_threshold(
        graph: &'g G,
        initials: impl IntoIterator<Item = VertexOf<G>>,
        visited: V,
        parallel_threshold: usize,
    ) -> Self {
        Self::with_expansion_and_visited_and_parallel_threshold(
            BackwardExpansion::new(graph),
            initials,
            visited,
            parallel_threshold,
        )
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &'g G {
        self.expansion().graph()
    }
}

impl<'g, G, V> ParallelSearch<BackwardExpansion<'g, G>, V>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy + Debug,
    V: Visited<VertexOf<G>> + Default,
{
    /// Creates a parallel backward BFS with the default visited structure and
    /// the default parallel threshold.
    #[must_use]
    #[inline]
    pub fn new(graph: &'g G, initials: impl IntoIterator<Item = VertexOf<G>>) -> Self {
        Self::with_parallel_threshold(graph, initials, DEFAULT_PARALLEL_THRESHOLD)
    }

    /// Creates a parallel backward BFS with the default visited structure and
    /// a custom parallel threshold.
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        arc::{Arc, FromArcs},
        backward::Backward,
        csr::CSR,
        forward::Forward,
        graph::Directed,
        reachability::Reachability,
        structure::FiniteVertices,
        worklist::Worklist,
    };
    use crate::lattices::{bit_vector::BitVector, set::Set};

    use proptest::prelude::*;
    use std::collections::VecDeque;

    fn arbitrary_instance() -> impl Strategy<Value = (Vec<(usize, usize)>, Vec<usize>)> {
        prop::collection::vec((0usize..16, 0usize..16), 0..64).prop_flat_map(|edges| {
            let vertex_count = edges
                .iter()
                .map(|&(source, destination)| source.max(destination))
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

        while let Some(source) = queue.pop_front() {
            for edge in graph.successors(source) {
                let destination = graph.destination(edge);
                if visited.visit(destination) {
                    queue.push_back(destination);
                }
            }
        }

        visited
    }

    fn reference_backward_reachable(graph: &CSR, initials: &[usize]) -> Set<usize> {
        let mut visited = Set::default();
        let mut queue = VecDeque::new();

        for &destination in initials {
            if destination < graph.vertex_count() && visited.visit(destination) {
                queue.push_back(destination);
            }
        }

        while let Some(destination) = queue.pop_front() {
            for edge in graph.predecessors(destination) {
                let source = graph.source(edge);
                if visited.visit(source) {
                    queue.push_back(source);
                }
            }
        }

        visited
    }

    #[test]
    fn default_parallel_threshold_is_exposed() {
        let graph = CSR::from_arcs([Arc::new(0, 1)]);
        let bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        assert_eq!(bfs.parallel_threshold(), DEFAULT_PARALLEL_THRESHOLD);
    }

    #[test]
    fn parallel_threshold_can_be_changed() {
        let graph = CSR::from_arcs([Arc::new(0, 1)]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        bfs.set_parallel_threshold(7);

        assert_eq!(bfs.parallel_threshold(), 7);
    }

    #[test]
    fn initial_layer_is_deduplicated_through_visited() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(2, 3)]);
        let bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0, 0, 2, 2]);

        assert_eq!(bfs.layer(), &[0, 2]);
        assert_eq!(bfs.visited(), &[0, 2].into_iter().collect::<Set<_>>());
    }

    #[test]
    fn forward_bfs_empty_graph_without_initials_is_empty() {
        let graph = CSR::from_arcs(std::iter::empty::<Arc<usize>>());
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> =
            ParallelGraphBFS::new(&graph, std::iter::empty());

        assert!(bfs.next().is_none());
        assert!(bfs.is_finished());
        assert!(bfs.visited().is_empty());
    }

    #[test]
    fn forward_bfs_line_graph_visits_vertices_in_layer_order() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 3)]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        let order: Vec<_> = bfs.by_ref().collect();

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn forward_bfs_branching_graph_reaches_all_reachable_vertices() {
        let graph = CSR::from_arcs([
            Arc::new(0, 1),
            Arc::new(0, 2),
            Arc::new(1, 3),
            Arc::new(2, 3),
        ]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        let mut seen: Vec<_> = bfs.by_ref().collect();
        seen.sort_unstable();

        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn forward_bfs_cycle_graph_terminates() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 1)]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        let mut seen: Vec<_> = bfs.by_ref().collect();
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(bfs.into_visited(), [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn forward_worklist_on_disconnected_graph_depends_on_initials() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(2, 3)]);

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
    fn forward_worklist_handles_loops() {
        let graph = CSR::from_arcs([Arc::new(0, 0), Arc::new(0, 1), Arc::new(1, 1)]);

        let reachable: Set<usize> =
            ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, [0]).worklist();

        assert_eq!(reachable, [0, 1].into_iter().collect());
    }

    #[test]
    fn backward_bfs_empty_graph_without_initials_is_empty() {
        let graph = CSR::from_arcs(std::iter::empty::<Arc<usize>>());
        let mut bfs: ParallelBackwardBFS<CSR, Set<usize>> =
            ParallelBackwardBFS::new(&graph, std::iter::empty());

        assert!(bfs.next().is_none());
        assert!(bfs.is_finished());
        assert!(bfs.visited().is_empty());
    }

    #[test]
    fn backward_bfs_line_graph_visits_predecessors_in_layer_order() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 3)]);
        let mut bfs: ParallelBackwardBFS<CSR, Set<usize>> = ParallelBackwardBFS::new(&graph, [3]);

        let order: Vec<_> = bfs.by_ref().collect();

        assert_eq!(order, vec![3, 2, 1, 0]);
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn backward_bfs_branching_graph_reaches_all_predecessors() {
        let graph = CSR::from_arcs([Arc::new(0, 2), Arc::new(1, 2), Arc::new(2, 3)]);
        let mut bfs: ParallelBackwardBFS<CSR, Set<usize>> = ParallelBackwardBFS::new(&graph, [3]);

        let mut seen: Vec<_> = bfs.by_ref().collect();
        seen.sort_unstable();

        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(bfs.into_visited(), [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn backward_worklist_on_disconnected_graph_depends_on_initials() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(2, 3)]);

        let from_left: Set<usize> =
            ParallelBackwardBFS::<CSR, Set<usize>>::new(&graph, [1]).worklist();
        let from_right: Set<usize> =
            ParallelBackwardBFS::<CSR, Set<usize>>::new(&graph, [3]).worklist();
        let from_both: Set<usize> =
            ParallelBackwardBFS::<CSR, Set<usize>>::new(&graph, [1, 3]).worklist();

        assert_eq!(from_left, [0, 1].into_iter().collect());
        assert_eq!(from_right, [2, 3].into_iter().collect());
        assert_eq!(from_both, [0, 1, 2, 3].into_iter().collect());
    }

    #[test]
    fn forward_sequential_and_parallel_step_agree() {
        let graph = CSR::from_arcs([
            Arc::new(0, 1),
            Arc::new(0, 2),
            Arc::new(1, 3),
            Arc::new(2, 3),
            Arc::new(3, 4),
            Arc::new(4, 5),
            Arc::new(5, 3),
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

    #[test]
    fn backward_sequential_and_parallel_step_agree() {
        let graph = CSR::from_arcs([
            Arc::new(0, 2),
            Arc::new(1, 2),
            Arc::new(2, 3),
            Arc::new(3, 4),
            Arc::new(4, 5),
            Arc::new(5, 3),
        ]);

        let mut sequential: ParallelBackwardBFS<CSR, Set<usize>> =
            ParallelBackwardBFS::new(&graph, [5]);
        let mut parallel: ParallelBackwardBFS<CSR, Set<usize>> =
            ParallelBackwardBFS::new(&graph, [5]);

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

    #[test]
    fn forward_reachability_finds_present_goal() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 3)]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        assert!(bfs.reachable(3));
    }

    #[test]
    fn forward_reachability_rejects_absent_goal() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(2, 3)]);
        let mut bfs: ParallelGraphBFS<CSR, Set<usize>> = ParallelGraphBFS::new(&graph, [0]);

        assert!(!bfs.reachable(3));
    }

    #[test]
    fn backward_reachability_finds_present_goal() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 3)]);
        let mut bfs: ParallelBackwardBFS<CSR, Set<usize>> = ParallelBackwardBFS::new(&graph, [3]);

        assert!(bfs.reachable(0));
    }

    #[test]
    fn backward_reachability_rejects_absent_goal() {
        let graph = CSR::from_arcs([Arc::new(0, 1), Arc::new(2, 3)]);
        let mut bfs: ParallelBackwardBFS<CSR, Set<usize>> = ParallelBackwardBFS::new(&graph, [1]);

        assert!(!bfs.reachable(3));
    }

    proptest! {
        #[test]
        fn prop_forward_worklist_matches_reference_bfs((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter()
                    .copied()
                    .map(|(source, destination)| Arc::new(source, destination)),
            );

            let actual: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            let expected = reference_reachable(&graph, &initials);

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn prop_backward_worklist_matches_reference_bfs((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter()
                    .copied()
                    .map(|(source, destination)| Arc::new(source, destination)),
            );

            let actual: Set<usize> =
                ParallelBackwardBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            let expected = reference_backward_reachable(&graph, &initials);

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn prop_forward_bitvector_and_set_visited_agree((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter()
                    .copied()
                    .map(|(source, destination)| Arc::new(source, destination)),
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
        fn prop_forward_every_initial_is_reached((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter()
                    .copied()
                    .map(|(source, destination)| Arc::new(source, destination)),
            );

            let reachable: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            for &initial in &initials {
                prop_assert!(reachable.contains(&initial));
            }
        }

        #[test]
        fn prop_backward_every_initial_is_reached((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter()
                    .copied()
                    .map(|(source, destination)| Arc::new(source, destination)),
            );

            let reachable: Set<usize> =
                ParallelBackwardBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            for &initial in &initials {
                prop_assert!(reachable.contains(&initial));
            }
        }

        #[test]
        fn prop_forward_worklist_is_idempotent((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter()
                    .copied()
                    .map(|(source, destination)| Arc::new(source, destination)),
            );

            let first: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            let second: Set<usize> =
                ParallelGraphBFS::<CSR, Set<usize>>::new(&graph, first.iter().copied()).worklist();

            prop_assert_eq!(first, second);
        }

        #[test]
        fn prop_backward_worklist_is_idempotent((edges, initials) in arbitrary_instance()) {
            let graph = CSR::from_arcs(
                edges.iter()
                    .copied()
                    .map(|(source, destination)| Arc::new(source, destination)),
            );

            let first: Set<usize> =
                ParallelBackwardBFS::<CSR, Set<usize>>::new(&graph, initials.iter().copied()).worklist();

            let second: Set<usize> =
                ParallelBackwardBFS::<CSR, Set<usize>>::new(&graph, first.iter().copied()).worklist();

            prop_assert_eq!(first, second);
        }
    }
}
