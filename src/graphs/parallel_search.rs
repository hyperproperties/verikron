use std::{fmt::Debug, hash::Hash};

use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};

use crate::graphs::{
    backward::Backward,
    expansion::{
        BackwardExpansion, Expansion, ExpansionStateOf, ForwardExpansion, HyperBackwardExpansion,
        HyperForwardExpansion,
    },
    forward::Forward,
    frontier::IncrementalFrontier,
    layered_frontier::LayeredFrontier,
    search::{Discovery, Search},
    structure::{VertexOf, VertexType},
    visited::Visited,
};

/// Parallel search over a forward graph.
pub type ParallelForwardSearch<'g, G, V> = ParallelSearch<ForwardExpansion<'g, G>, V>;
/// Parallel search over a backward graph.
pub type ParallelBackwardSearch<'g, G, V> = ParallelSearch<BackwardExpansion<'g, G>, V>;

/// Parallel search over a forward hypergraph.
pub type ParallelHyperForwardSearch<'g, G, V> = ParallelSearch<HyperForwardExpansion<'g, G>, V>;
/// Parallel search over a backward hypergraph.
pub type ParallelHyperBackwardSearch<'g, G, V> = ParallelSearch<HyperBackwardExpansion<'g, G>, V>;

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
    frontier: LayeredFrontier<Discovery<ExpansionStateOf<X>>>,
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
                initial_layer.push(Discovery::root(state));
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

    /// Returns the current search layer as discovered vertices.
    #[must_use]
    #[inline]
    pub fn layer(&self) -> Vec<ExpansionStateOf<X>> {
        self.frontier
            .layer()
            .iter()
            .map(Discovery::vertex)
            .collect()
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
        self.frontier
            .increment(|current, next| {
                for &discovery in current {
                    let state = discovery.vertex();

                    for successor in self.expansion.successors(state) {
                        if self.visited.visit(successor) {
                            next.push(Discovery::child(state, successor));
                        }
                    }
                }

                debug_assert!(
                    next.iter()
                        .all(|discovery| self.visited.is_visited(&discovery.vertex()))
                );
            })
            .map(|layer| {
                layer
                    .into_iter()
                    .map(|discovery| discovery.vertex())
                    .collect()
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
        self.frontier
            .increment(|current, next| {
                self.scratch.clear();

                self.scratch.par_extend(
                    current
                        .par_iter()
                        .flat_map_iter(|discovery| self.expansion.successors(discovery.vertex())),
                );

                for successor in self.scratch.drain(..) {
                    if self.visited.visit(successor) {
                        next.push(Discovery::child(current[0].vertex(), successor));
                    }
                }

                debug_assert!(
                    next.iter()
                        .all(|discovery| self.visited.is_visited(&discovery.vertex()))
                );
            })
            .map(|layer| {
                layer
                    .into_iter()
                    .map(|discovery| discovery.vertex())
                    .collect()
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

impl<X, V> VertexType for ParallelSearch<X, V>
where
    X: Expansion + Sync,
    ExpansionStateOf<X>: Eq + Hash + Copy + Send + Sync + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
    type Vertex = X::Vertex;
}

impl<X, V> Search for ParallelSearch<X, V>
where
    X: Expansion + Sync,
    ExpansionStateOf<X>: Eq + Hash + Copy + Send + Sync + Debug,
    V: Visited<ExpansionStateOf<X>>,
{
    #[inline]
    fn discover(&mut self) -> Option<Discovery<Self::Vertex>> {
        loop {
            if self.frontier.is_empty() {
                return None;
            }

            if self.index < self.frontier.len() {
                let discovery = self.frontier[self.index];
                self.index += 1;
                return Some(discovery);
            }

            self.step()?;
            self.index = 0;
        }
    }
}

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
