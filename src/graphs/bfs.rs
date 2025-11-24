use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};
use rustc_hash::FxHashSet;

use crate::graphs::frontier::Frontier;
use crate::graphs::visited::Visited;
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
