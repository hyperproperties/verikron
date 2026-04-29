use std::collections::VecDeque;

use crate::{
    graphs::{graph::Directed, structure::Vertices},
    lattices::{
        fixpoint::Fixpoint,
        monotone::{Direction, StatefulMonotone},
    },
};

/// Worklist-based fixed-point solver.
///
/// The direction determines which neighbors provide input facts and where
/// changes are propagated.
///
/// The worklist works for monotone analyses over finite-height domains,
/// provided the analysis initializes, reads, updates, and propagates facts
/// consistently with the fixed-point equation.
pub struct Worklist<'g, G: Directed, D: Direction<Vertex = G::Vertex>> {
    graph: &'g G,
    direction: D,
}

impl<'g, G: Directed, D: Direction<Vertex = G::Vertex>> Worklist<'g, G, D> {
    /// Creates a worklist solver over `graph` using `direction`.
    pub fn new(graph: &'g G, direction: D) -> Self {
        Self { graph, direction }
    }
}

impl<'g, G, D, A> Fixpoint<A> for Worklist<'g, G, D>
where
    G: Directed,
    G::Vertex: Copy,
    D: Direction<Vertex = G::Vertex>,
    A: StatefulMonotone<G>,
{
    type Solution = A::Output;

    fn solve(&self, mut analysis: A) -> Self::Solution {
        let mut worklist = VecDeque::new();

        analysis.initialize(self.graph);

        for node in self.graph.vertex_store().vertices() {
            worklist.push_back(node);
        }

        while let Some(node) = worklist.pop_front() {
            let input = analysis.merge(
                &node,
                self.direction
                    .dependencies(node)
                    .map(|dependency| analysis.fact(&dependency)),
            );

            let mut new_fact = analysis.transfer(&node, &input);

            if let Some(boundary_fact) = analysis.boundary_fact(&node) {
                new_fact = boundary_fact;
            }

            if analysis.set(&node, &new_fact) {
                for dependent in self.direction.dependents(node) {
                    worklist.push_back(dependent);
                }
            }
        }

        analysis.finish()
    }
}
