use std::collections::VecDeque;

use crate::{
    games::{
        arena::Arena, attractor::Attractor, controllable_predecessors::ControllablePredecessors,
        region::Region,
    },
    graphs::{backward::Backward, expansion::{BackwardExpansion, Expansion}},
    lattices::frontier::{Frontier, QueueFrontier},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct WorklistAttractor;

impl WorklistAttractor {
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl<A, R> Attractor<A, R> for WorklistAttractor
where
    A: Arena,
    R: Region<A::Position>,
    A::Player: ControllablePredecessors<A, R>,
    for<'g> BackwardExpansion<'g, A>: Expansion<Vertex = A::Position>,
{
    fn attractor_closure_from(&self, arena: &A, player: <A as Arena>::Player, mut region: R) -> R {
        // Frontier needs to implement from IntoIter.
        let mut worklist: QueueFrontier<A::Position> = VecDeque::from(region.positions());
        while let Some(position) = worklist.pop() {
            // No graph is needed just implement Backward.
            for predecessor in arena.predecessors(position) {
                let source = arena.source(predecessor);

                // This should be the incremental operator abstraction:
                if region.includes(source) {
                    continue;
                }

                if player.is_controllable_predecessor(arena, &region, source) {
                    region.expand(source);
                    worklist.push(source);
                } 
            }
        }

        region
    }
}
