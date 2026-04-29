use std::collections::VecDeque;

use crate::{
    games::{
        arena::Arena,
        attractor::AttractorStrategySynthesis,
        controllable_predecessors::ControllablePredecessors,
        region::Region,
        strategy::{PositionalStrategy, PositionalSynthesisResult},
    },
    graphs::backward::Backward,
    lattices::frontier::{Frontier, QueueFrontier},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct WorklistAttractorStrategySynthesis;

impl WorklistAttractorStrategySynthesis {
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl<A, R, S> AttractorStrategySynthesis<A, R, S> for WorklistAttractorStrategySynthesis
where
    A: Arena,
    R: Region<A::Position>,
    S: PositionalStrategy<Arena = A>,
    A::Player: ControllablePredecessors<A, R>,
{
    fn synthesize(
        &self,
        arena: &A,
        player: A::Player,
        initial: A::Position,
        mut target: R,
    ) -> PositionalSynthesisResult<R, S> {
        let mut strategy = S::empty(player);
        if target.includes(initial) {
            return PositionalSynthesisResult::winning(target, strategy);
        }

        // Frontier needs to implement from IntoIter.
        let mut worklist: QueueFrontier<<S::Arena as Arena>::Position> =
            VecDeque::from(target.positions());

        while let Some(position) = worklist.pop() {
            // No graph is needed just implement Backward.
            for predecessor in arena.predecessors(position) {
                let source = arena.source(predecessor);

                // This should be the incremental operator abstraction:
                if target.includes(source) {
                    continue;
                }

                if player.is_controllable_predecessor(arena, &target, source) {

                    // Should be some abstraction for the general attractor computation that one can hook into.
                    if let Some(successor) = player.strategy_successor(arena, &target, source) {
                        strategy.insert_choice(source, successor);
                    }

                    target.expand(source);
                    worklist.push(source);

                    if source == initial {
                        return PositionalSynthesisResult::winning(target, strategy);
                    }
                }
            }
        }

        if target.includes(initial) {
            return PositionalSynthesisResult::winning(target, strategy);
        } else {
            return PositionalSynthesisResult::losing(target);
        }
    }
}
