use std::ops::ControlFlow;

use crate::games::{
    arena::Arena,
    attractor::{Attractor, AttractorStrategySynthesis, AttractorVisitor},
    controllable_predecessors::ControllablePredecessors,
    region::Region,
    strategy::{PositionalStrategy, PositionalSynthesisResult},
    worklist_attractor::WorklistAttractor,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct WorklistAttractorStrategySynthesis;

impl WorklistAttractorStrategySynthesis {
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

pub struct PositionalSynthesisVisitor<S>
where
    S: PositionalStrategy,
{
    initial: <S::Arena as Arena>::Position,
    strategy: S,
    reached_initial: bool,
}

impl<S> PositionalSynthesisVisitor<S>
where
    S: PositionalStrategy,
{
    #[inline]
    pub fn new(
        player: <S::Arena as Arena>::Player,
        initial: <S::Arena as Arena>::Position,
    ) -> Self {
        Self {
            initial,
            strategy: S::empty(player),
            reached_initial: false,
        }
    }

    #[inline]
    pub const fn reached_initial(&self) -> bool {
        self.reached_initial
    }

    #[inline]
    pub fn into_strategy(self) -> S {
        self.strategy
    }
}

impl<A, R, S> AttractorVisitor<A, R> for PositionalSynthesisVisitor<S>
where
    A: Arena,
    R: Region<A::Position>,
    S: PositionalStrategy<Arena = A>,
    A::Player: ControllablePredecessors<A, R>,
{
    #[inline]
    fn before_insertion(
        &mut self,
        arena: &A,
        player: A::Player,
        region: &R,
        source: A::Position,
    ) -> ControlFlow<()> {
        if let Some(successor) = player.strategy_successor(arena, region, source) {
            self.strategy.insert_choice(source, successor);
        }

        ControlFlow::Continue(())
    }

    #[inline]
    fn after_insertion(&mut self, source: A::Position) -> ControlFlow<()> {
        if source == self.initial {
            self.reached_initial = true;
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<A, R, S> AttractorStrategySynthesis<A, R, S> for WorklistAttractorStrategySynthesis
where
    A: Arena,
    R: Region<A::Position>,
    S: PositionalStrategy<Arena = A>,
    A::Player: ControllablePredecessors<A, R>,
{
    #[inline]
    fn synthesize(
        &self,
        arena: &A,
        player: A::Player,
        initial: A::Position,
        target: R,
    ) -> PositionalSynthesisResult<R, S> {
        let mut visitor = PositionalSynthesisVisitor::<S>::new(player, initial);

        if target.includes(initial) {
            return PositionalSynthesisResult::winning(target, visitor.into_strategy());
        }

        let attractor =
            WorklistAttractor::new().attractor_with_visitor(arena, player, target, &mut visitor);

        let reached_initial = visitor.reached_initial();
        let strategy = visitor.into_strategy();

        if reached_initial {
            PositionalSynthesisResult::winning(attractor, strategy)
        } else {
            PositionalSynthesisResult::losing(attractor)
        }
    }
}
