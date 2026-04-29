use std::ops::ControlFlow;

use crate::games::{
    arena::Arena,
    controllable_predecessors::ControllablePredecessors,
    region::Region,
    strategy::{PositionalStrategy, PositionalSynthesisResult},
};

/// A strategy for computing reachability attractors.
pub trait Attractor<A, R>
where
    A: Arena,
    R: Region<A::Position>,
    A::Player: ControllablePredecessors<A, R>,
{
    fn attractor_closure_from(&self, arena: &A, player: A::Player, target: R) -> R;
}

pub trait AttractorVisitor<A, R>
where
    A: Arena,
    R: Region<A::Position>,
{
    /// Called before the attractor computation starts.
    #[inline]
    fn start(&mut self, _arena: &A, _player: A::Player, _region: &R) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called for each initial target position added to the worklist.
    #[inline]
    fn seed(&mut self, _position: A::Position) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called when a position is popped from the worklist.
    #[inline]
    fn pop(&mut self, _position: A::Position) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called when inspecting a predecessor of the current position.
    #[inline]
    fn consider_predecessor(
        &mut self,
        _source: A::Position,
        _target: A::Position,
    ) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called when a predecessor is skipped because it is already in the attractor.
    #[inline]
    fn skip_known(&mut self, _source: A::Position) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called when a predecessor fails the controllable-predecessor test.
    #[inline]
    fn reject_predecessor(&mut self, _source: A::Position) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called after a predecessor is accepted but before it is inserted.
    #[inline]
    fn before_insertion(
        &mut self,
        _arena: &A,
        _player: A::Player,
        _region: &R,
        _source: A::Position,
    ) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called immediately after a new position is inserted into the attractor.
    #[inline]
    fn after_insertion(&mut self, _source: A::Position) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called immediately after a new position is pushed onto the worklist.
    #[inline]
    fn after_push(&mut self, _source: A::Position) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }

    /// Called after the computation finishes or stops early.
    #[inline]
    fn finish(&mut self, _region: &R) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopAttractorVisitor;

impl<A, R> AttractorVisitor<A, R> for NoopAttractorVisitor
where
    A: Arena,
    R: Region<A::Position>,
{
}

pub trait AttractorStrategySynthesis<A, R, S>
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
        target: R,
    ) -> PositionalSynthesisResult<R, S>;
}
