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
