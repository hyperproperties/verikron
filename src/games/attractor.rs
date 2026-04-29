use crate::games::{
    arena::Arena, controllable_predecessors::ControllablePredecessors, region::Region,
};

/// A strategy for computing reachability attractors.
pub trait Attractor<A, R>
where
    A: Arena,
    R: Region<A::Position>,
    A::Player: ControllablePredecessors<A, R>,
{
    fn attractor_closure_from(&self, arena: &A, player: A::Player, region: R) -> R;
}
