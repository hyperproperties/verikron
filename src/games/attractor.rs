use crate::
    games::{arena::Arena, controllable_predecessors::ControllablePredecessors, region::Region}
;

/// A strategy for computing reachability attractors.
pub trait Attractor<A, R>
where
    A: Arena,
    R: Region<A::Position>,
    A::Player: ControllablePredecessors<A, R>,
{
    fn attractor_closure_from<I>(
        &self,
        arena: &A,
        player: A::Player,
        region: &mut R,
        frontier: I,
    ) -> bool
    where
        I: IntoIterator<Item = A::Position>;

    fn attractor_from<I>(&self, arena: &A, player: A::Player, mut region: R, target: I) -> R
    where
        I: IntoIterator<Item = A::Position>,
    {
        let target: Vec<A::Position> = target.into_iter().collect();

        for position in target.iter().copied() {
            region.expand(position);
        }

        self.attractor_closure_from(arena, player, &mut region, target);

        region
    }
}