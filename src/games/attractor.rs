use crate::{
    games::{arena::Arena, controllable_predecessors::ControllablePredecessors, region::Region},
    graphs::expansion::{BackwardExpansion, Expansion},
};

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

/// Worklist-based attractor computation.
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
    fn attractor_closure_from<I>(
        &self,
        arena: &A,
        player: A::Player,
        region: &mut R,
        frontier: I,
    ) -> bool
    where
        I: IntoIterator<Item = A::Position>,
    {
        let backward = BackwardExpansion::new(arena);

        let mut queue: Vec<A::Position> = frontier.into_iter().collect();
        let mut changed = false;

        while let Some(position) = queue.pop() {
            for predecessor in backward.successors(position) {
                if region.includes(predecessor) {
                    continue;
                }

                if player.is_controllable_predecessor(arena, region, predecessor)
                    && region.expand(predecessor)
                {
                    changed = true;
                    queue.push(predecessor);
                }
            }
        }

        changed
    }
}
