use crate::{games::{arena::Arena, attractor::Attractor, controllable_predecessors::ControllablePredecessors, region::Region}, graphs::expansion::{BackwardExpansion, Expansion}};


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
