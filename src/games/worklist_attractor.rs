use std::collections::VecDeque;

use crate::{
    games::{
        arena::Arena,
        attractor::{Attractor, AttractorVisitor},
        controllable_predecessors::ControllablePredecessors,
        region::Region,
    },
    graphs::backward::Backward,
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
{
    fn attractor_with_visitor<V>(
        &self,
        arena: &A,
        player: A::Player,
        mut region: R,
        visitor: &mut V,
    ) -> R
    where
        V: AttractorVisitor<A, R>,
    {
        if visitor.start(arena, player, &region).is_break() {
            let _ = visitor.finish(&region);
            return region;
        }

        let mut worklist: QueueFrontier<A::Position> = VecDeque::from(region.positions());

        for position in region.positions() {
            if visitor.seed(position).is_break() {
                let _ = visitor.finish(&region);
                return region;
            }
        }

        while let Some(position) = worklist.pop() {
            if visitor.pop(position).is_break() {
                break;
            }

            for predecessor in arena.predecessors(position) {
                let source = arena.source(predecessor);

                if visitor.consider_predecessor(source, position).is_break() {
                    let _ = visitor.finish(&region);
                    return region;
                }

                if region.includes(source) {
                    if visitor.skip_known(source).is_break() {
                        let _ = visitor.finish(&region);
                        return region;
                    }

                    continue;
                }

                if !player.is_controllable_predecessor(arena, &region, source) {
                    if visitor.reject_predecessor(source).is_break() {
                        let _ = visitor.finish(&region);
                        return region;
                    }

                    continue;
                }

                if visitor
                    .before_insertion(arena, player, &region, source)
                    .is_break()
                {
                    let _ = visitor.finish(&region);
                    return region;
                }

                region.expand(source);

                if visitor.after_insertion(source).is_break() {
                    let _ = visitor.finish(&region);
                    return region;
                }

                worklist.push(source);

                if visitor.after_push(source).is_break() {
                    let _ = visitor.finish(&region);
                    return region;
                }
            }
        }

        let _ = visitor.finish(&region);
        region
    }
}
