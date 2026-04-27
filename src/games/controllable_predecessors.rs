use crate::{
    games::{
        arena::Arena,
        players::{SinglePlayer, TwoPlayer},
        region::Region,
    },
    graphs::expansion::{Expansion, ForwardExpansion},
};

/// Controllable predecessor semantics for a player model.
///
/// This trait defines the one-step predecessor operator used by attractor
/// computations.
pub trait ControllablePredecessors<A, R>: Sized
where
    A: Arena<Player = Self>,
    R: Region<A::Position>,
{
    fn is_controllable_predecessor(
        &self,
        arena: &A,
        target: &R,
        position: A::Position,
    ) -> bool;

    fn strategy_successor(
        &self,
        arena: &A,
        target: &R,
        position: A::Position,
    ) -> Option<A::Position>;
}

impl<A, R> ControllablePredecessors<A, R> for SinglePlayer
where
    A: Arena<Player = SinglePlayer>,
    R: Region<A::Position>,
    for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = A::Position>,
{
    fn is_controllable_predecessor(
        &self,
        arena: &A,
        target: &R,
        position: A::Position,
    ) -> bool {
        ForwardExpansion::new(arena)
            .successors(position)
            .any(|successor| target.includes(successor))
    }

    fn strategy_successor(
        &self,
        arena: &A,
        target: &R,
        position: A::Position,
    ) -> Option<A::Position> {
        ForwardExpansion::new(arena)
            .successors(position)
            .find(|successor| target.includes(*successor))
    }
}

impl<A, R> ControllablePredecessors<A, R> for TwoPlayer
where
    A: Arena<Player = TwoPlayer>,
    R: Region<A::Position>,
    for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = A::Position>,
{
    fn is_controllable_predecessor(
        &self,
        arena: &A,
        target: &R,
        position: A::Position,
    ) -> bool {
        let forward = ForwardExpansion::new(arena);
        let mut successors = forward.successors(position);

        if arena.owner(position) == *self {
            successors.any(|successor| target.includes(successor))
        } else {
            successors.all(|successor| target.includes(successor))
        }
    }

    fn strategy_successor(
        &self,
        arena: &A,
        target: &R,
        position: A::Position,
    ) -> Option<A::Position> {
        if arena.owner(position) != *self {
            return None;
        }

        ForwardExpansion::new(arena)
            .successors(position)
            .find(|successor| target.includes(*successor))
    }
}
