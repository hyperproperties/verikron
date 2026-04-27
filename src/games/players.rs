use crate::{
    games::{arena::Arena, attractor::AttractorPredecessor, region::Region},
    graphs::expansion::{Expansion, ForwardExpansion},
};

/// A player/owner type usable in arenas and games.
pub trait Player: Copy + Eq {}

impl<T> Player for T where T: Copy + Eq {}

/// A player type with a unique adversarial opponent.
pub trait OpposedPlayer: Player {
    /// Returns the opposing player.
    fn opponent(self) -> Self;
}

/// A single-player type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SinglePlayer {
    #[default]
    Eve,
}

impl AttractorPredecessor for SinglePlayer {
    #[inline]
    fn attractor_successor<A, R>(
        self,
        arena: &A,
        target: &R,
        position: R::Position,
    ) -> Option<R::Position>
    where
        A: Arena<Player = Self, Position = R::Position>,
        R: Region<Player = Self>,
        for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = R::Position>,
    {
        ForwardExpansion::new(arena)
            .successors(position)
            .find(|successor| target.contains(successor))
    }
}

/// A two-player adversarial type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TwoPlayer {
    Eve,
    Adam,
}

impl OpposedPlayer for TwoPlayer {
    #[inline]
    fn opponent(self) -> Self {
        match self {
            Self::Eve => Self::Adam,
            Self::Adam => Self::Eve,
        }
    }
}

impl AttractorPredecessor for TwoPlayer {
    #[inline]
    fn attractor_successor<A, R>(
        self,
        arena: &A,
        target: &R,
        position: R::Position,
    ) -> Option<R::Position>
    where
        A: Arena<Player = Self, Position = R::Position>,
        R: Region<Player = Self>,
        for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = R::Position>,
    {
        let binding = ForwardExpansion::new(arena);
        let mut successors = binding.successors(position);

        if arena.owner(position) == self {
            successors.find(|successor| target.contains(successor))
        } else if successors.all(|successor| target.contains(&successor)) {
            // Opponent-owned position: no choice for this player is needed.
            // Return `position` only as a witness that the predecessor condition holds.
            Some(position)
        } else {
            None
        }
    }
}
