use crate::games::{arena::Arena, play::Play, region::Region, strategy::Strategy};

/// A game consisting of an arena and a winning condition on plays.
pub trait Game {
    /// Underlying arena.
    type Arena: Arena;

    /// Play type evaluated by the game.
    type Play: Play<Position = <Self::Arena as Arena>::Position>;

    /// Returns the underlying arena.
    fn arena(&self) -> &Self::Arena;

    /// Returns the unique winner of `play`, if one is defined.
    fn is_winning(&self, player: <Self::Arena as Arena>::Player, play: &Self::Play) -> bool;
}

/// A game that supports strategy synthesis.
pub trait SolvableGame: Game {
    /// Strategy type synthesized for this game.
    type Strategy: Strategy<Arena = Self::Arena>;

    /// Returns whether `strategy` wins for `player` from `position`.
    fn is_winning_strategy_from(
        &self,
        player: <Self::Arena as Arena>::Player,
        strategy: &Self::Strategy,
        position: <Self::Arena as Arena>::Position,
    ) -> bool;

    /// Returns a winning strategy for `player` from `position`, if one exists.
    fn winning_strategy_from(
        &self,
        player: <Self::Arena as Arena>::Player,
        position: <Self::Arena as Arena>::Position,
    ) -> Option<Self::Strategy>;

    /// Returns whether `player` has a winning strategy from `position`.
    #[must_use]
    #[inline]
    fn has_winning_strategy_from(
        &self,
        player: <Self::Arena as Arena>::Player,
        position: <Self::Arena as Arena>::Position,
    ) -> bool {
        self.winning_strategy_from(player, position).is_some()
    }
}

/// A game whose whole winning regions can be computed.
pub trait RegionSolvableGame: Game {
    /// Region type used for winning regions.
    type Region: Region<<Self::Arena as Arena>::Position>;

    /// Returns the winning region of `player`.
    fn winning_region(&self, player: <Self::Arena as Arena>::Player) -> Self::Region;
}
