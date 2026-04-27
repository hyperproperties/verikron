use crate::games::arena::Arena;

/// A strategy for an arena.
///
/// A strategy consists of a player, a memory state, a memory update function,
/// and a choice function on positions.
pub trait Strategy {
    /// Arena on which the strategy is defined.
    type Arena: Arena;

    /// Internal memory state.
    type Memory: Copy + Eq;

    /// Returns the player following the strategy.
    fn player(&self) -> <Self::Arena as Arena>::Player;

    /// Returns the initial memory state.
    fn initial_memory(&self) -> Self::Memory;

    /// Updates memory after observing `position`.
    fn update(
        &self,
        memory: &Self::Memory,
        position: <Self::Arena as Arena>::Position,
    ) -> Self::Memory;

    /// Returns the chosen successor from `position` under `memory`, if defined.
    fn choice(
        &self,
        memory: &Self::Memory,
        position: <Self::Arena as Arena>::Position,
    ) -> Option<<Self::Arena as Arena>::Position>;
}

/// A positional strategy.
///
/// This is the special case of a strategy with trivial memory.
pub trait PositionalStrategy: Strategy<Memory = ()> {}

impl<T> PositionalStrategy for T where T: Strategy<Memory = ()> {}
