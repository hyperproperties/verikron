use crate::games::arena::Arena;

/// A strategy for an arena.
///
/// A strategy chooses successors for one player. Strategies may be memoryless
/// or may carry a finite memory state.
pub trait Strategy {
    /// Arena on which the strategy is defined.
    type Arena: Arena;

    /// Internal memory state.
    ///
    /// Use `()` for positional strategies.
    type Memory: Copy + Eq;

    /// Returns the player following this strategy.
    fn player(&self) -> <Self::Arena as Arena>::Player;

    /// Returns the initial memory state.
    fn initial_memory(&self) -> Self::Memory;

    /// Updates memory after the play moves to `position`.
    #[inline]
    fn update(
        &self,
        memory: Self::Memory,
        _position: <Self::Arena as Arena>::Position,
    ) -> Self::Memory {
        memory
    }

    /// Returns the chosen successor from `position` under `memory`, if defined.
    fn choice(
        &self,
        memory: Self::Memory,
        position: <Self::Arena as Arena>::Position,
    ) -> Option<<Self::Arena as Arena>::Position>;
}

/// A positional strategy.
///
/// This is the special case of a strategy with trivial memory.
pub trait PositionalStrategy: Strategy<Memory = ()> {
    /// Returns the chosen successor from `position`, if defined.
    fn positional_choice(
        &self,
        position: <Self::Arena as Arena>::Position,
    ) -> Option<<Self::Arena as Arena>::Position> {
        self.choice((), position)
    }
}

impl<T> PositionalStrategy for T where T: Strategy<Memory = ()> {}
