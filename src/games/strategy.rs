use crate::games::{arena::Arena, region::Region};

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

    /// Creates an empty strategy for `player`.
    fn empty(player: <Self::Arena as Arena>::Player) -> Self;

    /// Returns the player following this strategy.
    fn player(&self) -> <Self::Arena as Arena>::Player;

    /// Returns the initial memory state.
    fn initial_memory(&self) -> Self::Memory;

    /// Updates memory after the play moves to `position`.
    ///
    /// Positional strategies can keep the default implementation.
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

/// Result of synthesizing a strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynthesisResult<S>
where
    S: Strategy,
{
    Winning { strategy: S },
    Losing,
}

impl<S> SynthesisResult<S>
where
    S: Strategy,
{
    pub fn winning(strategy: S) -> Self {
        Self::Winning { strategy }
    }

    pub const fn losing() -> Self {
        Self::Losing
    }

    pub const fn is_winning(&self) -> bool {
        matches!(self, Self::Winning { .. })
    }

    pub const fn is_losing(&self) -> bool {
        matches!(self, Self::Losing)
    }

    pub const fn strategy(&self) -> Option<&S> {
        match self {
            Self::Winning { strategy } => Some(strategy),
            Self::Losing => None,
        }
    }

    pub fn into_strategy(self) -> Option<S> {
        match self {
            Self::Winning { strategy } => Some(strategy),
            Self::Losing => None,
        }
    }
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

    /// Inserts or replaces the chosen successor for `position`.
    fn insert_choice(
        &mut self,
        position: <Self::Arena as Arena>::Position,
        successor: <Self::Arena as Arena>::Position,
    );
}

/// Result of synthesizing a positional strategy from an attractor computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionalSynthesisResult<R, S>
where
    S: PositionalStrategy,
    R: Region<<S::Arena as Arena>::Position>,
{
    Winning { attractor: R, strategy: S },
    Losing { attractor: R },
}

impl<R, S> PositionalSynthesisResult<R, S>
where
    S: PositionalStrategy,
    R: Region<<S::Arena as Arena>::Position>,
{
    pub fn winning(attractor: R, strategy: S) -> Self {
        Self::Winning {
            attractor,
            strategy,
        }
    }

    pub fn losing(attractor: R) -> Self {
        Self::Losing { attractor }
    }

    pub const fn is_winning(&self) -> bool {
        matches!(self, Self::Winning { .. })
    }

    pub const fn is_losing(&self) -> bool {
        matches!(self, Self::Losing { .. })
    }

    pub const fn attractor(&self) -> &R {
        match self {
            Self::Winning { attractor, .. } => attractor,
            Self::Losing { attractor } => attractor,
        }
    }

    pub const fn strategy(&self) -> Option<&S> {
        match self {
            Self::Winning { strategy, .. } => Some(strategy),
            Self::Losing { .. } => None,
        }
    }
}

impl<R, S> From<PositionalSynthesisResult<R, S>> for SynthesisResult<S>
where
    S: PositionalStrategy,
    R: Region<<S::Arena as Arena>::Position>,
{
    fn from(value: PositionalSynthesisResult<R, S>) -> Self {
        match value {
            PositionalSynthesisResult::Winning { strategy, .. } => Self::winning(strategy),
            PositionalSynthesisResult::Losing { .. } => Self::losing(),
        }
    }
}
