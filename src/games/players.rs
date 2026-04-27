/// A player/owner type usable in arenas and games.
///
/// A player is only an identifier: it can be copied and compared.
/// Game-theoretic behavior belongs to arenas and algorithms, not to players.
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
