use std::hash::Hash;

/// A play as a sequence of positions.
pub trait Play {
    /// Position type occurring in the play.
    type Position: Eq + Hash + Copy;

    /// Iterator over the positions of the play.
    type Sequence<'a>: Iterator<Item = Self::Position>
    where
        Self: 'a;

    /// Iterator over the distinct positions visited by the play.
    type Visited<'a>: Iterator<Item = Self::Position>
    where
        Self: 'a;

    /// Returns the play sequence.
    fn sequence(&self) -> Self::Sequence<'_>;

    /// Returns the distinct positions visited by the play.
    fn visited(&self) -> Self::Visited<'_>;
}

/// An infinite play.
///
/// Provides access to the positions visited infinitely often.
pub trait InfinitePlay: Play {
    /// Iterator over positions visited infinitely often.
    type InfinitelyOften<'a>: Iterator<Item = Self::Position>
    where
        Self: 'a;

    /// Returns the positions visited infinitely often.
    fn infinitely_often(&self) -> Self::InfinitelyOften<'_>;
}

/// A finite play.
///
/// Provides access to the positions visited only finitely often.
pub trait FinitePlay: Play {
    /// Iterator over positions visited finitely often.
    type FinitelyOften<'a>: Iterator<Item = Self::Position>
    where
        Self: 'a;

    /// Returns the positions visited finitely often.
    fn finitely_often(&self) -> Self::FinitelyOften<'_>;
}
