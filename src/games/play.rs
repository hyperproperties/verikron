use std::hash::Hash;

/// A play as an ordered sequence of positions.
pub trait Play {
    type Position: Eq + Hash + Copy;

    type Positions<'a>: Iterator<Item = Self::Position>
    where
        Self: 'a;

    fn positions(&self) -> Self::Positions<'_>;
}

/// A play whose distinct visited positions can be enumerated.
pub trait VisitedPlay: Play {
    type Visited<'a>: Iterator<Item = Self::Position>
    where
        Self: 'a;

    fn visited(&self) -> Self::Visited<'_>;
}

/// A finite play.
pub trait FinitePlay: VisitedPlay {
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn first(&self) -> Option<Self::Position>;

    fn last(&self) -> Option<Self::Position>;
}

/// An infinite play.
pub trait InfinitePlay: Play {}
