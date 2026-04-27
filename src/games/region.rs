use std::hash::Hash;

use crate::{games::arena::Arena, lattices::lattice::Lattice};

/// A region of positions.
///
/// Regions are subsets of positions, typically ordered by inclusion and used
/// as fixpoint carriers in game solving.
///
/// This trait intentionally does not require access to an arena. Some region
/// representations may be useful as pure sets. Algorithms that need arena
/// access should require [`ArenaRegion`].
pub trait Region: Lattice {
    /// Player/controller type associated with this region.
    type Player: Copy + Eq;

    /// Position type contained in the region.
    type Position: Eq + Hash + Copy;

    /// Iterator over positions in the region.
    type Positions<'a>: Iterator<Item = Self::Position>
    where
        Self: 'a;

    /// Returns the owner/controller of the region.
    fn owner(&self) -> Self::Player;

    /// Returns an iterator over the positions in the region.
    fn positions(&self) -> Self::Positions<'_>;

    /// Returns whether `position` is in the region.
    fn contains(&self, position: &Self::Position) -> bool;

    /// Returns the number of positions in the region.
    ///
    /// This default implementation is O(|region|). Implementors may override
    /// it with a cached/cardinality-based version.
    #[must_use]
    #[inline]
    fn len(&self) -> usize {
        self.positions().count()
    }

    /// Returns whether the region is empty.
    #[must_use]
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A region attached to a concrete arena.
///
/// This is useful for algorithms such as attractor computation, where the
/// region itself stores the arena reference.
pub trait ArenaRegion<'arena>: Region {
    /// Arena associated with this region.
    type Arena: Arena<Player = Self::Player, Position = Self::Position> + 'arena;

    /// Returns the arena this region belongs to.
    fn arena(&self) -> &'arena Self::Arena;
}

/// A region supporting insertion.
pub trait RegionInsertion: Region {
    /// Inserts `position`.
    ///
    /// Returns whether the region changed.
    fn insert(&mut self, position: Self::Position) -> bool;
}

/// A region supporting removal.
pub trait RegionRemoval: Region {
    /// Removes `position`.
    ///
    /// Returns whether the region changed.
    fn remove(&mut self, position: &Self::Position) -> bool;

    /// Clears the region.
    fn clear(&mut self);
}
