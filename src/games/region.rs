use std::hash::Hash;

use crate::lattices::{
    bit_vector::BitVector,
    lattice::{Lattice, MembershipLattice},
    set::Set,
};

pub trait Region<P>: Lattice {
    /// Returns true iff `position` is included in the region.
    fn includes(&self, position: P) -> bool;

    /// Expands the region by `position`.
    ///
    /// Returns true iff the region changed.
    fn expand(&mut self, position: P) -> bool;

    /// Removes `position` from the region.
    ///
    /// Returns true iff the region changed.
    fn contract(&mut self, position: P) -> bool;
}

/// Dense region for finite arenas whose positions are `usize`.
pub type DenseRegion = BitVector;

/// Sparse region for generic position types.
pub type SparseRegion<P> = Set<P>;

impl Region<usize> for BitVector {
    fn includes(&self, position: usize) -> bool {
        self.contains(&position)
    }

    fn expand(&mut self, position: usize) -> bool {
        self.insert(position)
    }

    fn contract(&mut self, position: usize) -> bool {
        let member = self.includes(position);
        BitVector::set(self, position, false);
        member
    }
}

impl<P> Region<P> for Set<P>
where
    P: Eq + Hash + Clone,
{
    fn includes(&self, position: P) -> bool {
        Set::contains(self, &position)
    }

    fn expand(&mut self, position: P) -> bool {
        Set::insert(self, position)
    }

    fn contract(&mut self, position: P) -> bool {
        Set::remove(self, &position)
    }
}
