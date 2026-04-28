use crate::lattices::partial_order::PartialOrder;

/// A join-semilattice.
///
/// `join` returns the least upper bound of two elements.
///
/// Laws: associative, commutative, idempotent, and compatible with the
/// [`PartialOrder`] order.
pub trait JoinSemiLattice: PartialOrder + Sized {
    fn join(&self, other: &Self) -> Self;
}

/// A meet-semilattice.
///
/// `meet` returns the greatest lower bound of two elements.
///
/// Laws: associative, commutative, idempotent, and compatible with the
/// [`PartialOrder`] order.
pub trait MeetSemiLattice: PartialOrder + Sized {
    fn meet(&self, other: &Self) -> Self;
}

/// A lattice: both a join-semilattice and a meet-semilattice.
pub trait Lattice: JoinSemiLattice + MeetSemiLattice {}

impl<T: JoinSemiLattice + MeetSemiLattice> Lattice for T {}

/// A carrier with a least element.
pub trait Bottom {
    /// Returns the least element, ⊥.
    fn bottom() -> Self;
}

/// A carrier with a greatest element.
pub trait Top {
    /// Returns the greatest element, ⊤.
    fn top() -> Self;
}

/// A lattice with both bottom and top elements.
pub trait BoundedLattice: Lattice + Top + Bottom {}

/// A bounded lattice supporting finite joins and meets over collections.
///
/// Empty joins should return bottom.
/// Empty meets should return top.
pub trait CompleteLattice: BoundedLattice {
    /// Returns the finite join of all elements.
    fn join_all<I: IntoIterator<Item = Self>>(iterator: I) -> Self;

    /// Returns the finite meet of all elements.
    fn meet_all<I: IntoIterator<Item = Self>>(iterator: I) -> Self;
}

/// A lattice where join and meet distribute over each other.
pub trait DistributiveLattice: JoinSemiLattice + MeetSemiLattice {}

/// A join-semilattice with element membership.
pub trait MembershipLattice<V>: JoinSemiLattice {
    /// Inserts `value`.
    ///
    /// Returns true iff the lattice element changed.
    fn insert(&mut self, value: V) -> bool;

    /// Returns true iff `value` is contained.
    fn contains(&self, value: &V) -> bool;
}
