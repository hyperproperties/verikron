use crate::lattices::partial_order::PartialOrder;

pub trait Semilattice: PartialOrder {}

/// A join-semilattice.
///
/// `join` returns the least upper bound of two elements.
///
/// Laws: associative, commutative, idempotent, and compatible with the
/// [`PartialOrder`] order.
pub trait JoinSemiLattice: Semilattice + Sized {
    fn join(&self, other: &Self) -> Self;
}

/// A meet-semilattice.
///
/// `meet` returns the greatest lower bound of two elements.
///
/// Laws: associative, commutative, idempotent, and compatible with the
/// [`PartialOrder`] order.
pub trait MeetSemiLattice: Semilattice + Sized {
    fn meet(&self, other: &Self) -> Self;
}

/// A lattice: both a join-semilattice and a meet-semilattice.
pub trait Lattice: JoinSemiLattice + MeetSemiLattice {}

impl<T: JoinSemiLattice + MeetSemiLattice> Lattice for T {}

/// A carrier with a least element.
pub trait Bottom {
    type Context;

    /// Returns the least element, ⊥, for the given context.
    fn bottom_with(context: &Self::Context) -> Self;

    /// Returns the least element using the default context.
    fn bottom() -> Self
    where
        Self: Sized,
        Self::Context: Default,
    {
        Self::bottom_with(&Self::Context::default())
    }
}

/// A carrier with a greatest element.
pub trait Top {
    type Context;

    /// Returns the greatest element, ⊤, for the given context.
    fn top_with(context: &Self::Context) -> Self;

    /// Returns the greatest element using the default context.
    fn top() -> Self
    where
        Self: Sized,
        Self::Context: Default,
    {
        Self::top_with(&Self::Context::default())
    }
}

/// A lattice with both bottom and top elements.
pub trait BoundedLattice: Lattice + Top + Bottom {}

/// A bounded lattice where every element has a complement.
pub trait ComplementedLattice: BoundedLattice {
    /// Returns a complement of `self`.
    ///
    /// Laws:
    /// - `self.join(&self.complement()) == Self::top()`
    /// - `self.meet(&self.complement()) == Self::bottom()`
    ///
    /// Complements are not necessarily unique in arbitrary complemented
    /// lattices.
    fn complement(self) -> Self;
}

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

/// A distributive complemented bounded lattice.
///
/// In Boolean algebras, complements are unique and set difference can be
/// defined as `x ∧ ¬y`.
pub trait BooleanLattice: ComplementedLattice + DistributiveLattice {}

/// A join-semilattice with element membership.
pub trait MembershipLattice<V>: JoinSemiLattice {
    /// Inserts `value`.
    ///
    /// Returns true iff the lattice element changed.
    fn insert(&mut self, value: V) -> bool;

    /// Returns true iff `value` is contained.
    fn contains(&self, value: &V) -> bool;
}
