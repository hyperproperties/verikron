use crate::lattices::partial_order::PartialOrder;

/// A join-semilattice: every pair of elements has a least upper bound (join).
///
/// # Laws
///
/// For all `a, b, c`:
///
/// * **Associativity**: `a.join(&b).join(&c) == a.join(&b.join(&c))`
/// * **Commutativity**: `a.join(&b) == b.join(&a)`
/// * **Idempotence**:   `a.join(&a) == a`
///
/// And in terms of the partial order `⊑` given by [`PartialOrder`]:
///
/// * `a ⊑ a.join(&b)` and `b ⊑ a.join(&b)`
/// * If `a ⊑ c` and `b ⊑ c` then `a.join(&b) ⊑ c`
///
/// Implementors are responsible for ensuring these laws hold.
pub trait JoinSemiLattice: PartialOrder + Sized {
    /// Returns the least upper bound (join) of `self` and `other`.
    ///
    /// Intuitively, this is the "merge" or "union" of two elements.
    fn join(&self, other: &Self) -> Self;
}

/// A meet-semilattice: every pair of elements has a greatest lower bound (meet).
///
/// # Laws
///
/// For all `a, b, c`:
///
/// * **Associativity**: `a.meet(&b).meet(&c) == a.meet(&b.meet(&c))`
/// * **Commutativity**: `a.meet(&b) == b.meet(&a)`
/// * **Idempotence**:   `a.meet(&a) == a`
///
/// And in terms of the partial order `⊑` given by [`PartialOrder`]:
///
/// * `a.meet(&b) ⊑ a` and `a.meet(&b) ⊑ b`
/// * If `c ⊑ a` and `c ⊑ b` then `c ⊑ a.meet(&b)`
///
/// Implementors are responsible for ensuring these laws hold.
pub trait MeetSemiLattice: PartialOrder + Sized {
    /// Returns the greatest lower bound (meet) of `self` and `other`.
    ///
    /// Intuitively, this is the "intersection" or "common part" of two elements.
    fn meet(&self, other: &Self) -> Self;
}

/// A lattice: a type that is both a join-semilattice and a meet-semilattice.
///
/// Every pair of elements has both a least upper bound (`join`) and a greatest
/// lower bound (`meet`).
///
/// This trait is automatically implemented for any type that implements
/// [`JoinSemiLattice`] and [`MeetSemiLattice`].
pub trait Lattice: JoinSemiLattice + MeetSemiLattice {}

impl<T: JoinSemiLattice + MeetSemiLattice> Lattice for T {}

/// A bounded lattice: a lattice with both a bottom and a top element.
///
/// * `bottom()` (⊥) is the least element: for all `x`, `bottom() ⊑ x`.
/// * `top()`    (⊤) is the greatest element: for all `x`, `x ⊑ top()`.
///
/// # Laws
///
/// For all `x`:
///
/// * `bottom().leq(&x)`
/// * `x.leq(&top())`
pub trait BoundedLattice: Lattice {
    /// Returns the least element (⊥) of the lattice.
    fn bottom() -> Self;

    /// Returns the greatest element (⊤) of the lattice.
    fn top() -> Self;
}

/// A complete lattice: a bounded lattice that supports finite joins and meets
/// over arbitrary collections of elements.
///
/// This trait provides *finite* joins and meets (`join_all`, `meet_all`) over
/// an [`IntoIterator`] of elements. (For truly arbitrary / infinite joins and
/// meets you would usually reason mathematically rather than encode them as
/// Rust functions.)
///
/// # Laws
///
/// For any finite family `{x_i}`:
///
/// * `join_all` returns their least upper bound:
///   * each `x_i ⊑ join_all({x_i})`
///   * if each `x_i ⊑ y` then `join_all({x_i}) ⊑ y`
///
/// * `meet_all` returns their greatest lower bound:
///   * `meet_all({x_i}) ⊑ x_i` for each `i`
///   * if `y ⊑ x_i` for all `i` then `y ⊑ meet_all({x_i})`
pub trait CompleteLattice: BoundedLattice {
    /// Returns the finite join (least upper bound) of all elements in `iterator`.
    ///
    /// For an empty iterator, a common convention is to return [`Self::bottom`],
    /// but the exact behavior should be documented by each implementation.
    fn join_all<I: IntoIterator<Item = Self>>(iterator: I) -> Self;

    /// Returns the finite meet (greatest lower bound) of all elements in `iterator`.
    ///
    /// For an empty iterator, a common convention is to return [`Self::top`],
    /// but the exact behavior should be documented by each implementation.
    fn meet_all<I: IntoIterator<Item = Self>>(iterator: I) -> Self;
}
