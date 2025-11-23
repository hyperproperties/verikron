/// A marker trait for types that form a *partial order* under their
/// [`PartialOrd`] implementation.
///
/// In mathematics, a partial order `⊑` on a set `X` satisfies:
///
/// * **Reflexivity**:    `x ⊑ x`
/// * **Antisymmetry**:   if `x ⊑ y` and `y ⊑ x` then `x == y`
/// * **Transitivity**:   if `x ⊑ y` and `y ⊑ z` then `x ⊑ z`
///
/// This trait does **not** add any methods. It simply documents and
/// constrains the expectation that the existing [`PartialOrd`] instance
/// really is a partial order in this sense.
///
/// In particular, [`PartialOrd::partial_cmp`] is allowed to return
/// `None` for **incomparable** elements (e.g. in a non-total order),
/// which naturally models a partial order.
///
/// Most of the time you should not implement `PartialOrder` manually:
/// just derive or write a lawful [`PartialEq`] + [`PartialOrd`], and
/// rely on the blanket implementation.
///
/// This trait exists so that other lattice/poset traits (such as
/// [`JoinSemiLattice`](crate::lattices::lattice::JoinSemiLattice))
/// can require “poset-like” behavior without redefining the ordering
/// operations themselves.
pub trait PartialOrder: PartialEq + PartialOrd {}

/// Blanket implementation: any type that implements [`PartialEq`] and
/// [`PartialOrd`] automatically implements [`PartialOrder`].
///
/// This lets you use ordinary Rust ordering (`<`, `<=`, etc.) together
/// with the lattice traits in this crate, as long as your `PartialOrd`
/// really does define a partial order.
impl<T: PartialEq + PartialOrd + ?Sized> PartialOrder for T {}
