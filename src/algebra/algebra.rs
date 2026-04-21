use crate::lattices::lattice::BoundedLattice;

/// Embeds values from an underlying bounded lattice into a syntax type.
pub trait Embed: Sized {
    type Value: BoundedLattice;

    #[must_use]
    fn embed(value: Self::Value) -> Self;
}

/// Injects generators into a syntax type.
pub trait Generate: Sized {
    type Symbol;

    #[must_use]
    fn generate(generator: Self::Symbol) -> Self;
}

/// Type-level label for additive structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Addition;

/// A type with additive syntax/structure.
///
/// `into_additive` returns the immediate additive children when `self` is an
/// additive node, and `Err(self)` otherwise.
pub trait Additive: Sized {
    #[must_use]
    fn sum(self, other: Self) -> Self;

    fn into_additive(self) -> Result<Vec<Self>, Self>;
}

/// Type-level label for multiplicative structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Multiplication;

/// A type with multiplicative syntax/structure.
///
/// `into_multiplicative` returns the immediate multiplicative children when
/// `self` is a multiplicative node, and `Err(self)` otherwise.
pub trait Multiplicative: Sized {
    #[must_use]
    fn multiply(self, other: Self) -> Self;

    fn into_multiplicative(self) -> Result<Vec<Self>, Self>;
}

/// Type-level label for complementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Complement;

/// A type with a unary complement / negation operation.
///
/// `into_complement` returns the immediate complemented subterm when `self` is
/// a complement node, and `Err(self)` otherwise.
pub trait Complemented: Sized {
    #[must_use]
    fn complement(self) -> Self;

    fn into_complement(self) -> Result<Self, Self>;
}

/// Type-level distributivity law.
///
/// `Distributor` distributes over `Distributed`.
///
/// Examples:
/// - `Distributive<Multiplication, Addition>`
/// - `Distributive<Addition, Multiplication>`
pub trait Distributive<Distributor, Distributed>: Sized {
    #[must_use]
    fn distribute(self, other: Self) -> Self;
}

/// Type-level De Morgan law.
///
/// Examples:
/// - `DeMorgan<Addition, Multiplication>` for `¬(x ∨ y) = ¬x ∧ ¬y`
/// - `DeMorgan<Multiplication, Addition>` for `¬(x ∧ y) = ¬x ∨ ¬y`
pub trait DeMorgan<Outer, Inner>: Sized {
    #[must_use]
    fn demorgan(self) -> Self;
}
