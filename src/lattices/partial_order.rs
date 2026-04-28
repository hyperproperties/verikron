/// A type whose [`PartialOrd`] defines a partial order.
///
/// Laws: reflexive, antisymmetric, and transitive.
///
/// [`PartialOrd::partial_cmp`] may return `None` for incomparable elements.
pub trait PartialOrder: PartialEq + PartialOrd {}

/// Any lawful [`PartialEq`] + [`PartialOrd`] type is a partial order.
impl<T: PartialEq + PartialOrd + ?Sized> PartialOrder for T {}
