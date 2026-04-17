use crate::graphs::structure::VertexType;

/// Common vertex and quotient-class identifier types.
///
/// A quotient partitions vertices into equivalence classes.
/// Each vertex belongs to exactly one class.
pub trait QuotientType: VertexType {
    /// Type used to identify quotient classes.
    type Class: Eq + Copy;
}

/// Quotient structure over vertices.
///
/// This trait provides:
/// - global enumeration of quotient classes,
/// - the class containing a given vertex,
/// - and enumeration of the vertices in a class.
///
/// Suitable for finite or infinite quotients.
pub trait Quotient: QuotientType {
    /// Iterator over all quotient classes.
    type Classes<'a>: Iterator<Item = Self::Class>
    where
        Self: 'a;

    /// Iterator over the vertices in a quotient class.
    type Members<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    /// Returns an iterator over all quotient classes.
    fn classes(&self) -> Self::Classes<'_>;

    /// Returns the class containing `vertex`.
    fn class(&self, vertex: Self::Vertex) -> Self::Class;

    /// Returns an iterator over the vertices in `class`.
    fn members(&self, class: Self::Class) -> Self::Members<'_>;
}

/// Finite quotient structure.
pub trait FiniteQuotient: Quotient {
    /// Returns the number of quotient classes.
    fn class_count(&self) -> usize {
        self.classes().count()
    }

    /// Returns whether `class` exists.
    fn contains_class(&self, class: &Self::Class) -> bool {
        self.classes().any(|c| &c == class)
    }
}

impl<T> FiniteQuotient for T where T: Quotient {}

pub type ClassOf<Q> = <Q as QuotientType>::Class;
