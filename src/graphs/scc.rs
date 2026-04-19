use crate::graphs::{properties::Properties, quotient::Quotient};

/// Strongly-connected-component decomposition.
///
/// Vertices are partitioned by the equivalence relation of mutual reachability.
/// Each class is a strongly connected component.
pub trait SCC: Quotient + Properties<Key = Self::Vertex, Property = Self::Class> {
    /// Returns a representative vertex of `class`.
    fn representative(&self, class: Self::Class) -> Self::Vertex;

    /// Returns whether `class` is recurrent, i.e. can sustain an infinite path.
    ///
    /// For finite directed graphs, this is equivalent to saying that the
    /// component contains a directed cycle.
    fn is_recurrent(&self, class: Self::Class) -> bool;

    /// Returns whether `class` is trivial.
    ///
    /// A trivial class contains exactly one vertex and no self-loop.
    fn is_trivial(&self, class: Self::Class) -> bool {
        !self.is_recurrent(class)
    }

    /// Returns whether `u` and `v` lie in the same strongly connected component.
    fn are_strongly_connected(&self, u: Self::Vertex, v: Self::Vertex) -> bool {
        self.class(u) == self.class(v)
    }
}
