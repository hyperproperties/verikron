/// Visited-set abstraction for search algorithms.
///
/// Implementations store which values have already been seen. The trait is
/// designed so search code can both test membership and insert in one step.
pub trait Visited<V>: Default {
    /// Marks `value` as visited.
    ///
    /// Returns `true` if `value` was not visited before, and `false`
    /// otherwise.
    fn visit(&mut self, value: V) -> bool;

    /// Returns whether `value` has already been visited.
    fn is_visited(&self, value: &V) -> bool;
}
