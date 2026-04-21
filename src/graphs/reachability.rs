use crate::graphs::search::Search;

/// Extension trait for searches that can test whether a goal is reachable.
pub trait Reachability: Search {
    /// Returns whether `goal` appears in the remaining search.
    fn reachable(&mut self, goal: Self::State) -> bool
    where
        Self::State: PartialEq;
}

/// Opt-in marker for the default linear reachability implementation.
///
/// Implement this for search types that should use the standard
/// iterator-based `reachable` behavior.
pub trait LinearReachability: Search {}

impl<T> Reachability for T
where
    T: LinearReachability,
{
    #[inline]
    fn reachable(&mut self, goal: Self::State) -> bool
    where
        Self::State: PartialEq,
    {
        self.any(|state| state == goal)
    }
}
