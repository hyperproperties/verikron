use crate::graphs::search::VisitedSearch;

/// Extension trait for searches that can be exhausted into their visited set.
pub trait Worklist: VisitedSearch {
    /// Runs the search to completion and returns the visited set.
    #[must_use]
    fn worklist(self) -> Self::Visited
    where
        Self: Sized;
}

/// Opt-in marker for the default exhaustive worklist implementation.
///
/// Implement this for search types that should use the standard
/// exhaust-then-return-visited behavior.
pub trait ExhaustiveWorklist: VisitedSearch {}

impl<T> Worklist for T
where
    T: ExhaustiveWorklist,
{
    #[inline]
    fn worklist(mut self) -> Self::Visited
    where
        Self: Sized,
    {
        while self.next().is_some() {}
        self.into_visited()
    }
}
