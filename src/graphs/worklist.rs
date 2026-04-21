use crate::graphs::search::VisitedSearch;

/// Extension trait for searches that can be exhausted into their visited set.
pub trait Worklist: VisitedSearch {
    /// Runs the search to completion and returns the visited set.
    #[must_use]
    fn worklist(mut self) -> Self::Visited
    where
        Self: Sized,
    {
        for _ in self.by_ref() {}
        self.into_visited()
    }
}

impl<T> Worklist for T where T: VisitedSearch {}
