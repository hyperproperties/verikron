use crate::graphs::visited::Visited;

/// Iterator-based graph search.
///
/// This is a domain marker over [`Iterator`].
pub trait Search: Iterator {}

impl<I> Search for I where I: Iterator {}

/// A search that owns a visited structure and can be exhausted into it.
pub trait VisitedSearch: Search {
    /// The visited structure maintained by this search.
    type Visited: Visited<Self::Item>;

    /// Consumes the search state and returns the visited structure.
    #[must_use]
    fn into_visited(self) -> Self::Visited
    where
        Self: Sized;

    /// Borrows the visited structure while the search is still running.
    #[must_use]
    fn visited(&self) -> &Self::Visited;
}
