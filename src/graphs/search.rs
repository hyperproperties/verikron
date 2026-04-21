use crate::graphs::visited::Visited;

pub type StateOf<S> = <S as Search>::State;
pub type VisitedOf<S> = <S as VisitedSearch>::Visited;

/// Iterator-based search.
///
/// This is a domain marker over [`Iterator`].
///
/// The searched state may be a graph vertex, a hyperedge, an automaton state,
/// or any other search state.
pub trait Search: Iterator<Item = Self::State> {
    /// State yielded by the search.
    type State;
}

impl<I> Search for I
where
    I: Iterator,
{
    type State = I::Item;
}

/// A search that owns a visited structure and can be exhausted into it.
pub trait VisitedSearch: Search {
    /// The visited structure maintained by this search.
    type Visited: Visited<Self::State>;

    /// Consumes the search state and returns the visited structure.
    #[must_use]
    fn into_visited(self) -> Self::Visited
    where
        Self: Sized;

    /// Borrows the visited structure while the search is still running.
    #[must_use]
    fn visited(&self) -> &Self::Visited;
}
