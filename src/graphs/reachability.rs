use crate::graphs::search::Search;

/// Extension trait for searches that can test whether a goal is reachable.
pub trait Reachability: Search {
    /// Returns whether `goal` appears in the remaining search.
    fn reachable(&mut self, goal: Self::Item) -> bool
    where
        Self::Item: PartialEq;
}

impl<T> Reachability for T
where
    T: Search,
{
    fn reachable(&mut self, goal: Self::Item) -> bool
    where
        Self::Item: PartialEq,
    {
        self.any(|successor| successor == goal)
    }
}
