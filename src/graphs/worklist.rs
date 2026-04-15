use crate::graphs::visited::Visited;

/// Search or traversal that can be exhausted into its visited set.
pub trait Worklist<T, V>: Sized
where
    V: Visited<T>,
{
    /// Runs the worklist to completion and returns the visited set.
    fn worklist(self) -> V;
}
