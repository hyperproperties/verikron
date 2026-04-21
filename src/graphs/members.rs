use crate::graphs::graph::Graph;

/// Members of an undirected hyperedge.
///
/// `members` is the collection of vertices incident to the hyperedge.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Members<T> {
    /// Member vertices.
    pub members: T,
}

impl<T> Members<T> {
    /// Creates a new hyperedge membership description.
    #[must_use]
    #[inline]
    pub const fn new(members: T) -> Self {
        Self { members }
    }

    /// Maps the member collection through `f`.
    #[must_use]
    #[inline]
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Members<U> {
        Members {
            members: f(self.members),
        }
    }
}

/// Undirected hypergraph constructible from owned member collections.
pub trait FromMembers: Sized + Graph {
    /// Creates an undirected hypergraph from owned member collections.
    fn from_members<I, S>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = Members<S>>,
        S: IntoIterator<Item = Self::Vertex>;
}
