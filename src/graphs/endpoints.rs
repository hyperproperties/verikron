use crate::graphs::graph::Graph;

/// Undirected endpoints of an edge.
///
/// `u` and `v` are the two endpoints, with no implied order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Endpoints<T> {
    /// First endpoint.
    pub u: T,
    /// Second endpoint.
    pub v: T,
}

impl<T> Endpoints<T> {
    /// Creates a new undirected incidence description.
    #[must_use]
    #[inline]
    pub const fn new(u: T, v: T) -> Self {
        Self { u, v }
    }

    /// Returns the same endpoints with swapped order.
    ///
    /// For undirected edges this represents the same abstract edge.
    #[must_use]
    #[inline]
    pub fn reversed(self) -> Self {
        Self {
            u: self.v,
            v: self.u,
        }
    }

    /// Maps both endpoints through `f`.
    #[must_use]
    #[inline]
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Endpoints<U> {
        Endpoints {
            u: f(self.u),
            v: f(self.v),
        }
    }
}

/// Undirected graph constructible from owned endpoint pairs.
pub trait FromEndpoints: Sized + Graph {
    /// Creates an undirected graph from owned endpoint pairs.
    fn from_endpoints<I>(edges: I) -> Self
    where
        I: IntoIterator<Item = Endpoints<Self::Vertex>>;
}
