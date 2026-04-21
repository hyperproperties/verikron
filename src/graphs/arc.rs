use crate::graphs::graph::Graph;

/// Directed incidence descriptor.
///
/// For ordinary graphs, this describes a directed edge from `source` to
/// `destination`.
///
/// For directed hypergraphs, `source` and `destination` typically contain the
/// tail and head vertex collections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arc<T> {
    /// Source side.
    pub source: T,
    /// Destination side.
    pub destination: T,
}

impl<T> Arc<T> {
    /// Creates a new directed incidence description.
    #[must_use]
    #[inline]
    pub const fn new(source: T, destination: T) -> Self {
        Self {
            source,
            destination,
        }
    }

    /// Returns the same arc with reversed direction.
    #[must_use]
    #[inline]
    pub fn reversed(self) -> Self {
        Self {
            source: self.destination,
            destination: self.source,
        }
    }

    /// Maps both sides through `f`.
    #[must_use]
    #[inline]
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Arc<U> {
        Arc {
            source: f(self.source),
            destination: f(self.destination),
        }
    }
}

/// Directed graph constructible from owned arcs.
pub trait FromArcs: Sized + Graph {
    /// Creates a directed graph from owned arcs.
    fn from_arcs<I>(arcs: I) -> Self
    where
        I: IntoIterator<Item = Arc<Self::Vertex>>;
}

/// Directed hypergraph constructible from owned hyperarcs.
pub trait FromHyperarcs: Sized + Graph {
    /// Creates a directed hypergraph from owned tail/head collections.
    fn from_hyperarcs<I, S>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = Arc<S>>,
        S: IntoIterator<Item = Self::Vertex>;
}
