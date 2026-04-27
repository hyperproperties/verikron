use crate::graphs::structure::VertexType;

/// A discovery event produced by a search.
///
/// Each event reports a newly discovered vertex together with the parent from
/// which it was first discovered. Roots have no parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Discovery<V> {
    parent: Option<V>,
    vertex: V,
}

impl<V> Discovery<V> {
    /// Creates a root discovery.
    #[must_use]
    #[inline]
    pub fn root(vertex: V) -> Self {
        Self {
            parent: None,
            vertex,
        }
    }

    /// Creates a non-root discovery.
    #[must_use]
    #[inline]
    pub fn child(source: V, destination: V) -> Self {
        Self {
            parent: Some(source),
            vertex: destination,
        }
    }

    /// Returns the source, if any.
    #[must_use]
    #[inline]
    pub fn parent(&self) -> Option<V>
    where
        V: Copy,
    {
        self.parent
    }

    /// Returns the discovered vertex.
    #[must_use]
    #[inline]
    pub fn vertex(&self) -> V
    where
        V: Copy,
    {
        self.vertex
    }

    /// Returns whether this is a root discovery.
    #[must_use]
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }
}

/// A search that exposes discovery events.
pub trait Search: VertexType {
    fn discover(&mut self) -> Option<Discovery<Self::Vertex>>;

    /// Advances the search until `goal` is discovered.
    fn find(&mut self, goal: Self::Vertex) -> bool
    where
        Self::Vertex: Eq,
    {
        while let Some(current) = self.discover() {
            if current.vertex() == goal {
                return true;
            }
        }

        false
    }
}
