use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::graphs::{
    search::{Discovery, Search},
    structure::VertexType,
};

/// A search that records how vertices were first discovered.
pub trait PathSearch: Search {
    /// Returns the predecessor of `vertex` in the search tree.
    ///
    /// Returns `None` for a root or for an undiscovered vertex.
    fn predecessor(&self, vertex: &Self::Vertex) -> Option<Self::Vertex>;

    /// Returns whether `vertex` has been discovered.
    fn discovered(&self, vertex: &Self::Vertex) -> bool;

    /// Returns the predecessor chain starting at `from`.
    ///
    /// The returned vector is ordered from `from` backwards to a root.
    fn backtrack(&self, from: Self::Vertex) -> Option<Vec<Self::Vertex>>
    where
        Self::Vertex: Eq + Copy,
    {
        if !self.discovered(&from) {
            return None;
        }

        let mut path = vec![from];
        let mut current = from;

        while let Some(parent) = self.predecessor(&current) {
            path.push(parent);
            current = parent;
        }

        Some(path)
    }
}

/// A sparse path-search wrapper that records predecessors lazily from
/// discovery events.
#[derive(Debug, Clone)]
pub struct SparsePathSearch<T>
where
    T: Search,
    T::Vertex: Eq + Hash + Copy,
{
    search: T,
    predecessors: FxHashMap<T::Vertex, Option<T::Vertex>>,
}

impl<T> SparsePathSearch<T>
where
    T: Search,
    T::Vertex: Eq + Hash + Copy,
{
    #[must_use]
    #[inline]
    pub fn new(search: T) -> Self {
        Self {
            search,
            predecessors: FxHashMap::default(),
        }
    }

    #[must_use]
    #[inline]
    pub fn search(&self) -> &T {
        &self.search
    }

    #[must_use]
    #[inline]
    pub fn into_inner(self) -> T {
        self.search
    }

    #[must_use]
    #[inline]
    pub fn predecessors(&self) -> &FxHashMap<T::Vertex, Option<T::Vertex>> {
        &self.predecessors
    }
}

impl<T> VertexType for SparsePathSearch<T>
where
    T: Search,
    T::Vertex: Eq + Hash + Copy,
{
    type Vertex = T::Vertex;
}

impl<T> Search for SparsePathSearch<T>
where
    T: Search,
    T::Vertex: Eq + Hash + Copy,
{
    #[inline]
    fn discover(&mut self) -> Option<Discovery<Self::Vertex>> {
        let discovery = self.search.discover()?;
        self.predecessors
            .entry(discovery.vertex())
            .or_insert(discovery.parent());
        Some(discovery)
    }
}

impl<T> PathSearch for SparsePathSearch<T>
where
    T: Search,
    T::Vertex: Eq + Hash + Copy,
{
    #[inline]
    fn predecessor(&self, vertex: &Self::Vertex) -> Option<Self::Vertex> {
        self.predecessors.get(vertex).copied().flatten()
    }

    #[inline]
    fn discovered(&self, vertex: &Self::Vertex) -> bool {
        self.predecessors.contains_key(vertex)
    }
}
