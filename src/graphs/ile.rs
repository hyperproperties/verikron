use std::mem;

use crate::graphs::{
    graph::{EdgeType, Edges, FiniteEdges, InsertEdge, RemoveEdge, VertexType},
    labeled::{InsertLabeledEdge, LabeledEdgeType, LabeledEdges, WriteEdgeLabel},
};

/// Edge store with labels indexed directly by `usize` edge ids.
///
/// The label for edge `e` is stored in `labels[e]`, so label lookup is a fast
/// vector access.
///
/// This wrapper is intended for edge stores whose edge ids are dense `usize`
/// indices and whose structural updates reindex edges consistently with
/// `Vec::insert` and `Vec::remove`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedLabeledEdges<E, L> {
    edges: E,
    labels: Vec<L>,
}

impl<E, L> IndexedLabeledEdges<E, L>
where
    E: FiniteEdges<Edge = usize>,
{
    /// Wraps an empty edge store.
    ///
    /// Panics if `edges` is not empty.
    #[must_use]
    #[inline]
    pub fn new(edges: E) -> Self {
        assert_eq!(
            edges.edge_count(),
            0,
            "IndexedLabeledEdges::new requires an empty edge store",
        );

        Self {
            edges,
            labels: Vec::new(),
        }
    }

    /// Wraps `edges` with one label per edge.
    ///
    /// `labels[e]` is the label of edge `e`.
    ///
    /// Panics if `labels.len() != edges.edge_count()`.
    #[must_use]
    #[inline]
    pub fn with_labels(edges: E, labels: Vec<L>) -> Self {
        assert_eq!(
            labels.len(),
            edges.edge_count(),
            "IndexedLabeledEdges requires exactly one label per edge",
        );

        Self { edges, labels }
    }
}

impl<E, L> IndexedLabeledEdges<E, L>
where
    E: EdgeType<Edge = usize>,
{
    /// Returns the wrapped edge store.
    #[must_use]
    #[inline]
    pub fn edge_store(&self) -> &E {
        &self.edges
    }

    /// Returns the labels in edge-id order.
    #[must_use]
    #[inline]
    pub fn labels(&self) -> &[L] {
        &self.labels
    }

    /// Returns the labels mutably in edge-id order.
    #[must_use]
    #[inline]
    pub fn labels_mut(&mut self) -> &mut [L] {
        &mut self.labels
    }

    /// Consumes `self` and returns the wrapped edge store.
    ///
    /// Stored labels are discarded.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> E {
        self.edges
    }

    /// Consumes `self` and returns `(edges, labels)`.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (E, Vec<L>) {
        (self.edges, self.labels)
    }

    /// Returns the number of labels.
    ///
    /// This equals the number of edges when the wrapper invariant holds.
    #[must_use]
    #[inline]
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }
}

impl<E, L> VertexType for IndexedLabeledEdges<E, L>
where
    E: VertexType + EdgeType<Edge = usize>,
{
    type Vertex = E::Vertex;
}

impl<E, L> EdgeType for IndexedLabeledEdges<E, L>
where
    E: EdgeType<Edge = usize>,
{
    type Edge = usize;
}

impl<E, L> Edges for IndexedLabeledEdges<E, L>
where
    E: Edges<Edge = usize>,
{
    type Edges<'a>
        = E::Edges<'a>
    where
        Self: 'a,
        L: 'a;

    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        self.edges.edges()
    }
}

impl<E, L> LabeledEdgeType for IndexedLabeledEdges<E, L>
where
    E: EdgeType<Edge = usize>,
{
    type Label = L;
}

/// Iterator over all labeled edges in an indexed edge store.
#[derive(Clone, Debug)]
pub struct IndexedLabeledEdgeIter<'a, I, L> {
    edges: I,
    labels: &'a [L],
}

impl<'a, I, V, L> Iterator for IndexedLabeledEdgeIter<'a, I, L>
where
    I: Iterator<Item = (V, usize, V)>,
{
    type Item = (V, usize, &'a L, V);

    fn next(&mut self) -> Option<Self::Item> {
        let (from, edge, to) = self.edges.next()?;
        let label = self
            .labels
            .get(edge)
            .expect("IndexedLabeledEdges invariant violated: missing label for edge");
        Some((from, edge, label, to))
    }
}

impl<E, L> LabeledEdges for IndexedLabeledEdges<E, L>
where
    E: Edges<Edge = usize>,
{
    type LabeledEdges<'a>
        = IndexedLabeledEdgeIter<'a, E::Edges<'a>, L>
    where
        Self: 'a,
        L: 'a;

    #[inline]
    fn labeled_edges(&self) -> Self::LabeledEdges<'_> {
        IndexedLabeledEdgeIter {
            edges: self.edges.edges(),
            labels: &self.labels,
        }
    }

    #[inline]
    fn label(&self, edge: Self::Edge) -> Option<&Self::Label> {
        self.labels.get(edge)
    }
}

impl<E, L> WriteEdgeLabel for IndexedLabeledEdges<E, L>
where
    E: EdgeType<Edge = usize>,
{
    #[inline]
    fn label_mut(&mut self, edge: Self::Edge) -> Option<&mut Self::Label> {
        self.labels.get_mut(edge)
    }

    #[inline]
    fn set_label(&mut self, edge: Self::Edge, label: Self::Label) -> Option<Self::Label> {
        self.labels
            .get_mut(edge)
            .map(|slot| mem::replace(slot, label))
    }
}

impl<E, L> InsertLabeledEdge for IndexedLabeledEdges<E, L>
where
    E: InsertEdge<Edge = usize>,
{
    #[inline]
    fn insert_labeled_edge(
        &mut self,
        from: Self::Vertex,
        label: Self::Label,
        to: Self::Vertex,
    ) -> Option<Self::Edge> {
        let edge = self.edges.insert_edge(from, to)?;

        assert!(
            edge <= self.labels.len(),
            "IndexedLabeledEdges requires inserted edges to stay within 0..=edge_count",
        );

        self.labels.insert(edge, label);
        Some(edge)
    }
}

impl<E, L> RemoveEdge for IndexedLabeledEdges<E, L>
where
    E: RemoveEdge<Edge = usize>,
{
    #[inline]
    fn remove_edge(&mut self, edge: Self::Edge) -> bool {
        if !self.edges.remove_edge(edge) {
            return false;
        }

        self.labels.remove(edge);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    use crate::graphs::{
        graph::{Edges, InsertVertex},
        mcsr::MCSR,
    };

    fn empty_store(vertex_count: usize) -> IndexedLabeledEdges<MCSR, char> {
        let mut edges = MCSR::new();
        for _ in 0..vertex_count {
            edges.insert_vertex().unwrap();
        }
        IndexedLabeledEdges::new(edges)
    }

    fn sample_store() -> IndexedLabeledEdges<MCSR, char> {
        let mut store = empty_store(4);
        store.insert_labeled_edge(0, 'a', 1).unwrap();
        store.insert_labeled_edge(1, 'b', 2).unwrap();
        store.insert_labeled_edge(2, 'c', 3).unwrap();
        store
    }

    #[test]
    fn label_lookup_and_iteration_follow_edge_indices() {
        let store = sample_store();

        assert_eq!(store.labels(), &['a', 'b', 'c']);
        assert_eq!(store.label(0), Some(&'a'));
        assert_eq!(store.label(1), Some(&'b'));
        assert_eq!(store.label(2), Some(&'c'));
        assert_eq!(store.label(3), None);

        let labeled = store
            .labeled_edges()
            .map(|(from, edge, label, to)| (from, edge, *label, to))
            .collect::<Vec<_>>();

        assert_eq!(
            labeled,
            vec![(0, 0, 'a', 1), (1, 1, 'b', 2), (2, 2, 'c', 3)]
        );
    }

    #[test]
    fn set_label_and_label_mut_update_existing_labels() {
        let mut store = sample_store();

        assert_eq!(store.set_label(1, 'x'), Some('b'));
        assert_eq!(store.label(1), Some(&'x'));

        *store.label_mut(2).unwrap() = 'z';
        assert_eq!(store.label(2), Some(&'z'));

        assert_eq!(store.set_label(9, 'q'), None);
        assert!(store.label_mut(9).is_none());
    }

    #[test]
    fn insert_labeled_edge_appends_edge_and_label() {
        let mut store = empty_store(2);

        let edge = store.insert_labeled_edge(0, 'x', 1).unwrap();

        assert_eq!(edge, 0);
        assert_eq!(store.labels(), &['x']);

        let labeled = store
            .labeled_edges()
            .map(|(from, edge, label, to)| (from, edge, *label, to))
            .collect::<Vec<_>>();

        assert_eq!(labeled, vec![(0, 0, 'x', 1)]);
    }

    #[test]
    fn remove_edge_removes_the_matching_label_slot() {
        let mut store = sample_store();

        assert!(store.remove_edge(1));

        assert_eq!(store.labels(), &['a', 'c']);
        assert_eq!(store.label(0), Some(&'a'));
        assert_eq!(store.label(1), Some(&'c'));
        assert_eq!(store.label(2), None);
    }

    #[test]
    #[should_panic(expected = "IndexedLabeledEdges::new requires an empty edge store")]
    fn new_panics_for_nonempty_edge_store() {
        let mut edges = MCSR::new();
        edges.insert_vertex().unwrap();
        edges.insert_vertex().unwrap();
        edges.insert_edge(0, 1).unwrap();

        let _ = IndexedLabeledEdges::<_, char>::new(edges);
    }

    #[test]
    #[should_panic(expected = "IndexedLabeledEdges requires exactly one label per edge")]
    fn with_labels_panics_for_wrong_label_count() {
        let mut edges = MCSR::new();
        for _ in 0..2 {
            edges.insert_vertex().unwrap();
        }
        edges.insert_edge(0, 1).unwrap();

        let _ = IndexedLabeledEdges::with_labels(edges, Vec::<char>::new());
    }

    fn arb_store() -> impl Strategy<Value = IndexedLabeledEdges<MCSR, u8>> {
        prop::collection::vec((0usize..5, 0usize..5, 0u8..10), 0..12).prop_map(|entries| {
            let vertex_count = entries
                .iter()
                .flat_map(|(from, to, _)| [*from, *to])
                .max()
                .map_or(0, |m| m + 1);

            let mut edges = MCSR::new();
            for _ in 0..vertex_count {
                edges.insert_vertex().unwrap();
            }

            let mut store = IndexedLabeledEdges::new(edges);
            for (from, to, label) in entries {
                store.insert_labeled_edge(from, label, to).unwrap();
            }

            store
        })
    }

    proptest! {
        #[test]
        fn prop_label_lookup_matches_label_vector(store in arb_store()) {
            for edge in 0..store.label_count() {
                prop_assert_eq!(store.label(edge).copied(), Some(store.labels()[edge]));
            }

            prop_assert_eq!(store.label(store.label_count()).copied(), None);
        }

        #[test]
        fn prop_labeled_edges_match_edges_plus_labels(store in arb_store()) {
            let got = store
                .labeled_edges()
                .map(|(from, edge, label, to)| (from, edge, *label, to))
                .collect::<Vec<_>>();

            let expected = Edges::edges(&store)
                .map(|(from, edge, to)| (from, edge, store.labels()[edge], to))
                .collect::<Vec<_>>();

            prop_assert_eq!(got, expected);
        }

        #[test]
        fn prop_remove_edge_removes_the_same_label_slot(
            mut store in arb_store(),
            edge in 0usize..12,
        ) {
            let edge_count = store.label_count();

            if edge >= edge_count {
                prop_assert!(!store.remove_edge(edge));
            } else {
                let mut expected_labels = store.labels().to_vec();
                expected_labels.remove(edge);

                prop_assert!(store.remove_edge(edge));
                prop_assert_eq!(store.labels(), expected_labels.as_slice());
            }
        }
    }
}
