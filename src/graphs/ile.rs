use crate::graphs::{
    graph::{EdgeType, Edges, InsertEdge, RemoveEdge, VertexType},
    labeled::{InsertLabeledEdge, LabeledEdgeType, LabeledEdges, WriteEdgeLabel},
};

/// Edge store with labels indexed directly by `usize` edge ids.
///
/// The label for edge `e` is stored in `labels[e]`, so label lookup is a fast
/// vector access rather than a map lookup.
///
/// This is intended for edge stores whose edge ids are dense `usize` indices,
/// and whose removal behavior stays consistent with `Vec::remove`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedLabeledEdges<E, L> {
    edges: E,
    labels: Vec<Option<L>>,
}

impl<E, L> IndexedLabeledEdges<E, L>
where
    E: EdgeType<Edge = usize>,
{
    /// Wraps `edges` with no labels.
    #[must_use]
    #[inline]
    pub fn new(edges: E) -> Self {
        Self {
            edges,
            labels: Vec::new(),
        }
    }

    /// Wraps `edges` with an existing label vector.
    ///
    /// `labels[e]` is the label of edge `e`. Missing slots or `None` mean that
    /// the edge is unlabeled.
    #[must_use]
    #[inline]
    pub fn with_labels(edges: E, labels: Vec<Option<L>>) -> Self {
        Self { edges, labels }
    }

    /// Returns the wrapped edge store.
    #[must_use]
    #[inline]
    pub fn edges(&self) -> &E {
        &self.edges
    }

    /// Returns the wrapped edge store mutably.
    #[must_use]
    #[inline]
    pub fn edges_mut(&mut self) -> &mut E {
        &mut self.edges
    }

    /// Returns the label slots.
    #[must_use]
    #[inline]
    pub fn labels(&self) -> &[Option<L>] {
        &self.labels
    }

    /// Returns the label slots mutably.
    #[must_use]
    #[inline]
    pub fn labels_mut(&mut self) -> &mut [Option<L>] {
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
    pub fn into_parts(self) -> (E, Vec<Option<L>>) {
        (self.edges, self.labels)
    }

    /// Returns whether `edge` has a label.
    #[must_use]
    #[inline]
    pub fn is_labeled(&self, edge: usize) -> bool {
        self.labels.get(edge).is_some_and(Option::is_some)
    }

    /// Returns the number of currently labeled edges.
    #[must_use]
    #[inline]
    pub fn label_count(&self) -> usize {
        self.labels.iter().filter(|label| label.is_some()).count()
    }

    /// Removes the label of `edge` and returns it if present.
    #[inline]
    pub fn clear_label(&mut self, edge: usize) -> Option<L> {
        self.labels.get_mut(edge).and_then(Option::take)
    }

    #[inline]
    fn ensure_slot(&mut self, edge: usize) {
        if edge >= self.labels.len() {
            self.labels.resize_with(edge + 1, || None);
        }
    }
}

impl<E, L> Default for IndexedLabeledEdges<E, L>
where
    E: Default + EdgeType<Edge = usize>,
{
    #[inline]
    fn default() -> Self {
        Self::new(E::default())
    }
}

impl<E, L> From<E> for IndexedLabeledEdges<E, L>
where
    E: EdgeType<Edge = usize>,
{
    #[inline]
    fn from(edges: E) -> Self {
        Self::new(edges)
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

/// Iterator over the labeled subset of an indexed edge store.
#[derive(Clone, Debug)]
pub struct IndexedLabeledEdgeIter<'a, I, L> {
    edges: I,
    labels: &'a [Option<L>],
}

impl<'a, I, V, L> Iterator for IndexedLabeledEdgeIter<'a, I, L>
where
    I: Iterator<Item = (V, usize, V)>,
{
    type Item = (V, usize, &'a L, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (from, edge, to) = self.edges.next()?;
            let Some(label) = self.labels.get(edge).and_then(Option::as_ref) else {
                continue;
            };
            return Some((from, edge, label, to));
        }
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
        self.labels.get(edge).and_then(Option::as_ref)
    }
}

impl<E, L> WriteEdgeLabel for IndexedLabeledEdges<E, L>
where
    E: EdgeType<Edge = usize>,
{
    #[inline]
    fn label_mut(&mut self, edge: Self::Edge) -> Option<&mut Self::Label> {
        self.labels.get_mut(edge).and_then(Option::as_mut)
    }

    #[inline]
    fn set_label(&mut self, edge: Self::Edge, label: Self::Label) -> Option<Self::Label> {
        self.ensure_slot(edge);
        self.labels[edge].replace(label)
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
        self.ensure_slot(edge);
        self.labels[edge] = Some(label);
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

        if edge < self.labels.len() {
            self.labels.remove(edge);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    use crate::graphs::{
        graph::{InsertEdge, InsertVertex},
        mcsr::MCSR,
    };

    fn sample_store() -> IndexedLabeledEdges<MCSR, char> {
        let mut edges = MCSR::new();

        for _ in 0..4 {
            edges.insert_vertex().unwrap();
        }

        edges.insert_edge(0, 1).unwrap();
        edges.insert_edge(1, 2).unwrap();
        edges.insert_edge(2, 3).unwrap();

        IndexedLabeledEdges::new(edges)
    }

    #[test]
    fn set_label_grows_slots_and_replaces_old_value() {
        let mut store = sample_store();

        assert_eq!(store.set_label(2, 'c'), None);
        assert_eq!(store.labels(), &[None, None, Some('c')]);
        assert_eq!(store.label(2), Some(&'c'));

        assert_eq!(store.set_label(2, 'z'), Some('c'));
        assert_eq!(store.label(2), Some(&'z'));
    }

    #[test]
    fn label_mut_and_clear_label_work() {
        let mut store = sample_store();
        store.set_label(1, 'b');

        *store.label_mut(1).unwrap() = 'x';
        assert_eq!(store.label(1), Some(&'x'));

        assert_eq!(store.clear_label(1), Some('x'));
        assert_eq!(store.label(1), None);
        assert!(!store.is_labeled(1));
    }

    #[test]
    fn labeled_edges_skip_unlabeled_edges_without_stopping() {
        let mut store = sample_store();
        store.set_label(2, 'c');

        let labeled = store
            .labeled_edges()
            .map(|(from, edge, label, to)| (from, edge, *label, to))
            .collect::<Vec<_>>();

        assert_eq!(labeled, vec![(2, 2, 'c', 3)]);
    }

    #[test]
    fn insert_labeled_edge_inserts_both_edge_and_label() {
        let mut store = IndexedLabeledEdges::<MCSR, char>::new(MCSR::new());

        for _ in 0..2 {
            store.edges_mut().insert_vertex().unwrap();
        }

        let edge = store.insert_labeled_edge(0, 'x', 1).unwrap();

        assert_eq!(edge, 0);
        assert_eq!(store.label(0), Some(&'x'));

        let labeled = store
            .labeled_edges()
            .map(|(from, edge, label, to)| (from, edge, *label, to))
            .collect::<Vec<_>>();

        assert_eq!(labeled, vec![(0, 0, 'x', 1)]);
    }

    #[test]
    fn remove_edge_keeps_labels_aligned_with_shifted_indices() {
        let mut store = sample_store();
        store.set_label(0, 'a');
        store.set_label(1, 'b');
        store.set_label(2, 'c');

        assert!(store.remove_edge(1));

        assert_eq!(store.labels(), &[Some('a'), Some('c')]);
        assert_eq!(store.label(0), Some(&'a'));
        assert_eq!(store.label(1), Some(&'c'));

        let labeled = store
            .labeled_edges()
            .map(|(from, edge, label, to)| (from, edge, *label, to))
            .collect::<Vec<_>>();

        assert_eq!(labeled, vec![(0, 0, 'a', 1), (2, 1, 'c', 3)]);
    }

    fn arb_store() -> impl Strategy<Value = IndexedLabeledEdges<MCSR, u8>> {
        prop::collection::vec((0u8..5, 0u8..5, prop::option::of(0u8..10)), 0..12).prop_map(
            |entries| {
                let vertex_count = entries
                    .iter()
                    .flat_map(|(from, to, _)| [*from as usize, *to as usize])
                    .max()
                    .map_or(0, |m| m + 1);

                let mut edges = MCSR::new();
                for _ in 0..vertex_count {
                    edges.insert_vertex().unwrap();
                }

                let mut labels = Vec::with_capacity(entries.len());
                for (from, to, label) in entries {
                    edges.insert_edge(from as usize, to as usize).unwrap();
                    labels.push(label);
                }

                IndexedLabeledEdges::with_labels(edges, labels)
            },
        )
    }

    proptest! {
        #[test]
        fn prop_label_lookup_matches_label_slots(store in arb_store()) {
            for edge in 0..store.labels().len() {
                prop_assert_eq!(store.label(edge).copied(), store.labels()[edge]);
                prop_assert_eq!(store.is_labeled(edge), store.labels()[edge].is_some());
            }
        }

    #[test]
    fn prop_labeled_edges_match_filtering_labels(store in arb_store()) {
        let got = store
            .labeled_edges()
            .map(|(from, edge, label, to)| (from, edge, *label, to))
            .collect::<Vec<_>>();

        let expected = Edges::edges(&store)
            .filter_map(|(from, edge, to)| {
                store.labels()[edge].map(|label| (from, edge, label, to))
            })
            .collect::<Vec<_>>();

        prop_assert_eq!(got, expected);
    }

        #[test]
        fn prop_remove_edge_removes_matching_label_slot(
            mut store in arb_store(),
            edge in 0usize..12,
        ) {
            let edge_count = store.labels().len();

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
