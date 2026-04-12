use crate::graphs::edges::{Edges, ReadEdges, RemoveEdge};

/// Common vertex, edge, and label types used by labeled-edge traits.
pub trait LabeledEdges: Edges {
    /// Type carried by each edge.
    ///
    /// This is the transition label in automata terminology.
    type Label;
}

/// A graph that supports read-only access to edge labels.
pub trait ReadLabeledEdges: ReadEdges + LabeledEdges {
    /// Iterator over all labeled edges in the graph.
    ///
    /// Each item is a quadruple `(source, edge, label, destination)`.
    /// The label is yielded by shared reference to avoid forcing `Label: Copy`.
    type LabeledEdges<'a>: Iterator<
        Item = (Self::Vertex, Self::Edge, &'a Self::Label, Self::Vertex),
    >
    where
        Self: 'a,
        Self::Label: 'a;

    /// Returns an iterator over all labeled edges in the graph.
    fn labeled_edges(&self) -> Self::LabeledEdges<'_>;

    /// Returns the label of an existing edge.
    ///
    /// Returns `None` if `edge` does not exist.
    fn label(&self, edge: Self::Edge) -> Option<&Self::Label>;
}

/// A graph that supports mutation of existing edge labels.
pub trait WriteEdgeLabel: LabeledEdges {
    /// Returns a mutable reference to the label of an existing edge.
    ///
    /// Returns `None` if `edge` does not exist.
    fn label_mut(&mut self, edge: Self::Edge) -> Option<&mut Self::Label>;

    /// Replaces the label of an existing edge.
    ///
    /// Returns the old label on success, or `None` if `edge` does not exist.
    fn set_label(&mut self, edge: Self::Edge, label: Self::Label) -> Option<Self::Label>;
}

/// A graph that supports insertion of labeled edges.
///
/// This is usually the most natural insertion operation for automata,
/// because transitions should not exist in an unlabeled intermediate state.
pub trait InsertLabeledEdge: LabeledEdges {
    /// Inserts a new labeled edge.
    ///
    /// The `endpoints` parameter is `(source, destination)` for directed graphs.
    ///
    /// On success, returns `Some(edge)` identifying the inserted edge.
    /// If the edge cannot be inserted, returns `None`.
    fn insert_labeled_edge(
        &mut self,
        endpoints: (Self::Vertex, Self::Vertex),
        label: Self::Label,
    ) -> Option<Self::Edge>;
}

/// Convenience alias for a labeled graph that supports both querying and mutation.
pub trait LabeledEdgesMut:
    LabeledEdges + ReadEdges + ReadLabeledEdges + InsertLabeledEdge + RemoveEdge + WriteEdgeLabel
{
}

impl<T> LabeledEdgesMut for T where
    T: LabeledEdges
        + ReadEdges
        + ReadLabeledEdges
        + InsertLabeledEdge
        + RemoveEdge
        + WriteEdgeLabel
{
}
