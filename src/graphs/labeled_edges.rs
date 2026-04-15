use crate::graphs::graph::{EdgeType, Endpoints, RemoveEdge};

/// Edge type with labels.
pub trait LabeledEdgeType: EdgeType {
    /// Label carried by each edge.
    type Label;
}

/// Read access to labeled edges.
pub trait LabeledEdges: LabeledEdgeType {
    /// Iterator over labeled edges as `(from, edge, label, to)`.
    type LabeledEdges<'a>: Iterator<
        Item = (Self::Vertex, Self::Edge, &'a Self::Label, Self::Vertex),
    >
    where
        Self: 'a,
        Self::Label: 'a;

    /// Returns all labeled edges.
    fn labeled_edges(&self) -> Self::LabeledEdges<'_>;

    /// Returns the label of `edge`.
    fn label(&self, edge: Self::Edge) -> Option<&Self::Label>;
}

/// Write access to existing edge labels.
pub trait WriteEdgeLabel: LabeledEdgeType {
    /// Returns a mutable reference to the label of `edge`.
    fn label_mut(&mut self, edge: Self::Edge) -> Option<&mut Self::Label>;

    /// Replaces the label of `edge` and returns the old label.
    fn set_label(&mut self, edge: Self::Edge, label: Self::Label) -> Option<Self::Label>;
}

/// Insertion of labeled edges.
pub trait InsertLabeledEdge: LabeledEdgeType {
    /// Inserts a labeled edge and returns its id on success.
    fn insert_labeled_edge(
        &mut self,
        endpoints: Endpoints<Self::Vertex>,
        label: Self::Label,
    ) -> Option<Self::Edge>;
}

/// Mutable labeled edge store.
pub trait LabeledEdgesMut: LabeledEdges + InsertLabeledEdge + WriteEdgeLabel + RemoveEdge {}

impl<T> LabeledEdgesMut for T where T: LabeledEdges + InsertLabeledEdge + WriteEdgeLabel + RemoveEdge
{}
