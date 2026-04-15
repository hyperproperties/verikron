use std::ops::Range;

use crate::graphs::graph::{
    Directed, EdgeType, Edges, Endpoints, FiniteDirected, FiniteEdges, FiniteVertices,
    FromEndpoints, Graph, InsertEdge, InsertVertex, RemoveEdge, RemoveVertex, VertexType, Vertices,
};

/// Mutable CSR-style directed multigraph.
///
/// Vertices are `0..vertex_count()`. Outgoing edges of each vertex occupy a
/// contiguous range in `indices`, described by `offsets`.
///
/// This representation favors fast queries. Mutation updates the CSR layout
/// eagerly, so edge ids are not stable across insertions and removals.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MCSR {
    /// CSR row offsets of length `vertex_count() + 1`.
    offsets: Vec<usize>,

    /// Concatenated edge targets.
    indices: Vec<usize>,

    /// Source vertex of each edge.
    ///
    /// This duplicates row information to make `source(edge)` and mutable
    /// operations faster.
    sources: Vec<usize>,
}

impl MCSR {
    /// Creates an empty graph.
    #[inline]
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            indices: Vec::new(),
            sources: Vec::new(),
        }
    }

    #[inline]
    fn row_range(&self, vertex: usize) -> Option<(usize, usize)> {
        if vertex >= self.vertex_count() {
            return None;
        }

        let start = self.offsets[vertex];
        let end = self.offsets[vertex + 1];

        debug_assert!(start <= end);
        debug_assert!(end <= self.indices.len());

        Some((start, end))
    }
}

impl FromEndpoints for MCSR {
    /// Builds a graph from owned directed edges.
    fn from_endpoints<I>(edges: I) -> Self
    where
        I: IntoIterator<Item = Endpoints<Self::Vertex>>,
    {
        let mut edges: Vec<_> = edges.into_iter().collect();

        if edges.is_empty() {
            return Self::new();
        }

        let vertex_count = edges
            .iter()
            .map(|edge| edge.from.max(edge.to))
            .max()
            .unwrap()
            + 1;

        edges.sort_unstable_by_key(|edge| edge.from);

        let mut offsets = vec![0; vertex_count + 1];
        let mut indices = Vec::with_capacity(edges.len());
        let mut sources = Vec::with_capacity(edges.len());

        for edge in edges {
            offsets[edge.from + 1] += 1;
            indices.push(edge.to);
            sources.push(edge.from);
        }

        for vertex in 1..=vertex_count {
            offsets[vertex] += offsets[vertex - 1];
        }

        Self {
            offsets,
            indices,
            sources,
        }
    }
}

impl VertexType for MCSR {
    type Vertex = usize;
}

impl Vertices for MCSR {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Returns all vertices.
    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count()
    }
}

impl FiniteVertices for MCSR {
    /// Returns the number of vertices.
    #[inline]
    fn vertex_count(&self) -> usize {
        self.offsets.len() - 1
    }
}

impl InsertVertex for MCSR {
    /// Appends a new isolated vertex.
    #[inline]
    fn insert_vertex(&mut self) -> Option<Self::Vertex> {
        let vertex = self.vertex_count();
        let last = *self.offsets.last().unwrap();
        self.offsets.push(last);

        Some(vertex)
    }
}

impl RemoveVertex for MCSR {
    /// Removes `vertex` and all incident edges.
    ///
    /// Remaining vertex ids greater than `vertex` are shifted down by one.
    fn remove_vertex(&mut self, vertex: Self::Vertex) -> bool {
        if vertex >= self.vertex_count() {
            return false;
        }

        let old_vertex_count = self.vertex_count();
        let old_edge_count = self.edge_count();

        if old_vertex_count == 1 {
            self.offsets.clear();
            self.offsets.push(0);
            self.indices.clear();
            self.sources.clear();

            debug_assert_eq!(self.vertex_count(), 0);
            return true;
        }

        let new_vertex_count = old_vertex_count - 1;
        let mut offsets = vec![0usize; new_vertex_count + 1];
        let mut indices = Vec::with_capacity(old_edge_count);
        let mut sources = Vec::with_capacity(old_edge_count);

        for edge in 0..old_edge_count {
            let from = self.sources[edge];
            let to = self.indices[edge];

            if from == vertex || to == vertex {
                continue;
            }

            let new_from = if from > vertex { from - 1 } else { from };
            let new_to = if to > vertex { to - 1 } else { to };

            offsets[new_from + 1] += 1;
            indices.push(new_to);
            sources.push(new_from);
        }

        for v in 1..=new_vertex_count {
            offsets[v] += offsets[v - 1];
        }

        self.offsets = offsets;
        self.indices = indices;
        self.sources = sources;

        true
    }
}

impl EdgeType for MCSR {
    type Edge = usize;
}

impl Edges for MCSR {
    type Edges<'a>
        = MCSREdges<'a>
    where
        Self: 'a;

    /// Returns all edges.
    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        MCSREdges::all(self)
    }
}

impl FiniteEdges for MCSR {
    /// Returns the number of edges.
    #[inline]
    fn edge_count(&self) -> usize {
        self.indices.len()
    }
}

impl InsertEdge for MCSR {
    /// Inserts a directed edge.
    ///
    /// The edge is appended to the outgoing row of `from`.
    fn insert_edge(&mut self, from: Self::Vertex, to: Self::Vertex) -> Option<Self::Edge> {
        if from >= self.vertex_count() || to >= self.vertex_count() {
            return None;
        }

        let edge = self.offsets[from + 1];

        debug_assert!(edge <= self.indices.len());

        self.indices.insert(edge, to);
        self.sources.insert(edge, from);

        for offset in &mut self.offsets[from + 1..] {
            *offset += 1;
        }

        debug_assert_eq!(self.sources[edge], from);
        debug_assert_eq!(self.indices[edge], to);

        Some(edge)
    }
}

impl RemoveEdge for MCSR {
    /// Removes `edge`.
    fn remove_edge(&mut self, edge: Self::Edge) -> bool {
        if edge >= self.edge_count() {
            return false;
        }

        let from = self.sources[edge];

        self.indices.remove(edge);
        self.sources.remove(edge);

        for offset in &mut self.offsets[from + 1..] {
            *offset -= 1;
        }

        true
    }
}

impl Directed for MCSR {
    type Outgoing<'a>
        = MCSREdges<'a>
    where
        Self: 'a;

    type Ingoing<'a>
        = MCSREdges<'a>
    where
        Self: 'a;

    type Connections<'a>
        = MCSREdges<'a>
    where
        Self: 'a;

    /// Returns the source of `edge`.
    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(edge < self.edge_count(), "MCSR::source: edge out of bounds");
        let source = self.sources[edge];
        debug_assert!(source < self.vertex_count());
        source
    }

    /// Returns the target of `edge`.
    #[inline]
    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(edge < self.edge_count(), "MCSR::target: edge out of bounds");
        let target = self.indices[edge];
        debug_assert!(target < self.vertex_count());
        target
    }

    /// Returns all outgoing edges from `source`.
    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        debug_assert!(
            source < self.vertex_count(),
            "MCSR::outgoing: vertex out of bounds"
        );
        MCSREdges::outgoing(self, source)
    }

    /// Returns all incoming edges to `destination`.
    #[inline]
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        debug_assert!(
            destination < self.vertex_count(),
            "MCSR::ingoing: vertex out of bounds"
        );
        MCSREdges::ingoing(self, destination)
    }

    /// Returns all edges from `from` to `to`.
    #[inline]
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        debug_assert!(
            from < self.vertex_count(),
            "MCSR::connections: source out of bounds"
        );
        debug_assert!(
            to < self.vertex_count(),
            "MCSR::connections: target out of bounds"
        );
        MCSREdges::connections(self, from, to)
    }
}

impl FiniteDirected for MCSR {
    /// Returns the number of outgoing edges from `vertex`.
    #[inline]
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        debug_assert!(
            vertex < self.vertex_count(),
            "MCSR::outgoing_degree: vertex out of bounds"
        );

        let (start, end) = self.row_range(vertex).unwrap();
        end - start
    }

    /// Returns the number of incoming edges to `vertex`.
    #[inline]
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        debug_assert!(
            vertex < self.vertex_count(),
            "MCSR::ingoing_degree: vertex out of bounds"
        );

        self.indices
            .iter()
            .filter(|&&target| target == vertex)
            .count()
    }

    /// Returns the number of loop edges at `vertex`.
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        debug_assert!(
            vertex < self.vertex_count(),
            "MCSR::loop_degree: vertex out of bounds"
        );

        let (start, end) = self.row_range(vertex).unwrap();
        self.indices[start..end]
            .iter()
            .filter(|&&target| target == vertex)
            .count()
    }
}

impl Graph for MCSR {
    type Vertices = Self;
    type Edges = Self;

    /// Returns the edge store.
    #[inline]
    fn edge_store(&self) -> &Self::Edges {
        self
    }

    /// Returns the vertex store.
    #[inline]
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

/// Iterator over selected edges of an [`MCSR`].
pub struct MCSREdges<'a> {
    graph: &'a MCSR,
    kind: MCSREdgesKind,
}

enum MCSREdgesKind {
    All {
        edge: usize,
        end: usize,
    },
    Outgoing {
        source: usize,
        edge: usize,
        end: usize,
    },
    Ingoing {
        destination: usize,
        edge: usize,
        end: usize,
    },
    Connections {
        source: usize,
        destination: usize,
        edge: usize,
        end: usize,
    },
}

impl<'a> MCSREdges<'a> {
    #[inline]
    fn all(graph: &'a MCSR) -> Self {
        Self {
            graph,
            kind: MCSREdgesKind::All {
                edge: 0,
                end: graph.edge_count(),
            },
        }
    }

    #[inline]
    fn outgoing(graph: &'a MCSR, source: usize) -> Self {
        let (edge, end) = graph.row_range(source).unwrap();
        Self {
            graph,
            kind: MCSREdgesKind::Outgoing { source, edge, end },
        }
    }

    #[inline]
    fn ingoing(graph: &'a MCSR, destination: usize) -> Self {
        Self {
            graph,
            kind: MCSREdgesKind::Ingoing {
                destination,
                edge: 0,
                end: graph.edge_count(),
            },
        }
    }

    #[inline]
    fn connections(graph: &'a MCSR, source: usize, destination: usize) -> Self {
        let (edge, end) = graph.row_range(source).unwrap();
        Self {
            graph,
            kind: MCSREdgesKind::Connections {
                source,
                destination,
                edge,
                end,
            },
        }
    }
}

impl<'a> Iterator for MCSREdges<'a> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.kind {
            MCSREdgesKind::All { edge, end } => {
                if *edge >= *end {
                    return None;
                }

                let current = *edge;
                *edge += 1;
                Some((
                    self.graph.sources[current],
                    current,
                    self.graph.indices[current],
                ))
            }

            MCSREdgesKind::Outgoing { source, edge, end } => {
                if *edge >= *end {
                    return None;
                }

                let current = *edge;
                *edge += 1;
                Some((*source, current, self.graph.indices[current]))
            }

            MCSREdgesKind::Ingoing {
                destination,
                edge,
                end,
            } => {
                while *edge < *end {
                    let current = *edge;
                    *edge += 1;

                    if self.graph.indices[current] == *destination {
                        return Some((self.graph.sources[current], current, *destination));
                    }
                }
                None
            }

            MCSREdgesKind::Connections {
                source,
                destination,
                edge,
                end,
            } => {
                while *edge < *end {
                    let current = *edge;
                    *edge += 1;

                    if self.graph.indices[current] == *destination {
                        return Some((*source, current, *destination));
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        csr::CSR,
        graph::{Endpoints, FromEndpoints},
    };

    use proptest::prelude::*;
    use std::collections::HashMap;

    fn edge_triples<G>(graph: &G) -> Vec<(usize, usize, usize)>
    where
        G: Edges<Vertex = usize, Edge = usize>,
    {
        graph.edges().collect()
    }

    fn edge_pairs<G>(graph: &G) -> Vec<(usize, usize)>
    where
        G: Edges<Vertex = usize, Edge = usize>,
    {
        graph.edges().map(|(from, _, to)| (from, to)).collect()
    }

    fn sorted_edge_pairs<G>(graph: &G) -> Vec<(usize, usize)>
    where
        G: Edges<Vertex = usize, Edge = usize>,
    {
        let mut edges = edge_pairs(graph);
        edges.sort_unstable();
        edges
    }

    fn pair_counts(edges: &[(usize, usize)]) -> HashMap<(usize, usize), usize> {
        let mut counts = HashMap::new();
        for &(from, to) in edges {
            *counts.entry((from, to)).or_insert(0) += 1;
        }
        counts
    }

    fn assert_graph_matches_state(
        graph: &MCSR,
        expected_vertex_count: usize,
        expected_edges: &[(usize, usize)],
    ) {
        assert_eq!(graph.vertex_count(), expected_vertex_count);
        assert_eq!(graph.edge_count(), expected_edges.len());
        assert_eq!(
            graph.vertices().collect::<Vec<_>>(),
            (0..expected_vertex_count).collect::<Vec<_>>()
        );

        let expected_counts = pair_counts(expected_edges);
        let actual_counts = pair_counts(&edge_pairs(graph));

        assert_eq!(actual_counts, expected_counts);

        for vertex in 0..expected_vertex_count {
            let expected_outgoing = expected_edges
                .iter()
                .filter(|&&(from, _)| from == vertex)
                .count();
            let expected_ingoing = expected_edges
                .iter()
                .filter(|&&(_, to)| to == vertex)
                .count();
            let expected_loops = expected_edges
                .iter()
                .filter(|&&(from, to)| from == vertex && to == vertex)
                .count();

            assert_eq!(graph.outgoing_degree(vertex), expected_outgoing);
            assert_eq!(graph.ingoing_degree(vertex), expected_ingoing);
            assert_eq!(graph.loop_degree(vertex), expected_loops);

            let outgoing_targets: Vec<_> = graph.outgoing(vertex).map(|(_, _, to)| to).collect();
            assert_eq!(outgoing_targets.len(), expected_outgoing);

            let ingoing_sources: Vec<_> = graph.ingoing(vertex).map(|(from, _, _)| from).collect();
            assert_eq!(ingoing_sources.len(), expected_ingoing);
        }

        for &(from, to) in expected_edges {
            assert!(graph.is_connected(from, to));
        }
    }

    fn assert_graph_matches_edges(graph: &MCSR, edges: &[(usize, usize)]) {
        let expected_vertex_count = edges
            .iter()
            .map(|&(from, to)| from.max(to))
            .max()
            .map(|v| v + 1)
            .unwrap_or(0);

        assert_graph_matches_state(graph, expected_vertex_count, edges);
    }

    fn assert_mcsr_matches_csr(edges: &[(usize, usize)]) {
        let mcsr = MCSR::from_endpoints(
            edges
                .iter()
                .copied()
                .map(|(from, to)| Endpoints::new(from, to)),
        );
        let csr = CSR::from_endpoints(
            edges
                .iter()
                .copied()
                .map(|(from, to)| Endpoints::new(from, to)),
        );

        assert_eq!(mcsr.vertex_count(), csr.vertex_count());
        assert_eq!(mcsr.edge_count(), csr.edge_count());
        assert_eq!(sorted_edge_pairs(&mcsr), sorted_edge_pairs(&csr));

        for vertex in 0..mcsr.vertex_count() {
            assert_eq!(mcsr.outgoing_degree(vertex), csr.outgoing_degree(vertex));
            assert_eq!(mcsr.ingoing_degree(vertex), csr.ingoing_degree(vertex));
            assert_eq!(mcsr.loop_degree(vertex), csr.loop_degree(vertex));
        }
    }

    fn arbitrary_edges() -> impl Strategy<Value = Vec<(usize, usize)>> {
        prop::collection::vec((0usize..16, 0usize..16), 0..64)
    }

    #[test]
    fn new_is_empty() {
        let graph = MCSR::new();

        assert_eq!(graph.vertex_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.vertices().collect::<Vec<_>>().is_empty());
        assert!(graph.edges().collect::<Vec<_>>().is_empty());
        assert_eq!(graph.row_range(0), None);
    }

    #[test]
    fn from_endpoints_empty_is_empty() {
        let graph = MCSR::from_endpoints(std::iter::empty::<Endpoints<usize>>());

        assert_eq!(graph.vertex_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.edges().collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn from_endpoints_matches_input_multiset() {
        let edges = vec![(0, 1), (0, 2), (2, 1), (2, 1), (3, 3)];
        let graph = MCSR::from_endpoints(
            edges
                .iter()
                .copied()
                .map(|(from, to)| Endpoints::new(from, to)),
        );

        assert_graph_matches_edges(&graph, &edges);
    }

    #[test]
    fn from_endpoints_matches_csr() {
        let edges = vec![(0, 1), (0, 2), (2, 1), (2, 1), (3, 3)];
        assert_mcsr_matches_csr(&edges);
    }

    #[test]
    fn insert_vertex_appends_isolated_vertex() {
        let mut graph = MCSR::from_endpoints([Endpoints::new(0, 1), Endpoints::new(1, 2)]);

        let new_vertex = graph.insert_vertex();

        assert_eq!(new_vertex, Some(3));
        assert_eq!(graph.vertex_count(), 4);
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.outgoing_degree(3), 0);
        assert_eq!(graph.ingoing_degree(3), 0);
        assert_eq!(sorted_edge_pairs(&graph), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn insert_edge_adds_edge_when_endpoints_exist() {
        let mut graph = MCSR::from_endpoints([Endpoints::new(0, 1), Endpoints::new(1, 2)]);

        let edge = graph.insert_edge(2, 0);

        assert!(edge.is_some());
        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        assert_graph_matches_edges(&graph, &[(0, 1), (1, 2), (2, 0)]);
    }

    #[test]
    fn insert_edge_rejects_out_of_bounds_vertices() {
        let mut graph = MCSR::from_endpoints([Endpoints::new(0, 1)]);

        assert_eq!(graph.insert_edge(0, 2), None);
        assert_eq!(graph.insert_edge(2, 0), None);
        assert_eq!(graph.edge_count(), 1);
        assert_graph_matches_edges(&graph, &[(0, 1)]);
    }

    #[test]
    fn remove_edge_removes_requested_edge() {
        let mut graph = MCSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 2),
        ]);

        let removed = graph.remove_edge(1);

        assert!(removed);
        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert_graph_matches_edges(&graph, &[(0, 1), (1, 2)]);
    }

    #[test]
    fn remove_edge_rejects_out_of_bounds_edge() {
        let mut graph = MCSR::from_endpoints([Endpoints::new(0, 1)]);

        assert!(!graph.remove_edge(1));
        assert_eq!(graph.edge_count(), 1);
        assert_graph_matches_edges(&graph, &[(0, 1)]);
    }

    #[test]
    fn remove_vertex_removes_incident_edges_and_shifts_vertices() {
        let mut graph = MCSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
            Endpoints::new(3, 0),
            Endpoints::new(3, 3),
        ]);

        let removed = graph.remove_vertex(1);

        assert!(removed);
        assert_eq!(graph.vertex_count(), 3);

        assert_graph_matches_state(&graph, 3, &[(1, 2), (2, 0), (2, 2)]);
    }

    #[test]
    fn remove_vertex_rejects_out_of_bounds_vertex() {
        let mut graph = MCSR::from_endpoints([Endpoints::new(0, 1)]);

        assert!(!graph.remove_vertex(2));
        assert_eq!(graph.vertex_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_graph_matches_state(&graph, 2, &[(0, 1)]);
    }

    #[test]
    fn remove_only_vertex_yields_empty_graph() {
        let mut graph = MCSR::new();
        assert_eq!(graph.insert_vertex(), Some(0));

        assert!(graph.remove_vertex(0));
        assert_eq!(graph.vertex_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.vertices().collect::<Vec<_>>().is_empty());
        assert!(graph.edges().collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn source_target_and_edge_iterator_agree() {
        let graph = MCSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(2, 2),
        ]);

        for (from, edge, to) in graph.edges() {
            assert_eq!(graph.source(edge), from);
            assert_eq!(graph.target(edge), to);
            assert!(graph.has_edge(from, edge, to));
        }
    }

    #[test]
    fn connections_match_edge_multiplicity() {
        let graph = MCSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 1),
        ]);

        assert_eq!(graph.connections(0, 1).count(), 2);
        assert_eq!(graph.connections(0, 2).count(), 1);
        assert_eq!(graph.connections(1, 1).count(), 1);
        assert_eq!(graph.connections(2, 2).count(), 0);
    }

    proptest! {
        #[test]
        fn prop_from_endpoints_matches_input(edges in arbitrary_edges()) {
            let graph = MCSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            assert_graph_matches_edges(&graph, &edges);
        }

        #[test]
        fn prop_from_endpoints_matches_csr(edges in arbitrary_edges()) {
            assert_mcsr_matches_csr(&edges);
        }

        #[test]
        fn prop_insert_vertex_preserves_existing_edges(edges in arbitrary_edges()) {
            let mut graph = MCSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let old_vertex_count = graph.vertex_count();
            let old_edges = sorted_edge_pairs(&graph);

            let inserted = graph.insert_vertex();

            prop_assert_eq!(inserted, Some(old_vertex_count));
            prop_assert_eq!(graph.vertex_count(), old_vertex_count + 1);
            prop_assert_eq!(sorted_edge_pairs(&graph), old_edges);
            prop_assert_eq!(graph.outgoing_degree(old_vertex_count), 0);
            prop_assert_eq!(graph.ingoing_degree(old_vertex_count), 0);
        }

        #[test]
        fn prop_insert_edge_adds_one_occurrence(
            edges in arbitrary_edges(),
            from in 0usize..16,
            to in 0usize..16,
        ) {
            let mut graph = MCSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(u, v)| Endpoints::new(u, v)),
            );

            if from < graph.vertex_count() && to < graph.vertex_count() {
                let old_counts = pair_counts(&edge_pairs(&graph));
                let result = graph.insert_edge(from, to);

                prop_assert!(result.is_some());

                let mut expected = old_counts;
                *expected.entry((from, to)).or_insert(0) += 1;

                let actual = pair_counts(&edge_pairs(&graph));
                prop_assert_eq!(actual, expected);
            } else {
                let old_edges = sorted_edge_pairs(&graph);
                prop_assert_eq!(graph.insert_edge(from, to), None);
                prop_assert_eq!(sorted_edge_pairs(&graph), old_edges);
            }
        }

        #[test]
        fn prop_remove_edge_decreases_edge_count_when_valid(
            edges in arbitrary_edges(),
            edge_hint in 0usize..80,
        ) {
            let mut graph = MCSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let old_edge_count = graph.edge_count();

            if old_edge_count == 0 {
                prop_assert!(!graph.remove_edge(0));
                prop_assert_eq!(graph.edge_count(), 0);
            } else {
                let edge = edge_hint % old_edge_count;
                let removed_pair = {
                    let triples = edge_triples(&graph);
                    let (from, _, to) = triples[edge];
                    (from, to)
                };

                let mut expected = pair_counts(&edge_pairs(&graph));
                let count = expected.get_mut(&removed_pair).unwrap();
                *count -= 1;
                if *count == 0 {
                    expected.remove(&removed_pair);
                }

                prop_assert!(graph.remove_edge(edge));
                prop_assert_eq!(graph.edge_count(), old_edge_count - 1);
                prop_assert_eq!(pair_counts(&edge_pairs(&graph)), expected);
            }
        }

        #[test]
        fn prop_remove_vertex_matches_reference_transformation(
            edges in arbitrary_edges(),
            vertex_hint in 0usize..20,
        ) {
            let mut graph = MCSR::from_endpoints(
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to)),
            );

            let old_vertex_count = graph.vertex_count();

            if old_vertex_count == 0 {
                prop_assert!(!graph.remove_vertex(0));
                prop_assert_eq!(graph.vertex_count(), 0);
            } else {
                let vertex = vertex_hint % old_vertex_count;

                let expected_edges: Vec<_> = edge_pairs(&graph)
                    .into_iter()
                    .filter_map(|(from, to)| {
                        if from == vertex || to == vertex {
                            None
                        } else {
                            Some((
                                if from > vertex { from - 1 } else { from },
                                if to > vertex { to - 1 } else { to },
                            ))
                        }
                    })
                    .collect();

                let expected_vertex_count = old_vertex_count - 1;

                prop_assert!(graph.remove_vertex(vertex));
                assert_graph_matches_state(&graph, expected_vertex_count, &expected_edges);
            }
        }
    }
}
