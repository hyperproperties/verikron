use std::ops::Range;

use crate::{
    graphs::{
        graph::{Directed, Endpoints, FiniteDirected, FromEndpoints, Graph, IndexedDirected},
        structure::{
            EdgeType, Edges, FiniteEdges, FiniteVertices, InsertVertex, Structure, VertexType,
            Vertices,
        },
    },
    mem::boxed_slices::grow_boxed_slice,
};

/// Compressed sparse row representation of a directed multigraph.
///
/// Vertices are `0..vertex_count()`. For each vertex `u`, its outgoing edges
/// occupy the range `offsets[u]..offsets[u + 1]` in `indices`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CSR {
    /// Row offsets into `indices`.
    offsets: Box<[usize]>,

    /// Edge targets stored row by row.
    indices: Box<[usize]>,
}

impl Default for CSR {
    /// Returns the empty graph.
    fn default() -> Self {
        Self {
            offsets: Box::from([0]),
            indices: Box::new([]),
        }
    }
}

impl CSR {
    /// Returns the outgoing edge range of `vertex`, or `None` if it is invalid.
    #[must_use]
    #[inline]
    pub fn neighbor_range(&self, vertex: usize) -> Option<(usize, usize)> {
        let start = *self.offsets.get(vertex)?;
        let end = *self.offsets.get(vertex + 1)?;

        debug_assert!(start <= end);
        debug_assert!(end <= self.indices.len());

        Some((start, end))
    }

    #[must_use]
    #[inline]
    fn source_of_edge(&self, edge: usize) -> usize {
        debug_assert!(edge < self.edge_count());

        let position = self.offsets.partition_point(|&offset| offset <= edge);
        debug_assert!(position > 0);
        debug_assert!(position <= self.vertex_count());

        position - 1
    }
}

impl FromEndpoints for CSR {
    /// Builds a CSR graph from directed edge endpoints.
    ///
    /// The vertex set is `0..=m`, where `m` is the largest endpoint.
    /// An empty input yields the empty graph.
    fn from_endpoints<I>(edges: I) -> Self
    where
        I: IntoIterator<Item = Endpoints<Self::Vertex>>,
    {
        let mut edges: Vec<_> = edges.into_iter().map(|e| (e.from, e.to)).collect();

        if edges.is_empty() {
            return Self::default();
        }

        edges.sort_unstable_by_key(|&(from, _)| from);

        let vertex_count = edges.iter().map(|&(from, to)| from.max(to)).max().unwrap() + 1;
        let edge_count = edges.len();

        let mut offsets = vec![0usize; vertex_count + 1];
        for &(from, _) in &edges {
            offsets[from + 1] += 1;
        }
        for i in 1..=vertex_count {
            offsets[i] += offsets[i - 1];
        }

        let indices = edges.into_iter().map(|(_, to)| to).collect::<Vec<_>>();

        debug_assert_eq!(offsets.len(), vertex_count + 1);
        debug_assert_eq!(offsets[0], 0);
        debug_assert_eq!(offsets[vertex_count], edge_count);
        debug_assert!(offsets.windows(2).all(|w| w[0] <= w[1]));
        debug_assert_eq!(indices.len(), edge_count);

        Self {
            offsets: offsets.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
        }
    }
}

impl VertexType for CSR {
    type Vertex = usize;
}

impl EdgeType for CSR {
    type Edge = usize;
}

impl Vertices for CSR {
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

impl FiniteVertices for CSR {
    /// Returns the number of vertices.
    #[inline]
    fn vertex_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Returns whether `vertex` exists.
    #[inline]
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        *vertex < self.vertex_count()
    }
}

impl Edges for CSR {
    type Edges<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Returns all edge identifiers.
    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        0..self.edge_count()
    }
}

impl FiniteEdges for CSR {
    /// Returns the number of edges.
    #[inline]
    fn edge_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns whether `edge` exists.
    #[inline]
    fn contains_edge(&self, edge: &Self::Edge) -> bool {
        *edge < self.edge_count()
    }
}

impl InsertVertex for CSR {
    /// Inserts a new isolated vertex.
    #[inline]
    fn insert_vertex(&mut self) -> Option<Self::Vertex> {
        let vertex = self.vertex_count();
        let last_offset = self.edge_count();

        if !grow_boxed_slice(&mut self.offsets, 1, |tail| {
            tail[0] = last_offset;
        }) {
            return None;
        }

        Some(vertex)
    }
}

impl Structure for CSR {
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

impl Graph for CSR {}

impl Directed for CSR {
    type Outgoing<'a>
        = CsrEdges<'a>
    where
        Self: 'a;

    type Ingoing<'a>
        = CsrEdges<'a>
    where
        Self: 'a;

    type Connections<'a>
        = CsrEdges<'a>
    where
        Self: 'a;

    /// Returns the source of `edge`.
    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.source_of_edge(edge)
    }

    /// Returns the destination of `edge`.
    #[inline]
    fn destination(&self, edge: Self::Edge) -> Self::Vertex {
        self.indices[edge]
    }

    /// Returns all outgoing edges from `source`.
    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        match self.neighbor_range(source) {
            Some((start, end)) => CsrEdges::new(self, start, end, None),
            None => CsrEdges::empty(self),
        }
    }

    /// Returns all incoming edges to `destination`.
    ///
    /// This scans the full edge set.
    #[inline]
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        if !self.contains(&destination) {
            return CsrEdges::empty(self);
        }
        CsrEdges::new(self, 0, self.edge_count(), Some(destination))
    }

    /// Returns all edges from `from` to `to`.
    #[inline]
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        match self.neighbor_range(from) {
            Some((start, end)) if self.contains(&to) => CsrEdges::new(self, start, end, Some(to)),
            _ => CsrEdges::empty(self),
        }
    }
}

impl FiniteDirected for CSR {
    /// Returns the outgoing degree of `vertex`.
    #[inline]
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        match self.neighbor_range(vertex) {
            Some((start, end)) => end - start,
            None => 0,
        }
    }

    /// Returns the incoming degree of `vertex`.
    #[inline]
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        if !self.contains(&vertex) {
            return 0;
        }

        self.indices
            .iter()
            .filter(|&&destination| destination == vertex)
            .count()
    }

    /// Returns the number of loops at `vertex`.
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        match self.neighbor_range(vertex) {
            Some((start, end)) => self.indices[start..end]
                .iter()
                .filter(|&&destination| destination == vertex)
                .count(),
            None => 0,
        }
    }
}

impl IndexedDirected for CSR {
    #[inline]
    fn outgoing_count(&self, vertex: Self::Vertex) -> usize {
        let n = self.vertex_count();
        if vertex >= n {
            return 0;
        }

        self.offsets[vertex + 1] - self.offsets[vertex]
    }

    #[inline]
    fn outgoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex> {
        let n = self.vertex_count();
        if vertex >= n {
            return None;
        }

        let start = self.offsets[vertex];
        let end = self.offsets[vertex + 1];

        if index < end - start {
            Some(self.indices[start + index])
        } else {
            None
        }
    }

    #[inline]
    fn ingoing_count(&self, vertex: Self::Vertex) -> usize {
        let n = self.vertex_count();
        if vertex >= n {
            return 0;
        }

        let indices = &self.indices;
        let mut count = 0;
        let mut edge = 0;

        while edge < indices.len() {
            count += (indices[edge] == vertex) as usize;
            edge += 1;
        }

        count
    }

    #[inline]
    fn ingoing_at(&self, vertex: Self::Vertex, mut index: usize) -> Option<Self::Vertex> {
        let n = self.vertex_count();
        if vertex >= n {
            return None;
        }

        let offsets = &self.offsets;
        let indices = &self.indices;
        let mut from = 0;
        let mut edge = 0;
        let edge_count = indices.len();

        while edge < edge_count {
            while edge == offsets[from + 1] {
                from += 1;
            }

            if indices[edge] == vertex {
                if index == 0 {
                    return Some(from);
                }
                index -= 1;
            }

            edge += 1;
        }

        None
    }
}

/// Iterator over selected CSR edges.
///
/// Yields `(source, edge, destination)`.
#[derive(Clone, Debug)]
pub struct CsrEdges<'a> {
    csr: &'a CSR,
    /// Current edge index.
    index: usize,
    /// Exclusive end index.
    end: usize,
    /// Optional destination filter.
    destination: Option<usize>,
}

impl<'a> CsrEdges<'a> {
    /// Creates a CSR edge iterator over `index..end`.
    #[must_use]
    pub fn new(csr: &'a CSR, index: usize, end: usize, destination: Option<usize>) -> Self {
        debug_assert!(index <= end);
        debug_assert!(end <= csr.edge_count());
        debug_assert!(destination.is_none_or(|v| v < csr.vertex_count()));

        Self {
            csr,
            index,
            end,
            destination,
        }
    }

    /// Returns the empty iterator.
    #[must_use]
    #[inline]
    pub fn empty(csr: &'a CSR) -> Self {
        Self::new(csr, 0, 0, None)
    }
}

impl<'a> Iterator for CsrEdges<'a> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.end {
            let edge = self.index;
            self.index += 1;

            let destination = self.csr.indices[edge];
            if self.destination.is_some_and(|filter| destination != filter) {
                continue;
            }

            let source = self.csr.source_of_edge(edge);
            return Some((source, edge, destination));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::graph::FromEndpoints;

    use proptest::prelude::*;
    use std::collections::HashMap;

    fn assert_csr_invariants(graph: &CSR) {
        assert_eq!(graph.offsets.len(), graph.vertex_count() + 1);
        assert_eq!(*graph.offsets.first().unwrap(), 0);
        assert_eq!(*graph.offsets.last().unwrap(), graph.indices.len());

        for window in graph.offsets.windows(2) {
            assert!(window[0] <= window[1], "offsets not non-decreasing");
        }

        for &to in &graph.indices {
            assert!(to < graph.vertex_count());
        }
    }

    fn assert_edges_match(graph: &CSR, expected: &[(usize, usize)]) {
        let mut expected_counts = HashMap::new();
        for &(from, to) in expected {
            *expected_counts.entry((from, to)).or_insert(0usize) += 1;
        }

        let mut actual_counts = HashMap::new();
        for edge in graph.edges() {
            let from = graph.source(edge);
            let to = graph.destination(edge);
            *actual_counts.entry((from, to)).or_insert(0usize) += 1;
        }

        assert_eq!(actual_counts, expected_counts);
    }

    fn expected_adjacency(
        edges: &[(usize, usize)],
        vertex_count: usize,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>) {
        let mut outgoing = vec![Vec::new(); vertex_count];
        let mut ingoing = vec![Vec::new(); vertex_count];
        let mut loops = vec![0usize; vertex_count];

        for &(from, to) in edges {
            outgoing[from].push(to);
            ingoing[to].push(from);
            if from == to {
                loops[from] += 1;
            }
        }

        (outgoing, ingoing, loops)
    }

    fn arbitrary_csr_instance() -> impl Strategy<Value = Vec<(usize, usize)>> {
        prop::collection::vec((0usize..16, 0usize..16), 0..64)
    }

    #[test]
    fn default_is_empty_and_consistent() {
        let graph = CSR::default();

        assert_eq!(graph.vertex_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.vertices().count(), 0);
        assert_eq!(graph.edges().count(), 0);
        assert_eq!(graph.offsets.as_ref(), &[0]);
        assert!(graph.indices.is_empty());

        assert_csr_invariants(&graph);
    }

    #[test]
    fn from_endpoints_empty_matches_default() {
        let graph = CSR::from_endpoints(std::iter::empty::<Endpoints<usize>>());

        assert_eq!(graph, CSR::default());
        assert_csr_invariants(&graph);
    }

    #[test]
    fn single_loop_at_zero_layout_and_queries() {
        let graph = CSR::from_endpoints([Endpoints::new(0, 0)]);

        assert_eq!(graph.vertex_count(), 1);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.offsets.as_ref(), &[0, 1]);
        assert_eq!(graph.indices.as_ref(), &[0]);

        assert_eq!(graph.neighbor_range(0), Some((0, 1)));
        assert_eq!(graph.neighbor_range(1), None);

        assert_eq!(graph.source(0), 0);
        assert_eq!(graph.destination(0), 0);

        assert_eq!(graph.outgoing(0).collect::<Vec<_>>(), vec![(0, 0, 0)]);
        assert_eq!(graph.ingoing(0).collect::<Vec<_>>(), vec![(0, 0, 0)]);

        assert_eq!(graph.outgoing_degree(0), 1);
        assert_eq!(graph.ingoing_degree(0), 1);
        assert_eq!(graph.loop_degree(0), 1);

        assert_csr_invariants(&graph);
    }

    #[test]
    fn vertex_queries_and_edges_are_consistent() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 0),
            Endpoints::new(2, 1),
        ]);

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);

        assert_eq!(graph.vertices().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(graph.neighbor_range(0), Some((0, 1)));
        assert_eq!(graph.neighbor_range(1), Some((1, 2)));
        assert_eq!(graph.neighbor_range(2), Some((2, 3)));
        assert_eq!(graph.neighbor_range(3), None);

        assert_eq!(graph.edges().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_edges_match(&graph, &[(0, 1), (1, 0), (2, 1)]);
        assert_csr_invariants(&graph);
    }

    #[test]
    fn directed_queries_match_expected_structure() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 0),
            Endpoints::new(2, 1),
        ]);

        assert_eq!(graph.outgoing(0).collect::<Vec<_>>(), vec![(0, 0, 1)]);
        assert_eq!(graph.outgoing(1).collect::<Vec<_>>(), vec![(1, 1, 0)]);
        assert_eq!(graph.outgoing(2).collect::<Vec<_>>(), vec![(2, 2, 1)]);

        assert_eq!(graph.ingoing(0).collect::<Vec<_>>(), vec![(1, 1, 0)]);
        assert_eq!(
            graph.ingoing(1).collect::<Vec<_>>(),
            vec![(0, 0, 1), (2, 2, 1)]
        );
        assert_eq!(
            graph.ingoing(2).collect::<Vec<_>>(),
            Vec::<(usize, usize, usize)>::new()
        );

        assert_eq!(graph.outgoing_degree(0), 1);
        assert_eq!(graph.outgoing_degree(1), 1);
        assert_eq!(graph.outgoing_degree(2), 1);

        assert_eq!(graph.ingoing_degree(0), 1);
        assert_eq!(graph.ingoing_degree(1), 2);
        assert_eq!(graph.ingoing_degree(2), 0);

        assert_eq!(graph.loop_degree(0), 0);
        assert_eq!(graph.loop_degree(1), 0);
        assert_eq!(graph.loop_degree(2), 0);

        assert!(graph.is_connected(0, 1));
        assert!(graph.is_connected(1, 0));
        assert!(graph.is_connected(2, 1));
        assert!(!graph.is_connected(1, 2));
    }

    #[test]
    fn parallel_edges_and_loops_are_preserved() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 0),
            Endpoints::new(0, 0),
            Endpoints::new(1, 0),
        ]);

        assert_eq!(graph.vertex_count(), 2);
        assert_eq!(graph.edge_count(), 3);

        assert_eq!(graph.outgoing_degree(0), 2);
        assert_eq!(graph.outgoing_degree(1), 1);

        assert_eq!(graph.ingoing_degree(0), 3);
        assert_eq!(graph.ingoing_degree(1), 0);

        assert_eq!(graph.loop_degree(0), 2);
        assert_eq!(graph.loop_degree(1), 0);

        assert_eq!(graph.connections(0, 0).count(), 2);
        assert_eq!(graph.connections(1, 0).count(), 1);

        assert_edges_match(&graph, &[(0, 0), (0, 0), (1, 0)]);
        assert_csr_invariants(&graph);
    }

    #[test]
    fn insert_vertex_appends_new_isolated_vertex() {
        let mut graph = CSR::from_endpoints([Endpoints::new(0, 1), Endpoints::new(0, 1)]);

        let old_edges: Vec<_> = graph.edges().collect();
        let old_vertex_count = graph.vertex_count();
        let old_edge_count = graph.edge_count();

        let vertex = graph.insert_vertex().unwrap();

        assert_eq!(vertex, old_vertex_count);
        assert_eq!(graph.vertex_count(), old_vertex_count + 1);
        assert_eq!(graph.edge_count(), old_edge_count);
        assert_eq!(graph.edges().collect::<Vec<_>>(), old_edges);
        assert_eq!(
            graph.neighbor_range(vertex),
            Some((old_edge_count, old_edge_count))
        );

        assert_csr_invariants(&graph);
    }

    #[test]
    fn indexed_directed_matches_expected() {
        let graph = CSR::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(2, 1),
            Endpoints::new(2, 1),
        ]);

        assert_eq!(graph.outgoing_count(0), 2);
        assert_eq!(graph.outgoing_count(1), 0);
        assert_eq!(graph.outgoing_count(2), 2);
        assert_eq!(graph.outgoing_count(3), 0);

        assert_eq!(graph.outgoing_at(0, 0), Some(1));
        assert_eq!(graph.outgoing_at(0, 1), Some(2));
        assert_eq!(graph.outgoing_at(0, 2), None);

        assert_eq!(graph.ingoing_count(0), 0);
        assert_eq!(graph.ingoing_count(1), 3);
        assert_eq!(graph.ingoing_count(2), 1);
        assert_eq!(graph.ingoing_count(3), 0);

        assert_eq!(graph.ingoing_at(1, 0), Some(0));
        assert_eq!(graph.ingoing_at(1, 1), Some(2));
        assert_eq!(graph.ingoing_at(1, 2), Some(2));
        assert_eq!(graph.ingoing_at(1, 3), None);
    }

    proptest! {
        #[test]
        fn prop_constructor_preserves_edges_and_invariants(edges in arbitrary_csr_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            if edges.is_empty() {
                prop_assert_eq!(graph.vertex_count(), 0);
                prop_assert_eq!(graph.edge_count(), 0);
                prop_assert_eq!(graph.offsets.as_ref(), &[0]);
                prop_assert!(graph.indices.is_empty());
            } else {
                let expected_vertex_count =
                    edges.iter().map(|&(from, to)| from.max(to)).max().unwrap() + 1;

                prop_assert_eq!(graph.vertex_count(), expected_vertex_count);
                prop_assert_eq!(graph.edge_count(), edges.len());

                for vertex in 0..graph.vertex_count() {
                    prop_assert_eq!(
                        graph.neighbor_range(vertex),
                        Some((graph.offsets[vertex], graph.offsets[vertex + 1]))
                    );
                }
            }

            assert_edges_match(&graph, &edges);
            assert_csr_invariants(&graph);
        }

        #[test]
        fn prop_adjacency_and_degrees_match_input(edges in arbitrary_csr_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            if edges.is_empty() {
                prop_assert_eq!(graph.vertex_count(), 0);
                prop_assert_eq!(graph.edge_count(), 0);
                return Ok(());
            }

            let (expected_outgoing, expected_ingoing, expected_loops) =
                expected_adjacency(&edges, graph.vertex_count());

            for vertex in 0..graph.vertex_count() {
                let mut actual_outgoing: Vec<_> =
                    graph.outgoing(vertex).map(|(_, _, to)| to).collect();
                let mut want_outgoing = expected_outgoing[vertex].clone();
                actual_outgoing.sort_unstable();
                want_outgoing.sort_unstable();

                prop_assert_eq!(actual_outgoing, want_outgoing);
                prop_assert_eq!(graph.outgoing_degree(vertex), expected_outgoing[vertex].len());

                let mut actual_ingoing: Vec<_> =
                    graph.ingoing(vertex).map(|(from, _, _)| from).collect();
                let mut want_ingoing = expected_ingoing[vertex].clone();
                actual_ingoing.sort_unstable();
                want_ingoing.sort_unstable();

                prop_assert_eq!(actual_ingoing, want_ingoing);
                prop_assert_eq!(graph.ingoing_degree(vertex), expected_ingoing[vertex].len());

                prop_assert_eq!(graph.loop_degree(vertex), expected_loops[vertex]);
            }

            let total_outgoing: usize =
                (0..graph.vertex_count()).map(|vertex| graph.outgoing_degree(vertex)).sum();
            let total_ingoing: usize =
                (0..graph.vertex_count()).map(|vertex| graph.ingoing_degree(vertex)).sum();

            prop_assert_eq!(total_outgoing, graph.edge_count());
            prop_assert_eq!(total_ingoing, graph.edge_count());
        }

        #[test]
        fn prop_connections_match_input_multiplicity(
            edges in arbitrary_csr_instance(),
            from_hint in 0usize..32,
            to_hint in 0usize..32,
        ) {
            let graph = CSR::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            if graph.vertex_count() == 0 {
                prop_assert_eq!(graph.edge_count(), 0);
                return Ok(());
            }

            let from = from_hint % graph.vertex_count();
            let to = to_hint % graph.vertex_count();

            let expected = edges
                .iter()
                .filter(|&&(u, v)| u == from && v == to)
                .count();

            prop_assert_eq!(graph.connections(from, to).count(), expected);
        }

        #[test]
        fn prop_indexed_directed_matches_iterators(edges in arbitrary_csr_instance()) {
            let graph = CSR::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            for vertex in 0..graph.vertex_count() {
                let outgoing: Vec<_> = graph.outgoing(vertex).map(|(_, _, to)| to).collect();
                prop_assert_eq!(graph.outgoing_count(vertex), outgoing.len());

                for index in 0..outgoing.len() {
                    prop_assert_eq!(graph.outgoing_at(vertex, index), Some(outgoing[index]));
                }
                prop_assert_eq!(graph.outgoing_at(vertex, outgoing.len()), None);

                let ingoing: Vec<_> = graph.ingoing(vertex).map(|(from, _, _)| from).collect();
                prop_assert_eq!(graph.ingoing_count(vertex), ingoing.len());

                for index in 0..ingoing.len() {
                    prop_assert_eq!(graph.ingoing_at(vertex, index), Some(ingoing[index]));
                }
                prop_assert_eq!(graph.ingoing_at(vertex, ingoing.len()), None);
            }

            let invalid = graph.vertex_count();

            prop_assert_eq!(graph.outgoing_count(invalid), 0);
            prop_assert_eq!(graph.outgoing_at(invalid, 0), None);
            prop_assert_eq!(graph.ingoing_count(invalid), 0);
            prop_assert_eq!(graph.ingoing_at(invalid, 0), None);
        }
    }
}
