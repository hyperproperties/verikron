use std::ops::Range;

use crate::graphs::{directed::Directed, edges::Edges, graph::Graph, vertices::Vertices};

/// Compressed Sparse Row (CSR) representation of a directed multi graph.
///
/// Vertices are numbered from zero up to `vertex_count - 1`.
/// Outgoing neighbors of a vertex `u` are stored in a contiguous
/// segment in the indices array.
/// The segment for `u` is given by the half open range
/// `offsets[u]` up to `offsets[u + one]`.
///
/// The length of `offsets` is `vertex_count + one`.
/// `offsets[0]` is always zero.
/// `offsets[vertex_count]` is always equal to `indices.len`.
/// The length of `indices` is equal to the number of directed edges
pub struct CSR {
    /// Row offsets for the CSR layout.
    ///
    /// For a vertex `u` the outgoing neighbors of `u` live in
    /// `indices[offsets[u]` up to `offsets[u + one]]`
    offsets: Box<[usize]>,

    /// Concatenated list of destination vertices for all edges.
    ///
    /// Each entry is the target vertex of one directed edge.
    indices: Box<[usize]>,
}

impl Default for CSR {
    /// Empty CSR graph with no vertices and no edges.
    fn default() -> Self {
        Self {
            offsets: Box::from([0]),
            indices: Box::new([]),
        }
    }
}

impl From<Vec<(usize, usize)>> for CSR {
    /// Build a CSR graph from a list of directed edges.
    ///
    /// The vertex set is the integers from zero up to one plus
    /// the largest source vertex in the edge list.
    /// Destination vertices must be in the same range.
    fn from(mut edges: Vec<(usize, usize)>) -> Self {
        if edges.is_empty() {
            return Self::default();
        }

        // Sort by source so adjacency lists become contiguous.
        edges.sort_unstable_by_key(|&(source, _)| source);

        let vertex_count = edges.iter().map(|&(from, to)| from.max(to)).max().unwrap() + 1;
        let edge_count = edges.len();

        // offsets[i + 1] will count how many edges originate from vertex i.
        let mut offsets = vec![0usize; vertex_count + 1];

        for (source, _) in &edges {
            // We assume `source < vertex_count` by construction.
            offsets[source + 1] += 1;
        }

        // Prefix sum: now offsets[u] is the starting index in `indices`
        // for edges from vertex `u`.
        for i in 1..=vertex_count {
            offsets[i] += offsets[i - 1];
        }

        debug_assert_eq!(offsets[0], 0);
        debug_assert_eq!(offsets[vertex_count], edge_count);

        // Edges are already sorted by `source`, so we can just copy
        // destinations in order; they will match the CSR ranges.
        let indices: Vec<usize> = edges.into_iter().map(|(_, dest)| dest).collect();

        CSR {
            offsets: offsets.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
        }
    }
}

impl CSR {
    /// Returns the half open range of indices for the outgoing neighbors of a vertex.
    ///
    /// On success this returns start and end such that the neighbors
    /// of vertex are `indices[start..end]`.
    /// Returns None when vertex is out of range
    #[inline]
    pub fn neighbor_range(&self, vertex: usize) -> Option<(usize, usize)> {
        let start = *self.offsets.get(vertex)?;
        // CSR is closed by an additional offset marking the length of the indices.
        let end = *self.offsets.get(vertex + 1)?;
        Some((start, end))
    }
}

impl Directed for CSR {
    /// Source vertex of an edge.
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.offsets.partition_point(|&offset| offset <= edge) - 1
    }

    /// Destination vertex of an edge.
    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        self.indices[edge]
    }

    /// Iterator over all edges whose source equals the given vertex.
    fn outgoing(&self, source: Self::Vertex) -> Self::Edges<'_> {
        match self.neighbor_range(source) {
            Some((start, end)) => Self::Edges::new(self, start, end, None),
            None => Self::Edges::empty(self),
        }
    }

    /// Number of outgoing edges for a vertex.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        match self.neighbor_range(vertex) {
            Some((start, end)) => end - start,
            None => 0,
        }
    }

    /// Iterator over all edges whose destination equals the given vertex.
    ///
    /// This scans all edges and filters by destination
    /// which is optimal given a single CSR layout.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Edges<'_> {
        Self::Edges::new(self, 0, self.edge_count(), Some(destination))
    }

    /// Number of incoming edges for a vertex.
    ///
    /// This counts matches in the indices array.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.indices
            .iter()
            .filter(|&destination| *destination == vertex)
            .count()
    }

    /// Number of edges pointing back to the source vertex.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        match self.neighbor_range(vertex) {
            Some((start, end)) => Self::Edges::new(self, start, end, Some(vertex)).count(),
            None => 0,
        }
    }

    /// Returns an iterator over all edges whose source is from,
    /// and whose destination is to.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Edges<'_> {
        match self.neighbor_range(from) {
            Some((start, end)) => Self::Edges::new(self, start, end, Some(to)),
            None => Self::Edges::empty(self),
        }
    }
}

impl Edges for CSR {
    type Vertex = usize;

    type Edge = usize;

    type Edges<'a>
        = CsrEdges<'a>
    where
        Self: 'a;

    /// Iterator over all edges in the graph.
    fn edges(&self) -> Self::Edges<'_> {
        Self::Edges::new(self, 0, self.edge_count(), None)
    }

    /// Number of edges.
    fn edge_count(&self) -> usize {
        self.indices.len()
    }
}

impl Vertices for CSR {
    type Vertex = usize;

    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Iterator over all vertices in the graph.
    ///
    /// Vertices are the integers from zero up to `vertex_count`.
    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count()
    }

    /// Number of vertices.
    fn vertex_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

impl Graph for CSR {
    type Vertices = Self;
    type Edges = Self;

    /// Access to the edge store.
    fn edge_store(&self) -> &Self::Edges {
        self
    }

    /// Access to the vertex store.
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

/// Iterator over edges in a CSR graph.
///
/// The iterator yields triples: `(source, edge, destination)`.
/// It walks a half-open integer interval and optionally filter by destination vertex.
pub struct CsrEdges<'a> {
    csr: &'a CSR,
    /// Current index in the `indices`.
    index: usize,
    /// The index to iterate to in the `indices`.
    end: usize,
    /// Optionally destination filter.
    /// When present only edges whose destination equals this value is yeilded.
    destination: Option<usize>,
}

impl<'a> CsrEdges<'a> {
    pub fn new(csr: &'a CSR, index: usize, end: usize, destination: Option<usize>) -> Self {
        Self {
            csr,
            index,
            end,
            destination,
        }
    }

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

            let source = self.csr.source(edge);

            return Some((source, edge, destination));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::{directed::Directed, edges::Edges, graph::Graph, vertices::Vertices};

    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    #[test]
    fn default_is_empty_and_has_consistent_invariants() {
        let csr = CSR::default();

        // basic counts
        assert_eq!(csr.vertex_count(), 0, "default has no vertices");
        assert_eq!(csr.edge_count(), 0, "default has no edges");
        assert_eq!(csr.size(), 0, "size must be zero for an empty graph");
        assert!(csr.is_empty(), "is_empty must be true for default graph");

        // iterators
        assert_eq!(csr.vertices().count(), 0, "no vertices to iterate");
        assert_eq!(csr.edges().count(), 0, "no edges to iterate");

        // internal layout
        assert_eq!(csr.offsets.as_ref(), &[0], "offsets for empty graph");
        assert!(csr.indices.is_empty(), "indices must be empty");
    }

    #[test]
    fn from_empty_vec_matches_default() {
        let from_empty = CSR::from(Vec::<(usize, usize)>::new());
        let default = CSR::default();

        assert_eq!(from_empty.vertex_count(), default.vertex_count());
        assert_eq!(from_empty.edge_count(), default.edge_count());
        assert_eq!(from_empty.size(), default.size());
        assert_eq!(from_empty.offsets.as_ref(), default.offsets.as_ref());
        assert_eq!(from_empty.indices.as_ref(), default.indices.as_ref());
    }

    #[test]
    fn single_loop_at_zero_layout_and_queries() {
        let csr = CSR::from(vec![(0, 0)]);

        // layout
        assert_eq!(csr.vertex_count(), 1, "only vertex 0 exists");
        assert_eq!(csr.edge_count(), 1, "exactly one edge");
        assert_eq!(csr.offsets.as_ref(), &[0, 1], "CSR offsets for one vertex");
        assert_eq!(csr.indices.as_ref(), &[0], "single edge 0 -> 0");

        // neighbor range
        assert_eq!(csr.neighbor_range(0), Some((0, 1)));
        assert_eq!(csr.neighbor_range(1), None, "vertex 1 is out of range");

        // directed view
        assert_eq!(csr.source(0), 0, "source of edge 0 is vertex 0");
        assert_eq!(csr.target(0), 0, "target of edge 0 is vertex 0");

        // outgoing and ingoing
        let out0: Vec<_> = csr.outgoing(0).collect();
        let in0: Vec<_> = csr.ingoing(0).collect();
        assert_eq!(out0, vec![(0, 0, 0)], "outgoing from 0");
        assert_eq!(in0, vec![(0, 0, 0)], "ingoing to 0");

        assert_eq!(csr.outgoing_degree(0), 1);
        assert_eq!(csr.ingoing_degree(0), 1);
        assert_eq!(csr.loop_degree(0), 1);

        // graph view
        assert_eq!(csr.size(), csr.vertex_count() + csr.edge_count());
        assert!(!csr.is_empty());
    }

    #[test]
    fn single_loop_at_one_with_gap_at_zero() {
        let csr = CSR::from(vec![(1, 1)]);

        // layout
        assert_eq!(csr.vertex_count(), 2, "vertices 0 and 1 exist");
        assert_eq!(csr.edge_count(), 1, "exactly one edge");

        // vertex 0 has no outgoing edges
        assert_eq!(csr.neighbor_range(0), Some((0, 0)));
        // vertex 1 has the single edge
        assert_eq!(csr.neighbor_range(1), Some((0, 1)));

        assert_eq!(csr.outgoing_degree(0), 0);
        assert_eq!(csr.ingoing_degree(0), 0);
        assert_eq!(csr.loop_degree(0), 0);

        assert_eq!(csr.outgoing_degree(1), 1);
        assert_eq!(csr.ingoing_degree(1), 1);
        assert_eq!(csr.loop_degree(1), 1);
    }

    #[test]
    fn simple_graph_with_three_vertices_layout_and_iterators() {
        // edges: 0 -> 1, 1 -> 0, 2 -> 1
        let edges = vec![(0, 1), (1, 0), (2, 1)];
        let csr = CSR::from(edges.clone());

        // counts
        assert_eq!(csr.vertex_count(), 3);
        assert_eq!(csr.edge_count(), 3);

        // internal layout
        // edges are sorted by source so the order is stable here
        assert_eq!(csr.offsets.as_ref(), &[0, 1, 2, 3]);
        assert_eq!(csr.indices.as_ref(), &[1, 0, 1]);

        // neighbor ranges
        assert_eq!(csr.neighbor_range(0), Some((0, 1)));
        assert_eq!(csr.neighbor_range(1), Some((1, 2)));
        assert_eq!(csr.neighbor_range(2), Some((2, 3)));
        assert_eq!(csr.neighbor_range(3), None);

        // source and target should reconstruct the sorted edge list
        let mut sorted = edges.clone();
        sorted.sort_unstable_by_key(|(s, _)| *s);
        for (e, (src, dst)) in sorted.iter().enumerate() {
            assert_eq!(csr.source(e), *src, "source mismatch for edge {e}");
            assert_eq!(csr.target(e), *dst, "target mismatch for edge {e}");
        }

        // edges iterator must match the same triples
        let triples: Vec<_> = csr.edges().collect();
        let expected: Vec<_> = sorted
            .iter()
            .enumerate()
            .map(|(e, (s, d))| (*s, e, *d))
            .collect();
        assert_eq!(triples, expected, "edges iterator must yield all edges");
    }

    #[test]
    fn outgoing_and_ingoing_iterators_and_degrees() {
        // edges: 0 -> 1, 1 -> 0, 2 -> 1
        let csr = CSR::from(vec![(0, 1), (1, 0), (2, 1)]);

        // outgoing neighbors
        let out0: Vec<_> = csr.outgoing(0).collect();
        assert_eq!(out0.len(), 1);
        assert_eq!(out0[0].0, 0);
        assert_eq!(out0[0].2, 1);

        let out1: Vec<_> = csr.outgoing(1).collect();
        assert_eq!(out1.len(), 1);
        assert_eq!(out1[0].0, 1);
        assert_eq!(out1[0].2, 0);

        let out2: Vec<_> = csr.outgoing(2).collect();
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].0, 2);
        assert_eq!(out2[0].2, 1);

        // ingoing neighbors
        let in0: Vec<_> = csr.ingoing(0).collect();
        let in1: Vec<_> = csr.ingoing(1).collect();
        let in2: Vec<_> = csr.ingoing(2).collect();

        assert_eq!(in0.len(), 1, "only 1 -> 0 points to 0");
        assert_eq!(in1.len(), 2, "0 -> 1 and 2 -> 1 point to 1");
        assert_eq!(in2.len(), 0, "no edge points to 2");

        // degrees
        assert_eq!(csr.outgoing_degree(0), 1);
        assert_eq!(csr.outgoing_degree(1), 1);
        assert_eq!(csr.outgoing_degree(2), 1);

        assert_eq!(csr.ingoing_degree(0), 1);
        assert_eq!(csr.ingoing_degree(1), 2);
        assert_eq!(csr.ingoing_degree(2), 0);

        // no loops in this graph
        for v in 0..csr.vertex_count() {
            assert_eq!(csr.loop_degree(v), 0);
        }
    }

    #[test]
    fn degrees_sum_to_edge_count() {
        // a slightly larger example with multiple edges to the same destination
        let edges = vec![(0, 0), (1, 0), (1, 1), (2, 1)];
        let csr = CSR::from(edges);

        let total_out: usize = (0..csr.vertex_count())
            .map(|v| csr.outgoing_degree(v))
            .sum();
        let total_in: usize = (0..csr.vertex_count()).map(|v| csr.ingoing_degree(v)).sum();

        assert_eq!(total_out, csr.edge_count(), "sum of outgoing degrees");
        assert_eq!(total_in, csr.edge_count(), "sum of ingoing degrees");
    }

    #[test]
    fn parallel_loops_and_multigraph_behavior() {
        // multiple loops on the same vertex and one extra edge
        // edges: 0 -> 0, 0 -> 0, 1 -> 0
        let edges = vec![(0, 0), (0, 0), (1, 0)];
        let csr = CSR::from(edges);

        assert_eq!(csr.vertex_count(), 2);
        assert_eq!(csr.edge_count(), 3);

        // vertex 0 has two outgoing loops
        assert_eq!(csr.outgoing_degree(0), 2);
        assert_eq!(csr.loop_degree(0), 2);

        // vertex 1 has one outgoing edge
        assert_eq!(csr.outgoing_degree(1), 1);
        assert_eq!(csr.loop_degree(1), 0);

        // all three edges go into vertex 0
        assert_eq!(csr.ingoing_degree(0), 3);
        assert_eq!(csr.ingoing_degree(1), 0);
    }

    #[test]
    fn vertices_and_graph_traits_are_consistent() {
        let csr = CSR::from(vec![(0, 0), (1, 0), (1, 1)]);

        // vertices iterator matches vertex_count
        let vs: Vec<_> = csr.vertices().collect();
        assert_eq!(vs.len(), csr.vertex_count());
        assert_eq!(vs, (0..csr.vertex_count()).collect::<Vec<_>>());

        // edges iterator matches edge_count
        let es: Vec<_> = csr.edges().collect();
        assert_eq!(es.len(), csr.edge_count());

        // size is vertices plus edges
        assert_eq!(csr.size(), csr.vertex_count() + csr.edge_count());
        assert!(!csr.is_empty());
    }

    #[test]
    fn csr_offsets_are_monotonic_and_neighbor_ranges_match() {
        let edges = vec![
            (0, 0),
            (0, 1),
            (3, 2), // gap at 1 and 2
            (5, 4), // gap at 4 with another gap at 4 as a source
        ];
        let csr = CSR::from(edges);

        assert!(csr.vertex_count() >= 6);

        // offsets must be increasing
        for v in 0..csr.vertex_count() {
            assert!(
                csr.offsets[v] <= csr.offsets[v + 1],
                "offsets[{}] = {} > offsets[{}] = {}",
                v,
                csr.offsets[v],
                v + 1,
                csr.offsets[v + 1],
            );

            // neighbor_range must match offsets exactly
            assert_eq!(
                csr.neighbor_range(v),
                Some((csr.offsets[v], csr.offsets[v + 1])),
            );
        }
    }

    // Generate a random edge list respecting CSR's documented vertex set:
    // - Vertex set: 0 ..= max_source
    // - Destinations are always in 0 ..= max_source as well.
    prop_compose! {
        fn csr_edge_list()
            (
                edges in prop::collection::vec((0u8..=15, 0u8..=15), 0..=64)
            ) -> Vec<(usize, usize)>
        {
            if edges.is_empty() {
                return Vec::new();
            }

            edges
                .into_iter()
                .map(|(from, to)|
                    (from as usize, to as usize)
                )
                .collect()
        }
    }

    proptest! {
        // Basic structural invariants of CSR::from(edges).
        #[test]
        fn prop_csr_basic_invariants(edges in csr_edge_list()) {
            let csr = CSR::from(edges.clone());

            if edges.is_empty() {
                prop_assert_eq!(csr.vertex_count(), 0);
                prop_assert_eq!(csr.edge_count(), 0);
                prop_assert_eq!(csr.offsets.as_ref(), &[0]);
                prop_assert!(csr.indices.is_empty());
            } else {
                prop_assert_eq!(csr.edge_count(), edges.len());
                prop_assert_eq!(csr.offsets.len(), csr.vertex_count() + 1);
                prop_assert_eq!(csr.offsets[0], 0);
                prop_assert_eq!(csr.offsets[csr.vertex_count()], csr.indices.len());

                for v in 0..csr.vertex_count() {
                    prop_assert!(
                        csr.offsets[v] <= csr.offsets[v + 1],
                        "offsets[{}] = {} > offsets[{}] = {}",
                        v,
                        csr.offsets[v],
                        v + 1,
                        csr.offsets[v + 1]
                    );

                    let actual = csr.neighbor_range(v);
                    let expected = Some((csr.offsets[v], csr.offsets[v + 1]));
                    prop_assert_eq!(actual, expected);
                }
            }
        }

        // Edges iterator must represent the same multiset of (source, dest)
        // as the input edge list.
        #[test]
        fn prop_csr_edges_iterator_matches_input(edges in csr_edge_list()) {
            let csr = CSR::from(edges.clone());

            let mut expected_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for (s, d) in edges {
                *expected_counts.entry((s, d)).or_insert(0) += 1;
            }

            let mut actual_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for (src, _, dst) in csr.edges() {
                *actual_counts.entry((src, dst)).or_insert(0) += 1;
            }

            prop_assert_eq!(actual_counts, expected_counts);
        }

        // outgoing(), ingoing(), *degree() and loop_degree()
        // must match the multigraph structure of the input.
        #[test]
        fn prop_csr_adjacency_and_degrees_match(edges in csr_edge_list()) {
            let csr = CSR::from(edges.clone());

            if edges.is_empty() {
                prop_assert_eq!(csr.vertex_count(), 0);
                prop_assert_eq!(csr.edge_count(), 0);
            } else {
                // Build expected adjacency structures.
                let mut out_to: Vec<Vec<usize>> = vec![Vec::new(); csr.vertex_count()];
                let mut in_from: Vec<Vec<usize>> = vec![Vec::new(); csr.vertex_count()];
                let mut loops: Vec<usize> = vec![0; csr.vertex_count()];

                for (from, to) in &edges {
                    out_to[*from].push(*to);
                    in_from[*to].push(*from);
                    if *from == *to {
                        loops[*from] += 1;
                    }
                }

                for v in 0..csr.vertex_count() {
                    // outgoing neighbors: collect destinations
                    let mut actual_out: Vec<usize> =
                        csr.outgoing(v).map(|(_, _, dst)| dst).collect();
                    let mut expected_out = out_to[v].clone();
                    actual_out.sort_unstable();
                    expected_out.sort_unstable();
                    prop_assert_eq!(
                        actual_out,
                        expected_out,
                        "outgoing({}) mismatch",
                        v
                    );

                    prop_assert_eq!(
                        csr.outgoing_degree(v),
                        out_to[v].len(),
                        "outgoing_degree({}) mismatch",
                        v
                    );

                    // ingoing neighbors: collect sources
                    let mut actual_in: Vec<usize> =
                        csr.ingoing(v).map(|(src, _, _)| src).collect();
                    let mut expected_in = in_from[v].clone();
                    actual_in.sort_unstable();
                    expected_in.sort_unstable();
                    prop_assert_eq!(
                        actual_in,
                        expected_in,
                        "ingoing({}) mismatch",
                        v
                    );

                    prop_assert_eq!(
                        csr.ingoing_degree(v),
                        in_from[v].len(),
                        "ingoing_degree({}) mismatch",
                        v
                    );

                    // loops
                    prop_assert_eq!(
                        csr.loop_degree(v),
                        loops[v],
                        "loop_degree({}) mismatch",
                        v
                    );
                }

                // Sum of degrees equals edge_count.
                let total_out: usize =
                    (0..csr.vertex_count()).map(|v| csr.outgoing_degree(v)).sum();
                let total_in: usize =
                    (0..csr.vertex_count()).map(|v| csr.ingoing_degree(v)).sum();

                prop_assert_eq!(
                    total_out,
                    csr.edge_count(),
                    "sum of outgoing degrees mismatch"
                );
                prop_assert_eq!(
                    total_in,
                    csr.edge_count(),
                    "sum of ingoing degrees mismatch"
                );
            }
        }

        // connections(from, to) must count exactly the multiplicity of (from, to).
        #[test]
        fn prop_csr_connections_match_edges(
            edges in csr_edge_list(),
            q_from in 0u8..=15,
            q_to   in 0u8..=15,
        ) {
            let csr = CSR::from(edges.clone());

            if edges.is_empty() {
                prop_assert_eq!(csr.edge_count(), 0);
            } else {
                let from = (q_from as usize) % csr.vertex_count();
                let to = (q_to as usize) % csr.vertex_count();

                let expected = edges
                    .iter()
                    .filter(|(s, d)| *s == from && *d == to)
                    .count();
                let actual = csr.connections(from, to).count();

                prop_assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn random_csr_invariants_and_edges_multiset() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4353_525F_494E_565F);

        for _ in 0..100 {
            let edge_count = rng.random_range(0..=100usize);
            let mut edges = Vec::with_capacity(edge_count);

            for _ in 0..edge_count {
                let s: u8 = rng.random_range(0..=15);
                let d: u8 = rng.random_range(0..=15);
                edges.push((s as usize, d as usize));
            }

            let csr = CSR::from(edges.clone());

            if edges.is_empty() {
                assert_eq!(csr.vertex_count(), 0);
                assert_eq!(csr.edge_count(), 0);
                assert_eq!(csr.offsets.as_ref(), &[0]);
                assert!(csr.indices.is_empty());
                continue;
            }

            // basic structure
            assert_eq!(csr.edge_count(), edges.len());
            assert_eq!(csr.offsets.len(), csr.vertex_count() + 1);
            assert_eq!(csr.offsets[0], 0);
            assert_eq!(csr.offsets[csr.vertex_count()], csr.indices.len());

            // offsets monotonic and neighbor_range matches
            for v in 0..csr.vertex_count() {
                assert!(
                    csr.offsets[v] <= csr.offsets[v + 1],
                    "offsets[{}] = {} > offsets[{}] = {}",
                    v,
                    csr.offsets[v],
                    v + 1,
                    csr.offsets[v + 1]
                );
                assert_eq!(
                    csr.neighbor_range(v),
                    Some((csr.offsets[v], csr.offsets[v + 1]))
                );
            }

            // sum of degrees
            let total_out: usize = (0..csr.vertex_count())
                .map(|v| csr.outgoing_degree(v))
                .sum();
            let total_in: usize = (0..csr.vertex_count()).map(|v| csr.ingoing_degree(v)).sum();
            assert_eq!(total_in, total_out);
            assert_eq!(total_in, csr.edge_count());
            assert_eq!(total_out, csr.edge_count());

            // multiset of (source, dest) pairs must match
            let mut expected: HashMap<(usize, usize), usize> = HashMap::new();
            for (from, to) in &edges {
                *expected.entry((*from, *to)).or_insert(0) += 1;
            }

            let mut actual: HashMap<(usize, usize), usize> = HashMap::new();
            for (from, _, to) in csr.edges() {
                *actual.entry((from, to)).or_insert(0) += 1;
            }

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn random_csr_connections_and_loop_degree() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4353_525F_434F_4E4E);

        for _ in 0..100 {
            let edge_count = rng.random_range(0..=80usize);
            let mut edges = Vec::with_capacity(edge_count);

            for _ in 0..edge_count {
                let from: usize = rng.random_range(0..=10);
                let to: usize = rng.random_range(0..=10);
                edges.push((from, to));
            }

            let csr = CSR::from(edges.clone());

            if edges.is_empty() {
                assert_eq!(csr.vertex_count(), 0);
                assert_eq!(csr.edge_count(), 0);
                continue;
            }

            // Precompute counts for (from, to) and loops.
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
            let mut loop_counts: HashMap<usize, usize> = HashMap::new();

            for (from, to) in &edges {
                *pair_counts.entry((*from, *to)).or_insert(0) += 1;
                if from == to {
                    *loop_counts.entry(*from).or_insert(0) += 1;
                }
            }

            // Randomly sample some (from, to) queries inside vertex set.
            for _ in 0..50 {
                let from = rng.random_range(0..csr.vertex_count());
                let to = rng.random_range(0..csr.vertex_count());

                let expected = *pair_counts.get(&(from, to)).unwrap_or(&0);
                let actual = csr.connections(from, to).count();
                assert_eq!(actual, expected, "connections({from}, {to}) mismatch");
            }

            // Check loop_degree against precomputed counts.
            for v in 0..csr.vertex_count() {
                let expected_loops = *loop_counts.get(&v).unwrap_or(&0);
                assert_eq!(csr.loop_degree(v), expected_loops);
            }
        }
    }
}
