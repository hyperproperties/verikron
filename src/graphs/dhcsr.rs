use std::ops::Range;

use crate::graphs::{
    graph::Graph,
    hyper::{DirectedHyperedge, DirectedHypergraph, FromDirectedHyperedges},
    structure::{
        EdgeType, Edges, FiniteEdges, FiniteVertices, Structure, VertexOf, VertexType, Vertices,
    },
};

/// Directed hypergraph in CSR-style form.
///
/// Each hyperedge is stored twice:
/// - once by edge, as its tail and head members,
/// - once by vertex, as outgoing and ingoing incident edges.
///
/// All identifiers are dense `usize` values:
/// - vertices are `0..vertex_count()`
/// - hyperedges are `0..edge_count()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DHCSR {
    vertex_count: usize,
    edge_count: usize,

    /// Concatenated offset tables:
    /// `[tail_offsets | head_offsets | out_offsets | in_offsets]`.
    offsets: Box<[usize]>,

    /// Concatenated index tables:
    /// `[tail_members | head_members | out_hyperedges | in_hyperedges]`.
    indices: Box<[usize]>,

    head_members_start: usize,
    out_hyperedges_start: usize,
    in_hyperedges_start: usize,
}

impl Default for DHCSR {
    #[inline]
    fn default() -> Self {
        Self {
            vertex_count: 0,
            edge_count: 0,
            offsets: vec![0, 0, 0, 0].into_boxed_slice(),
            indices: Box::new([]),
            head_members_start: 0,
            out_hyperedges_start: 0,
            in_hyperedges_start: 0,
        }
    }
}

impl DHCSR {
    /// Creates a directed hypergraph from owned directed hyperedges.
    #[must_use]
    #[inline]
    pub fn new<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = DirectedHyperedge<usize>>,
    {
        Self::from_directed_hyperedges(hyperedges)
    }

    /// Returns the number of hyperedges.
    #[must_use]
    #[inline]
    pub fn hyperedge_count(&self) -> usize {
        self.edge_count
    }

    #[inline]
    fn debug_assert_invariants(&self) {
        let edge_offsets_len = self.edge_count + 1;
        let vertex_offsets_len = self.vertex_count + 1;

        debug_assert_eq!(
            self.offsets.len(),
            2 * edge_offsets_len + 2 * vertex_offsets_len,
        );

        debug_assert_eq!(self.tail_offsets().len(), edge_offsets_len);
        debug_assert_eq!(self.head_offsets().len(), edge_offsets_len);
        debug_assert_eq!(self.out_offsets().len(), vertex_offsets_len);
        debug_assert_eq!(self.in_offsets().len(), vertex_offsets_len);

        debug_assert_eq!(self.tail_offsets()[0], 0);
        debug_assert_eq!(self.head_offsets()[0], 0);
        debug_assert_eq!(self.out_offsets()[0], 0);
        debug_assert_eq!(self.in_offsets()[0], 0);

        debug_assert!(self.tail_offsets().windows(2).all(|w| w[0] <= w[1]));
        debug_assert!(self.head_offsets().windows(2).all(|w| w[0] <= w[1]));
        debug_assert!(self.out_offsets().windows(2).all(|w| w[0] <= w[1]));
        debug_assert!(self.in_offsets().windows(2).all(|w| w[0] <= w[1]));

        debug_assert_eq!(
            *self.tail_offsets().last().unwrap(),
            self.head_members_start,
        );
        debug_assert_eq!(
            *self.head_offsets().last().unwrap(),
            self.out_hyperedges_start - self.head_members_start,
        );
        debug_assert_eq!(
            *self.out_offsets().last().unwrap(),
            self.in_hyperedges_start - self.out_hyperedges_start,
        );
        debug_assert_eq!(
            *self.in_offsets().last().unwrap(),
            self.indices.len() - self.in_hyperedges_start,
        );

        debug_assert!(
            self.tail_members().iter().all(|&v| v < self.vertex_count),
            "tail member out of range",
        );
        debug_assert!(
            self.head_members().iter().all(|&v| v < self.vertex_count),
            "head member out of range",
        );
        debug_assert!(
            self.out_hyperedges().iter().all(|&e| e < self.edge_count),
            "outgoing incident hyperedge out of range",
        );
        debug_assert!(
            self.in_hyperedges().iter().all(|&e| e < self.edge_count),
            "ingoing incident hyperedge out of range",
        );
    }

    #[must_use]
    #[inline]
    fn tail_offsets(&self) -> &[usize] {
        &self.offsets[..self.edge_count + 1]
    }

    #[must_use]
    #[inline]
    fn head_offsets(&self) -> &[usize] {
        let start = self.edge_count + 1;
        let end = start + self.edge_count + 1;
        &self.offsets[start..end]
    }

    #[must_use]
    #[inline]
    fn out_offsets(&self) -> &[usize] {
        let start = 2 * (self.edge_count + 1);
        let end = start + self.vertex_count + 1;
        &self.offsets[start..end]
    }

    #[must_use]
    #[inline]
    fn in_offsets(&self) -> &[usize] {
        let start = 2 * (self.edge_count + 1) + (self.vertex_count + 1);
        &self.offsets[start..]
    }

    #[must_use]
    #[inline]
    fn tail_members(&self) -> &[usize] {
        &self.indices[..self.head_members_start]
    }

    #[must_use]
    #[inline]
    fn head_members(&self) -> &[usize] {
        &self.indices[self.head_members_start..self.out_hyperedges_start]
    }

    #[must_use]
    #[inline]
    fn out_hyperedges(&self) -> &[usize] {
        &self.indices[self.out_hyperedges_start..self.in_hyperedges_start]
    }

    #[must_use]
    #[inline]
    fn in_hyperedges(&self) -> &[usize] {
        &self.indices[self.in_hyperedges_start..]
    }

    #[must_use]
    #[inline]
    fn row_range(offsets: &[usize], row: usize) -> (usize, usize) {
        let start = offsets[row];
        let end = offsets[row + 1];
        debug_assert!(start <= end);
        (start, end)
    }

    /// Returns the tail-member range of hyperedge `edge`.
    #[must_use]
    #[inline]
    pub fn tail_range(&self, edge: usize) -> Option<(usize, usize)> {
        if edge >= self.edge_count {
            return None;
        }

        let range = Self::row_range(self.tail_offsets(), edge);
        debug_assert!(range.1 <= self.tail_members().len());
        Some(range)
    }

    /// Returns the head-member range of hyperedge `edge`.
    #[must_use]
    #[inline]
    pub fn head_range(&self, edge: usize) -> Option<(usize, usize)> {
        if edge >= self.edge_count {
            return None;
        }

        let range = Self::row_range(self.head_offsets(), edge);
        debug_assert!(range.1 <= self.head_members().len());
        Some(range)
    }

    /// Returns the outgoing-incident range of vertex `vertex`.
    #[must_use]
    #[inline]
    pub fn outgoing_range(&self, vertex: usize) -> Option<(usize, usize)> {
        if vertex >= self.vertex_count {
            return None;
        }

        let range = Self::row_range(self.out_offsets(), vertex);
        debug_assert!(range.1 <= self.out_hyperedges().len());
        Some(range)
    }

    /// Returns the ingoing-incident range of vertex `vertex`.
    #[must_use]
    #[inline]
    pub fn ingoing_range(&self, vertex: usize) -> Option<(usize, usize)> {
        if vertex >= self.vertex_count {
            return None;
        }

        let range = Self::row_range(self.in_offsets(), vertex);
        debug_assert!(range.1 <= self.in_hyperedges().len());
        Some(range)
    }
}

impl FromDirectedHyperedges for DHCSR {
    #[inline]
    fn from_directed_hyperedges<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = DirectedHyperedge<VertexOf<Self>>>,
    {
        let hyperedges: Vec<_> = hyperedges.into_iter().collect();

        if hyperedges.is_empty() {
            return Self::default();
        }

        let edge_count = hyperedges.len();
        let vertex_count = hyperedges
            .iter()
            .flat_map(|edge| edge.tail.iter().chain(edge.head.iter()))
            .copied()
            .max()
            .map_or(0, |vertex| vertex + 1);

        let mut tail_offsets = Vec::with_capacity(edge_count + 1);
        let mut head_offsets = Vec::with_capacity(edge_count + 1);
        let mut tail_members = Vec::new();
        let mut head_members = Vec::new();

        tail_offsets.push(0);
        head_offsets.push(0);

        for edge in &hyperedges {
            tail_members.extend_from_slice(&edge.tail);
            head_members.extend_from_slice(&edge.head);

            tail_offsets.push(tail_members.len());
            head_offsets.push(head_members.len());
        }

        let mut out_offsets = vec![0usize; vertex_count + 1];
        let mut in_offsets = vec![0usize; vertex_count + 1];

        for edge in &hyperedges {
            for &vertex in &edge.tail {
                out_offsets[vertex + 1] += 1;
            }
            for &vertex in &edge.head {
                in_offsets[vertex + 1] += 1;
            }
        }

        for vertex in 1..=vertex_count {
            out_offsets[vertex] += out_offsets[vertex - 1];
            in_offsets[vertex] += in_offsets[vertex - 1];
        }

        let mut out_hyperedges = vec![0usize; tail_members.len()];
        let mut in_hyperedges = vec![0usize; head_members.len()];

        let mut out_cursor = out_offsets[..vertex_count].to_vec();
        let mut in_cursor = in_offsets[..vertex_count].to_vec();

        for (edge, hyperedge) in hyperedges.iter().enumerate() {
            for &vertex in &hyperedge.tail {
                let pos = out_cursor[vertex];
                out_hyperedges[pos] = edge;
                out_cursor[vertex] += 1;
            }

            for &vertex in &hyperedge.head {
                let pos = in_cursor[vertex];
                in_hyperedges[pos] = edge;
                in_cursor[vertex] += 1;
            }
        }

        let head_members_start = tail_members.len();
        let out_hyperedges_start = head_members_start + head_members.len();
        let in_hyperedges_start = out_hyperedges_start + out_hyperedges.len();

        let mut offsets = Vec::with_capacity(2 * (edge_count + 1) + 2 * (vertex_count + 1));
        offsets.extend_from_slice(&tail_offsets);
        offsets.extend_from_slice(&head_offsets);
        offsets.extend_from_slice(&out_offsets);
        offsets.extend_from_slice(&in_offsets);

        let mut indices = Vec::with_capacity(
            tail_members.len() + head_members.len() + out_hyperedges.len() + in_hyperedges.len(),
        );
        indices.extend_from_slice(&tail_members);
        indices.extend_from_slice(&head_members);
        indices.extend_from_slice(&out_hyperedges);
        indices.extend_from_slice(&in_hyperedges);

        let hypergraph = Self {
            vertex_count,
            edge_count,
            offsets: offsets.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
            head_members_start,
            out_hyperedges_start,
            in_hyperedges_start,
        };

        hypergraph.debug_assert_invariants();
        hypergraph
    }
}

impl VertexType for DHCSR {
    type Vertex = usize;
}

impl EdgeType for DHCSR {
    type Edge = usize;
}

impl Vertices for DHCSR {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count
    }
}

impl FiniteVertices for DHCSR {
    #[inline]
    fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    #[inline]
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        *vertex < self.vertex_count
    }
}

impl Edges for DHCSR {
    type Edges<'a>
        = Range<usize>
    where
        Self: 'a;

    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        0..self.edge_count
    }
}

impl FiniteEdges for DHCSR {
    #[inline]
    fn edge_count(&self) -> usize {
        self.edge_count
    }

    #[inline]
    fn contains_edge(&self, edge: &Self::Edge) -> bool {
        *edge < self.edge_count
    }
}

impl Structure for DHCSR {
    type Vertices = Self;
    type Edges = Self;

    #[inline]
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }

    #[inline]
    fn edge_store(&self) -> &Self::Edges {
        self
    }
}

impl Graph for DHCSR {}

impl DirectedHypergraph for DHCSR {
    type Tail<'a>
        = std::iter::Copied<std::slice::Iter<'a, usize>>
    where
        Self: 'a;

    type Head<'a>
        = std::iter::Copied<std::slice::Iter<'a, usize>>
    where
        Self: 'a;

    type Outgoing<'a>
        = std::iter::Copied<std::slice::Iter<'a, usize>>
    where
        Self: 'a;

    type Ingoing<'a>
        = std::iter::Copied<std::slice::Iter<'a, usize>>
    where
        Self: 'a;

    #[inline]
    fn tail(&self, edge: Self::Edge) -> Self::Tail<'_> {
        let (start, end) = self.tail_range(edge).expect("hyperedge out of bounds");
        self.tail_members()[start..end].iter().copied()
    }

    #[inline]
    fn head(&self, edge: Self::Edge) -> Self::Head<'_> {
        let (start, end) = self.head_range(edge).expect("hyperedge out of bounds");
        self.head_members()[start..end].iter().copied()
    }

    #[inline]
    fn outgoing(&self, vertex: Self::Vertex) -> Self::Outgoing<'_> {
        let (start, end) = self.outgoing_range(vertex).expect("vertex out of bounds");
        self.out_hyperedges()[start..end].iter().copied()
    }

    #[inline]
    fn ingoing(&self, vertex: Self::Vertex) -> Self::Ingoing<'_> {
        let (start, end) = self.ingoing_range(vertex).expect("vertex out of bounds");
        self.in_hyperedges()[start..end].iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::graphs::hyper::FiniteDirectedHypergraph;

    use super::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn reference_vertex_count(hyperedges: &[DirectedHyperedge<usize>]) -> usize {
        hyperedges
            .iter()
            .flat_map(|edge| edge.tail.iter().chain(edge.head.iter()))
            .copied()
            .max()
            .map_or(0, |vertex| vertex + 1)
    }

    fn reference_outgoing(hyperedges: &[DirectedHyperedge<usize>], vertex: usize) -> Vec<usize> {
        let mut outgoing = Vec::new();

        for (edge, hyperedge) in hyperedges.iter().enumerate() {
            for &member in &hyperedge.tail {
                if member == vertex {
                    outgoing.push(edge);
                }
            }
        }

        outgoing
    }

    fn reference_ingoing(hyperedges: &[DirectedHyperedge<usize>], vertex: usize) -> Vec<usize> {
        let mut ingoing = Vec::new();

        for (edge, hyperedge) in hyperedges.iter().enumerate() {
            for &member in &hyperedge.head {
                if member == vertex {
                    ingoing.push(edge);
                }
            }
        }

        ingoing
    }

    fn assert_matches_input(hypergraph: &DHCSR, hyperedges: &[DirectedHyperedge<usize>]) {
        let vertex_count = reference_vertex_count(hyperedges);

        assert_eq!(hypergraph.vertex_count(), vertex_count);
        assert_eq!(hypergraph.edge_count(), hyperedges.len());
        assert_eq!(
            hypergraph.vertices().collect::<Vec<_>>(),
            (0..vertex_count).collect::<Vec<_>>(),
        );
        assert_eq!(
            hypergraph.edges().collect::<Vec<_>>(),
            (0..hyperedges.len()).collect::<Vec<_>>(),
        );

        for (edge, hyperedge) in hyperedges.iter().enumerate() {
            assert_eq!(hypergraph.tail(edge).collect::<Vec<_>>(), hyperedge.tail);
            assert_eq!(hypergraph.head(edge).collect::<Vec<_>>(), hyperedge.head);
            assert_eq!(hypergraph.tail_cardinality(edge), hyperedge.tail.len());
            assert_eq!(hypergraph.head_cardinality(edge), hyperedge.head.len());

            for vertex in 0..vertex_count {
                assert_eq!(
                    hypergraph.in_tail(edge, vertex),
                    hyperedge.tail.contains(&vertex),
                );
                assert_eq!(
                    hypergraph.in_head(edge, vertex),
                    hyperedge.head.contains(&vertex),
                );
            }
        }

        for vertex in 0..vertex_count {
            let outgoing = reference_outgoing(hyperedges, vertex);
            let ingoing = reference_ingoing(hyperedges, vertex);

            assert_eq!(hypergraph.outgoing(vertex).collect::<Vec<_>>(), outgoing);
            assert_eq!(hypergraph.ingoing(vertex).collect::<Vec<_>>(), ingoing);
            assert_eq!(hypergraph.outgoing_degree(vertex), outgoing.len());
            assert_eq!(hypergraph.ingoing_degree(vertex), ingoing.len());
        }
    }

    #[test]
    fn default_is_empty() {
        let hypergraph = DHCSR::default();

        assert_eq!(hypergraph.vertex_count(), 0);
        assert_eq!(hypergraph.edge_count(), 0);
        assert_eq!(
            hypergraph.vertices().collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        assert_eq!(hypergraph.edges().collect::<Vec<_>>(), Vec::<usize>::new());

        assert_eq!(hypergraph.tail_range(0), None);
        assert_eq!(hypergraph.head_range(0), None);
        assert_eq!(hypergraph.outgoing_range(0), None);
        assert_eq!(hypergraph.ingoing_range(0), None);
    }

    #[test]
    fn single_hyperedge_is_stored_correctly() {
        let input = vec![DirectedHyperedge::new([0, 2], [1, 3])];
        let hypergraph = DHCSR::from_directed_hyperedges(input.clone());

        assert_matches_input(&hypergraph, &input);
    }

    #[test]
    fn repeated_vertices_are_preserved() {
        let input = vec![
            DirectedHyperedge::new([0, 0, 1], [2, 2]),
            DirectedHyperedge::new([1], [1, 1, 1]),
        ];
        let hypergraph = DHCSR::from_directed_hyperedges(input.clone());

        assert_eq!(hypergraph.tail(0).collect::<Vec<_>>(), vec![0, 0, 1]);
        assert_eq!(hypergraph.head(0).collect::<Vec<_>>(), vec![2, 2]);
        assert_eq!(hypergraph.tail(1).collect::<Vec<_>>(), vec![1]);
        assert_eq!(hypergraph.head(1).collect::<Vec<_>>(), vec![1, 1, 1]);

        assert_eq!(hypergraph.outgoing(0).collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(hypergraph.ingoing(2).collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(hypergraph.ingoing(1).collect::<Vec<_>>(), vec![1, 1, 1]);

        assert_matches_input(&hypergraph, &input);
    }

    #[test]
    fn sparse_vertex_ids_induce_full_prefix_vertex_set() {
        let input = vec![DirectedHyperedge::new([2], [5])];
        let hypergraph = DHCSR::from_directed_hyperedges(input);

        assert_eq!(hypergraph.vertex_count(), 6);
        assert_eq!(
            hypergraph.vertices().collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5],
        );

        assert_eq!(
            hypergraph.outgoing(0).collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        assert_eq!(
            hypergraph.ingoing(0).collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        assert_eq!(
            hypergraph.outgoing(3).collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        assert_eq!(
            hypergraph.ingoing(4).collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn multiple_hyperedges_are_stored_correctly() {
        let input = vec![
            DirectedHyperedge::new([0, 1], [2]),
            DirectedHyperedge::new([2], [1, 3]),
            DirectedHyperedge::new([1], [1]),
            DirectedHyperedge::new([], [0, 2]),
            DirectedHyperedge::new([3], []),
        ];
        let hypergraph = DHCSR::from_directed_hyperedges(input.clone());

        assert_matches_input(&hypergraph, &input);
    }

    #[test]
    fn empty_hyperedges_without_vertices_are_supported() {
        let input = vec![
            DirectedHyperedge::new([], []),
            DirectedHyperedge::new([], []),
            DirectedHyperedge::new([], []),
        ];
        let hypergraph = DHCSR::from_directed_hyperedges(input.clone());

        assert_eq!(hypergraph.vertex_count(), 0);
        assert_eq!(hypergraph.edge_count(), 3);
        assert_eq!(hypergraph.edges().collect::<Vec<_>>(), vec![0, 1, 2]);

        for edge in 0..hypergraph.edge_count() {
            assert_eq!(hypergraph.tail(edge).count(), 0);
            assert_eq!(hypergraph.head(edge).count(), 0);
        }

        assert_matches_input(&hypergraph, &input);
    }

    #[test]
    fn range_methods_match_materialized_iterators() {
        let hypergraph = DHCSR::from_directed_hyperedges([
            DirectedHyperedge::new([0, 1], [2]),
            DirectedHyperedge::new([2], [1, 3]),
            DirectedHyperedge::new([1], [1]),
        ]);

        for edge in 0..hypergraph.edge_count() {
            let (tail_start, tail_end) = hypergraph.tail_range(edge).unwrap();
            let (head_start, head_end) = hypergraph.head_range(edge).unwrap();

            assert_eq!(hypergraph.tail(edge).count(), tail_end - tail_start);
            assert_eq!(hypergraph.head(edge).count(), head_end - head_start);
        }

        for vertex in 0..hypergraph.vertex_count() {
            let (out_start, out_end) = hypergraph.outgoing_range(vertex).unwrap();
            let (in_start, in_end) = hypergraph.ingoing_range(vertex).unwrap();

            assert_eq!(hypergraph.outgoing(vertex).count(), out_end - out_start);
            assert_eq!(hypergraph.ingoing(vertex).count(), in_end - in_start);
        }
    }

    #[test]
    fn randomized_cases_match_reference() {
        let mut rng = StdRng::seed_from_u64(0x5EED_BAAD_F00D);

        for _ in 0..500 {
            let vertex_bound = rng.random_range(0..=20);
            let edge_count = rng.random_range(0..=40);

            let mut input = Vec::with_capacity(edge_count);

            for _ in 0..edge_count {
                let tail_len = rng.random_range(0..=8);
                let head_len = rng.random_range(0..=8);

                let tail = (0..tail_len)
                    .map(|_| {
                        if vertex_bound == 0 {
                            0
                        } else {
                            rng.random_range(0..vertex_bound)
                        }
                    })
                    .collect::<Vec<_>>();

                let head = (0..head_len)
                    .map(|_| {
                        if vertex_bound == 0 {
                            0
                        } else {
                            rng.random_range(0..vertex_bound)
                        }
                    })
                    .collect::<Vec<_>>();

                input.push(DirectedHyperedge::new(tail, head));
            }

            let input = if vertex_bound == 0 { Vec::new() } else { input };
            let hypergraph = DHCSR::from_directed_hyperedges(input.clone());

            assert_matches_input(&hypergraph, &input);
        }
    }

    fn arb_hyperedge(vertex_bound: usize) -> impl Strategy<Value = DirectedHyperedge<usize>> {
        (
            prop::collection::vec(0..vertex_bound, 0..8),
            prop::collection::vec(0..vertex_bound, 0..8),
        )
            .prop_map(|(tail, head)| DirectedHyperedge::new(tail, head))
    }

    fn arb_hypergraph() -> impl Strategy<Value = Vec<DirectedHyperedge<usize>>> {
        (0usize..20, 0usize..40).prop_flat_map(|(vertex_bound, edge_count)| {
            if vertex_bound == 0 {
                prop::collection::vec(Just(DirectedHyperedge::new(vec![], vec![])), edge_count)
                    .boxed()
            } else {
                prop::collection::vec(arb_hyperedge(vertex_bound), edge_count).boxed()
            }
        })
    }

    proptest! {
        #[test]
        fn prop_matches_reference(input in arb_hypergraph()) {
            let hypergraph = DHCSR::from_directed_hyperedges(input.clone());
            assert_matches_input(&hypergraph, &input);
        }

        #[test]
        fn prop_ranges_are_consistent(input in arb_hypergraph()) {
            let hypergraph = DHCSR::from_directed_hyperedges(input);

            for edge in 0..hypergraph.edge_count() {
                let (tail_start, tail_end) = hypergraph.tail_range(edge).unwrap();
                let (head_start, head_end) = hypergraph.head_range(edge).unwrap();

                prop_assert!(tail_start <= tail_end);
                prop_assert!(head_start <= head_end);
                prop_assert_eq!(tail_end - tail_start, hypergraph.tail(edge).count());
                prop_assert_eq!(head_end - head_start, hypergraph.head(edge).count());
            }

            for vertex in 0..hypergraph.vertex_count() {
                let (out_start, out_end) = hypergraph.outgoing_range(vertex).unwrap();
                let (in_start, in_end) = hypergraph.ingoing_range(vertex).unwrap();

                prop_assert!(out_start <= out_end);
                prop_assert!(in_start <= in_end);
                prop_assert_eq!(out_end - out_start, hypergraph.outgoing(vertex).count());
                prop_assert_eq!(in_end - in_start, hypergraph.ingoing(vertex).count());
            }
        }

        #[test]
        fn prop_out_of_bounds_ranges_return_none(input in arb_hypergraph()) {
            let hypergraph = DHCSR::from_directed_hyperedges(input);

            prop_assert_eq!(hypergraph.tail_range(hypergraph.edge_count()), None);
            prop_assert_eq!(hypergraph.head_range(hypergraph.edge_count()), None);
            prop_assert_eq!(hypergraph.outgoing_range(hypergraph.vertex_count()), None);
            prop_assert_eq!(hypergraph.ingoing_range(hypergraph.vertex_count()), None);
        }

        #[test]
        fn prop_degrees_match_iterators(input in arb_hypergraph()) {
            let hypergraph = DHCSR::from_directed_hyperedges(input);

            for edge in 0..hypergraph.edge_count() {
                prop_assert_eq!(hypergraph.tail_cardinality(edge), hypergraph.tail(edge).count());
                prop_assert_eq!(hypergraph.head_cardinality(edge), hypergraph.head(edge).count());
            }

            for vertex in 0..hypergraph.vertex_count() {
                prop_assert_eq!(hypergraph.outgoing_degree(vertex), hypergraph.outgoing(vertex).count());
                prop_assert_eq!(hypergraph.ingoing_degree(vertex), hypergraph.ingoing(vertex).count());
            }
        }
    }
}
