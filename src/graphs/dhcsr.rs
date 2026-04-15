use std::ops::Range;

use crate::graphs::{
    graph::{FiniteVertices, VertexType, Vertices},
    hyper::{
        DirectedHyperedge, DirectedHypergraph, FromDirectedHyperedges, HyperedgeType, Hyperedges,
        InfiniteDirectedHypergraph,
    },
};

/// Directed hypergraph in CSR-style form.
///
/// Hyperedges are stored twice: once by hyperedge (`tail`/`head` members) and
/// once by vertex (`outgoing`/`ingoing` incident hyperedges).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DHCSR {
    vertex_count: usize,
    hyperedge_count: usize,

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
    fn default() -> Self {
        Self {
            vertex_count: 0,
            hyperedge_count: 0,
            offsets: vec![0, 0, 0, 0].into_boxed_slice(),
            indices: Box::new([]),
            head_members_start: 0,
            out_hyperedges_start: 0,
            in_hyperedges_start: 0,
        }
    }
}

impl DHCSR {
    /// Creates a graph from directed hyperedges.
    #[inline]
    pub fn new<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = DirectedHyperedge<usize>>,
    {
        Self::from_directed_hyperedges(hyperedges)
    }

    #[inline]
    fn tail_offsets(&self) -> &[usize] {
        &self.offsets[..self.hyperedge_count + 1]
    }

    #[inline]
    fn head_offsets(&self) -> &[usize] {
        let start = self.hyperedge_count + 1;
        let end = start + self.hyperedge_count + 1;
        &self.offsets[start..end]
    }

    #[inline]
    fn out_offsets(&self) -> &[usize] {
        let start = 2 * (self.hyperedge_count + 1);
        let end = start + self.vertex_count + 1;
        &self.offsets[start..end]
    }

    #[inline]
    fn in_offsets(&self) -> &[usize] {
        let start = 2 * (self.hyperedge_count + 1) + (self.vertex_count + 1);
        &self.offsets[start..]
    }

    #[inline]
    fn tail_members(&self) -> &[usize] {
        &self.indices[..self.head_members_start]
    }

    #[inline]
    fn head_members(&self) -> &[usize] {
        &self.indices[self.head_members_start..self.out_hyperedges_start]
    }

    #[inline]
    fn out_hyperedges(&self) -> &[usize] {
        &self.indices[self.out_hyperedges_start..self.in_hyperedges_start]
    }

    #[inline]
    fn in_hyperedges(&self) -> &[usize] {
        &self.indices[self.in_hyperedges_start..]
    }

    #[inline]
    fn row_range(offsets: &[usize], row: usize) -> (usize, usize) {
        let start = offsets[row];
        let end = offsets[row + 1];
        debug_assert!(start <= end);
        (start, end)
    }

    /// Returns the tail member range of hyperedge `e`.
    #[inline]
    pub fn tail_range(&self, e: usize) -> Option<(usize, usize)> {
        if e >= self.hyperedge_count {
            return None;
        }

        let range = Self::row_range(self.tail_offsets(), e);
        debug_assert!(range.1 <= self.tail_members().len());
        Some(range)
    }

    /// Returns the head member range of hyperedge `e`.
    #[inline]
    pub fn head_range(&self, e: usize) -> Option<(usize, usize)> {
        if e >= self.hyperedge_count {
            return None;
        }

        let range = Self::row_range(self.head_offsets(), e);
        debug_assert!(range.1 <= self.head_members().len());
        Some(range)
    }

    /// Returns the outgoing incident range of vertex `v`.
    #[inline]
    pub fn outgoing_range(&self, v: usize) -> Option<(usize, usize)> {
        if v >= self.vertex_count {
            return None;
        }

        let range = Self::row_range(self.out_offsets(), v);
        debug_assert!(range.1 <= self.out_hyperedges().len());
        Some(range)
    }

    /// Returns the ingoing incident range of vertex `v`.
    #[inline]
    pub fn ingoing_range(&self, v: usize) -> Option<(usize, usize)> {
        if v >= self.vertex_count {
            return None;
        }

        let range = Self::row_range(self.in_offsets(), v);
        debug_assert!(range.1 <= self.in_hyperedges().len());
        Some(range)
    }
}

impl FromDirectedHyperedges for DHCSR {
    /// Builds a graph from owned directed hyperedges.
    fn from_directed_hyperedges<I>(hyperedges: I) -> Self
    where
        I: IntoIterator<Item = DirectedHyperedge<Self::Vertex>>,
    {
        let hyperedges: Vec<_> = hyperedges.into_iter().collect();

        if hyperedges.is_empty() {
            return Self::default();
        }

        let hyperedge_count = hyperedges.len();
        let vertex_count = hyperedges
            .iter()
            .flat_map(|edge| edge.tail.iter().chain(edge.head.iter()))
            .copied()
            .max()
            .map_or(0, |v| v + 1);

        let mut tail_offsets = Vec::with_capacity(hyperedge_count + 1);
        let mut head_offsets = Vec::with_capacity(hyperedge_count + 1);
        let mut tail_members = Vec::new();
        let mut head_members = Vec::new();

        tail_offsets.push(0);
        head_offsets.push(0);

        for edge in &hyperedges {
            debug_assert!(edge.tail.iter().all(|&v| v < vertex_count));
            debug_assert!(edge.head.iter().all(|&v| v < vertex_count));

            tail_members.extend_from_slice(&edge.tail);
            head_members.extend_from_slice(&edge.head);

            tail_offsets.push(tail_members.len());
            head_offsets.push(head_members.len());
        }

        let mut out_offsets = vec![0usize; vertex_count + 1];
        let mut in_offsets = vec![0usize; vertex_count + 1];

        for edge in &hyperedges {
            for &v in &edge.tail {
                out_offsets[v + 1] += 1;
            }
            for &v in &edge.head {
                in_offsets[v + 1] += 1;
            }
        }

        for v in 1..=vertex_count {
            out_offsets[v] += out_offsets[v - 1];
            in_offsets[v] += in_offsets[v - 1];
        }

        let mut out_hyperedges = vec![0usize; tail_members.len()];
        let mut in_hyperedges = vec![0usize; head_members.len()];

        let mut out_cursor = out_offsets[..vertex_count].to_vec();
        let mut in_cursor = in_offsets[..vertex_count].to_vec();

        for (e, edge) in hyperedges.iter().enumerate() {
            debug_assert!(e < hyperedge_count);

            for &v in &edge.tail {
                let pos = out_cursor[v];
                debug_assert!(pos < out_hyperedges.len());
                out_hyperedges[pos] = e;
                out_cursor[v] += 1;
            }

            for &v in &edge.head {
                let pos = in_cursor[v];
                debug_assert!(pos < in_hyperedges.len());
                in_hyperedges[pos] = e;
                in_cursor[v] += 1;
            }
        }

        let head_members_start = tail_members.len();
        let out_hyperedges_start = head_members_start + head_members.len();
        let in_hyperedges_start = out_hyperedges_start + out_hyperedges.len();

        let mut offsets = Vec::with_capacity(2 * (hyperedge_count + 1) + 2 * (vertex_count + 1));
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

        let graph = Self {
            vertex_count,
            hyperedge_count,
            offsets: offsets.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
            head_members_start,
            out_hyperedges_start,
            in_hyperedges_start,
        };

        graph
    }
}

impl VertexType for DHCSR {
    type Vertex = usize;
}

impl HyperedgeType for DHCSR {
    type Hyperedge = usize;
}

impl Hyperedges for DHCSR {
    type Hyperedges<'a>
        = Range<usize>
    where
        Self: 'a;

    fn hyperedges(&self) -> Self::Hyperedges<'_> {
        0..self.hyperedge_count
    }

    fn hyperedge_count(&self) -> usize {
        self.hyperedge_count
    }
}

impl Vertices for DHCSR {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count
    }
}

impl FiniteVertices for DHCSR {
    fn vertex_count(&self) -> usize {
        self.vertex_count
    }
}

impl InfiniteDirectedHypergraph for DHCSR {
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

    fn tail(&self, e: Self::Hyperedge) -> Self::Tail<'_> {
        let (start, end) = self.tail_range(e).expect("hyperedge out of bounds");
        self.tail_members()[start..end].iter().copied()
    }

    fn head(&self, e: Self::Hyperedge) -> Self::Head<'_> {
        let (start, end) = self.head_range(e).expect("hyperedge out of bounds");
        self.head_members()[start..end].iter().copied()
    }

    fn outgoing(&self, v: Self::Vertex) -> Self::Outgoing<'_> {
        let (start, end) = self.outgoing_range(v).expect("vertex out of bounds");
        self.out_hyperedges()[start..end].iter().copied()
    }

    fn ingoing(&self, v: Self::Vertex) -> Self::Ingoing<'_> {
        let (start, end) = self.ingoing_range(v).expect("vertex out of bounds");
        self.in_hyperedges()[start..end].iter().copied()
    }
}

impl DirectedHypergraph for DHCSR {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn reference_vertex_count(hyperedges: &[DirectedHyperedge<usize>]) -> usize {
        hyperedges
            .iter()
            .flat_map(|edge| edge.tail.iter().chain(edge.head.iter()))
            .copied()
            .max()
            .map_or(0, |v| v + 1)
    }

    fn reference_outgoing(hyperedges: &[DirectedHyperedge<usize>], vertex: usize) -> Vec<usize> {
        let mut outgoing = Vec::new();

        for (e, edge) in hyperedges.iter().enumerate() {
            for &u in &edge.tail {
                if u == vertex {
                    outgoing.push(e);
                }
            }
        }

        outgoing
    }

    fn reference_ingoing(hyperedges: &[DirectedHyperedge<usize>], vertex: usize) -> Vec<usize> {
        let mut ingoing = Vec::new();

        for (e, edge) in hyperedges.iter().enumerate() {
            for &u in &edge.head {
                if u == vertex {
                    ingoing.push(e);
                }
            }
        }

        ingoing
    }

    fn assert_matches_input(graph: &DHCSR, hyperedges: &[DirectedHyperedge<usize>]) {
        let vertex_count = reference_vertex_count(hyperedges);

        assert_eq!(graph.vertex_count(), vertex_count);
        assert_eq!(graph.hyperedge_count(), hyperedges.len());
        assert_eq!(
            graph.vertices().collect::<Vec<_>>(),
            (0..vertex_count).collect::<Vec<_>>()
        );
        assert_eq!(
            graph.hyperedges().collect::<Vec<_>>(),
            (0..hyperedges.len()).collect::<Vec<_>>()
        );

        for (e, edge) in hyperedges.iter().enumerate() {
            assert_eq!(graph.tail(e).collect::<Vec<_>>(), edge.tail);
            assert_eq!(graph.head(e).collect::<Vec<_>>(), edge.head);
            assert_eq!(graph.tail_cardinality(e), edge.tail.len());
            assert_eq!(graph.head_cardinality(e), edge.head.len());

            for v in 0..vertex_count {
                assert_eq!(graph.in_tail(e, v), edge.tail.contains(&v));
                assert_eq!(graph.in_head(e, v), edge.head.contains(&v));
            }
        }

        for v in 0..vertex_count {
            let outgoing = reference_outgoing(hyperedges, v);
            let ingoing = reference_ingoing(hyperedges, v);

            assert_eq!(graph.outgoing(v).collect::<Vec<_>>(), outgoing);
            assert_eq!(graph.ingoing(v).collect::<Vec<_>>(), ingoing);
            assert_eq!(
                graph.outgoing_degree(v),
                reference_outgoing(hyperedges, v).len()
            );
            assert_eq!(
                graph.ingoing_degree(v),
                reference_ingoing(hyperedges, v).len()
            );
        }
    }

    #[test]
    fn default_is_empty() {
        let graph = DHCSR::default();

        assert_eq!(graph.vertex_count(), 0);
        assert_eq!(graph.hyperedge_count(), 0);
        assert_eq!(graph.vertices().collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(graph.hyperedges().collect::<Vec<_>>(), Vec::<usize>::new());

        assert_eq!(graph.tail_range(0), None);
        assert_eq!(graph.head_range(0), None);
        assert_eq!(graph.outgoing_range(0), None);
        assert_eq!(graph.ingoing_range(0), None);
    }

    #[test]
    fn single_hyperedge_is_stored_correctly() {
        let input = vec![DirectedHyperedge::new([0, 2], [1, 3])];
        let graph = DHCSR::from_directed_hyperedges(input.clone());

        assert_matches_input(&graph, &input);
    }

    #[test]
    fn repeated_vertices_are_preserved() {
        let input = vec![
            DirectedHyperedge::new([0, 0, 1], [2, 2]),
            DirectedHyperedge::new([1], [1, 1, 1]),
        ];
        let graph = DHCSR::from_directed_hyperedges(input.clone());

        assert_eq!(graph.tail(0).collect::<Vec<_>>(), vec![0, 0, 1]);
        assert_eq!(graph.head(0).collect::<Vec<_>>(), vec![2, 2]);
        assert_eq!(graph.tail(1).collect::<Vec<_>>(), vec![1]);
        assert_eq!(graph.head(1).collect::<Vec<_>>(), vec![1, 1, 1]);

        assert_eq!(graph.outgoing(0).collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(graph.ingoing(2).collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(graph.ingoing(1).collect::<Vec<_>>(), vec![1, 1, 1]);

        assert_matches_input(&graph, &input);
    }

    #[test]
    fn sparse_vertex_ids_induce_full_prefix_vertex_set() {
        let input = vec![DirectedHyperedge::new([2], [5])];
        let graph = DHCSR::from_directed_hyperedges(input);

        assert_eq!(graph.vertex_count(), 6);
        assert_eq!(graph.vertices().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);

        assert_eq!(graph.outgoing(0).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(graph.ingoing(0).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(graph.outgoing(3).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(graph.ingoing(4).collect::<Vec<_>>(), Vec::<usize>::new());
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
        let graph = DHCSR::from_directed_hyperedges(input.clone());

        assert_matches_input(&graph, &input);
    }

    #[test]
    fn range_methods_match_materialized_iterators() {
        let graph = DHCSR::from_directed_hyperedges([
            DirectedHyperedge::new([0, 1], [2]),
            DirectedHyperedge::new([2], [1, 3]),
            DirectedHyperedge::new([1], [1]),
        ]);

        for e in 0..graph.hyperedge_count() {
            let (tail_start, tail_end) = graph.tail_range(e).unwrap();
            let (head_start, head_end) = graph.head_range(e).unwrap();

            assert_eq!(graph.tail(e).count(), tail_end - tail_start);
            assert_eq!(graph.head(e).count(), head_end - head_start);
        }

        for v in 0..graph.vertex_count() {
            let (out_start, out_end) = graph.outgoing_range(v).unwrap();
            let (in_start, in_end) = graph.ingoing_range(v).unwrap();

            assert_eq!(graph.outgoing(v).count(), out_end - out_start);
            assert_eq!(graph.ingoing(v).count(), in_end - in_start);
        }
    }

    #[test]
    fn randomized_cases_match_reference() {
        let mut rng = StdRng::seed_from_u64(0x5EED_BAAD_F00D);

        for _ in 0..500 {
            let vertex_bound = rng.random_range(0..=20);
            let hyperedge_count = rng.random_range(0..=40);

            let mut input = Vec::with_capacity(hyperedge_count);

            for _ in 0..hyperedge_count {
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
            let graph = DHCSR::from_directed_hyperedges(input.clone());

            assert_matches_input(&graph, &input);
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
        (0usize..20, 0usize..40).prop_flat_map(|(vertex_bound, hyperedge_count)| {
            if vertex_bound == 0 {
                Just(Vec::new()).boxed()
            } else {
                prop::collection::vec(arb_hyperedge(vertex_bound), hyperedge_count).boxed()
            }
        })
    }

    proptest! {
        #[test]
        fn prop_matches_reference(input in arb_hypergraph()) {
            let graph = DHCSR::from_directed_hyperedges(input.clone());
            assert_matches_input(&graph, &input);
        }

        #[test]
        fn prop_ranges_are_consistent(input in arb_hypergraph()) {
            let graph = DHCSR::from_directed_hyperedges(input);

            for e in 0..graph.hyperedge_count() {
                let (tail_start, tail_end) = graph.tail_range(e).unwrap();
                let (head_start, head_end) = graph.head_range(e).unwrap();

                prop_assert!(tail_start <= tail_end);
                prop_assert!(head_start <= head_end);
                prop_assert_eq!(tail_end - tail_start, graph.tail(e).count());
                prop_assert_eq!(head_end - head_start, graph.head(e).count());
            }

            for v in 0..graph.vertex_count() {
                let (out_start, out_end) = graph.outgoing_range(v).unwrap();
                let (in_start, in_end) = graph.ingoing_range(v).unwrap();

                prop_assert!(out_start <= out_end);
                prop_assert!(in_start <= in_end);
                prop_assert_eq!(out_end - out_start, graph.outgoing(v).count());
                prop_assert_eq!(in_end - in_start, graph.ingoing(v).count());
            }
        }

        #[test]
        fn prop_out_of_bounds_ranges_return_none(input in arb_hypergraph()) {
            let graph = DHCSR::from_directed_hyperedges(input);

            prop_assert_eq!(graph.tail_range(graph.hyperedge_count()), None);
            prop_assert_eq!(graph.head_range(graph.hyperedge_count()), None);
            prop_assert_eq!(graph.outgoing_range(graph.vertex_count()), None);
            prop_assert_eq!(graph.ingoing_range(graph.vertex_count()), None);
        }

        #[test]
        fn prop_degrees_match_iterators(input in arb_hypergraph()) {
            let graph = DHCSR::from_directed_hyperedges(input);

            for e in 0..graph.hyperedge_count() {
                prop_assert_eq!(graph.tail_cardinality(e), graph.tail(e).count());
                prop_assert_eq!(graph.head_cardinality(e), graph.head(e).count());
            }

            for v in 0..graph.vertex_count() {
                prop_assert_eq!(graph.outgoing_degree(v), graph.outgoing(v).count());
                prop_assert_eq!(graph.ingoing_degree(v), graph.ingoing(v).count());
            }
        }
    }
}
