use std::ops::Range;

use crate::graphs::{
    graph::{VertexType, Vertices},
    hyper::{DirectedHypergraph, Hyperedges},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DHCSR {
    vertex_count: usize,
    hyperedge_count: usize,

    // [tail_offsets | head_offsets | out_offsets | in_offsets]
    offsets: Box<[usize]>,

    // [tail_members | head_members | out_hyperedges | in_hyperedges]
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
    pub fn new(hyperedges: Vec<(Vec<usize>, Vec<usize>)>) -> Self {
        Self::from(hyperedges)
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    #[inline]
    pub fn hyperedge_count(&self) -> usize {
        self.hyperedge_count
    }

    #[inline]
    fn tail_offsets(&self) -> &[usize] {
        let len = self.hyperedge_count + 1;
        &self.offsets[..len]
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

    #[inline]
    pub fn tail_range(&self, e: usize) -> Option<(usize, usize)> {
        if e >= self.hyperedge_count {
            return None;
        }
        let range = Self::row_range(self.tail_offsets(), e);
        debug_assert!(range.1 <= self.tail_members().len());
        Some(range)
    }

    #[inline]
    pub fn head_range(&self, e: usize) -> Option<(usize, usize)> {
        if e >= self.hyperedge_count {
            return None;
        }
        let range = Self::row_range(self.head_offsets(), e);
        debug_assert!(range.1 <= self.head_members().len());
        Some(range)
    }

    #[inline]
    pub fn outgoing_range(&self, v: usize) -> Option<(usize, usize)> {
        if v >= self.vertex_count {
            return None;
        }
        let range = Self::row_range(self.out_offsets(), v);
        debug_assert!(range.1 <= self.out_hyperedges().len());
        Some(range)
    }

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

impl From<Vec<(Vec<usize>, Vec<usize>)>> for DHCSR {
    fn from(hyperedges: Vec<(Vec<usize>, Vec<usize>)>) -> Self {
        if hyperedges.is_empty() {
            return Self::default();
        }

        let hyperedge_count = hyperedges.len();

        let vertex_count = hyperedges
            .iter()
            .flat_map(|(tail, head)| tail.iter().chain(head.iter()))
            .copied()
            .max()
            .map_or(0, |v| v + 1);

        let mut tail_offsets = Vec::with_capacity(hyperedge_count + 1);
        let mut head_offsets = Vec::with_capacity(hyperedge_count + 1);
        tail_offsets.push(0);
        head_offsets.push(0);

        let mut tail_members = Vec::new();
        let mut head_members = Vec::new();

        for (tail, head) in &hyperedges {
            tail_members.extend_from_slice(tail);
            head_members.extend_from_slice(head);

            tail_offsets.push(tail_members.len());
            head_offsets.push(head_members.len());
        }

        let mut out_offsets = vec![0usize; vertex_count + 1];
        let mut in_offsets = vec![0usize; vertex_count + 1];

        for (e, (tail, head)) in hyperedges.iter().enumerate() {
            debug_assert!(e < hyperedge_count);

            for &v in tail {
                debug_assert!(v < vertex_count);
                out_offsets[v + 1] += 1;
            }

            for &v in head {
                debug_assert!(v < vertex_count);
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

        for (e, (tail, head)) in hyperedges.iter().enumerate() {
            for &v in tail {
                let pos = out_cursor[v];
                debug_assert!(pos < out_hyperedges.len());
                out_hyperedges[pos] = e;
                out_cursor[v] += 1;
            }

            for &v in head {
                let pos = in_cursor[v];
                debug_assert!(pos < in_hyperedges.len());
                in_hyperedges[pos] = e;
                in_cursor[v] += 1;
            }
        }

        debug_assert_eq!(tail_offsets.len(), hyperedge_count + 1);
        debug_assert_eq!(head_offsets.len(), hyperedge_count + 1);
        debug_assert_eq!(out_offsets.len(), vertex_count + 1);
        debug_assert_eq!(in_offsets.len(), vertex_count + 1);

        debug_assert_eq!(tail_offsets[0], 0);
        debug_assert_eq!(head_offsets[0], 0);
        debug_assert_eq!(out_offsets[0], 0);
        debug_assert_eq!(in_offsets[0], 0);

        debug_assert_eq!(*tail_offsets.last().unwrap(), tail_members.len());
        debug_assert_eq!(*head_offsets.last().unwrap(), head_members.len());
        debug_assert_eq!(*out_offsets.last().unwrap(), out_hyperedges.len());
        debug_assert_eq!(*in_offsets.last().unwrap(), in_hyperedges.len());

        debug_assert_eq!(tail_members.len(), out_hyperedges.len());
        debug_assert_eq!(head_members.len(), in_hyperedges.len());

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

        debug_assert_eq!(
            offsets.len(),
            2 * (hyperedge_count + 1) + 2 * (vertex_count + 1)
        );
        debug_assert!(head_members_start <= out_hyperedges_start);
        debug_assert!(out_hyperedges_start <= in_hyperedges_start);
        debug_assert!(in_hyperedges_start <= indices.len());

        Self {
            vertex_count,
            hyperedge_count,
            offsets: offsets.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
            head_members_start,
            out_hyperedges_start,
            in_hyperedges_start,
        }
    }
}

impl Hyperedges for DHCSR {
    type Vertex = usize;
    type Hyperedge = usize;

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

impl VertexType for DHCSR {
    type Vertex = usize;
}

impl Vertices for DHCSR {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count
    }

    fn vertex_count(&self) -> usize {
        self.vertex_count
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn naive_vertex_count(hyperedges: &[(Vec<usize>, Vec<usize>)]) -> usize {
        hyperedges
            .iter()
            .flat_map(|(tail, head)| tail.iter().chain(head.iter()))
            .copied()
            .max()
            .map_or(0, |v| v + 1)
    }

    fn naive_outgoing(hyperedges: &[(Vec<usize>, Vec<usize>)], v: usize) -> Vec<usize> {
        let mut outgoing = Vec::new();

        for (e, (tail, _)) in hyperedges.iter().enumerate() {
            for &u in tail {
                if u == v {
                    outgoing.push(e);
                }
            }
        }

        outgoing
    }

    fn naive_ingoing(hyperedges: &[(Vec<usize>, Vec<usize>)], v: usize) -> Vec<usize> {
        let mut ingoing = Vec::new();

        for (e, (_, head)) in hyperedges.iter().enumerate() {
            for &u in head {
                if u == v {
                    ingoing.push(e);
                }
            }
        }

        ingoing
    }

    fn assert_graph_matches_input(g: &DHCSR, hyperedges: &[(Vec<usize>, Vec<usize>)]) {
        let vertex_count = naive_vertex_count(hyperedges);

        assert_eq!(g.vertex_count(), vertex_count);
        assert_eq!(g.hyperedge_count(), hyperedges.len());
        assert_eq!(
            g.vertices().collect::<Vec<_>>(),
            (0..vertex_count).collect::<Vec<_>>()
        );
        assert_eq!(
            g.hyperedges().collect::<Vec<_>>(),
            (0..hyperedges.len()).collect::<Vec<_>>()
        );

        for (e, (tail, head)) in hyperedges.iter().enumerate() {
            assert_eq!(
                g.tail(e).collect::<Vec<_>>(),
                *tail,
                "tail mismatch for hyperedge {e}"
            );
            assert_eq!(
                g.head(e).collect::<Vec<_>>(),
                *head,
                "head mismatch for hyperedge {e}"
            );
            assert_eq!(
                g.tail_cardinality(e),
                tail.len(),
                "tail cardinality mismatch for hyperedge {e}"
            );
            assert_eq!(
                g.head_cardinality(e),
                head.len(),
                "head cardinality mismatch for hyperedge {e}"
            );

            for v in 0..vertex_count {
                assert_eq!(
                    g.in_tail(e, v),
                    tail.contains(&v),
                    "in_tail mismatch for hyperedge {e}, vertex {v}"
                );
                assert_eq!(
                    g.in_head(e, v),
                    head.contains(&v),
                    "in_head mismatch for hyperedge {e}, vertex {v}"
                );
            }
        }

        for v in 0..vertex_count {
            let expected_out = naive_outgoing(hyperedges, v);
            let expected_in = naive_ingoing(hyperedges, v);

            assert_eq!(
                g.outgoing(v).collect::<Vec<_>>(),
                expected_out,
                "outgoing mismatch for vertex {v}"
            );
            assert_eq!(
                g.ingoing(v).collect::<Vec<_>>(),
                expected_in,
                "ingoing mismatch for vertex {v}"
            );
            assert_eq!(
                g.outgoing_degree(v),
                expected_out.len(),
                "outgoing degree mismatch for vertex {v}"
            );
            assert_eq!(
                g.ingoing_degree(v),
                expected_in.len(),
                "ingoing degree mismatch for vertex {v}"
            );
        }
    }

    #[test]
    fn empty_graph() {
        let g = DHCSR::default();

        assert_eq!(g.vertex_count(), 0);
        assert_eq!(g.hyperedge_count(), 0);
        assert_eq!(g.vertices().collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(g.hyperedges().collect::<Vec<_>>(), Vec::<usize>::new());

        assert_eq!(g.tail_range(0), None);
        assert_eq!(g.head_range(0), None);
        assert_eq!(g.outgoing_range(0), None);
        assert_eq!(g.ingoing_range(0), None);
    }

    #[test]
    fn single_hyperedge() {
        let input = vec![(vec![0, 2], vec![1, 3])];
        let g = DHCSR::from(input.clone());

        assert_graph_matches_input(&g, &input);
    }

    #[test]
    fn repeated_vertices_inside_hyperedge_are_preserved() {
        let input = vec![(vec![0, 0, 1], vec![2, 2]), (vec![1], vec![1, 1, 1])];
        let g = DHCSR::from(input.clone());

        assert_eq!(g.tail(0).collect::<Vec<_>>(), vec![0, 0, 1]);
        assert_eq!(g.head(0).collect::<Vec<_>>(), vec![2, 2]);
        assert_eq!(g.tail(1).collect::<Vec<_>>(), vec![1]);
        assert_eq!(g.head(1).collect::<Vec<_>>(), vec![1, 1, 1]);

        assert_eq!(g.outgoing(0).collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(g.ingoing(2).collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(g.ingoing(1).collect::<Vec<_>>(), vec![1, 1, 1]);

        assert_graph_matches_input(&g, &input);
    }

    #[test]
    fn isolated_vertices_are_not_representable_through_from() {
        let input = vec![(vec![2], vec![5])];
        let g = DHCSR::from(input);

        assert_eq!(g.vertex_count(), 6);
        assert_eq!(g.vertices().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);

        assert_eq!(g.outgoing(0).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(g.ingoing(0).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(g.outgoing(3).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(g.ingoing(4).collect::<Vec<_>>(), Vec::<usize>::new());
    }

    #[test]
    fn multiple_hyperedges() {
        let input = vec![
            (vec![0, 1], vec![2]),
            (vec![2], vec![1, 3]),
            (vec![1], vec![1]),
            (vec![], vec![0, 2]),
            (vec![3], vec![]),
        ];
        let g = DHCSR::from(input.clone());

        assert_graph_matches_input(&g, &input);
    }

    #[test]
    fn range_methods_match_materialized_rows() {
        let input = vec![
            (vec![0, 1], vec![2]),
            (vec![2], vec![1, 3]),
            (vec![1], vec![1]),
        ];
        let g = DHCSR::from(input);

        for e in 0..g.hyperedge_count() {
            let (ts, te) = g.tail_range(e).unwrap();
            let (hs, he) = g.head_range(e).unwrap();

            assert_eq!(g.tail(e).count(), te - ts);
            assert_eq!(g.head(e).count(), he - hs);
        }

        for v in 0..g.vertex_count() {
            let (os, oe) = g.outgoing_range(v).unwrap();
            let (is, ie) = g.ingoing_range(v).unwrap();

            assert_eq!(g.outgoing(v).count(), oe - os);
            assert_eq!(g.ingoing(v).count(), ie - is);
        }
    }

    #[test]
    fn randomized_against_naive_reference() {
        let mut rng = StdRng::seed_from_u64(0x5EED_BAAD_F00D);

        for _case in 0..500 {
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

                input.push((tail, head));
            }

            let normalized_input = if vertex_bound == 0 { Vec::new() } else { input };

            let g = DHCSR::from(normalized_input.clone());
            assert_graph_matches_input(&g, &normalized_input);
        }
    }

    #[test]
    fn randomized_dense_cases_against_naive_reference() {
        let mut rng = StdRng::seed_from_u64(123);

        for _case in 0..200 {
            let vertex_bound = rng.random_range(1..=8);
            let hyperedge_count = rng.random_range(1..=30);

            let mut input = Vec::with_capacity(hyperedge_count);

            for _ in 0..hyperedge_count {
                let mut tail = Vec::new();
                let mut head = Vec::new();

                for v in 0..vertex_bound {
                    if rng.random_bool(0.5) {
                        tail.push(v);
                    }
                    if rng.random_bool(0.5) {
                        head.push(v);
                    }
                }

                input.push((tail, head));
            }

            let g = DHCSR::from(input.clone());
            assert_graph_matches_input(&g, &input);
        }
    }

    use proptest::collection::vec;
    use proptest::prelude::*;

    prop_compose! {
        fn arb_hyperedge(vertex_bound: usize)
            (tail in vec(0..vertex_bound, 0..8),
             head in vec(0..vertex_bound, 0..8)) -> (Vec<usize>, Vec<usize>) {
            (tail, head)
        }
    }

    prop_compose! {
        fn arb_hypergraph()
            (vertex_bound in 0usize..20, hyperedge_count in 0usize..40)
            (
                edges in if vertex_bound == 0 {
                    Just(Vec::<(Vec<usize>, Vec<usize>)>::new()).boxed()
                } else {
                    vec(arb_hyperedge(vertex_bound), hyperedge_count).boxed()
                }
            ) -> Vec<(Vec<usize>, Vec<usize>)> {
            edges
        }
    }

    proptest! {
        #[test]
        fn prop_matches_naive_reference(input in arb_hypergraph()) {
            let g = DHCSR::from(input.clone());
            assert_graph_matches_input(&g, &input);
        }

        #[test]
        fn prop_tail_and_head_ranges_are_consistent(input in arb_hypergraph()) {
            let g = DHCSR::from(input);

            for e in 0..g.hyperedge_count() {
                let (ts, te) = g.tail_range(e).unwrap();
                let (hs, he) = g.head_range(e).unwrap();

                prop_assert!(ts <= te);
                prop_assert!(hs <= he);
                prop_assert_eq!(te - ts, g.tail(e).count());
                prop_assert_eq!(he - hs, g.head(e).count());
            }
        }

        #[test]
        fn prop_outgoing_and_ingoing_ranges_are_consistent(input in arb_hypergraph()) {
            let g = DHCSR::from(input);

            for v in 0..g.vertex_count() {
                let (os, oe) = g.outgoing_range(v).unwrap();
                let (is, ie) = g.ingoing_range(v).unwrap();

                prop_assert!(os <= oe);
                prop_assert!(is <= ie);
                prop_assert_eq!(oe - os, g.outgoing(v).count());
                prop_assert_eq!(ie - is, g.ingoing(v).count());
            }
        }

        #[test]
        fn prop_out_of_bounds_ranges_return_none(input in arb_hypergraph()) {
            let g = DHCSR::from(input);

            prop_assert_eq!(g.tail_range(g.hyperedge_count()), None);
            prop_assert_eq!(g.head_range(g.hyperedge_count()), None);
            prop_assert_eq!(g.outgoing_range(g.vertex_count()), None);
            prop_assert_eq!(g.ingoing_range(g.vertex_count()), None);
        }

        #[test]
        fn prop_degrees_match_materialized_iterators(input in arb_hypergraph()) {
            let g = DHCSR::from(input);

            for e in 0..g.hyperedge_count() {
                prop_assert_eq!(g.tail_cardinality(e), g.tail(e).count());
                prop_assert_eq!(g.head_cardinality(e), g.head(e).count());
            }

            for v in 0..g.vertex_count() {
                prop_assert_eq!(g.outgoing_degree(v), g.outgoing(v).count());
                prop_assert_eq!(g.ingoing_degree(v), g.ingoing(v).count());
            }
        }
    }
}
