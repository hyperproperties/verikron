use std::ops::Range;

use bit_vec::BitVec;

use crate::graphs::{
    bipartite::{RepartitionVertex, Side},
    colored::{ColoredVertices, InsertColoredVertex, ReadColoredVertices},
    csr::{CSR, CsrEdges},
    graph::{Directed, Edges, InsertVertex, ReadEdges, ReadGraph, ReadVertices, Vertices},
};

/// Bipartite graph backed by a CSR edge store and a dense bit-vector coloring.
///
/// This representation stores the graph structure in an inner [`CSR`], while
/// the bipartition is stored separately in `colors`.
///
/// Each vertex has exactly one side:
/// - `false` means [`Side::Left`]
/// - `true` means [`Side::Right`]
///
/// The key invariant is:
/// every edge connects vertices on opposite sides.
///
/// This type is intended for forward-heavy graph algorithms, where the CSR
/// layout provides efficient outgoing-edge traversal while the bit-vector
/// provides compact O(1) side lookup.
#[derive(Debug)]
pub struct BCSR {
    /// Directed graph structure.
    csr: CSR,

    /// Total bipartition coloring.
    ///
    /// `colors[v]` is the side of vertex `v`.
    /// `false` encodes [`Side::Left`], `true` encodes [`Side::Right`].
    colors: BitVec,
}

impl Default for BCSR {
    /// Empty bipartite CSR graph with no vertices and no edges.
    fn default() -> Self {
        let graph = Self {
            csr: Default::default(),
            colors: Default::default(),
        };

        debug_assert_eq!(graph.vertex_count(), 0);
        debug_assert_eq!(graph.edge_count(), 0);
        debug_assert_eq!(graph.colors.len(), 0);

        graph
    }
}

impl BCSR {
    /// Checks the internal representation invariant in debug builds.
    ///
    /// The number of stored colors must always match the number of vertices
    /// in the underlying CSR graph.
    #[inline]
    fn debug_check_invariant(&self) {
        debug_assert_eq!(
            self.colors.len(),
            self.csr.vertex_count(),
            "BCSR invariant violated: colors.len() = {}, vertex_count = {}",
            self.colors.len(),
            self.csr.vertex_count()
        );
    }
}

impl Edges for BCSR {
    type Vertex = usize;
    type Edge = usize;
}

impl ReadEdges for BCSR {
    type Edges<'a>
        = CsrEdges<'a>
    where
        Self: 'a;

    /// Iterator over all edges in the graph.
    fn edges(&self) -> Self::Edges<'_> {
        self.debug_check_invariant();
        self.csr.edges()
    }

    /// Number of edges.
    ///
    /// Delegated directly to the inner CSR store to avoid counting by iteration.
    fn edge_count(&self) -> usize {
        self.debug_check_invariant();
        self.csr.edge_count()
    }
}

impl Directed for BCSR {
    /// Source vertex of an edge.
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.debug_check_invariant();
        self.csr.source(edge)
    }

    /// Destination vertex of an edge.
    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        self.debug_check_invariant();
        self.csr.target(edge)
    }

    /// Iterator over all edges whose source equals the given vertex.
    fn outgoing(&self, source: Self::Vertex) -> Self::Edges<'_> {
        self.debug_check_invariant();
        self.csr.outgoing(source)
    }

    /// Number of outgoing edges for a vertex.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.debug_check_invariant();
        self.csr.outgoing_degree(vertex)
    }

    /// Iterator over all edges whose destination equals the given vertex.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Edges<'_> {
        self.debug_check_invariant();
        self.csr.ingoing(destination)
    }

    /// Number of incoming edges for a vertex.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.debug_check_invariant();
        self.csr.ingoing_degree(vertex)
    }

    /// Number of loop edges at a vertex.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.debug_check_invariant();
        self.csr.loop_degree(vertex)
    }

    /// Returns an iterator over all edges from `from` to `to`.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Edges<'_> {
        self.debug_check_invariant();
        self.csr.connections(from, to)
    }
}

impl Vertices for BCSR {
    type Vertex = usize;
}

impl ReadVertices for BCSR {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Iterator over all vertices in the graph.
    fn vertices(&self) -> Self::Vertices<'_> {
        self.debug_check_invariant();
        self.csr.vertices()
    }

    /// Number of vertices.
    ///
    /// Delegated directly to the inner CSR store to avoid counting by iteration.
    fn vertex_count(&self) -> usize {
        self.debug_check_invariant();
        self.csr.vertex_count()
    }
}

impl ReadGraph for BCSR {
    type Vertex = usize;
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

impl ColoredVertices for BCSR {
    type Color = Side;
}

impl ReadColoredVertices for BCSR {
    /// Returns the side of the given vertex.
    ///
    /// On success, returns a reference to the side of the vertex.
    /// If `vertex` is out of range, returns `None`.
    #[inline]
    fn vertex_color(&self, vertex: Self::Vertex) -> Option<&Self::Color> {
        self.debug_check_invariant();
        self.colors.get(vertex).map(Side::from_bit_ref)
    }
}

impl InsertColoredVertex for BCSR {
    /// Inserts a new isolated vertex with the given side.
    ///
    /// The vertex is inserted into the underlying CSR structure and its side
    /// is stored in the color bit-vector.
    ///
    /// On success, returns the identifier of the new vertex.
    #[inline]
    fn insert_colored_vertex(&mut self, color: Self::Color) -> Option<Self::Vertex> {
        self.debug_check_invariant();

        let vertex = self.csr.insert_vertex()?;
        let bit = color.as_bit();

        if vertex == self.colors.len() {
            self.colors.push(bit);
        } else if vertex < self.colors.len() {
            self.colors.set(vertex, bit);
        } else {
            self.colors.grow(vertex + 1 - self.colors.len(), false);
            self.colors.set(vertex, bit);
        }

        self.debug_check_invariant();
        Some(vertex)
    }
}

impl RepartitionVertex for BCSR {
    /// Attempts to move an existing vertex to the given side.
    ///
    /// Repartitioning succeeds only if all incident edges continue to connect
    /// vertices on opposite sides after the change.
    ///
    /// On success, returns the old side of the vertex.
    /// Returns `Err(())` if the vertex does not exist or if the change would
    /// violate the bipartite invariant.
    #[inline]
    fn repartition_vertex(&mut self, vertex: Self::Vertex, side: Side) -> Result<Side, ()> {
        self.debug_check_invariant();

        let old_bit = self.colors.get(vertex).ok_or(())?;
        let old_side = Side::from_bit(old_bit);
        let new_bit = side.as_bit();

        if old_bit == new_bit {
            return Ok(old_side);
        }

        // All outgoing neighbors must remain on the opposite side.
        for (_, _, target) in self.csr.outgoing(vertex) {
            if self.colors.get(target) == Some(new_bit) {
                return Err(());
            }
        }

        // All incoming neighbors must remain on the opposite side.
        for (source, _, _) in self.csr.ingoing(vertex) {
            if self.colors.get(source) == Some(new_bit) {
                return Err(());
            }
        }

        self.colors.set(vertex, new_bit);

        self.debug_check_invariant();
        Ok(old_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::{bipartite::ReadBipartiteVertices, graph::ReadGraph};

    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashMap;

    // ============================================================
    // Helpers
    // ============================================================

    /// Build a BCSR from an edge list and a total side assignment.
    ///
    /// Isolated vertices are added to the inner CSR as needed.
    fn bcsr_from_parts(edges: &[(usize, usize)], colors: &[Side]) -> BCSR {
        assert!(
            edges
                .iter()
                .all(|&(u, v)| u < colors.len() && v < colors.len()),
            "edge endpoint out of range for provided color vector"
        );

        let mut csr = CSR::from(edges.to_vec());
        while csr.vertex_count() < colors.len() {
            assert!(csr.insert_vertex().is_some());
        }

        let mut bits = BitVec::from_elem(colors.len(), false);
        for (i, side) in colors.iter().copied().enumerate() {
            bits.set(i, side.as_bit());
        }

        let g = BCSR { csr, colors: bits };
        assert_bcsr_invariants(&g);
        g
    }

    /// Check core BCSR invariants.
    fn assert_bcsr_invariants(g: &BCSR) {
        assert_eq!(
            g.colors.len(),
            g.csr.vertex_count(),
            "colors length must equal vertex count"
        );

        for v in 0..g.vertex_count() {
            assert!(
                g.vertex_color(v).is_some(),
                "every in-range vertex must have a color"
            );
        }

        for (src, _, dst) in g.edges() {
            let s = g.vertex_color(src).copied();
            let t = g.vertex_color(dst).copied();
            assert_eq!(
                s.is_some() && t.is_some() && s != t,
                true,
                "edge ({src}, {dst}) does not cross the bipartition"
            );
        }

        let total_out: usize = (0..g.vertex_count()).map(|v| g.outgoing_degree(v)).sum();
        let total_in: usize = (0..g.vertex_count()).map(|v| g.ingoing_degree(v)).sum();
        assert_eq!(total_out, g.edge_count());
        assert_eq!(total_in, g.edge_count());
    }

    /// Build a multiplicity map for directed edges.
    fn build_pair_counts(edges: &[(usize, usize)]) -> HashMap<(usize, usize), usize> {
        let mut counts = HashMap::new();
        for &(from, to) in edges {
            *counts.entry((from, to)).or_insert(0) += 1;
        }
        counts
    }

    /// Returns whether repartitioning a vertex to `new_side` should be valid.
    fn repartition_allowed(g: &BCSR, vertex: usize, new_side: Side) -> bool {
        let Some(old_side) = g.vertex_color(vertex).copied() else {
            return false;
        };

        if old_side == new_side {
            return true;
        }

        for (_, _, target) in g.outgoing(vertex) {
            if g.vertex_color(target).copied() == Some(new_side) {
                return false;
            }
        }

        for (source, _, _) in g.ingoing(vertex) {
            if g.vertex_color(source).copied() == Some(new_side) {
                return false;
            }
        }

        true
    }

    // ============================================================
    // Unit tests: construction and invariants
    // ============================================================

    #[test]
    fn default_is_empty_and_consistent() {
        let g = BCSR::default();

        assert_eq!(g.vertex_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.size(), 0);
        assert!(g.is_empty());

        assert_eq!(g.vertices().count(), 0);
        assert_eq!(g.edges().count(), 0);

        assert_eq!(g.colors.len(), 0);
        assert!(g.csr.edges().next().is_none());

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn read_colors_and_bipartite_helpers_work() {
        let colors = [Side::Left, Side::Right, Side::Left];
        let edges = [(0, 1), (1, 2)];
        let g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.vertex_color(0).copied(), Some(Side::Left));
        assert_eq!(g.vertex_color(1).copied(), Some(Side::Right));
        assert_eq!(g.vertex_color(2).copied(), Some(Side::Left));
        assert_eq!(g.vertex_color(3).copied(), None);

        assert!(g.is_left(0));
        assert!(g.is_right(1));
        assert_eq!(g.same_side(0, 2), Some(true));
        assert_eq!(g.opposite_sides(0, 1), Some(true));
        assert_eq!(g.left_vertex_count(), 2);
        assert_eq!(g.right_vertex_count(), 1);

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn graph_traits_delegate_consistently() {
        let colors = [Side::Left, Side::Right, Side::Left];
        let edges = [(0, 1), (2, 1)];
        let g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.vertex_count(), 3);
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.size(), 5);
        assert!(!g.is_empty());

        let vertices: Vec<_> = g.vertices().collect();
        assert_eq!(vertices, vec![0, 1, 2]);

        let edge_triples: Vec<_> = g.edges().collect();
        assert_eq!(edge_triples.len(), 2);

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn multigraph_parallel_edges_are_preserved() {
        let colors = [Side::Left, Side::Right];
        let edges = [(0, 1), (0, 1), (0, 1)];
        let g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.edge_count(), 3);
        assert_eq!(g.outgoing_degree(0), 3);
        assert_eq!(g.ingoing_degree(1), 3);
        assert_eq!(g.connections(0, 1).count(), 3);

        assert_bcsr_invariants(&g);
    }

    // ============================================================
    // Unit tests: insertion
    // ============================================================

    #[test]
    fn insert_colored_vertex_on_empty_graph() {
        let mut g = BCSR::default();

        let v = g
            .insert_colored_vertex(Side::Right)
            .expect("insert_colored_vertex should succeed");

        assert_eq!(v, 0);
        assert_eq!(g.vertex_count(), 1);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.vertex_color(0).copied(), Some(Side::Right));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn insert_colored_vertex_preserves_existing_structure_and_colors() {
        let colors = [Side::Left, Side::Right];
        let edges = [(0, 1), (0, 1)];
        let mut g = bcsr_from_parts(&edges, &colors);

        let old_edges: Vec<_> = g.edges().collect();

        let v = g
            .insert_colored_vertex(Side::Left)
            .expect("insert_colored_vertex should succeed");

        assert_eq!(v, 2);
        assert_eq!(g.vertex_count(), 3);
        assert_eq!(g.edge_count(), 2);

        assert_eq!(g.vertex_color(0).copied(), Some(Side::Left));
        assert_eq!(g.vertex_color(1).copied(), Some(Side::Right));
        assert_eq!(g.vertex_color(2).copied(), Some(Side::Left));

        let new_edges: Vec<_> = g.edges().collect();
        assert_eq!(new_edges, old_edges);

        assert_bcsr_invariants(&g);
    }

    // ============================================================
    // Unit tests: repartition
    // ============================================================

    #[test]
    fn repartition_noop_returns_old_side() {
        let colors = [Side::Left, Side::Right];
        let edges = [(0, 1)];
        let mut g = bcsr_from_parts(&edges, &colors);

        let old = g.repartition_vertex(0, Side::Left).unwrap();
        assert_eq!(old, Side::Left);
        assert_eq!(g.vertex_color(0).copied(), Some(Side::Left));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn repartition_rejects_when_outgoing_edge_would_break_bipartiteness() {
        let colors = [Side::Left, Side::Right];
        let edges = [(0, 1)];
        let mut g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.repartition_vertex(0, Side::Right), Err(()));
        assert_eq!(g.vertex_color(0).copied(), Some(Side::Left));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn repartition_rejects_when_ingoing_edge_would_break_bipartiteness() {
        let colors = [Side::Left, Side::Right];
        let edges = [(1, 0)];
        let mut g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.repartition_vertex(0, Side::Right), Err(()));
        assert_eq!(g.vertex_color(0).copied(), Some(Side::Left));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn repartition_accepts_isolated_vertex() {
        let colors = [Side::Left, Side::Right, Side::Left];
        let edges = [(0, 1)];
        let mut g = bcsr_from_parts(&edges, &colors);

        let old = g.repartition_vertex(2, Side::Right).unwrap();
        assert_eq!(old, Side::Left);
        assert_eq!(g.vertex_color(2).copied(), Some(Side::Right));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn repartition_invalid_vertex_fails() {
        let colors = [Side::Left];
        let edges = [];
        let mut g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.repartition_vertex(1, Side::Right), Err(()));
        assert_bcsr_invariants(&g);
    }

    // ============================================================
    // Randomized tests (fixed-seed RNG)
    // ============================================================

    #[test]
    fn random_bcsr_invariants_and_edge_multiset() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4243_5352_5F49_4E56);

        for _ in 0..100 {
            let vertex_count = rng.random_range(0..=16usize);

            let colors: Vec<Side> = (0..vertex_count)
                .map(|_| Side::from_bit(rng.random()))
                .collect();

            let raw_edge_count = rng.random_range(0..=100usize);
            let mut edges = Vec::new();

            for _ in 0..raw_edge_count {
                if vertex_count == 0 {
                    break;
                }

                let from = rng.random_range(0..vertex_count);
                let to = rng.random_range(0..vertex_count);

                if colors[from] != colors[to] {
                    edges.push((from, to));
                }
            }

            let g = bcsr_from_parts(&edges, &colors);

            assert_bcsr_invariants(&g);

            let expected = build_pair_counts(&edges);
            let mut actual = HashMap::new();
            for (src, _, dst) in g.edges() {
                *actual.entry((src, dst)).or_insert(0) += 1;
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn random_insert_colored_vertex_preserves_existing_edges_and_colors() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4243_5352_5F49_4E53);

        for _ in 0..100 {
            let vertex_count = rng.random_range(0..=12usize);

            let colors: Vec<Side> = (0..vertex_count)
                .map(|_| Side::from_bit(rng.random()))
                .collect();

            let raw_edge_count = rng.random_range(0..=64usize);
            let mut edges = Vec::new();

            for _ in 0..raw_edge_count {
                if vertex_count == 0 {
                    break;
                }

                let from = rng.random_range(0..vertex_count);
                let to = rng.random_range(0..vertex_count);

                if colors[from] != colors[to] {
                    edges.push((from, to));
                }
            }

            let mut g = bcsr_from_parts(&edges, &colors);

            let old_vertex_count = g.vertex_count();
            let old_edge_count = g.edge_count();
            let old_edges: Vec<_> = g.edges().collect();
            let old_colors: Vec<_> = (0..g.vertex_count())
                .map(|v| g.vertex_color(v).copied().unwrap())
                .collect();

            let new_side = Side::from_bit(rng.random());
            let v = g.insert_colored_vertex(new_side).unwrap();

            assert_eq!(v, old_vertex_count);
            assert_eq!(g.vertex_count(), old_vertex_count + 1);
            assert_eq!(g.edge_count(), old_edge_count);
            assert_eq!(g.edges().collect::<Vec<_>>(), old_edges);

            for (i, &side) in old_colors.iter().enumerate() {
                assert_eq!(g.vertex_color(i).copied(), Some(side));
            }

            assert_eq!(g.vertex_color(v).copied(), Some(new_side));
            assert_bcsr_invariants(&g);
        }
    }

    #[test]
    fn random_repartition_matches_expected_feasibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4243_5352_5F524550);

        for _ in 0..100 {
            let vertex_count = rng.random_range(0..=12usize);

            let colors: Vec<Side> = (0..vertex_count)
                .map(|_| Side::from_bit(rng.random()))
                .collect();

            let raw_edge_count = rng.random_range(0..=64usize);
            let mut edges = Vec::new();

            for _ in 0..raw_edge_count {
                if vertex_count == 0 {
                    break;
                }

                let from = rng.random_range(0..vertex_count);
                let to = rng.random_range(0..vertex_count);

                if colors[from] != colors[to] {
                    edges.push((from, to));
                }
            }

            let mut g = bcsr_from_parts(&edges, &colors);

            if g.vertex_count() == 0 {
                assert_bcsr_invariants(&g);
                continue;
            }

            let vertex = rng.random_range(0..g.vertex_count());
            let new_side = Side::from_bit(rng.random());

            let expected_ok = repartition_allowed(&g, vertex, new_side);
            let old_side = g.vertex_color(vertex).copied().unwrap();

            let result = g.repartition_vertex(vertex, new_side);

            if expected_ok {
                assert_eq!(result, Ok(old_side));
                assert_eq!(g.vertex_color(vertex).copied(), Some(new_side));
            } else {
                assert_eq!(result, Err(()));
                assert_eq!(g.vertex_color(vertex).copied(), Some(old_side));
            }

            assert_bcsr_invariants(&g);
        }
    }

    // ============================================================
    // Property-based tests (proptest)
    // ============================================================

    fn arb_bcsr_instance() -> impl Strategy<Value = (Vec<Side>, Vec<(usize, usize)>)> {
        (0usize..16).prop_flat_map(|vertex_count| {
            let colors = prop::collection::vec(any::<bool>(), vertex_count)
                .prop_map(|bits| bits.into_iter().map(Side::from_bit).collect::<Vec<_>>());

            let raw_edges = prop::collection::vec(
                (0usize..vertex_count.max(1), 0usize..vertex_count.max(1)),
                0..64,
            );

            (Just(vertex_count), colors, raw_edges).prop_map(|(n, colors, raw_edges)| {
                let edges = if n == 0 {
                    Vec::new()
                } else {
                    raw_edges
                        .into_iter()
                        .filter(|&(u, v)| colors[u] != colors[v])
                        .collect()
                };

                (colors, edges)
            })
        })
    }

    proptest! {
        #[test]
        fn prop_bcsr_basic_invariants((colors, edges) in arb_bcsr_instance()) {
            let g = bcsr_from_parts(&edges, &colors);
            assert_bcsr_invariants(&g);

            prop_assert_eq!(g.vertex_count(), colors.len());
            for (v, side) in colors.iter().copied().enumerate() {
                prop_assert_eq!(g.vertex_color(v).copied(), Some(side));
            }
        }

        #[test]
        fn prop_bcsr_edges_iterator_matches_input_multiset((colors, edges) in arb_bcsr_instance()) {
            let g = bcsr_from_parts(&edges, &colors);

            let expected = build_pair_counts(&edges);
            let mut actual = HashMap::new();

            for (src, _, dst) in g.edges() {
                *actual.entry((src, dst)).or_insert(0) += 1;
            }

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn prop_insert_colored_vertex_preserves_existing_graph(
            (colors, edges) in arb_bcsr_instance(),
            new_side_bit in any::<bool>(),
        ) {
            let mut g = bcsr_from_parts(&edges, &colors);

            let old_vertex_count = g.vertex_count();
            let old_edge_count = g.edge_count();
            let old_edges: Vec<_> = g.edges().collect();
            let old_colors: Vec<_> = (0..g.vertex_count())
                .map(|v| g.vertex_color(v).copied().unwrap())
                .collect();

            let new_side = Side::from_bit(new_side_bit);
            let v = g.insert_colored_vertex(new_side).unwrap();

            prop_assert_eq!(v, old_vertex_count);
            prop_assert_eq!(g.vertex_count(), old_vertex_count + 1);
            prop_assert_eq!(g.edge_count(), old_edge_count);
            prop_assert_eq!(g.edges().collect::<Vec<_>>(), old_edges);

            for (i, side) in old_colors.iter().copied().enumerate() {
                prop_assert_eq!(g.vertex_color(i).copied(), Some(side));
            }

            prop_assert_eq!(g.vertex_color(v).copied(), Some(new_side));
            assert_bcsr_invariants(&g);
        }

        #[test]
        fn prop_repartition_matches_incident_constraints(
            (colors, edges) in arb_bcsr_instance(),
            side_bit in any::<bool>(),
            vertex_hint in 0usize..32,
        ) {
            let mut g = bcsr_from_parts(&edges, &colors);

            if g.vertex_count() == 0 {
                prop_assert_eq!(g.repartition_vertex(0, Side::from_bit(side_bit)), Err(()));
                return Ok(());
            }

            let vertex = vertex_hint % g.vertex_count();
            let new_side = Side::from_bit(side_bit);
            let expected_ok = repartition_allowed(&g, vertex, new_side);
            let old_side = g.vertex_color(vertex).copied().unwrap();

            let result = g.repartition_vertex(vertex, new_side);

            if expected_ok {
                prop_assert_eq!(result, Ok(old_side));
                prop_assert_eq!(g.vertex_color(vertex).copied(), Some(new_side));
            } else {
                prop_assert_eq!(result, Err(()));
                prop_assert_eq!(g.vertex_color(vertex).copied(), Some(old_side));
            }

            assert_bcsr_invariants(&g);
        }
    }
}
