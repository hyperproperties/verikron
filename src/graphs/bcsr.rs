use std::ops::Range;

use bit_vec::BitVec;

use crate::graphs::{
    bipartite::{RepartitionVertex, Side},
    colored::{ColoredVertices, InsertColoredVertex, VertexColor},
    csr::{CSR, CsrEdges},
    graph::{Directed, EdgeType, Edges, Graph, InsertVertex, VertexType, Vertices},
};

/// Bipartite graph backed by CSR plus a dense side map.
///
/// Structure is stored in `csr`.
/// The bipartition is stored in `colors`, where:
/// - `false` = [`Side::Left`]
/// - `true` = [`Side::Right`]
///
/// Invariant:
/// every edge connects vertices on opposite sides.
#[derive(Debug)]
pub struct BCSR {
    /// Directed graph structure.
    csr: CSR,

    /// Total bipartition coloring, one bit per vertex.
    colors: BitVec,
}

impl Default for BCSR {
    fn default() -> Self {
        Self {
            csr: CSR::default(),
            colors: BitVec::default(),
        }
    }
}

impl VertexType for BCSR {
    type Vertex = usize;
}

impl EdgeType for BCSR {
    type Edge = usize;
}

impl Vertices for BCSR {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Returns all vertices.
    fn vertices(&self) -> Self::Vertices<'_> {
        self.csr.vertices()
    }

    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize {
        self.csr.vertex_count()
    }
}

impl Edges for BCSR {
    type Edges<'a>
        = CsrEdges<'a>
    where
        Self: 'a;

    /// Returns all edges.
    fn edges(&self) -> Self::Edges<'_> {
        self.csr.edges()
    }

    /// Returns the number of edges.
    fn edge_count(&self) -> usize {
        self.csr.edge_count()
    }
}

impl Directed for BCSR {
    type Outgoing<'a>
        = <CSR as Directed>::Outgoing<'a>
    where
        Self: 'a;

    type Ingoing<'a>
        = <CSR as Directed>::Ingoing<'a>
    where
        Self: 'a;

    type Connections<'a>
        = <CSR as Directed>::Connections<'a>
    where
        Self: 'a;

    /// Returns the source of `edge`.
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.csr.source(edge)
    }

    /// Returns the destination of `edge`.
    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        self.csr.target(edge)
    }

    /// Returns all outgoing edges from `source`.
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        self.csr.outgoing(source)
    }

    /// Returns the outgoing degree of `vertex`.
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.csr.outgoing_degree(vertex)
    }

    /// Returns all incoming edges to `destination`.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        self.csr.ingoing(destination)
    }

    /// Returns the incoming degree of `vertex`.
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.csr.ingoing_degree(vertex)
    }

    /// Returns the number of loop edges at `vertex`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.csr.loop_degree(vertex)
    }

    /// Returns all edges from `from` to `to`.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        self.csr.connections(from, to)
    }
}

impl Graph for BCSR {
    type Vertices = Self;
    type Edges = Self;

    /// Returns the edge store.
    fn edge_store(&self) -> &Self::Edges {
        self
    }

    /// Returns the vertex store.
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

impl VertexColor for BCSR {
    type Color = Side;
}

impl ColoredVertices for BCSR {
    /// Returns the side of `vertex`, or `None` if it does not exist.
    #[inline]
    fn vertex_color(&self, vertex: Self::Vertex) -> Option<Self::Color> {
        self.colors.get(vertex).map(Side::from)
    }
}

impl InsertColoredVertex for BCSR {
    /// Inserts a new isolated vertex with the given side.
    #[inline]
    fn insert_colored_vertex(&mut self, color: Self::Color) -> Option<Self::Vertex> {
        let vertex = self.csr.insert_vertex()?;
        let bit = color.into();

        debug_assert_eq!(vertex, self.colors.len());
        self.colors.push(bit);

        Some(vertex)
    }
}

impl RepartitionVertex for BCSR {
    /// Moves `vertex` to `side` if all incident edges remain bipartite.
    ///
    /// Returns the previous side on success.
    #[inline]
    fn repartition_vertex(&mut self, vertex: Self::Vertex, side: Side) -> Result<Side, ()> {
        let old_bit = self.colors.get(vertex).ok_or(())?;
        let old_side = old_bit.into();
        let new_bit = side.into();

        if old_bit == new_bit {
            return Ok(old_side);
        }

        for (_, _, target) in self.csr.outgoing(vertex) {
            if self.colors.get(target) == Some(new_bit) {
                return Err(());
            }
        }

        for (source, _, _) in self.csr.ingoing(vertex) {
            if self.colors.get(source) == Some(new_bit) {
                return Err(());
            }
        }

        self.colors.set(vertex, new_bit);
        Ok(old_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::{bipartite::BipartiteGraph, graph::Graph};

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
            bits.set(i, side.into());
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
            let s = g.vertex_color(src);
            let t = g.vertex_color(dst);
            assert!(
                s.is_some() && t.is_some() && s != t,
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
        let Some(old_side) = g.vertex_color(vertex) else {
            return false;
        };

        if old_side == new_side {
            return true;
        }

        for (_, _, target) in g.outgoing(vertex) {
            if g.vertex_color(target) == Some(new_side) {
                return false;
            }
        }

        for (source, _, _) in g.ingoing(vertex) {
            if g.vertex_color(source) == Some(new_side) {
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

        assert_eq!(g.vertex_color(0), Some(Side::Left));
        assert_eq!(g.vertex_color(1), Some(Side::Right));
        assert_eq!(g.vertex_color(2), Some(Side::Left));
        assert_eq!(g.vertex_color(3), None);

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
        assert_eq!(g.vertex_color(0), Some(Side::Right));

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

        assert_eq!(g.vertex_color(0), Some(Side::Left));
        assert_eq!(g.vertex_color(1), Some(Side::Right));
        assert_eq!(g.vertex_color(2), Some(Side::Left));

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
        assert_eq!(g.vertex_color(0), Some(Side::Left));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn repartition_rejects_when_outgoing_edge_would_break_bipartiteness() {
        let colors = [Side::Left, Side::Right];
        let edges = [(0, 1)];
        let mut g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.repartition_vertex(0, Side::Right), Err(()));
        assert_eq!(g.vertex_color(0), Some(Side::Left));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn repartition_rejects_when_ingoing_edge_would_break_bipartiteness() {
        let colors = [Side::Left, Side::Right];
        let edges = [(1, 0)];
        let mut g = bcsr_from_parts(&edges, &colors);

        assert_eq!(g.repartition_vertex(0, Side::Right), Err(()));
        assert_eq!(g.vertex_color(0), Some(Side::Left));

        assert_bcsr_invariants(&g);
    }

    #[test]
    fn repartition_accepts_isolated_vertex() {
        let colors = [Side::Left, Side::Right, Side::Left];
        let edges = [(0, 1)];
        let mut g = bcsr_from_parts(&edges, &colors);

        let old = g.repartition_vertex(2, Side::Right).unwrap();
        assert_eq!(old, Side::Left);
        assert_eq!(g.vertex_color(2), Some(Side::Right));

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
                .map(|_| rng.random_bool(0.5).into())
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
                .map(|_| rng.random_bool(0.5).into())
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
                .map(|v| g.vertex_color(v).unwrap())
                .collect();

            let new_side = rng.random_bool(0.5).into();
            let v = g.insert_colored_vertex(new_side).unwrap();

            assert_eq!(v, old_vertex_count);
            assert_eq!(g.vertex_count(), old_vertex_count + 1);
            assert_eq!(g.edge_count(), old_edge_count);
            assert_eq!(g.edges().collect::<Vec<_>>(), old_edges);

            for (i, &side) in old_colors.iter().enumerate() {
                assert_eq!(g.vertex_color(i), Some(side));
            }

            assert_eq!(g.vertex_color(v), Some(new_side));
            assert_bcsr_invariants(&g);
        }
    }

    #[test]
    fn random_repartition_matches_expected_feasibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x4243_5352_5F524550);

        for _ in 0..100 {
            let vertex_count = rng.random_range(0..=12usize);

            let colors: Vec<Side> = (0..vertex_count)
                .map(|_| rng.random_bool(0.5).into())
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
            let new_side = rng.random_bool(0.5).into();

            let expected_ok = repartition_allowed(&g, vertex, new_side);
            let old_side = g.vertex_color(vertex).unwrap();

            let result = g.repartition_vertex(vertex, new_side);

            if expected_ok {
                assert_eq!(result, Ok(old_side));
                assert_eq!(g.vertex_color(vertex), Some(new_side));
            } else {
                assert_eq!(result, Err(()));
                assert_eq!(g.vertex_color(vertex), Some(old_side));
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
                .prop_map(|bits| bits.into_iter().map(bool::into).collect::<Vec<_>>());

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
                prop_assert_eq!(g.vertex_color(v), Some(side));
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
                .map(|v| g.vertex_color(v).unwrap())
                .collect();

            let new_side = new_side_bit.into();
            let v = g.insert_colored_vertex(new_side).unwrap();

            prop_assert_eq!(v, old_vertex_count);
            prop_assert_eq!(g.vertex_count(), old_vertex_count + 1);
            prop_assert_eq!(g.edge_count(), old_edge_count);
            prop_assert_eq!(g.edges().collect::<Vec<_>>(), old_edges);

            for (i, side) in old_colors.iter().copied().enumerate() {
                prop_assert_eq!(g.vertex_color(i), Some(side));
            }

            prop_assert_eq!(g.vertex_color(v), Some(new_side));
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
                prop_assert_eq!(g.repartition_vertex(0, side_bit.into()), Err(()));
                return Ok(());
            }

            let vertex = vertex_hint % g.vertex_count();
            let new_side = side_bit.into();
            let expected_ok = repartition_allowed(&g, vertex, new_side);
            let old_side = g.vertex_color(vertex).unwrap();

            let result = g.repartition_vertex(vertex, new_side);

            if expected_ok {
                prop_assert_eq!(result, Ok(old_side));
                prop_assert_eq!(g.vertex_color(vertex), Some(new_side));
            } else {
                prop_assert_eq!(result, Err(()));
                prop_assert_eq!(g.vertex_color(vertex), Some(old_side));
            }

            assert_bcsr_invariants(&g);
        }
    }
}
