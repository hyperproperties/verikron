use bit_vec::BitVec;

use crate::graphs::{
    bipartite::{BipartiteVertices, RepartitionError, RepartitionVertex, Side},
    colored::{ColoredVertices, FromColoredEndpoints, InsertColoredVertex, VertexColor},
    graph::{Directed, Endpoints, FiniteDirected, FromEndpoints, Graph},
    structure::{
        EdgeType, Edges, FiniteEdges, FiniteVertices, InsertEdge, InsertVertex, Structure,
        VertexType, Vertices,
    },
};

/// Bipartite wrapper over a graph with sides stored in a bit vector.
///
/// The inner graph is stored in `graph`.
/// Vertex sides are stored in `colors`, where `false` means left and `true`
/// means right.
///
/// This wrapper assumes dense `usize` vertices in `0..vertex_count()`.
#[derive(Debug, Clone, Default)]
pub struct BitBipartite<G> {
    /// Underlying graph structure.
    graph: G,

    /// One side bit per vertex.
    colors: BitVec,
}

impl<G> BitBipartite<G>
where
    G: Graph + Directed + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
    G::Edges: FiniteEdges<Vertex = usize, Edge = G::Edge>,
{
    #[must_use]
    #[inline]
    pub fn new(graph: G, colors: BitVec) -> Self {
        assert_eq!(
            graph.vertex_store().vertex_count(),
            colors.len(),
            "BitBipartite requires exactly one side per vertex",
        );

        let mut seen = BitVec::from_elem(colors.len(), false);
        for vertex in graph.vertex_store().vertices() {
            assert!(
                vertex < colors.len(),
                "BitBipartite requires dense vertex ids in 0..vertex_count()",
            );
            assert!(
                !seen.get(vertex).unwrap_or(false),
                "BitBipartite requires unique vertex ids",
            );
            seen.set(vertex, true);
        }

        assert!(
            seen.iter().all(|bit| bit),
            "BitBipartite requires dense vertex ids in 0..vertex_count()",
        );

        for edge in graph.edge_store().edges() {
            let source = graph.source(edge);
            let target = graph.destination(edge);

            assert_ne!(
                colors.get(source),
                colors.get(target),
                "BitBipartite requires every edge to cross the partition",
            );
        }

        Self { graph, colors }
    }
}

impl<G> BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    #[must_use]
    #[inline]
    fn side_bit(&self, vertex: usize) -> Option<bool> {
        self.graph
            .vertex_store()
            .contains(&vertex)
            .then(|| self.colors.get(vertex))
            .flatten()
    }
}

impl<G> VertexType for BitBipartite<G>
where
    G: VertexType<Vertex = usize>,
{
    type Vertex = G::Vertex;
}

impl<G> EdgeType for BitBipartite<G>
where
    G: EdgeType + VertexType<Vertex = usize>,
{
    type Edge = G::Edge;
}

impl<G> Vertices for BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: Vertices<Vertex = usize> + FiniteVertices<Vertex = usize>,
{
    type Vertices<'a>
        = <G::Vertices as Vertices>::Vertices<'a>
    where
        Self: 'a,
        G: 'a;

    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        self.graph.vertex_store().vertices()
    }
}

impl<G> FiniteVertices for BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    #[inline]
    fn vertex_count(&self) -> usize {
        self.graph.vertex_store().vertex_count()
    }

    #[inline]
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        self.graph.vertex_store().contains(vertex)
    }
}

impl<G> Edges for BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
    G::Edges: Edges<Vertex = usize, Edge = G::Edge>,
{
    type Edges<'a>
        = <G::Edges as Edges>::Edges<'a>
    where
        Self: 'a,
        G: 'a;

    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        self.graph.edge_store().edges()
    }
}

impl<G> FiniteEdges for BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
    G::Edges: FiniteEdges<Vertex = usize, Edge = G::Edge>,
{
    #[inline]
    fn edge_count(&self) -> usize {
        self.graph.edge_store().edge_count()
    }

    #[inline]
    fn contains_edge(&self, edge: &Self::Edge) -> bool {
        self.graph.edge_store().contains_edge(edge)
    }
}

impl<G> Structure for BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    type Vertices = Self;
    type Edges = G::Edges;

    #[inline]
    fn edge_store(&self) -> &Self::Edges {
        self.graph.edge_store()
    }

    #[inline]
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

impl<G> Graph for BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
}

impl<G> Directed for BitBipartite<G>
where
    G: Graph + Directed + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    type Outgoing<'a>
        = G::Outgoing<'a>
    where
        Self: 'a,
        G: 'a;

    type Ingoing<'a>
        = G::Ingoing<'a>
    where
        Self: 'a,
        G: 'a;

    type Connections<'a>
        = G::Connections<'a>
    where
        Self: 'a,
        G: 'a;

    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.source(edge)
    }

    #[inline]
    fn destination(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.destination(edge)
    }

    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        self.graph.outgoing(source)
    }

    #[inline]
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        self.graph.ingoing(destination)
    }

    #[inline]
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        self.graph.connections(from, to)
    }
}

impl<G> FiniteDirected for BitBipartite<G>
where
    G: Graph + FiniteDirected + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
    G::Edges: FiniteEdges<Vertex = usize, Edge = G::Edge>,
{
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.loop_degree(vertex)
    }

    #[inline]
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.outgoing_degree(vertex)
    }

    #[inline]
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.ingoing_degree(vertex)
    }

    #[inline]
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.graph.is_connected(from, to)
    }

    #[inline]
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.graph.has_edge(from, edge, to)
    }
}

impl<G> VertexColor for BitBipartite<G>
where
    G: VertexType<Vertex = usize>,
{
    type Color = Side;
}

impl<G> ColoredVertices for BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    #[inline]
    fn vertex_color(&self, vertex: Self::Vertex) -> Option<Self::Color> {
        self.side_bit(vertex).map(Side::from)
    }
}

impl<G> InsertColoredVertex for BitBipartite<G>
where
    G: Graph + InsertVertex + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    /// Inserts a new isolated vertex with the given side.
    ///
    /// This assumes append-only dense vertex insertion.
    #[inline]
    fn insert_colored_vertex(&mut self, color: Self::Color) -> Option<Self::Vertex> {
        let vertex = self.graph.insert_vertex()?;
        let bit: bool = color.into();

        debug_assert_eq!(
            vertex,
            self.colors.len(),
            "BitBipartite assumes dense append-only vertex insertion",
        );

        if vertex != self.colors.len() {
            return None;
        }

        self.colors.push(bit);
        Some(vertex)
    }
}

impl<G> InsertEdge for BitBipartite<G>
where
    G: Graph + InsertEdge + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    /// Inserts an edge if its endpoints lie on opposite sides.
    ///
    /// Returns `None` if an endpoint does not exist, both endpoints are on the
    /// same side, or the inner graph rejects the insertion.
    #[inline]
    fn insert_edge(&mut self, from: Self::Vertex, to: Self::Vertex) -> Option<Self::Edge> {
        if self.same_side(from, to)? {
            return None;
        }

        let edge = self.graph.insert_edge(from, to)?;
        debug_assert_ne!(self.vertex_color(from), self.vertex_color(to));
        Some(edge)
    }
}

impl<G> RepartitionVertex for BitBipartite<G>
where
    G: Graph + Directed + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    /// Moves `vertex` to `side` if all incident edges remain bipartite.
    ///
    /// Returns the previous side on success.
    #[inline]
    fn repartition_vertex(
        &mut self,
        vertex: Self::Vertex,
        side: Side,
    ) -> Result<Side, RepartitionError> {
        let old_bit = self
            .side_bit(vertex)
            .ok_or(RepartitionError::MissingVertex)?;
        let old_side = Side::from(old_bit);
        let new_bit: bool = side.into();

        if old_bit == new_bit {
            return Ok(old_side);
        }

        for (_, _, target) in self.graph.outgoing(vertex) {
            if self.colors.get(target) == Some(new_bit) {
                return Err(RepartitionError::ViolatesBipartiteness);
            }
        }

        for (source, _, _) in self.graph.ingoing(vertex) {
            if self.colors.get(source) == Some(new_bit) {
                return Err(RepartitionError::ViolatesBipartiteness);
            }
        }

        self.colors.set(vertex, new_bit);
        Ok(old_side)
    }
}

impl<G> FromColoredEndpoints for BitBipartite<G>
where
    G: FromEndpoints + Graph + Directed + InsertVertex + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
    G::Edges: FiniteEdges<Vertex = usize, Edge = G::Edge>,
{
    /// Builds a bipartite graph from endpoints and vertex colors.
    ///
    /// Missing isolated vertices are inserted so the graph has one vertex per
    /// color entry.
    #[inline]
    fn from_endpoints_and_colors<E, C>(edges: E, colors: C) -> Self
    where
        Self: Sized,
        E: IntoIterator<Item = Endpoints<Self::Vertex>>,
        C: IntoIterator<Item = Self::Color>,
    {
        let colors: Vec<Side> = colors.into_iter().collect();
        let mut graph = G::from_endpoints(edges);

        assert!(
            graph.vertex_store().vertex_count() <= colors.len(),
            "constructed graph has more vertices than provided colors",
        );

        while graph.vertex_store().vertex_count() < colors.len() {
            assert!(graph.insert_vertex().is_some());
        }

        let mut bits = BitVec::from_elem(colors.len(), false);
        for (vertex, color) in colors.into_iter().enumerate() {
            bits.set(vertex, bool::from(color));
        }

        Self::new(graph, bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        bipartite::{RepartitionError, RepartitionVertex, Side},
        colored::{ColoredVertices, FromColoredEndpoints, InsertColoredVertex},
        graph::{Directed, FiniteDirected},
        mcsr::MCSR,
        structure::{FiniteEdges, FiniteVertices, InsertEdge},
    };
    use proptest::prelude::*;

    fn make_bipartite(edges: &[(usize, usize)], sides: &[Side]) -> BitBipartite<MCSR> {
        BitBipartite::<MCSR>::from_endpoints_and_colors(
            edges
                .iter()
                .copied()
                .map(|(from, to)| Endpoints::new(from, to)),
            sides.iter().copied(),
        )
    }

    fn assert_edges_cross_partition(graph: &BitBipartite<MCSR>) {
        for edge in graph.edges() {
            let from = graph.source(edge);
            let to = graph.destination(edge);
            assert_ne!(graph.vertex_color(from), graph.vertex_color(to));
        }
    }

    fn repartition_is_allowed(graph: &BitBipartite<MCSR>, vertex: usize, new_side: Side) -> bool {
        let Some(old_side) = graph.vertex_color(vertex) else {
            return false;
        };

        if old_side == new_side {
            return true;
        }

        for (_, _, to) in graph.outgoing(vertex) {
            if graph.vertex_color(to) == Some(new_side) {
                return false;
            }
        }

        for (from, _, _) in graph.ingoing(vertex) {
            if graph.vertex_color(from) == Some(new_side) {
                return false;
            }
        }

        true
    }

    fn arbitrary_bipartite_instance() -> impl Strategy<Value = (Vec<Side>, Vec<(usize, usize)>)> {
        (0usize..16).prop_flat_map(|vertex_count| {
            let sides = prop::collection::vec(any::<bool>(), vertex_count)
                .prop_map(|bits| bits.into_iter().map(Side::from).collect::<Vec<_>>());

            let raw_edges = prop::collection::vec(
                (0usize..vertex_count.max(1), 0usize..vertex_count.max(1)),
                0..64,
            );

            (Just(vertex_count), sides, raw_edges).prop_map(|(n, sides, raw_edges)| {
                let edges = if n == 0 {
                    Vec::new()
                } else {
                    raw_edges
                        .into_iter()
                        .filter(|&(u, v)| sides[u] != sides[v])
                        .collect()
                };

                (sides, edges)
            })
        })
    }

    #[test]
    fn vertex_color_returns_expected_side() {
        let graph = make_bipartite(&[(0, 1)], &[Side::Left, Side::Right, Side::Left]);

        assert_eq!(graph.vertex_color(0), Some(Side::Left));
        assert_eq!(graph.vertex_color(1), Some(Side::Right));
        assert_eq!(graph.vertex_color(2), Some(Side::Left));
        assert_eq!(graph.vertex_color(3), None);
    }

    #[test]
    fn insert_colored_vertex_appends_new_vertex_and_side() {
        let mut graph = make_bipartite(&[(0, 1)], &[Side::Left, Side::Right]);

        let vertex = graph.insert_colored_vertex(Side::Left).unwrap();

        assert_eq!(vertex, 2);
        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.vertex_color(2), Some(Side::Left));
        assert_edges_cross_partition(&graph);
    }

    #[test]
    fn insert_edge_accepts_opposite_sides_and_rejects_same_sides() {
        let mut graph = make_bipartite(&[], &[Side::Left, Side::Right, Side::Left]);

        let ok = graph.insert_edge(0, 1);
        let bad = graph.insert_edge(0, 2);
        let missing = graph.insert_edge(0, 9);

        assert!(ok.is_some());
        assert_eq!(bad, None);
        assert_eq!(missing, None);

        assert_eq!(graph.edge_count(), 1);
        assert!(graph.is_connected(0, 1));
        assert!(!graph.is_connected(0, 2));
        assert_edges_cross_partition(&graph);
    }

    #[test]
    fn repartition_accepts_isolated_vertex() {
        let mut graph = make_bipartite(&[(0, 1)], &[Side::Left, Side::Right, Side::Left]);

        let old_side = graph.repartition_vertex(2, Side::Right).unwrap();

        assert_eq!(old_side, Side::Left);
        assert_eq!(graph.vertex_color(2), Some(Side::Right));
        assert_edges_cross_partition(&graph);
    }

    #[test]
    fn repartition_rejects_when_outgoing_edge_would_become_invalid() {
        let mut graph = make_bipartite(&[(0, 1)], &[Side::Left, Side::Right]);

        let result = graph.repartition_vertex(0, Side::Right);

        assert_eq!(result, Err(RepartitionError::ViolatesBipartiteness));
        assert_eq!(graph.vertex_color(0), Some(Side::Left));
        assert_edges_cross_partition(&graph);
    }

    #[test]
    fn repartition_rejects_when_incoming_edge_would_become_invalid() {
        let mut graph = make_bipartite(&[(1, 0)], &[Side::Left, Side::Right]);

        let result = graph.repartition_vertex(0, Side::Right);

        assert_eq!(result, Err(RepartitionError::ViolatesBipartiteness));
        assert_eq!(graph.vertex_color(0), Some(Side::Left));
        assert_edges_cross_partition(&graph);
    }

    #[test]
    fn repartition_rejects_missing_vertex() {
        let mut graph = make_bipartite(&[(0, 1)], &[Side::Left, Side::Right]);

        let result = graph.repartition_vertex(2, Side::Left);

        assert_eq!(result, Err(RepartitionError::MissingVertex));
    }

    #[test]
    fn constructor_rejects_non_bipartite_edges() {
        let result =
            std::panic::catch_unwind(|| make_bipartite(&[(0, 1)], &[Side::Left, Side::Left]));

        assert!(result.is_err());
    }

    #[test]
    fn constructor_rejects_wrong_number_of_colors() {
        let result = std::panic::catch_unwind(|| make_bipartite(&[(0, 1)], &[Side::Left]));

        assert!(result.is_err());
    }

    #[test]
    fn from_endpoints_and_colors_adds_isolated_vertices() {
        let graph = make_bipartite(&[(0, 1)], &[Side::Left, Side::Right, Side::Left]);

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.vertex_color(2), Some(Side::Left));
    }

    #[test]
    fn directed_queries_delegate_to_wrapped_graph() {
        let graph = make_bipartite(
            &[(0, 1), (2, 1), (2, 3)],
            &[Side::Left, Side::Right, Side::Left, Side::Right],
        );

        assert_eq!(graph.vertex_count(), 4);
        assert_eq!(graph.edge_count(), 3);

        assert_eq!(graph.outgoing_degree(0), 1);
        assert_eq!(graph.outgoing_degree(1), 0);
        assert_eq!(graph.outgoing_degree(2), 2);

        assert_eq!(graph.ingoing_degree(0), 0);
        assert_eq!(graph.ingoing_degree(1), 2);
        assert_eq!(graph.ingoing_degree(3), 1);

        assert_eq!(graph.loop_degree(0), 0);
        assert!(graph.is_connected(0, 1));
        assert!(graph.is_connected(2, 1));
        assert!(!graph.is_connected(1, 0));

        assert_edges_cross_partition(&graph);
    }

    proptest! {
        #[test]
        fn prop_constructor_preserves_sides_and_bipartite_edges(
            (sides, edges) in arbitrary_bipartite_instance()
        ) {
            let graph = make_bipartite(&edges, &sides);

            prop_assert_eq!(graph.vertex_count(), sides.len());

            for (vertex, side) in sides.iter().copied().enumerate() {
                prop_assert_eq!(graph.vertex_color(vertex), Some(side));
            }

            for edge in graph.edges() {
                let from = graph.source(edge);
                let to = graph.destination(edge);
                prop_assert_ne!(graph.vertex_color(from), graph.vertex_color(to));
            }
        }

        #[test]
        fn prop_insert_colored_vertex_preserves_existing_structure(
            (sides, edges) in arbitrary_bipartite_instance(),
            new_side_bit in any::<bool>(),
        ) {
            let mut graph = make_bipartite(&edges, &sides);

            let old_vertex_count = graph.vertex_count();
            let old_edge_count = graph.edge_count();
            let old_edges: Vec<_> = graph.edges().collect();
            let old_sides: Vec<_> = (0..old_vertex_count)
                .map(|vertex| graph.vertex_color(vertex).unwrap())
                .collect();

            let new_side = Side::from(new_side_bit);
            let new_vertex = graph.insert_colored_vertex(new_side).unwrap();

            prop_assert_eq!(new_vertex, old_vertex_count);
            prop_assert_eq!(graph.vertex_count(), old_vertex_count + 1);
            prop_assert_eq!(graph.edge_count(), old_edge_count);
            prop_assert_eq!(graph.edges().collect::<Vec<_>>(), old_edges);

            for (vertex, side) in old_sides.into_iter().enumerate() {
                prop_assert_eq!(graph.vertex_color(vertex), Some(side));
            }

            prop_assert_eq!(graph.vertex_color(new_vertex), Some(new_side));
            assert_edges_cross_partition(&graph);
        }

        #[test]
        fn prop_repartition_matches_local_feasibility(
            (sides, edges) in arbitrary_bipartite_instance(),
            vertex_hint in 0usize..32,
            new_side_bit in any::<bool>(),
        ) {
            let mut graph = make_bipartite(&edges, &sides);
            let new_side = Side::from(new_side_bit);

            if graph.vertex_count() == 0 {
                prop_assert_eq!(
                    graph.repartition_vertex(0, new_side),
                    Err(RepartitionError::MissingVertex),
                );
                return Ok(());
            }

            let vertex = vertex_hint % graph.vertex_count();
            let old_side = graph.vertex_color(vertex).unwrap();
            let allowed = repartition_is_allowed(&graph, vertex, new_side);

            let result = graph.repartition_vertex(vertex, new_side);

            if allowed {
                prop_assert_eq!(result, Ok(old_side));
                prop_assert_eq!(graph.vertex_color(vertex), Some(new_side));
            } else {
                prop_assert_eq!(result, Err(RepartitionError::ViolatesBipartiteness));
                prop_assert_eq!(graph.vertex_color(vertex), Some(old_side));
            }

            for edge in graph.edges() {
                let from = graph.source(edge);
                let to = graph.destination(edge);
                prop_assert_ne!(graph.vertex_color(from), graph.vertex_color(to));
            }
        }

        #[test]
        fn prop_insert_edge_matches_partition_rule(
            (sides, edges) in arbitrary_bipartite_instance(),
            from_hint in 0usize..32,
            to_hint in 0usize..32,
        ) {
            let mut graph = make_bipartite(&edges, &sides);

            if graph.vertex_count() == 0 {
                prop_assert_eq!(graph.insert_edge(0, 0), None);
                return Ok(());
            }

            let from = from_hint % graph.vertex_count();
            let to = to_hint % graph.vertex_count();
            let old_edge_count = graph.edge_count();

            let result = graph.insert_edge(from, to);

            if graph.vertex_color(from) != graph.vertex_color(to) {
                prop_assert!(result.is_some());
                prop_assert_eq!(graph.edge_count(), old_edge_count + 1);
            } else {
                prop_assert_eq!(result, None);
                prop_assert_eq!(graph.edge_count(), old_edge_count);
            }

            for edge in graph.edges() {
                let u = graph.source(edge);
                let v = graph.destination(edge);
                prop_assert_ne!(graph.vertex_color(u), graph.vertex_color(v));
            }
        }
    }
}
