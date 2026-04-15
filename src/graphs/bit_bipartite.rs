use bit_vec::BitVec;

use crate::graphs::{
    bipartite::{BipartiteVertices, RepartitionVertex, Side},
    colored::{ColoredVertices, FromColoredEndpoints, InsertColoredVertex, VertexColor},
    graph::{
        Directed, EdgeType, Edges, Endpoints, FiniteDirected, FiniteEdges, FiniteVertices,
        FromEndpoints, Graph, InsertEdge, InsertVertex, VertexType, Vertices,
    },
};

/// Bipartite wrapper over a graph with sides stored in a bit vector.
///
/// The inner graph is stored in `graph`.
/// Vertex sides are stored in `colors`, where `false` means left and `true`
/// means right.
///
/// This wrapper assumes dense `usize` vertices so they can index `colors`.
#[derive(Debug, Clone, Default)]
pub struct BitBipartite<G> {
    /// Underlying graph structure.
    graph: G,

    /// One side bit per vertex.
    colors: BitVec,
}

impl<G> BitBipartite<G>
where
    G: Graph + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
{
    /// Creates a bipartite wrapper from a graph and side map.
    ///
    /// Panics if the side map length does not match the vertex count.
    #[inline]
    pub fn new(graph: G, colors: BitVec) -> Self {
        assert_eq!(graph.vertex_store().vertex_count(), colors.len());
        Self { graph, colors }
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
    G: Vertices + VertexType<Vertex = usize>,
{
    type Vertices<'a>
        = G::Vertices<'a>
    where
        Self: 'a;

    /// Returns all vertices.
    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        self.graph.vertices()
    }
}

impl<G> FiniteVertices for BitBipartite<G>
where
    G: FiniteVertices + VertexType<Vertex = usize>,
{
    /// Returns the number of vertices.
    #[inline]
    fn vertex_count(&self) -> usize {
        debug_assert_eq!(
            self.colors.len(),
            self.graph.vertex_count(),
            "BitBipartite invariant violated: colors.len() must equal vertex_count()",
        );
        self.graph.vertex_count()
    }

    /// Returns whether `vertex` exists.
    #[inline]
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        debug_assert_eq!(
            self.colors.len(),
            self.graph.vertex_count(),
            "BitBipartite invariant violated: colors.len() must equal vertex_count()",
        );
        self.graph.contains(vertex)
    }
}

impl<G> Edges for BitBipartite<G>
where
    G: Edges + VertexType<Vertex = usize>,
{
    type Edges<'a>
        = G::Edges<'a>
    where
        Self: 'a;

    /// Returns all edges.
    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        self.graph.edges()
    }
}

impl<G> FiniteEdges for BitBipartite<G>
where
    G: FiniteEdges + VertexType<Vertex = usize>,
{
    /// Returns the number of edges.
    #[inline]
    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl<G> Directed for BitBipartite<G>
where
    G: Directed + VertexType<Vertex = usize>,
{
    type Outgoing<'a>
        = G::Outgoing<'a>
    where
        Self: 'a;

    type Ingoing<'a>
        = G::Ingoing<'a>
    where
        Self: 'a;

    type Connections<'a>
        = G::Connections<'a>
    where
        Self: 'a;

    /// Returns the source of `edge`.
    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.source(edge)
    }

    /// Returns the destination of `edge`.
    #[inline]
    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.target(edge)
    }

    /// Returns all outgoing edges from `source`.
    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        self.graph.outgoing(source)
    }

    /// Returns all incoming edges to `destination`.
    #[inline]
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        self.graph.ingoing(destination)
    }

    /// Returns all edges from `from` to `to`.
    #[inline]
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        self.graph.connections(from, to)
    }
}

impl<G> FiniteDirected for BitBipartite<G>
where
    G: FiniteDirected + VertexType<Vertex = usize>,
{
    /// Returns the outgoing degree of `vertex`.
    #[inline]
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.outgoing_degree(vertex)
    }

    /// Returns the incoming degree of `vertex`.
    #[inline]
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.ingoing_degree(vertex)
    }

    /// Returns the number of loop edges at `vertex`.
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.loop_degree(vertex)
    }
}

impl<G> Graph for BitBipartite<G>
where
    G: Graph + Vertices + Edges + VertexType<Vertex = usize>,
{
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

impl<G> VertexColor for BitBipartite<G>
where
    G: VertexType<Vertex = usize>,
{
    type Color = Side;
}

impl<G> ColoredVertices for BitBipartite<G>
where
    G: FiniteVertices + VertexType<Vertex = usize>,
{
    /// Returns the side of `vertex`, or `None` if it does not exist.
    #[inline]
    fn vertex_color(&self, vertex: Self::Vertex) -> Option<Self::Color> {
        self.colors.get(vertex).map(Side::from)
    }
}

impl<G> InsertColoredVertex for BitBipartite<G>
where
    G: InsertVertex + VertexType<Vertex = usize>,
{
    /// Inserts a new isolated vertex with the given side.
    #[inline]
    fn insert_colored_vertex(&mut self, color: Self::Color) -> Option<Self::Vertex> {
        let vertex = self.graph.insert_vertex()?;
        let bit: bool = color.into();

        debug_assert_eq!(
            vertex,
            self.colors.len(),
            "BitBipartite assumes dense append-only vertex insertion",
        );

        self.colors.push(bit);
        Some(vertex)
    }
}

impl<G> InsertEdge for BitBipartite<G>
where
    G: FiniteVertices + InsertEdge + VertexType<Vertex = usize>,
{
    /// Inserts an edge if its endpoints lie on opposite sides.
    ///
    /// Returns `None` if an endpoint does not exist, both endpoints are on the
    /// same side, or the inner graph rejects the insertion.
    #[inline]
    fn insert_edge(&mut self, from: Self::Vertex, to: Self::Vertex) -> Option<Self::Edge> {
        debug_assert_eq!(
            self.colors.len(),
            self.graph.vertex_count(),
            "BitBipartite invariant violated: colors.len() must equal vertex_count()",
        );

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
    G: FiniteVertices + Directed + VertexType<Vertex = usize>,
{
    /// Moves `vertex` to `side` if all incident edges remain bipartite.
    ///
    /// Returns the previous side on success.
    #[inline]
    fn repartition_vertex(&mut self, vertex: Self::Vertex, side: Side) -> Result<Side, ()> {
        let old_bit = self.colors.get(vertex).ok_or(())?;
        let old_side = Side::from(old_bit);
        let new_bit: bool = side.into();

        if old_bit == new_bit {
            return Ok(old_side);
        }

        for (_, _, target) in self.graph.outgoing(vertex) {
            if self.colors.get(target) == Some(new_bit) {
                return Err(());
            }
        }

        for (source, _, _) in self.graph.ingoing(vertex) {
            if self.colors.get(source) == Some(new_bit) {
                return Err(());
            }
        }

        self.colors.set(vertex, new_bit);
        Ok(old_side)
    }
}

impl<G> FromColoredEndpoints for BitBipartite<G>
where
    G: FromEndpoints + Graph + InsertVertex + VertexType<Vertex = usize>,
    G::Vertices: FiniteVertices<Vertex = usize>,
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

        debug_assert!(
            graph.vertex_store().vertex_count() <= colors.len() || !colors.is_empty(),
            "constructed graph has more vertices than provided colors",
        );

        while graph.vertex_store().vertex_count() < colors.len() {
            assert!(graph.insert_vertex().is_some());
        }

        debug_assert_eq!(
            graph.vertex_store().vertex_count(),
            colors.len(),
            "BitBipartite construction requires exactly one color per vertex",
        );

        let mut bits = BitVec::from_elem(colors.len(), false);
        for (vertex, color) in colors.into_iter().enumerate() {
            bits.set(vertex, bool::from(color));
        }

        debug_assert_eq!(bits.len(), graph.vertex_store().vertex_count());

        Self::new(graph, bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::{
        csr::CSR,
        graph::{Directed, Endpoints, FiniteDirected, FiniteEdges, FiniteVertices},
    };

    use proptest::prelude::*;

    fn make_bipartite(edges: &[(usize, usize)], sides: &[Side]) -> BitBipartite<CSR> {
        BitBipartite::<CSR>::from_endpoints_and_colors(
            edges
                .iter()
                .copied()
                .map(|(from, to)| Endpoints::new(from, to)),
            sides.iter().copied(),
        )
    }

    fn assert_edges_cross_partition(graph: &BitBipartite<CSR>) {
        for (source, _, target) in graph.edges() {
            let source_side = graph.vertex_color(source);
            let target_side = graph.vertex_color(target);

            assert!(source_side.is_some());
            assert!(target_side.is_some());
            assert_ne!(source_side, target_side);
        }
    }

    fn repartition_is_allowed(graph: &BitBipartite<CSR>, vertex: usize, new_side: Side) -> bool {
        let Some(old_side) = graph.vertex_color(vertex) else {
            return false;
        };

        if old_side == new_side {
            return true;
        }

        for (_, _, target) in graph.outgoing(vertex) {
            if graph.vertex_color(target) == Some(new_side) {
                return false;
            }
        }

        for (source, _, _) in graph.ingoing(vertex) {
            if graph.vertex_color(source) == Some(new_side) {
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

        assert_eq!(result, Err(()));
        assert_eq!(graph.vertex_color(0), Some(Side::Left));
        assert_edges_cross_partition(&graph);
    }

    #[test]
    fn repartition_rejects_when_incoming_edge_would_become_invalid() {
        let mut graph = make_bipartite(&[(1, 0)], &[Side::Left, Side::Right]);

        let result = graph.repartition_vertex(0, Side::Right);

        assert_eq!(result, Err(()));
        assert_eq!(graph.vertex_color(0), Some(Side::Left));
        assert_edges_cross_partition(&graph);
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

            for (source, _, target) in graph.edges() {
                prop_assert_ne!(graph.vertex_color(source), graph.vertex_color(target));
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

            for (source, _, target) in graph.edges() {
                prop_assert_ne!(graph.vertex_color(source), graph.vertex_color(target));
            }
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
                prop_assert_eq!(graph.repartition_vertex(0, new_side), Err(()));
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
                prop_assert_eq!(result, Err(()));
                prop_assert_eq!(graph.vertex_color(vertex), Some(old_side));
            }

            for (source, _, target) in graph.edges() {
                prop_assert_ne!(graph.vertex_color(source), graph.vertex_color(target));
            }
        }
    }
}
