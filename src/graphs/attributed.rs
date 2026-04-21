use crate::graphs::{
    graph::{Directed, FiniteDirected, Graph},
    structure::{
        EdgeOf, EdgeType, Edges, FiniteEdges, FiniteVertices, Structure, VertexOf, VertexType,
        Vertices,
    },
};

/// Graph with separately stored vertex and edge properties.
///
/// The graph provides the structure.
/// Vertex and edge properties are stored in separate side stores.
///
/// This keeps graph structure independent from attached data while still
/// allowing the whole bundle to behave as a graph.
///
/// The wrapper itself does not impose a specific property-store API.
/// That keeps it usable with indexed stores, sparse maps, attribute tables,
/// or empty stores.
///
/// Any consistency invariant between the graph and the property stores
/// is the responsibility of the chosen property-store types and constructors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributedGraph<G, VP, EP> {
    graph: G,
    vertex_properties: VP,
    edge_properties: EP,
}

impl<G, VP, EP> AttributedGraph<G, VP, EP> {
    /// Creates an attributed graph from a graph and two property stores.
    #[must_use]
    #[inline]
    pub fn new(graph: G, vertex_properties: VP, edge_properties: EP) -> Self {
        Self {
            graph,
            vertex_properties,
            edge_properties,
        }
    }

    /// Returns the underlying graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// Returns the underlying graph mutably.
    #[must_use]
    #[inline]
    pub fn graph_mut(&mut self) -> &mut G {
        &mut self.graph
    }

    /// Returns the vertex-property store.
    #[must_use]
    #[inline]
    pub fn vertex_properties(&self) -> &VP {
        &self.vertex_properties
    }

    /// Returns the vertex-property store mutably.
    #[must_use]
    #[inline]
    pub fn vertex_properties_mut(&mut self) -> &mut VP {
        &mut self.vertex_properties
    }

    /// Returns the edge-property store.
    #[must_use]
    #[inline]
    pub fn edge_properties(&self) -> &EP {
        &self.edge_properties
    }

    /// Returns the edge-property store mutably.
    #[must_use]
    #[inline]
    pub fn edge_properties_mut(&mut self) -> &mut EP {
        &mut self.edge_properties
    }

    /// Consumes the wrapper and returns its parts.
    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (G, VP, EP) {
        (self.graph, self.vertex_properties, self.edge_properties)
    }
}

impl<G, VP, EP> AttributedGraph<G, VP, EP>
where
    VP: Default,
    EP: Default,
{
    /// Wraps `graph` with default property stores.
    #[must_use]
    #[inline]
    pub fn with_default_properties(graph: G) -> Self {
        Self::new(graph, VP::default(), EP::default())
    }
}

impl<G, VP, EP> Default for AttributedGraph<G, VP, EP>
where
    G: Default,
    VP: Default,
    EP: Default,
{
    #[inline]
    fn default() -> Self {
        Self::new(G::default(), VP::default(), EP::default())
    }
}

impl<G, VP, EP> VertexType for AttributedGraph<G, VP, EP>
where
    G: Structure,
{
    type Vertex = VertexOf<G>;
}

impl<G, VP, EP> EdgeType for AttributedGraph<G, VP, EP>
where
    G: Structure,
{
    type Edge = EdgeOf<G>;
}

impl<G, VP, EP> Vertices for AttributedGraph<G, VP, EP>
where
    G: Structure,
    G::Vertices: Vertices<Vertex = VertexOf<G>>,
{
    type Vertices<'a>
        = <G::Vertices as Vertices>::Vertices<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a;

    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        self.graph.vertex_store().vertices()
    }
}

impl<G, VP, EP> FiniteVertices for AttributedGraph<G, VP, EP>
where
    G: Structure,
    G::Vertices: FiniteVertices<Vertex = VertexOf<G>>,
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

impl<G, VP, EP> Edges for AttributedGraph<G, VP, EP>
where
    G: Structure,
    G::Edges: Edges<Vertex = VertexOf<G>, Edge = EdgeOf<G>>,
{
    type Edges<'a>
        = <G::Edges as Edges>::Edges<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a;

    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        self.graph.edge_store().edges()
    }
}

impl<G, VP, EP> FiniteEdges for AttributedGraph<G, VP, EP>
where
    G: Structure,
    G::Edges: FiniteEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>>,
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

impl<G, VP, EP> Structure for AttributedGraph<G, VP, EP>
where
    G: Structure,
{
    type Vertices = G::Vertices;
    type Edges = G::Edges;

    #[inline]
    fn vertex_store(&self) -> &Self::Vertices {
        self.graph.vertex_store()
    }

    #[inline]
    fn edge_store(&self) -> &Self::Edges {
        self.graph.edge_store()
    }
}

impl<G, VP, EP> Graph for AttributedGraph<G, VP, EP> where G: Graph {}

impl<G, VP, EP> Directed for AttributedGraph<G, VP, EP>
where
    G: Directed,
{
    type Outgoing<'a>
        = G::Outgoing<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a;

    type Incoming<'a>
        = G::Incoming<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a;

    type Connections<'a>
        = G::Connections<'a>
    where
        Self: 'a,
        G: 'a,
        VP: 'a,
        EP: 'a;

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
    fn incoming(&self, destination: Self::Vertex) -> Self::Incoming<'_> {
        self.graph.incoming(destination)
    }

    #[inline]
    fn connections(
        &self,
        source: Self::Vertex,
        destination: Self::Vertex,
    ) -> Self::Connections<'_> {
        self.graph.connections(source, destination)
    }
}

impl<G, VP, EP> FiniteDirected for AttributedGraph<G, VP, EP>
where
    G: FiniteDirected,
    G::Vertices: FiniteVertices<Vertex = VertexOf<G>>,
    G::Edges: FiniteEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>>,
{
    #[inline]
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.outgoing_degree(vertex)
    }

    #[inline]
    fn incoming_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.incoming_degree(vertex)
    }

    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.loop_degree(vertex)
    }

    #[inline]
    fn is_connected(&self, source: Self::Vertex, destination: Self::Vertex) -> bool {
        self.graph.is_connected(source, destination)
    }

    #[inline]
    fn has_edge(&self, source: Self::Vertex, edge: Self::Edge, destination: Self::Vertex) -> bool {
        self.graph.has_edge(source, edge, destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        arc::{Arc, FromArcs},
        csr::CSR,
        mcsr::MCSR,
        structure::{InsertEdge, InsertVertex},
    };
    use proptest::prelude::*;

    fn sample_graph() -> CSR {
        CSR::from_arcs([
            Arc::new(0, 1),
            Arc::new(0, 2),
            Arc::new(2, 1),
            Arc::new(2, 2),
        ])
    }

    #[test]
    fn new_and_accessors_return_stored_parts() {
        let graph = sample_graph();
        let vertex_properties = vec!['a', 'b', 'c'];
        let edge_properties = vec![10u8, 20, 30, 40];

        let attributed = AttributedGraph::new(
            graph.clone(),
            vertex_properties.clone(),
            edge_properties.clone(),
        );

        assert_eq!(attributed.graph(), &graph);
        assert_eq!(attributed.vertex_properties(), &vertex_properties);
        assert_eq!(attributed.edge_properties(), &edge_properties);
    }

    #[test]
    fn property_store_mut_accessors_work() {
        let graph = sample_graph();
        let mut attributed = AttributedGraph::new(graph, vec!['a', 'b', 'c'], vec![1u8, 2, 3, 4]);

        attributed.vertex_properties_mut()[1] = 'x';
        attributed.edge_properties_mut()[2] = 99;

        assert_eq!(attributed.vertex_properties(), &['a', 'x', 'c']);
        assert_eq!(attributed.edge_properties(), &[1, 2, 99, 4]);
    }

    #[test]
    fn graph_mut_allows_mutating_the_wrapped_graph() {
        let mut graph = MCSR::new();
        for _ in 0..3 {
            graph.insert_vertex().unwrap();
        }
        graph.insert_edge(0, 1).unwrap();

        let mut attributed = AttributedGraph::new(graph, vec!["v0", "v1", "v2"], vec!["e0"]);

        let new_vertex = attributed.graph_mut().insert_vertex().unwrap();
        let new_edge = attributed.graph_mut().insert_edge(2, 3).unwrap();

        assert_eq!(new_vertex, 3);
        assert_eq!(new_edge, 1);
        assert_eq!(attributed.vertex_count(), 4);
        assert_eq!(attributed.edge_count(), 2);
        assert_eq!(attributed.vertex_properties(), &["v0", "v1", "v2"]);
        assert_eq!(attributed.edge_properties(), &["e0"]);
    }

    #[test]
    fn with_default_properties_uses_default_property_stores() {
        let graph = sample_graph();
        let attributed = AttributedGraph::<_, Vec<char>, Vec<u8>>::with_default_properties(graph);

        assert!(attributed.vertex_properties().is_empty());
        assert!(attributed.edge_properties().is_empty());
    }

    #[test]
    fn default_uses_defaults_for_all_parts() {
        let attributed = AttributedGraph::<CSR, Vec<char>, Vec<u8>>::default();

        assert_eq!(attributed.vertex_count(), 0);
        assert_eq!(attributed.edge_count(), 0);
        assert!(attributed.vertex_properties().is_empty());
        assert!(attributed.edge_properties().is_empty());
    }

    #[test]
    fn vertex_and_edge_enumeration_delegate_to_wrapped_graph() {
        let graph = sample_graph();
        let attributed = AttributedGraph::new(graph.clone(), (), ());

        assert_eq!(
            attributed.vertices().collect::<Vec<_>>(),
            graph.vertices().collect::<Vec<_>>()
        );
        assert_eq!(
            attributed.edges().collect::<Vec<_>>(),
            graph.edges().collect::<Vec<_>>()
        );
    }

    #[test]
    fn finite_queries_delegate_to_wrapped_graph() {
        let graph = sample_graph();
        let attributed = AttributedGraph::new(graph.clone(), (), ());

        assert_eq!(attributed.vertex_count(), graph.vertex_count());
        assert_eq!(attributed.edge_count(), graph.edge_count());

        for vertex in graph.vertices() {
            assert_eq!(attributed.contains(&vertex), graph.contains(&vertex));
        }

        for edge in graph.edges() {
            assert_eq!(attributed.contains_edge(&edge), graph.contains_edge(&edge));
        }
    }

    #[test]
    fn directed_queries_delegate_to_wrapped_graph() {
        let graph = sample_graph();
        let attributed = AttributedGraph::new(graph.clone(), (), ());

        for edge in graph.edges() {
            assert_eq!(attributed.source(edge), graph.source(edge));
            assert_eq!(attributed.destination(edge), graph.destination(edge));

            let source = graph.source(edge);
            let destination = graph.destination(edge);
            assert!(attributed.has_edge(source, edge, destination));
        }

        for vertex in graph.vertices() {
            assert_eq!(
                attributed.outgoing(vertex).collect::<Vec<_>>(),
                graph.outgoing(vertex).collect::<Vec<_>>()
            );
            assert_eq!(
                attributed.incoming(vertex).collect::<Vec<_>>(),
                graph.incoming(vertex).collect::<Vec<_>>()
            );

            assert_eq!(
                attributed.outgoing_degree(vertex),
                graph.outgoing_degree(vertex)
            );
            assert_eq!(
                attributed.incoming_degree(vertex),
                graph.incoming_degree(vertex)
            );
            assert_eq!(attributed.loop_degree(vertex), graph.loop_degree(vertex));
        }

        for source in graph.vertices() {
            for destination in graph.vertices() {
                assert_eq!(
                    attributed.is_connected(source, destination),
                    graph.is_connected(source, destination)
                );
                assert_eq!(
                    attributed
                        .connections(source, destination)
                        .collect::<Vec<_>>(),
                    graph.connections(source, destination).collect::<Vec<_>>()
                );
            }
        }
    }

    fn arbitrary_edges() -> impl Strategy<Value = Vec<(usize, usize)>> {
        prop::collection::vec((0usize..12, 0usize..12), 0..48)
    }

    proptest! {
        #[test]
        fn prop_wrapper_matches_inner_graph_for_structure(edges in arbitrary_edges()) {
            let graph = CSR::from_arcs(
                edges.iter().copied().map(|(source, destination)| Arc::new(source, destination))
            );
            let attributed = AttributedGraph::new(graph.clone(), vec![0u8; 3], vec![0u8; 5]);

            prop_assert_eq!(
                attributed.vertices().collect::<Vec<_>>(),
                graph.vertices().collect::<Vec<_>>()
            );
            prop_assert_eq!(
                attributed.edges().collect::<Vec<_>>(),
                graph.edges().collect::<Vec<_>>()
            );
            prop_assert_eq!(attributed.vertex_count(), graph.vertex_count());
            prop_assert_eq!(attributed.edge_count(), graph.edge_count());
        }

        #[test]
        fn prop_wrapper_matches_inner_graph_for_directed_queries(edges in arbitrary_edges()) {
            let graph = CSR::from_arcs(
                edges.iter().copied().map(|(source, destination)| Arc::new(source, destination))
            );
            let attributed = AttributedGraph::new(graph.clone(), (), ());

            for edge in graph.edges() {
                prop_assert_eq!(attributed.source(edge), graph.source(edge));
                prop_assert_eq!(attributed.destination(edge), graph.destination(edge));
            }

            for vertex in graph.vertices() {
                prop_assert_eq!(
                    attributed.outgoing(vertex).collect::<Vec<_>>(),
                    graph.outgoing(vertex).collect::<Vec<_>>()
                );
                prop_assert_eq!(
                    attributed.incoming(vertex).collect::<Vec<_>>(),
                    graph.incoming(vertex).collect::<Vec<_>>()
                );
                prop_assert_eq!(attributed.outgoing_degree(vertex), graph.outgoing_degree(vertex));
                prop_assert_eq!(attributed.incoming_degree(vertex), graph.incoming_degree(vertex));
                prop_assert_eq!(attributed.loop_degree(vertex), graph.loop_degree(vertex));
            }

            for source in graph.vertices() {
                for destination in graph.vertices() {
                    prop_assert_eq!(
                        attributed.is_connected(source, destination),
                        graph.is_connected(source, destination)
                    );
                    prop_assert_eq!(
                        attributed.connections(source, destination).collect::<Vec<_>>(),
                        graph.connections(source, destination).collect::<Vec<_>>()
                    );
                }
            }
        }

        #[test]
        fn prop_property_stores_do_not_affect_graph_behavior(
            edges in arbitrary_edges(),
            vertex_props in prop::collection::vec(any::<u8>(), 0..16),
            edge_props in prop::collection::vec(any::<u16>(), 0..16),
        ) {
            let graph = CSR::from_arcs(
                edges.iter().copied().map(|(source, destination)| Arc::new(source, destination))
            );

            let left = AttributedGraph::new(graph.clone(), vertex_props, edge_props);
            let right = AttributedGraph::new(graph.clone(), (), ());

            prop_assert_eq!(left.vertices().collect::<Vec<_>>(), right.vertices().collect::<Vec<_>>());
            prop_assert_eq!(left.edges().collect::<Vec<_>>(), right.edges().collect::<Vec<_>>());

            for edge in graph.edges() {
                prop_assert_eq!(left.source(edge), right.source(edge));
                prop_assert_eq!(left.destination(edge), right.destination(edge));
            }

            for vertex in graph.vertices() {
                prop_assert_eq!(left.outgoing(vertex).collect::<Vec<_>>(), right.outgoing(vertex).collect::<Vec<_>>());
                prop_assert_eq!(left.incoming(vertex).collect::<Vec<_>>(), right.incoming(vertex).collect::<Vec<_>>());
                prop_assert_eq!(left.outgoing_degree(vertex), right.outgoing_degree(vertex));
                prop_assert_eq!(left.incoming_degree(vertex), right.incoming_degree(vertex));
                prop_assert_eq!(left.loop_degree(vertex), right.loop_degree(vertex));
            }
        }
    }
}
