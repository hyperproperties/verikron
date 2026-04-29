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
}

impl<G, EP> AttributedGraph<G, (), EP> {
    /// Creates an attributed graph with no vertex properties.
    #[must_use]
    #[inline]
    pub fn with_edge_properties(graph: G, edge_properties: EP) -> Self {
        Self {
            graph,
            vertex_properties: (),
            edge_properties,
        }
    }
}

impl<G, VP> AttributedGraph<G, VP, ()> {
    /// Creates an attributed graph with no edge properties.
    #[must_use]
    #[inline]
    pub fn with_vertex_properties(graph: G, vertex_properties: VP) -> Self {
        Self {
            graph,
            vertex_properties,
            edge_properties: (),
        }
    }
}

impl<G> AttributedGraph<G, (), ()> {
    /// Creates an attributed graph with no vertex or edge properties.
    #[must_use]
    #[inline]
    pub fn from_graph(graph: G) -> Self {
        Self {
            graph,
            vertex_properties: (),
            edge_properties: (),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        arc::{Arc, FromArcs},
        csr::CSR,
        mcsr::MCSR,
        structure::{InsertEdge, InsertVertex},
    };

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

        assert!(attributed.vertex_properties().is_empty());
        assert!(attributed.edge_properties().is_empty());
    }
}
