use std::{ops::Range, slice};

use crate::graphs::{
    dbm::{DBM, DbmEdgeIds, DbmEdges},
    graph::{Directed, Endpoints, FiniteDirected, FromEndpoints, Graph, IndexedDirected},
    properties::{Properties, PropertyStoreType},
    quotient::{Quotient, QuotientType},
    scc::SCC,
    structure::{
        EdgeType, Edges, FiniteEdges, FiniteVertices, InsertVertex, Structure, VertexType, Vertices,
    },
};

/// Dense SCC quotient on vertices `0..vertex_count()`.
///
/// `classes[v]` is the SCC id of `v`.
/// `members` stores all original vertices grouped by SCC, so each SCC occupies
/// one contiguous range.
///
/// `graph` is the SCC quotient graph:
/// - `c -> d` exists iff some original edge goes from SCC `c` to SCC `d`,
/// - `c -> c` exists iff SCC `c` is recurrent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SCCQuotient {
    /// SCC id of each original vertex.
    classes: Box<[usize]>,

    /// Original vertices grouped contiguously by SCC id.
    members: Box<[usize]>,

    /// Quotient graph on SCC ids, with recurrence encoded as self-loops.
    graph: DBM,
}

impl SCCQuotient {
    /// Creates a dense SCC quotient.
    ///
    /// The caller must uphold the [`SCCQuotient`] invariants.
    #[must_use]
    pub fn new(classes: Box<[usize]>, members: Box<[usize]>, graph: DBM) -> Self {
        Self {
            classes,
            members,
            graph,
        }
    }

    /// Computes the dense SCC quotient of `graph` by iterative Tarjan.
    ///
    /// Vertices are assumed to be dense `usize` values in `0..vertex_count()`.
    /// SCC ids are assigned in pop order. The quotient graph contains edges
    /// between distinct SCCs and a self-loop on each recurrent SCC.
    #[must_use]
    #[inline]
    pub fn tarjan<G>(graph: &G) -> Self
    where
        G: IndexedDirected<Vertex = usize>,
        <G as Structure>::Vertices: FiniteVertices,
    {
        const UNDISCOVERED: usize = 0;
        const UNASSIGNED: usize = usize::MAX;

        let vertex_count = graph.vertex_store().vertex_count();

        // DFS stack of active vertices.
        let mut frontier = Vec::<usize>::with_capacity(vertex_count);

        // Tarjan stack of discovered but not yet assigned vertices.
        let mut active = Vec::<usize>::with_capacity(vertex_count);

        // Per-vertex resume position in the outgoing neighborhood.
        let mut next_out: Box<[usize]> = vec![0; vertex_count].into_boxed_slice();

        // Tarjan discovery times and lowlinks.
        // Discovery times start at 1 so 0 can mean "undiscovered".
        let mut discoveries: Box<[usize]> = vec![UNDISCOVERED; vertex_count].into_boxed_slice();
        let mut lowlinks: Box<[usize]> = vec![0; vertex_count].into_boxed_slice();

        // Vertex-to-SCC map and grouped SCC members.
        let mut classes: Box<[usize]> = vec![UNASSIGNED; vertex_count].into_boxed_slice();
        let mut members: Box<[usize]> = vec![0; vertex_count].into_boxed_slice();

        // Temporary recurrence flags; later encoded as quotient self-loops.
        let mut recurrent: Vec<bool> = Vec::new();

        let mut next_discovery = 1usize;
        let mut next_member = 0usize;

        // Start one DFS from each not-yet-discovered vertex.
        for root in graph.vertex_store().vertices() {
            if discoveries[root] != UNDISCOVERED {
                continue;
            }

            // Discover the root and seed both stacks.
            discoveries[root] = next_discovery;
            lowlinks[root] = next_discovery;
            next_discovery += 1;

            frontier.push(root);
            active.push(root);

            while let Some(&vertex) = frontier.last() {
                let next = next_out[vertex];

                // Explore the next outgoing neighbor of `vertex`.
                if let Some(successor) = graph.outgoing_at(vertex, next) {
                    next_out[vertex] += 1;

                    if discoveries[successor] == UNDISCOVERED {
                        // Tree edge: discover `successor` and descend.
                        discoveries[successor] = next_discovery;
                        lowlinks[successor] = next_discovery;
                        next_discovery += 1;

                        frontier.push(successor);
                        active.push(successor);
                    } else if classes[successor] == UNASSIGNED {
                        // Back edge into the active DFS region:
                        // lowlink uses the discovery time of the target.
                        lowlinks[vertex] = lowlinks[vertex].min(discoveries[successor]);
                    }

                    continue;
                }

                // All outgoing neighbors of `vertex` have been processed.
                frontier.pop();

                // If `vertex` is an SCC root, pop exactly that SCC.
                if lowlinks[vertex] == discoveries[vertex] {
                    let component = recurrent.len();
                    let start = next_member;

                    loop {
                        let current = active.pop().expect("Tarjan stack underflow");
                        classes[current] = component;
                        members[next_member] = current;
                        next_member += 1;

                        if current == vertex {
                            break;
                        }
                    }

                    // A finite SCC is recurrent iff it has size > 1 or a self-loop.
                    let size = next_member - start;
                    recurrent.push(size > 1 || graph.has_loop(vertex));
                }

                // Propagate the completed lowlink to the DFS parent, if any.
                if let Some(&parent) = frontier.last() {
                    lowlinks[parent] = lowlinks[parent].min(lowlinks[vertex]);
                }
            }
        }

        let component_count = recurrent.len();

        // Build the quotient graph:
        // - inter-component edges come from original edges,
        // - recurrence is encoded by quotient self-loops.
        let mut quotient = DBM::from_endpoints(
            (0..vertex_count)
                .flat_map(|from| {
                    let classes_ref = &classes;
                    (0..graph.outgoing_count(from)).filter_map(move |i| {
                        let to = graph.outgoing_at(from, i).unwrap();
                        let from_class = classes_ref[from];
                        let to_class = classes_ref[to];

                        (from_class != to_class).then(|| Endpoints::new(from_class, to_class))
                    })
                })
                .chain(
                    (0..component_count)
                        .filter(|&class| recurrent[class])
                        .map(|class| Endpoints::new(class, class)),
                ),
        );

        // `from_endpoints` infers vertices from present edges, so isolated SCCs
        // would disappear unless they are added explicitly.
        while quotient.vertex_count() < component_count {
            assert!(quotient.insert_vertex().is_some());
        }

        Self::new(classes, members, quotient)
    }

    /// Returns the number of original vertices.
    #[must_use]
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.classes.len()
    }

    /// Returns the number of SCCs.
    #[must_use]
    #[inline]
    pub fn component_count(&self) -> usize {
        self.graph.vertex_count()
    }

    /// Returns the SCC quotient graph.
    #[must_use]
    #[inline]
    pub fn graph(&self) -> &DBM {
        &self.graph
    }

    /// Returns the contiguous range of `members` belonging to `class`.
    #[must_use]
    #[inline]
    pub fn members_range(&self, class: usize) -> Range<usize> {
        let start = self
            .members
            .partition_point(|&vertex| self.classes[vertex] < class);

        let end = self
            .members
            .partition_point(|&vertex| self.classes[vertex] <= class);

        start..end
    }

    /// Returns the original vertices in `class`.
    #[must_use]
    #[inline]
    pub fn members_slice(&self, class: usize) -> &[usize] {
        let range = self.members_range(class);
        &self.members[range]
    }
}

impl VertexType for SCCQuotient {
    type Vertex = usize;
}

impl PropertyStoreType for SCCQuotient {
    type Key = usize;
    type Property = usize;
}

impl Properties for SCCQuotient {
    /// Returns the SCC id of `key`.
    #[inline]
    fn property(&self, key: Self::Key) -> Option<&Self::Property> {
        self.classes.get(key)
    }
}

impl QuotientType for SCCQuotient {
    type Class = usize;
}

impl Quotient for SCCQuotient {
    type Classes<'a>
        = Range<usize>
    where
        Self: 'a;

    type Members<'a>
        = std::iter::Copied<slice::Iter<'a, usize>>
    where
        Self: 'a;

    /// Returns all SCC ids.
    #[inline]
    fn classes(&self) -> Self::Classes<'_> {
        0..self.component_count()
    }

    /// Returns the SCC id of `vertex`.
    #[inline]
    fn class(&self, vertex: Self::Vertex) -> Self::Class {
        self.classes[vertex]
    }

    /// Returns the original vertices in `class`.
    #[inline]
    fn members(&self, class: Self::Class) -> Self::Members<'_> {
        self.members_slice(class).iter().copied()
    }
}

impl SCC for SCCQuotient {
    /// Returns an arbitrary representative of `class`.
    #[inline]
    fn representative(&self, class: Self::Class) -> Self::Vertex {
        self.members_slice(class)[0]
    }

    /// Returns whether `class` is recurrent.
    #[inline]
    fn is_recurrent(&self, class: Self::Class) -> bool {
        self.graph.is_connected(class, class)
    }
}

impl Vertices for SCCQuotient {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Returns all SCC ids as quotient vertices.
    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        self.graph.vertices()
    }
}

impl FiniteVertices for SCCQuotient {
    /// Returns the number of SCCs.
    #[inline]
    fn vertex_count(&self) -> usize {
        self.graph.vertex_count()
    }

    /// Returns whether `vertex` is a valid SCC id.
    #[inline]
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        self.graph.contains(vertex)
    }
}

impl EdgeType for SCCQuotient {
    type Edge = usize;
}

impl Edges for SCCQuotient {
    type Edges<'a>
        = DbmEdgeIds<'a>
    where
        Self: 'a;

    /// Returns all quotient-edge ids.
    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        self.graph.edges()
    }
}

impl FiniteEdges for SCCQuotient {
    /// Returns the number of quotient edges.
    #[inline]
    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns whether `edge` is a valid quotient-edge id.
    #[inline]
    fn contains_edge(&self, edge: &Self::Edge) -> bool {
        self.graph.contains_edge(edge)
    }
}

impl Structure for SCCQuotient {
    type Vertices = Self;
    type Edges = Self;

    /// Returns the quotient edge store.
    #[inline]
    fn edge_store(&self) -> &Self::Edges {
        self
    }

    /// Returns the quotient vertex store.
    #[inline]
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

impl Graph for SCCQuotient {}

impl Directed for SCCQuotient {
    type Outgoing<'a>
        = DbmEdges<'a>
    where
        Self: 'a;

    type Ingoing<'a>
        = DbmEdges<'a>
    where
        Self: 'a;

    type Connections<'a>
        = DbmEdges<'a>
    where
        Self: 'a;

    /// Returns the source SCC of `edge`.
    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.source(edge)
    }

    /// Returns the destination SCC of `edge`.
    #[inline]
    fn destination(&self, edge: Self::Edge) -> Self::Vertex {
        self.graph.destination(edge)
    }

    /// Returns all quotient edges outgoing from `source`.
    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        self.graph.outgoing(source)
    }

    /// Returns all quotient edges ingoing to `destination`.
    #[inline]
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        self.graph.ingoing(destination)
    }

    /// Returns all quotient edges from `from` to `to`.
    #[inline]
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        self.graph.connections(from, to)
    }
}

impl FiniteDirected for SCCQuotient {
    /// Returns the number of quotient self-loops at `vertex`.
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.loop_degree(vertex)
    }

    /// Returns the number of outgoing quotient edges from `vertex`.
    #[inline]
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.outgoing(vertex).count()
    }

    /// Returns the number of ingoing quotient edges to `vertex`.
    #[inline]
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        self.graph.ingoing(vertex).count()
    }

    /// Returns whether there is a quotient edge from `from` to `to`.
    #[inline]
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.graph.connections(from, to).next().is_some()
    }

    /// Returns whether `edge` is a quotient edge from `from` to `to`.
    #[inline]
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.graph.connections(from, to).any(|(_, e, _)| e == edge)
    }
}

impl IndexedDirected for SCCQuotient {
    /// Returns the number of outgoing neighboring SCCs of `vertex`.
    #[inline]
    fn outgoing_count(&self, vertex: Self::Vertex) -> usize {
        self.graph.outgoing_count(vertex)
    }

    /// Returns the `index`th outgoing neighboring SCC of `vertex`.
    #[inline]
    fn outgoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex> {
        self.graph.outgoing_at(vertex, index)
    }

    /// Returns the number of ingoing neighboring SCCs of `vertex`.
    #[inline]
    fn ingoing_count(&self, vertex: Self::Vertex) -> usize {
        self.graph.ingoing_count(vertex)
    }

    /// Returns the `index`th ingoing neighboring SCC of `vertex`.
    #[inline]
    fn ingoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex> {
        self.graph.ingoing_at(vertex, index)
    }

    /// Returns whether `vertex` has a quotient self-loop.
    #[inline]
    fn has_loop(&self, vertex: Self::Vertex) -> bool {
        self.graph.has_loop(vertex)
    }

    /// Returns whether `from` has `to` as an outgoing neighboring SCC.
    #[inline]
    fn connects_to(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.graph.connects_to(from, to)
    }

    /// Returns the outgoing neighboring SCCs of `vertex`.
    #[inline]
    fn outgoing_neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        self.graph.outgoing_neighbors(vertex)
    }

    /// Returns the ingoing neighboring SCCs of `vertex`.
    #[inline]
    fn ingoing_neighbors(&self, vertex: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_
    where
        Self: Sized,
    {
        self.graph.ingoing_neighbors(vertex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        dbm::DBM,
        graph::{Endpoints, FiniteDirected, FromEndpoints},
        properties::Properties,
        quotient::{FiniteQuotient, Quotient},
        scc::SCC,
        structure::InsertVertex,
    };

    use proptest::prelude::*;

    fn sample_dscc() -> SCCQuotient {
        // SCCs:
        // 0 -> {1, 4}
        // 1 -> {0, 2}
        // 2 -> {3}
        //
        // Quotient edges:
        // 1 -> 2
        // 0 and 1 are recurrent, 2 is not.
        SCCQuotient::new(
            vec![1, 0, 1, 2, 0].into_boxed_slice(),
            vec![1, 4, 0, 2, 3].into_boxed_slice(),
            DBM::from_endpoints([
                Endpoints::new(0, 0),
                Endpoints::new(1, 1),
                Endpoints::new(1, 2),
            ]),
        )
    }

    #[test]
    fn accessors_return_expected_sizes() {
        let dscc = sample_dscc();

        assert_eq!(dscc.vertex_count(), 5);
        assert_eq!(dscc.component_count(), 3);
    }

    #[test]
    fn properties_expose_vertex_classes() {
        let dscc = sample_dscc();

        assert_eq!(dscc.property(0), Some(&1));
        assert_eq!(dscc.property(1), Some(&0));
        assert_eq!(dscc.property(2), Some(&1));
        assert_eq!(dscc.property(3), Some(&2));
        assert_eq!(dscc.property(4), Some(&0));
        assert_eq!(dscc.property(5), None);
    }

    #[test]
    fn quotient_classes_are_dense() {
        let dscc = sample_dscc();

        assert_eq!(dscc.classes().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(dscc.class_count(), 3);
        assert!(dscc.contains_class(&0));
        assert!(dscc.contains_class(&1));
        assert!(dscc.contains_class(&2));
        assert!(!dscc.contains_class(&3));
    }

    #[test]
    fn class_lookup_matches_expected_partition() {
        let dscc = sample_dscc();

        assert_eq!(dscc.class(0), 1);
        assert_eq!(dscc.class(1), 0);
        assert_eq!(dscc.class(2), 1);
        assert_eq!(dscc.class(3), 2);
        assert_eq!(dscc.class(4), 0);
    }

    #[test]
    fn members_ranges_are_correct() {
        let dscc = sample_dscc();

        assert_eq!(dscc.members_range(0), 0..2);
        assert_eq!(dscc.members_range(1), 2..4);
        assert_eq!(dscc.members_range(2), 4..5);
    }

    #[test]
    fn members_slices_are_correct() {
        let dscc = sample_dscc();

        assert_eq!(dscc.members_slice(0), &[1, 4]);
        assert_eq!(dscc.members_slice(1), &[0, 2]);
        assert_eq!(dscc.members_slice(2), &[3]);
    }

    #[test]
    fn quotient_members_iterator_matches_members_slice() {
        let dscc = sample_dscc();

        assert_eq!(dscc.members(0).collect::<Vec<_>>(), vec![1, 4]);
        assert_eq!(dscc.members(1).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(dscc.members(2).collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn representative_is_first_member_of_each_class() {
        let dscc = sample_dscc();

        assert_eq!(dscc.representative(0), 1);
        assert_eq!(dscc.representative(1), 0);
        assert_eq!(dscc.representative(2), 3);
    }

    #[test]
    fn recurrence_queries_match_quotient_self_loops() {
        let dscc = sample_dscc();

        assert!(dscc.is_recurrent(0));
        assert!(dscc.is_recurrent(1));
        assert!(!dscc.is_recurrent(2));

        assert!(!dscc.is_trivial(0));
        assert!(!dscc.is_trivial(1));
        assert!(dscc.is_trivial(2));
    }

    #[test]
    fn strong_connectivity_matches_class_equality() {
        let dscc = sample_dscc();

        assert!(dscc.are_strongly_connected(0, 2));
        assert!(dscc.are_strongly_connected(1, 4));
        assert!(dscc.are_strongly_connected(3, 3));

        assert!(!dscc.are_strongly_connected(0, 1));
        assert!(!dscc.are_strongly_connected(2, 3));
        assert!(!dscc.are_strongly_connected(4, 3));
    }

    #[test]
    fn empty_dscc_is_valid() {
        let dscc = SCCQuotient::new(
            Vec::<usize>::new().into_boxed_slice(),
            Vec::<usize>::new().into_boxed_slice(),
            DBM::default(),
        );

        assert_eq!(dscc.vertex_count(), 0);
        assert_eq!(dscc.component_count(), 0);
        assert_eq!(dscc.classes().collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(dscc.property(0), None);
    }

    #[test]
    fn singleton_component_is_supported() {
        let dscc = SCCQuotient::new(
            vec![0].into_boxed_slice(),
            vec![0].into_boxed_slice(),
            DBM::new(1, false),
        );

        assert_eq!(dscc.vertex_count(), 1);
        assert_eq!(dscc.component_count(), 1);
        assert_eq!(dscc.class(0), 0);
        assert_eq!(dscc.members_slice(0), &[0]);
        assert_eq!(dscc.representative(0), 0);
        assert!(!dscc.is_recurrent(0));
        assert!(dscc.is_trivial(0));
    }

    #[test]
    fn tarjan_on_empty_graph_returns_empty_decomposition() {
        let graph = DBM::default();
        let dscc = SCCQuotient::tarjan(&graph);

        assert_eq!(dscc.vertex_count(), 0);
        assert_eq!(dscc.component_count(), 0);
        assert_eq!(dscc.classes().collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(dscc.graph().vertex_count(), 0);
    }

    #[test]
    fn tarjan_on_acyclic_graph_yields_nonrecurrent_singletons() {
        let graph = DBM::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
        ]);

        let dscc = SCCQuotient::tarjan(&graph);

        for vertex in 0..graph.vertex_count() {
            assert!(dscc.are_strongly_connected(vertex, vertex));
            assert!(dscc.is_trivial(dscc.class(vertex)));
        }

        assert!(!dscc.are_strongly_connected(0, 1));
        assert!(!dscc.are_strongly_connected(1, 2));
        assert!(!dscc.are_strongly_connected(2, 3));
    }

    #[test]
    fn tarjan_finds_cyclic_looped_and_isolated_components() {
        let mut graph = DBM::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 0),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
            Endpoints::new(3, 2),
            Endpoints::new(4, 4),
        ]);

        assert_eq!(graph.insert_vertex(), Some(5));

        let dscc = SCCQuotient::tarjan(&graph);

        assert!(dscc.are_strongly_connected(0, 1));
        assert!(dscc.are_strongly_connected(2, 3));
        assert!(dscc.are_strongly_connected(4, 4));
        assert!(dscc.are_strongly_connected(5, 5));

        assert!(!dscc.are_strongly_connected(1, 2));
        assert!(!dscc.are_strongly_connected(0, 4));
        assert!(!dscc.are_strongly_connected(4, 5));
    }

    #[test]
    fn tarjan_builds_quotient_graph() {
        let graph = DBM::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 0),
            Endpoints::new(1, 2),
            Endpoints::new(2, 3),
            Endpoints::new(3, 2),
        ]);

        let dscc = SCCQuotient::tarjan(&graph);

        let c01 = dscc.class(0);
        let c23 = dscc.class(2);

        assert_ne!(c01, c23);
        assert!(dscc.graph().is_connected(c01, c23));
        assert!(!dscc.graph().is_connected(c23, c01));
        assert!(dscc.graph().is_connected(c01, c01));
        assert!(dscc.graph().is_connected(c23, c23));
    }

    proptest! {
        #[test]
        fn prop_tarjan_matches_mutual_reachability_and_recurrence(
            edges in prop::collection::vec((0usize..12, 0usize..12), 0..48),
            extra_vertices in 0usize..4,
        ) {
            let mut graph = DBM::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            for _ in 0..extra_vertices {
                let _ = graph.insert_vertex();
            }

            let dscc = SCCQuotient::tarjan(&graph);
            let n = graph.vertex_count();

            prop_assert_eq!(dscc.vertex_count(), n);

            let mut all_members = dscc
                .classes()
                .flat_map(|class| dscc.members(class))
                .collect::<Vec<_>>();
            all_members.sort_unstable();

            prop_assert_eq!(all_members, (0..n).collect::<Vec<_>>());

            for class in dscc.classes() {
                let members = dscc.members(class).collect::<Vec<_>>();
                prop_assert!(!members.is_empty());

                let expected_recurrent =
                    members.len() > 1 || graph.is_connected(members[0], members[0]);

                prop_assert_eq!(dscc.is_recurrent(class), expected_recurrent);
                prop_assert_eq!(dscc.graph().is_connected(class, class), expected_recurrent);
            }
        }
    }
}
