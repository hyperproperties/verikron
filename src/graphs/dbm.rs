use std::ops::Range;

use bit_vec::BitVec;

use crate::graphs::{
    arc::{Arc, FromArcs},
    graph::{Directed, FiniteDirected, Graph, IndexedDirected},
    structure::{
        EdgeType, Edges, FiniteEdges, FiniteVertices, InsertVertex, Structure, VertexType, Vertices,
    },
};

/// Dense bit-matrix representation of a directed simple graph.
///
/// Vertices are `0..vertex_count()`.
///
/// Edge `(source, destination)` is stored as one bit in `matrix`.
/// Loops are allowed; parallel edges are not.
///
/// Edge identifiers are matrix indices. An index identifies a present edge
/// exactly when the corresponding bit is set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DBM {
    /// Flat adjacency matrix of length `vertex_count()^2`.
    matrix: BitVec,
}

impl DBM {
    /// Precomputed values of `n^2` for `1 <= n <= 128`.
    const SIZES: [usize; 128] = [
        1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400,
        441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1089, 1156, 1225, 1296, 1369,
        1444, 1521, 1600, 1681, 1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401, 2500, 2601, 2704,
        2809, 2916, 3025, 3136, 3249, 3364, 3481, 3600, 3721, 3844, 3969, 4096, 4225, 4356, 4489,
        4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929, 6084, 6241, 6400, 6561, 6724,
        6889, 7056, 7225, 7396, 7569, 7744, 7921, 8100, 8281, 8464, 8649, 8836, 9025, 9216, 9409,
        9604, 9801, 10000, 10201, 10404, 10609, 10816, 11025, 11236, 11449, 11664, 11881, 12100,
        12321, 12544, 12769, 12996, 13225, 13456, 13689, 13924, 14161, 14400, 14641, 14884, 15129,
        15376, 15625, 15876, 16129, 16384,
    ];

    /// Creates a graph with `vertices` vertices.
    ///
    /// When `complete` is `true`, all possible directed edges are present,
    /// including loops.
    #[must_use]
    pub fn new(vertices: usize, complete: bool) -> Self {
        let size = if vertices == 0 {
            0
        } else {
            Self::size(vertices)
        };
        Self {
            matrix: BitVec::from_elem(size, complete),
        }
    }

    /// Returns `vertices^2`.
    #[must_use]
    #[inline]
    pub fn size(vertices: usize) -> usize {
        if vertices == 0 {
            0
        } else if vertices <= 128 {
            Self::SIZES[vertices - 1]
        } else {
            vertices * vertices
        }
    }

    /// Returns `(vertices + 1)^2 - vertices^2`.
    #[must_use]
    #[inline]
    pub const fn growth(vertices: usize) -> usize {
        2 * vertices + 1
    }

    /// Maps `(source, destination)` to a matrix index.
    #[must_use]
    #[inline]
    pub fn index(source: usize, destination: usize) -> usize {
        let radius = source.max(destination);

        if radius == 0 {
            return 0;
        }

        let base = Self::size(radius);

        if destination == radius {
            base + source
        } else {
            base + 2 * radius - destination
        }
    }

    /// Inverse of [`DBM::index`].
    #[must_use]
    pub fn inverse_index(index: usize) -> (usize, usize) {
        if index == 0 {
            return (0, 0);
        }

        let radius = index.isqrt();
        let base = radius * radius;
        let offset = index - base;

        if offset <= radius {
            (offset, radius)
        } else {
            (radius, 2 * radius - offset)
        }
    }

    #[inline]
    fn contains_edge_index(&self, edge: usize) -> bool {
        self.matrix.get(edge).unwrap_or(false)
    }
}

impl FromArcs for DBM {
    /// Builds a graph from directed arcs.
    ///
    /// Duplicate arcs are collapsed.
    fn from_arcs<I, A>(arcs: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<Arc<Self::Vertex>>,
    {
        let arcs: Vec<_> = arcs.into_iter().map(Into::into).collect();

        let vertex_count = arcs
            .iter()
            .map(|arc| arc.source.max(arc.destination))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let mut dbm = Self::new(vertex_count, false);

        for arc in arcs {
            let index = Self::index(arc.source, arc.destination);
            dbm.matrix.set(index, true);
        }

        dbm
    }
}

impl VertexType for DBM {
    type Vertex = usize;
}

impl EdgeType for DBM {
    /// Edge id as a flat matrix index.
    type Edge = usize;
}

impl Vertices for DBM {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Returns all vertices.
    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count()
    }
}

impl FiniteVertices for DBM {
    /// Returns the number of vertices.
    #[inline]
    fn vertex_count(&self) -> usize {
        let len = self.matrix.len();
        if len == 0 {
            return 0;
        }

        let vertices = len.isqrt();
        debug_assert_eq!(len, Self::size(vertices));
        vertices
    }

    /// Returns whether `vertex` exists.
    #[inline]
    fn contains(&self, vertex: &Self::Vertex) -> bool {
        *vertex < self.vertex_count()
    }
}

impl Edges for DBM {
    type Edges<'a>
        = DbmEdgeIds<'a>
    where
        Self: 'a;

    /// Returns all present edge identifiers.
    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        DbmEdgeIds::new(self)
    }
}

impl FiniteEdges for DBM {
    /// Returns the number of present edges.
    #[inline]
    fn edge_count(&self) -> usize {
        self.matrix.count_ones() as usize
    }

    /// Returns whether `edge` exists.
    #[inline]
    fn contains_edge(&self, edge: &Self::Edge) -> bool {
        self.contains_edge_index(*edge)
    }
}

impl InsertVertex for DBM {
    /// Inserts a new isolated vertex.
    ///
    /// Existing edge indices remain unchanged.
    #[inline]
    fn insert_vertex(&mut self) -> Option<Self::Vertex> {
        let vertex = self.vertex_count();
        let growth = Self::growth(vertex);

        for _ in 0..growth {
            self.matrix.push(false);
        }

        Some(vertex)
    }
}

impl Structure for DBM {
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

impl Graph for DBM {}

impl Directed for DBM {
    type Outgoing<'a>
        = DbmEdges<'a>
    where
        Self: 'a;

    type Incoming<'a>
        = DbmEdges<'a>
    where
        Self: 'a;

    type Connections<'a>
        = DbmEdges<'a>
    where
        Self: 'a;

    /// Returns the source of `edge`.
    ///
    /// The caller is expected to pass the id of a present edge.
    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(self.contains_edge_index(edge));
        let (source, _) = Self::inverse_index(edge);
        source
    }

    /// Returns the destination of `edge`.
    ///
    /// The caller is expected to pass the id of a present edge.
    #[inline]
    fn destination(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(self.contains_edge_index(edge));
        let (_, destination) = Self::inverse_index(edge);
        destination
    }

    /// Returns all outgoing edges from `source`.
    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        if !self.contains(&source) {
            return DbmEdges::empty(self);
        }
        DbmEdges::outgoing(self, source)
    }

    /// Returns all incoming edges to `destination`.
    #[inline]
    fn incoming(&self, destination: Self::Vertex) -> Self::Incoming<'_> {
        if !self.contains(&destination) {
            return DbmEdges::empty(self);
        }
        DbmEdges::incoming(self, destination)
    }

    /// Returns all edges from `source` to `destination`.
    #[inline]
    fn connections(
        &self,
        source: Self::Vertex,
        destination: Self::Vertex,
    ) -> Self::Connections<'_> {
        if !(self.contains(&source) && self.contains(&destination)) {
            return DbmEdges::empty(self);
        }
        DbmEdges::between(self, source, destination)
    }
}

impl FiniteDirected for DBM {
    /// Returns the number of outgoing edges from `vertex`.
    #[inline]
    fn outgoing_degree(&self, vertex: Self::Vertex) -> usize {
        if !self.contains(&vertex) {
            return 0;
        }

        (0..self.vertex_count())
            .filter(|&destination| self.is_connected(vertex, destination))
            .count()
    }

    /// Returns the number of incoming edges to `vertex`.
    #[inline]
    fn incoming_degree(&self, vertex: Self::Vertex) -> usize {
        if !self.contains(&vertex) {
            return 0;
        }

        (0..self.vertex_count())
            .filter(|&source| self.is_connected(source, vertex))
            .count()
    }

    /// Returns the number of loops at `vertex`.
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        usize::from(self.is_connected(vertex, vertex))
    }

    /// Returns whether `(source, destination)` is present.
    #[inline]
    fn is_connected(&self, source: Self::Vertex, destination: Self::Vertex) -> bool {
        if !(self.contains(&source) && self.contains(&destination)) {
            return false;
        }

        self.matrix[Self::index(source, destination)]
    }

    /// Returns whether `edge` is exactly the edge `(source, destination)` and is present.
    #[inline]
    fn has_edge(&self, source: Self::Vertex, edge: Self::Edge, destination: Self::Vertex) -> bool {
        self.contains_edge_index(edge) && edge == Self::index(source, destination)
    }
}

impl IndexedDirected for DBM {
    #[inline]
    fn outgoing_count(&self, vertex: Self::Vertex) -> usize {
        if !self.contains(&vertex) {
            return 0;
        }

        (0..self.vertex_count())
            .filter(|&destination| self.is_connected(vertex, destination))
            .count()
    }

    #[inline]
    fn outgoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex> {
        if !self.contains(&vertex) {
            return None;
        }

        let mut seen = 0usize;
        for destination in 0..self.vertex_count() {
            if self.is_connected(vertex, destination) {
                if seen == index {
                    return Some(destination);
                }
                seen += 1;
            }
        }

        None
    }

    #[inline]
    fn incoming_count(&self, vertex: Self::Vertex) -> usize {
        if !self.contains(&vertex) {
            return 0;
        }

        (0..self.vertex_count())
            .filter(|&source| self.is_connected(source, vertex))
            .count()
    }

    #[inline]
    fn incoming_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex> {
        if !self.contains(&vertex) {
            return None;
        }

        let mut seen = 0usize;
        for source in 0..self.vertex_count() {
            if self.is_connected(source, vertex) {
                if seen == index {
                    return Some(source);
                }
                seen += 1;
            }
        }

        None
    }
}

/// Iterator over present DBM edge identifiers.
#[derive(Clone, Debug)]
pub struct DbmEdgeIds<'a> {
    dbm: &'a DBM,
    edge: usize,
}

impl<'a> DbmEdgeIds<'a> {
    #[must_use]
    #[inline]
    fn new(dbm: &'a DBM) -> Self {
        Self { dbm, edge: 0 }
    }
}

impl<'a> Iterator for DbmEdgeIds<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.edge < self.dbm.matrix.len() {
            let edge = self.edge;
            self.edge += 1;

            if self.dbm.matrix[edge] {
                return Some(edge);
            }
        }

        None
    }
}

/// Iterator over selected DBM edge identifiers.
#[derive(Clone, Debug)]
pub struct DbmEdges<'a> {
    dbm: &'a DBM,
    kind: DbmEdgesKind,
}

/// Internal iterator mode.
#[derive(Clone, Debug)]
enum DbmEdgesKind {
    Outgoing {
        source: usize,
        destination: usize,
    },
    Incoming {
        source: usize,
        destination: usize,
    },
    Between {
        source: usize,
        destination: usize,
        done: bool,
    },
    Empty,
}

impl<'a> DbmEdges<'a> {
    /// Iterates over edges outgoing from `source`.
    #[must_use]
    fn outgoing(dbm: &'a DBM, source: usize) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::Outgoing {
                source,
                destination: 0,
            },
        }
    }

    /// Iterates over edges incoming to `destination`.
    #[must_use]
    fn incoming(dbm: &'a DBM, destination: usize) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::Incoming {
                source: 0,
                destination,
            },
        }
    }

    /// Iterates over the directed edge from `source` to `destination`, if present.
    #[must_use]
    fn between(dbm: &'a DBM, source: usize, destination: usize) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::Between {
                source,
                destination,
                done: false,
            },
        }
    }

    /// Returns the empty iterator.
    #[must_use]
    fn empty(dbm: &'a DBM) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::Empty,
        }
    }
}

impl<'a> Iterator for DbmEdges<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.kind {
            DbmEdgesKind::Outgoing {
                source,
                destination,
            } => {
                while *destination < self.dbm.vertex_count() {
                    let current_destination = *destination;
                    *destination += 1;

                    let edge = DBM::index(*source, current_destination);
                    if self.dbm.matrix[edge] {
                        return Some(edge);
                    }
                }
                None
            }

            DbmEdgesKind::Incoming {
                source,
                destination,
            } => {
                while *source < self.dbm.vertex_count() {
                    let current_source = *source;
                    *source += 1;

                    let edge = DBM::index(current_source, *destination);
                    if self.dbm.matrix[edge] {
                        return Some(edge);
                    }
                }
                None
            }

            DbmEdgesKind::Between {
                source,
                destination,
                done,
            } => {
                if *done {
                    return None;
                }

                *done = true;
                let edge = DBM::index(*source, *destination);
                self.dbm.matrix[edge].then_some(edge)
            }

            DbmEdgesKind::Empty => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::arc::FromArcs;
    use proptest::prelude::*;
    use std::collections::HashSet;

    fn edge_set(graph: &DBM) -> HashSet<(usize, usize)> {
        graph
            .edges()
            .map(|edge| (graph.source(edge), graph.destination(edge)))
            .collect()
    }

    #[test]
    fn new_empty_and_complete_graphs_are_consistent() {
        let empty = DBM::new(4, false);
        assert_eq!(empty.vertex_count(), 4);
        assert_eq!(empty.edge_count(), 0);
        assert!(empty.edges().next().is_none());

        let complete = DBM::new(3, true);
        assert_eq!(complete.vertex_count(), 3);
        assert_eq!(complete.edge_count(), 9);

        for source in 0..3 {
            for destination in 0..3 {
                assert!(complete.is_connected(source, destination));
            }
        }
    }

    #[test]
    fn from_arcs_builds_expected_graph() {
        let graph = DBM::from_arcs([Arc::new(0, 1), Arc::new(1, 2), Arc::new(2, 2)]);

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);

        assert!(graph.is_connected(0, 1));
        assert!(graph.is_connected(1, 2));
        assert!(graph.is_connected(2, 2));
        assert!(!graph.is_connected(1, 0));
    }

    #[test]
    fn duplicate_arcs_do_not_create_parallel_edges() {
        let graph = DBM::from_arcs([Arc::new(0, 1), Arc::new(0, 1), Arc::new(0, 1)]);

        assert_eq!(graph.vertex_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.connections(0, 1).count(), 1);
    }

    #[test]
    fn index_and_inverse_index_roundtrip_on_documented_grid() {
        for source in 0..6 {
            for destination in 0..6 {
                let edge = DBM::index(source, destination);
                assert_eq!(DBM::inverse_index(edge), (source, destination));
            }
        }
    }

    #[test]
    fn documented_layout_for_six_by_six_matches() {
        let expected = [
            [0, 1, 4, 9, 16, 25],
            [3, 2, 5, 10, 17, 26],
            [8, 7, 6, 11, 18, 27],
            [15, 14, 13, 12, 19, 28],
            [24, 23, 22, 21, 20, 29],
            [35, 34, 33, 32, 31, 30],
        ];

        for source in 0..6 {
            for destination in 0..6 {
                let edge = DBM::index(source, destination);
                assert_eq!(edge, expected[source][destination]);
                assert_eq!(DBM::inverse_index(edge), (source, destination));
            }
        }
    }

    #[test]
    fn directed_queries_are_consistent() {
        let graph = DBM::from_arcs([
            Arc::new(0, 1),
            Arc::new(0, 2),
            Arc::new(2, 1),
            Arc::new(2, 2),
        ]);

        assert_eq!(
            graph
                .outgoing(0)
                .map(|edge| graph.destination(edge))
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            graph
                .incoming(1)
                .map(|edge| graph.source(edge))
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(graph.connections(2, 1).count(), 1);
        assert_eq!(graph.connections(1, 2).count(), 0);
        assert_eq!(graph.loop_degree(2), 1);
        assert!(graph.has_edge(2, DBM::index(2, 2), 2));
    }

    #[test]
    fn insert_vertex_appends_new_isolated_vertex() {
        let mut graph = DBM::from_arcs([Arc::new(0, 1), Arc::new(1, 1)]);

        let vertex = graph.insert_vertex().unwrap();

        assert_eq!(vertex, 2);
        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 2);

        for u in 0..3 {
            assert!(!graph.is_connected(u, 2));
            assert!(!graph.is_connected(2, u));
        }
    }

    #[test]
    fn edge_ids_enumerate_only_present_edges() {
        let graph = DBM::from_arcs([Arc::new(0, 1), Arc::new(2, 0), Arc::new(2, 2)]);

        let edges: Vec<_> = graph.edges().collect();

        assert_eq!(edges.len(), 3);
        for edge in edges {
            assert!(graph.contains_edge(&edge));
        }
    }

    #[test]
    fn size_and_growth_match_square_arithmetic() {
        for n in 1..=256 {
            assert_eq!(DBM::size(n), n * n);
            assert_eq!(DBM::growth(n), DBM::size(n + 1) - DBM::size(n));
        }
    }

    #[test]
    fn indexed_directed_queries_are_consistent() {
        let graph = DBM::from_arcs([
            Arc::new(0, 1),
            Arc::new(0, 2),
            Arc::new(2, 1),
            Arc::new(2, 2),
            Arc::new(2, 2),
        ]);

        assert_eq!(graph.outgoing_count(0), 2);
        assert_eq!(graph.outgoing_count(1), 0);
        assert_eq!(graph.outgoing_count(2), 2);
        assert_eq!(graph.outgoing_count(3), 0);

        assert_eq!(graph.outgoing_at(0, 0), Some(1));
        assert_eq!(graph.outgoing_at(0, 1), Some(2));
        assert_eq!(graph.outgoing_at(0, 2), None);

        assert_eq!(graph.outgoing_at(1, 0), None);

        assert_eq!(graph.outgoing_at(2, 0), Some(1));
        assert_eq!(graph.outgoing_at(2, 1), Some(2));
        assert_eq!(graph.outgoing_at(2, 2), None);

        assert_eq!(graph.incoming_count(0), 0);
        assert_eq!(graph.incoming_count(1), 2);
        assert_eq!(graph.incoming_count(2), 2);
        assert_eq!(graph.incoming_count(3), 0);

        assert_eq!(graph.incoming_at(0, 0), None);

        assert_eq!(graph.incoming_at(1, 0), Some(0));
        assert_eq!(graph.incoming_at(1, 1), Some(2));
        assert_eq!(graph.incoming_at(1, 2), None);

        assert_eq!(graph.incoming_at(2, 0), Some(0));
        assert_eq!(graph.incoming_at(2, 1), Some(2));
        assert_eq!(graph.incoming_at(2, 2), None);

        assert_eq!(graph.outgoing_count(99), 0);
        assert_eq!(graph.outgoing_at(99, 0), None);
        assert_eq!(graph.incoming_count(99), 0);
        assert_eq!(graph.incoming_at(99, 0), None);
    }

    proptest! {
        #[test]
        fn prop_size_matches_square(n in 1usize..=1_000) {
            prop_assert_eq!(DBM::size(n), n * n);
        }

        #[test]
        fn prop_growth_matches_size_difference(n in 0usize..=1_000) {
            prop_assert_eq!(DBM::growth(n), DBM::size(n + 1) - DBM::size(n));
        }

        #[test]
        fn prop_index_inverse_roundtrip(source in 0usize..10_000, destination in 0usize..10_000) {
            let edge = DBM::index(source, destination);
            prop_assert_eq!(DBM::inverse_index(edge), (source, destination));
        }

        #[test]
        fn prop_index_is_injective_within_bounded_grid(max in 0usize..32) {
            let mut seen = HashSet::new();

            for source in 0..=max {
                for destination in 0..=max {
                    let edge = DBM::index(source, destination);
                    prop_assert!(seen.insert(edge));
                }
            }

            prop_assert_eq!(seen.len(), (max + 1) * (max + 1));
        }

        #[test]
        fn prop_from_arcs_matches_expected_adjacency(
            edges in prop::collection::vec((0usize..16, 0usize..16), 0..64)
        ) {
            let graph = DBM::from_arcs(
                edges.iter().copied().map(|(source, destination)| Arc::new(source, destination))
            );

            let expected: HashSet<_> = edges.into_iter().collect();
            let actual = edge_set(&graph);

            prop_assert_eq!(graph.edge_count(), expected.len());
            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn prop_outgoing_and_incoming_match_membership(
            edges in prop::collection::vec((0usize..12, 0usize..12), 0..48)
        ) {
            let graph = DBM::from_arcs(
                edges.iter().copied().map(|(source, destination)| Arc::new(source, destination))
            );

            let expected: HashSet<_> = edges.into_iter().collect();

            for vertex in 0..graph.vertex_count() {
                let actual_outgoing: HashSet<_> =
                    graph.outgoing(vertex).map(|edge| (vertex, graph.destination(edge))).collect();
                let expected_outgoing: HashSet<_> =
                    expected.iter().copied().filter(|&(source, _)| source == vertex).collect();
                prop_assert_eq!(actual_outgoing, expected_outgoing);

                let actual_incoming: HashSet<_> =
                    graph.incoming(vertex).map(|edge| (graph.source(edge), vertex)).collect();
                let expected_incoming: HashSet<_> =
                    expected.iter().copied().filter(|&(_, destination)| destination == vertex).collect();
                prop_assert_eq!(actual_incoming, expected_incoming);
            }
        }

        #[test]
        fn prop_edges_enumerate_present_edge_ids_only(
            edges in prop::collection::vec((0usize..12, 0usize..12), 0..48)
        ) {
            let graph = DBM::from_arcs(
                edges.iter().copied().map(|(source, destination)| Arc::new(source, destination))
            );

            let enumerated: Vec<_> = graph.edges().collect();

            prop_assert_eq!(enumerated.len(), graph.edge_count());

            for edge in enumerated {
                prop_assert!(graph.contains_edge(&edge));
                let source = graph.source(edge);
                let destination = graph.destination(edge);
                prop_assert!(graph.is_connected(source, destination));
                prop_assert_eq!(DBM::index(source, destination), edge);
            }
        }

        #[test]
        fn prop_indexed_directed_matches_connectivity(
            edges in prop::collection::vec((0usize..12, 0usize..12), 0..48)
        ) {
            let graph = DBM::from_arcs(
                edges.iter().copied().map(|(source, destination)| Arc::new(source, destination))
            );

            let n = graph.vertex_count();

            for vertex in 0..n {
                let expected_outgoing: Vec<_> =
                    (0..n).filter(|&destination| graph.is_connected(vertex, destination)).collect();
                prop_assert_eq!(graph.outgoing_count(vertex), expected_outgoing.len());

                for (index, &destination) in expected_outgoing.iter().enumerate() {
                    prop_assert_eq!(graph.outgoing_at(vertex, index), Some(destination));
                }
                prop_assert_eq!(graph.outgoing_at(vertex, expected_outgoing.len()), None);

                let expected_incoming: Vec<_> =
                    (0..n).filter(|&source| graph.is_connected(source, vertex)).collect();
                prop_assert_eq!(graph.incoming_count(vertex), expected_incoming.len());

                for (index, &source) in expected_incoming.iter().enumerate() {
                    prop_assert_eq!(graph.incoming_at(vertex, index), Some(source));
                }
                prop_assert_eq!(graph.incoming_at(vertex, expected_incoming.len()), None);
            }

            let invalid = n;

            prop_assert_eq!(graph.outgoing_count(invalid), 0);
            prop_assert_eq!(graph.outgoing_at(invalid, 0), None);
            prop_assert_eq!(graph.incoming_count(invalid), 0);
            prop_assert_eq!(graph.incoming_at(invalid, 0), None);
        }
    }
}
