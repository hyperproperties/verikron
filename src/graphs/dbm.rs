use std::ops::Range;

use bit_vec::BitVec;

use crate::graphs::{
    graph::{Directed, Endpoints, FiniteDirected, FromEndpoints, Graph},
    structure::{
        EdgeType, Edges, FiniteEdges, FiniteVertices, InsertVertex, Structure, VertexType, Vertices,
    },
};

/// Dense bit-matrix representation of a directed simple graph.
///
/// Vertices are `0..vertex_count()`.
///
/// Edge `(from, to)` is stored as one bit in `matrix`.
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

    /// Maps `(from, to)` to a matrix index.
    #[must_use]
    #[inline]
    pub fn index(from: usize, to: usize) -> usize {
        let radius = from.max(to);

        if radius == 0 {
            return 0;
        }

        let base = Self::size(radius);

        if to == radius {
            base + from
        } else {
            base + 2 * radius - to
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

impl FromEndpoints for DBM {
    /// Builds a graph from directed edges.
    ///
    /// Duplicate edges are collapsed.
    fn from_endpoints<I>(edges: I) -> Self
    where
        I: IntoIterator<Item = Endpoints<Self::Vertex>>,
    {
        let edges: Vec<_> = edges.into_iter().collect();

        let vertex_count = edges
            .iter()
            .map(|edge| edge.from.max(edge.to))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let mut dbm = Self::new(vertex_count, false);

        for edge in edges {
            let index = Self::index(edge.from, edge.to);
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

    type Ingoing<'a>
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
        let (from, _) = Self::inverse_index(edge);
        from
    }

    /// Returns the target of `edge`.
    ///
    /// The caller is expected to pass the id of a present edge.
    #[inline]
    fn destination(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(self.contains_edge_index(edge));
        let (_, to) = Self::inverse_index(edge);
        to
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
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        if !self.contains(&destination) {
            return DbmEdges::empty(self);
        }
        DbmEdges::ingoing(self, destination)
    }

    /// Returns all edges from `from` to `to`.
    #[inline]
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        if !(self.contains(&from) && self.contains(&to)) {
            return DbmEdges::empty(self);
        }
        DbmEdges::between(self, from, to)
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
            .filter(|&to| self.is_connected(vertex, to))
            .count()
    }

    /// Returns the number of incoming edges to `vertex`.
    #[inline]
    fn ingoing_degree(&self, vertex: Self::Vertex) -> usize {
        if !self.contains(&vertex) {
            return 0;
        }

        (0..self.vertex_count())
            .filter(|&from| self.is_connected(from, vertex))
            .count()
    }

    /// Returns the number of loops at `vertex`.
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        usize::from(self.is_connected(vertex, vertex))
    }

    /// Returns whether `(from, to)` is present.
    #[inline]
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        if !(self.contains(&from) && self.contains(&to)) {
            return false;
        }

        self.matrix[Self::index(from, to)]
    }

    /// Returns whether `edge` is exactly the edge `(from, to)` and is present.
    #[inline]
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        self.contains_edge_index(edge) && edge == Self::index(from, to)
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

/// Iterator over selected DBM edges.
///
/// Yields `(from, edge, to)`.
#[derive(Clone, Debug)]
pub struct DbmEdges<'a> {
    dbm: &'a DBM,
    kind: DbmEdgesKind,
}

/// Internal iterator mode.
#[derive(Clone, Debug)]
enum DbmEdgesKind {
    All { edge: usize },
    Outgoing { from: usize, to: usize },
    Ingoing { from: usize, to: usize },
    Between { from: usize, to: usize, done: bool },
    Empty,
}

impl<'a> DbmEdges<'a> {
    /// Iterates over all present edges.
    #[must_use]
    fn all(dbm: &'a DBM) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::All { edge: 0 },
        }
    }

    /// Iterates over edges outgoing from `from`.
    #[must_use]
    fn outgoing(dbm: &'a DBM, from: usize) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::Outgoing { from, to: 0 },
        }
    }

    /// Iterates over edges incoming to `to`.
    #[must_use]
    fn ingoing(dbm: &'a DBM, to: usize) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::Ingoing { from: 0, to },
        }
    }

    /// Iterates over the directed edge from `from` to `to`, if present.
    #[must_use]
    fn between(dbm: &'a DBM, from: usize, to: usize) -> Self {
        Self {
            dbm,
            kind: DbmEdgesKind::Between {
                from,
                to,
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
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.kind {
            DbmEdgesKind::All { edge } => {
                while *edge < self.dbm.matrix.len() {
                    let current = *edge;
                    *edge += 1;

                    if self.dbm.matrix[current] {
                        let (from, to) = DBM::inverse_index(current);
                        return Some((from, current, to));
                    }
                }
                None
            }

            DbmEdgesKind::Outgoing { from, to } => {
                while *to < self.dbm.vertex_count() {
                    let current_to = *to;
                    *to += 1;

                    let edge = DBM::index(*from, current_to);
                    if self.dbm.matrix[edge] {
                        return Some((*from, edge, current_to));
                    }
                }
                None
            }

            DbmEdgesKind::Ingoing { from, to } => {
                while *from < self.dbm.vertex_count() {
                    let current_from = *from;
                    *from += 1;

                    let edge = DBM::index(current_from, *to);
                    if self.dbm.matrix[edge] {
                        return Some((current_from, edge, *to));
                    }
                }
                None
            }

            DbmEdgesKind::Between { from, to, done } => {
                if *done {
                    return None;
                }

                *done = true;
                let edge = DBM::index(*from, *to);
                self.dbm.matrix[edge].then_some((*from, edge, *to))
            }

            DbmEdgesKind::Empty => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::graph::FromEndpoints;
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

        for from in 0..3 {
            for to in 0..3 {
                assert!(complete.is_connected(from, to));
            }
        }
    }

    #[test]
    fn from_edges_builds_expected_graph() {
        let graph = DBM::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(1, 2),
            Endpoints::new(2, 2),
        ]);

        assert_eq!(graph.vertex_count(), 3);
        assert_eq!(graph.edge_count(), 3);

        assert!(graph.is_connected(0, 1));
        assert!(graph.is_connected(1, 2));
        assert!(graph.is_connected(2, 2));
        assert!(!graph.is_connected(1, 0));
    }

    #[test]
    fn duplicate_edges_do_not_create_parallel_edges() {
        let graph = DBM::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 1),
            Endpoints::new(0, 1),
        ]);

        assert_eq!(graph.vertex_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.connections(0, 1).count(), 1);
    }

    #[test]
    fn index_and_inverse_index_roundtrip_on_documented_grid() {
        for from in 0..6 {
            for to in 0..6 {
                let edge = DBM::index(from, to);
                assert_eq!(DBM::inverse_index(edge), (from, to));
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

        for from in 0..6 {
            for to in 0..6 {
                let edge = DBM::index(from, to);
                assert_eq!(edge, expected[from][to]);
                assert_eq!(DBM::inverse_index(edge), (from, to));
            }
        }
    }

    #[test]
    fn directed_queries_are_consistent() {
        let graph = DBM::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(0, 2),
            Endpoints::new(2, 1),
            Endpoints::new(2, 2),
        ]);

        assert_eq!(
            graph.outgoing(0).map(|(_, _, to)| to).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            graph
                .ingoing(1)
                .map(|(from, _, _)| from)
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
        let mut graph = DBM::from_endpoints([Endpoints::new(0, 1), Endpoints::new(1, 1)]);

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
        let graph = DBM::from_endpoints([
            Endpoints::new(0, 1),
            Endpoints::new(2, 0),
            Endpoints::new(2, 2),
        ]);

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
        fn prop_index_inverse_roundtrip(from in 0usize..10_000, to in 0usize..10_000) {
            let edge = DBM::index(from, to);
            prop_assert_eq!(DBM::inverse_index(edge), (from, to));
        }

        #[test]
        fn prop_index_is_injective_within_bounded_grid(max in 0usize..32) {
            let mut seen = HashSet::new();

            for from in 0..=max {
                for to in 0..=max {
                    let edge = DBM::index(from, to);
                    prop_assert!(seen.insert(edge));
                }
            }

            prop_assert_eq!(seen.len(), (max + 1) * (max + 1));
        }

        #[test]
        fn prop_from_edges_matches_expected_adjacency(
            edges in prop::collection::vec((0usize..16, 0usize..16), 0..64)
        ) {
            let graph = DBM::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            let expected: HashSet<_> = edges.into_iter().collect();
            let actual = edge_set(&graph);

            prop_assert_eq!(graph.edge_count(), expected.len());
            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn prop_outgoing_and_ingoing_match_membership(
            edges in prop::collection::vec((0usize..12, 0usize..12), 0..48)
        ) {
            let graph = DBM::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            let expected: HashSet<_> = edges.into_iter().collect();

            for vertex in 0..graph.vertex_count() {
                let actual_outgoing: HashSet<_> =
                    graph.outgoing(vertex).map(|(_, _, to)| (vertex, to)).collect();
                let expected_outgoing: HashSet<_> =
                    expected.iter().copied().filter(|&(from, _)| from == vertex).collect();
                prop_assert_eq!(actual_outgoing, expected_outgoing);

                let actual_ingoing: HashSet<_> =
                    graph.ingoing(vertex).map(|(from, _, _)| (from, vertex)).collect();
                let expected_ingoing: HashSet<_> =
                    expected.iter().copied().filter(|&(_, to)| to == vertex).collect();
                prop_assert_eq!(actual_ingoing, expected_ingoing);
            }
        }

        #[test]
        fn prop_edges_enumerate_present_edge_ids_only(
            edges in prop::collection::vec((0usize..12, 0usize..12), 0..48)
        ) {
            let graph = DBM::from_endpoints(
                edges.iter().copied().map(|(from, to)| Endpoints::new(from, to))
            );

            let enumerated: Vec<_> = graph.edges().collect();

            prop_assert_eq!(enumerated.len(), graph.edge_count());

            for edge in enumerated {
                prop_assert!(graph.contains_edge(&edge));
                let from = graph.source(edge);
                let to = graph.destination(edge);
                prop_assert!(graph.is_connected(from, to));
                prop_assert_eq!(DBM::index(from, to), edge);
            }
        }
    }
}
