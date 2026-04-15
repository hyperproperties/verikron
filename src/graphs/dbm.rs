use std::ops::Range;

use bit_vec::BitVec;

use crate::graphs::graph::{
    Directed, EdgeType, Edges, Endpoints, FiniteDirected, FiniteEdges, FiniteVertices,
    FromEndpoints, Graph, VertexType, Vertices,
};

/// Dense bit-matrix representation of a directed simple graph.
///
/// Vertices are `0..vertex_count()`. Edge `(from, to)` is stored as one bit in
/// `matrix`. Loops are allowed; parallel edges are not.
#[derive(Default)]
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
    /// When `complete` is `true`, all possible edges are present.
    pub fn new(vertices: usize, complete: bool) -> Self {
        if vertices == 0 {
            let dbm = Self {
                matrix: BitVec::default(),
            };
            debug_assert_eq!(dbm.vertex_count(), 0);
            debug_assert_eq!(dbm.matrix.len(), 0);
            return dbm;
        }

        let size = Self::size(vertices);
        let matrix = BitVec::from_elem(size, complete);

        debug_assert_eq!(
            matrix.len(),
            size,
            "DBM::new: matrix.len() = {} != size({vertices}) = {size}",
            matrix.len()
        );

        Self { matrix }
    }

    /// Returns `vertex^2`.
    pub fn size(vertex: usize) -> usize {
        debug_assert!(vertex > 0, "DBM::size: vertex must be positive");
        if vertex <= 128 {
            Self::SIZES[vertex - 1]
        } else {
            vertex.pow(2)
        }
    }

    /// Returns `(vertex + 1)^2 - vertex^2`.
    pub fn growth(vertex: usize) -> usize {
        2 * vertex + 1
    }

    /// Maps `(from, to)` to a bit index.
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
    pub fn inverse_index(index: usize) -> (usize, usize) {
        if index == 0 {
            return (0, 0);
        }

        let radius = index.isqrt();
        let base = radius * radius;
        let offset = index - base;

        debug_assert!(
            offset <= 2 * radius,
            "DBM::inverse_index: offset {} > 2 * radius {} for index {}",
            offset,
            2 * radius,
            index
        );

        if offset <= radius {
            (offset, radius)
        } else {
            (radius, 2 * radius - offset)
        }
    }
}

impl FromEndpoints for DBM {
    /// Builds a graph from directed edges.
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
            debug_assert!(
                edge.from < vertex_count && edge.to < vertex_count,
                "DBM::from_endpoints: endpoint ({}, {}) out of range {}",
                edge.from,
                edge.to,
                vertex_count
            );

            let index = Self::index(edge.from, edge.to);

            debug_assert!(
                index < dbm.matrix.len(),
                "DBM::from_endpoints: index {} out of bounds {} for ({}, {})",
                index,
                dbm.matrix.len(),
                edge.from,
                edge.to
            );

            dbm.matrix.set(index, true);
        }

        dbm
    }
}

impl VertexType for DBM {
    type Vertex = usize;
}

impl Vertices for DBM {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Returns all vertices.
    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count()
    }
}

impl FiniteVertices for DBM {
    /// Returns the number of vertices.
    fn vertex_count(&self) -> usize {
        let len = self.matrix.len();
        if len == 0 {
            return 0;
        }

        let vertices = len.isqrt();

        debug_assert_eq!(
            len,
            Self::size(vertices),
            "DBM::vertex_count: matrix.len() = {} is not a valid square size for {} vertices",
            len,
            vertices
        );

        vertices
    }
}

impl EdgeType for DBM {
    /// Edge id as a flat matrix index.
    type Edge = usize;
}

impl Edges for DBM {
    type Edges<'a>
        = CbmEdges<'a>
    where
        Self: 'a;

    /// Returns all edges.
    fn edges(&self) -> Self::Edges<'_> {
        let vertices = self.vertex_count();
        let expected_len = if vertices == 0 {
            0
        } else {
            Self::size(vertices)
        };

        debug_assert_eq!(
            self.matrix.len(),
            expected_len,
            "DBM::edges: matrix.len() = {} != size({vertices}) = {}",
            self.matrix.len(),
            expected_len
        );

        CbmEdges::all(self)
    }
}

impl FiniteEdges for DBM {
    /// Returns the number of present edges.
    fn edge_count(&self) -> usize {
        self.matrix.count_ones() as usize
    }
}

impl Directed for DBM {
    type Outgoing<'a>
        = CbmEdges<'a>
    where
        Self: 'a;

    type Ingoing<'a>
        = CbmEdges<'a>
    where
        Self: 'a;

    type Connections<'a>
        = CbmEdges<'a>
    where
        Self: 'a;

    /// Returns the source of `edge`.
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(
            edge < self.matrix.len(),
            "DBM::source: edge {} out of bounds {}",
            edge,
            self.matrix.len()
        );

        let (from, _) = Self::inverse_index(edge);

        debug_assert!(
            from < self.vertex_count(),
            "DBM::source: decoded source {} out of range {}",
            from,
            self.vertex_count()
        );

        from
    }

    /// Returns the target of `edge`.
    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(
            edge < self.matrix.len(),
            "DBM::target: edge {} out of bounds {}",
            edge,
            self.matrix.len()
        );

        let (_, to) = Self::inverse_index(edge);

        debug_assert!(
            to < self.vertex_count(),
            "DBM::target: decoded target {} out of range {}",
            to,
            self.vertex_count()
        );

        to
    }

    /// Returns all outgoing edges from `source`.
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        debug_assert!(
            source < self.vertex_count(),
            "DBM::outgoing: source {} out of range {}",
            source,
            self.vertex_count()
        );
        CbmEdges::outgoing(self, source)
    }

    /// Returns all incoming edges to `destination`.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        debug_assert!(
            destination < self.vertex_count(),
            "DBM::ingoing: destination {} out of range {}",
            destination,
            self.vertex_count()
        );
        CbmEdges::ingoing(self, destination)
    }

    /// Returns all edges from `from` to `to`.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        debug_assert!(
            from < self.vertex_count() && to < self.vertex_count(),
            "DBM::connections: ({from}, {to}) out of range {}",
            self.vertex_count()
        );
        CbmEdges::between(self, from, to)
    }
}

impl FiniteDirected for DBM {
    /// Returns the number of loops at `vertex`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        debug_assert!(
            vertex < self.vertex_count(),
            "DBM::loop_degree: vertex {} out of range {}",
            vertex,
            self.vertex_count()
        );
        usize::from(self.is_connected(vertex, vertex))
    }

    /// Returns whether `(from, to)` is present.
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        debug_assert!(
            from < self.vertex_count() && to < self.vertex_count(),
            "DBM::is_connected: ({from}, {to}) out of range {}",
            self.vertex_count()
        );

        let index = Self::index(from, to);

        debug_assert!(
            index < self.matrix.len(),
            "DBM::is_connected: index {} out of bounds {} for ({from}, {to})",
            index,
            self.matrix.len()
        );

        self.matrix[index]
    }

    /// Returns whether `edge` is exactly the edge `(from, to)` and is present.
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        debug_assert!(
            from < self.vertex_count() && to < self.vertex_count(),
            "DBM::has_edge: ({from}, {to}) out of range {}",
            self.vertex_count()
        );
        debug_assert!(
            edge < self.matrix.len(),
            "DBM::has_edge: edge {} out of bounds {}",
            edge,
            self.matrix.len()
        );

        let index = Self::index(from, to);

        debug_assert!(
            index < self.matrix.len(),
            "DBM::has_edge: index {} out of bounds {} for ({from}, {to})",
            index,
            self.matrix.len()
        );

        edge == index && self.matrix[index]
    }
}

impl Graph for DBM {
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

/// Iterator over selected DBM edges.
///
/// Yields `(from, edge, to)`.
pub struct CbmEdges<'a> {
    dbm: &'a DBM,
    kind: CbmEdgesKind,
}

/// Internal iterator mode.
enum CbmEdgesKind {
    All { edge: usize },
    Outgoing { from: usize, to: usize },
    Ingoing { from: usize, to: usize },
    Between { from: usize, to: usize, state: u8 },
}

impl<'a> CbmEdges<'a> {
    /// Iterates over all edges.
    fn all(dbm: &'a DBM) -> Self {
        let vertices = dbm.vertex_count();
        let expected_len = if vertices == 0 {
            0
        } else {
            DBM::size(vertices)
        };

        debug_assert_eq!(
            dbm.matrix.len(),
            expected_len,
            "CbmEdges::all: matrix.len() = {} != size({vertices}) = {}",
            dbm.matrix.len(),
            expected_len
        );

        Self {
            dbm,
            kind: CbmEdgesKind::All { edge: 0 },
        }
    }

    /// Iterates over edges outgoing from `from`.
    fn outgoing(dbm: &'a DBM, from: usize) -> Self {
        debug_assert!(
            from < dbm.vertex_count(),
            "CbmEdges::outgoing: from {} out of range {}",
            from,
            dbm.vertex_count()
        );

        Self {
            dbm,
            kind: CbmEdgesKind::Outgoing { from, to: 0 },
        }
    }

    /// Iterates over edges incoming to `to`.
    fn ingoing(dbm: &'a DBM, to: usize) -> Self {
        debug_assert!(
            to < dbm.vertex_count(),
            "CbmEdges::ingoing: to {} out of range {}",
            to,
            dbm.vertex_count()
        );

        Self {
            dbm,
            kind: CbmEdgesKind::Ingoing { from: 0, to },
        }
    }

    /// Iterates over edges between `from` and `to`.
    fn between(dbm: &'a DBM, from: usize, to: usize) -> Self {
        debug_assert!(
            from < dbm.vertex_count() && to < dbm.vertex_count(),
            "CbmEdges::between: ({from}, {to}) out of range {}",
            dbm.vertex_count()
        );

        Self {
            dbm,
            kind: CbmEdgesKind::Between { from, to, state: 0 },
        }
    }
}

impl<'a> Iterator for CbmEdges<'a> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.kind {
            CbmEdgesKind::All { edge } => {
                while *edge < self.dbm.matrix.len() {
                    let current = *edge;
                    *edge += 1;

                    if self.dbm.matrix[current] {
                        let (from, to) = DBM::inverse_index(current);

                        debug_assert!(
                            from < self.dbm.vertex_count() && to < self.dbm.vertex_count(),
                            "CbmEdges::All: decoded ({from}, {to}) out of range {} for edge {}",
                            self.dbm.vertex_count(),
                            current
                        );

                        return Some((from, current, to));
                    }
                }
                None
            }

            CbmEdgesKind::Outgoing { from, to } => {
                while *to < self.dbm.vertex_count() {
                    let current_to = *to;
                    *to += 1;

                    let edge = DBM::index(*from, current_to);

                    debug_assert!(
                        edge < self.dbm.matrix.len(),
                        "CbmEdges::Outgoing: index {} out of bounds {} for ({}, {})",
                        edge,
                        self.dbm.matrix.len(),
                        *from,
                        current_to
                    );

                    if self.dbm.matrix[edge] {
                        return Some((*from, edge, current_to));
                    }
                }
                None
            }

            CbmEdgesKind::Ingoing { from, to } => {
                while *from < self.dbm.vertex_count() {
                    let current_from = *from;
                    *from += 1;

                    let edge = DBM::index(current_from, *to);

                    debug_assert!(
                        edge < self.dbm.matrix.len(),
                        "CbmEdges::Ingoing: index {} out of bounds {} for ({}, {})",
                        edge,
                        self.dbm.matrix.len(),
                        current_from,
                        *to
                    );

                    if self.dbm.matrix[edge] {
                        return Some((current_from, edge, *to));
                    }
                }
                None
            }

            CbmEdgesKind::Between { from, to, state } => {
                if *state == 0 {
                    *state = if *from == *to { 2 } else { 1 };

                    let edge = DBM::index(*from, *to);

                    debug_assert!(
                        edge < self.dbm.matrix.len(),
                        "CbmEdges::Between: index {} out of bounds {} for ({}, {})",
                        edge,
                        self.dbm.matrix.len(),
                        *from,
                        *to
                    );

                    if self.dbm.matrix[edge] {
                        return Some((*from, edge, *to));
                    }
                }

                if *state == 1 {
                    *state = 2;

                    let edge = DBM::index(*to, *from);

                    debug_assert!(
                        edge < self.dbm.matrix.len(),
                        "CbmEdges::Between: reversed index {} out of bounds {} for ({}, {})",
                        edge,
                        self.dbm.matrix.len(),
                        *to,
                        *from
                    );

                    if self.dbm.matrix[edge] {
                        return Some((*to, edge, *from));
                    }
                }

                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphs::graph::Endpoints;
    use proptest::prelude::*;
    use std::collections::HashSet;

    fn edge_set(graph: &DBM) -> HashSet<(usize, usize)> {
        graph.edges().map(|(from, _, to)| (from, to)).collect()
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
        assert_eq!(graph.connections(1, 2).count(), 1);
        assert_eq!(graph.loop_degree(2), 1);
        assert!(graph.has_edge(2, DBM::index(2, 2), 2));
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
        fn prop_growth_matches_size_difference(n in 1usize..=1_000) {
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
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to))
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
                edges.iter()
                    .copied()
                    .map(|(from, to)| Endpoints::new(from, to))
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
    }
}
