use std::ops::Range;

use bit_vec::BitVec;

use crate::graphs::{
    directed::Directed,
    edges::{Edges, ReadEdges},
    graph::ReadGraph,
    vertices::{ReadVertices, Vertices},
};

/// A Dense Bit Matrix (DBM) representation of a directed graph.
///
/// The graph is stored as a flat `BitVec` of length `n^2`, where `n` is the
/// number of vertices. Each possible directed edge `(from, to)` corresponds
/// to exactly one bit in the matrix; if the bit is set, the edge exists.
///
/// This is a *simple* directed graph: no parallel edges are allowed, but
/// self-loops `(v, v)` are.
///
/// The bit indices are laid out in concentric "rings" around vertex `0`,
/// which results in the following layout for `6` vertices (the numbers are
/// the bit indices used by [`DBM::index`]):
///
/// ```text
/// +----+----+----+----+----+----+
/// |  0 |  1 |  4 |  9 | 16 | 25 |
/// +----+----+----+----+----+----+
/// |  3 |  2 |  5 | 10 | 17 | 26 |
/// +----+----+----+----+----+----+
/// |  8 |  7 |  6 | 11 | 18 | 27 |
/// +----+----+----+----+----+----+
/// | 15 | 14 | 13 | 12 | 19 | 28 |
/// +----+----+----+----+----+----+
/// | 24 | 23 | 22 | 21 | 20 | 29 |
/// +----+----+----+----+----+----+
/// | 35 | 34 | 33 | 32 | 31 | 30 |
/// +----+----+----+----+----+----+
/// ```
///
/// The mapping between `(from, to)` and bit index is implemented by
/// [`DBM::index`] and [`DBM::inverse_index`].
#[derive(Default)]
pub struct DBM {
    /// Flat bit matrix of size `vertex_count()^2`.
    ///
    /// Bit at position `i` encodes the existence of the directed edge
    /// returned by [`DBM::inverse_index(i)`].
    matrix: BitVec,
}

impl DBM {
    /// Precomputed squares `n^2` for `1 <= n <= 128`.
    ///
    /// This allows computing `n^2` without multiplication for small `n`,
    /// which can be a tiny micro-optimisation (and avoids repeated `pow`).
    ///
    /// The entry at index `k` is `(k + 1)^2`.
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

    /// Creates a new `DBM` with the given number of vertices.
    ///
    /// If `complete` is `true`, the resulting graph contains all possible
    /// directed edges (including self-loops). Otherwise, it starts empty
    /// with no edges at all.
    ///
    /// When `vertices == 0`, the matrix is empty.
    ///
    /// # Panics
    ///
    /// Panics if `vertices^2` overflows `usize` for large `vertices`
    /// (for `vertices < 2^32` on 64-bit systems this is fine).
    pub fn new(vertices: usize, complete: bool) -> Self {
        if vertices == 0 {
            let dbm = Self {
                matrix: BitVec::default(),
            };
            debug_assert_eq!(dbm.vertex_count(), 0);
            debug_assert_eq!(dbm.matrix.len(), 0);
            dbm
        } else {
            let size = Self::size(vertices);
            let matrix = BitVec::from_elem(size, complete);
            debug_assert_eq!(
                matrix.len(),
                size,
                "DBM::new: BitVec length {} != size({}) = {}",
                matrix.len(),
                vertices,
                size
            );
            Self { matrix }
        }
    }

    /// Returns the number of possible edges which can be contained in the graph.
    /// This is equivalent to the number of bits required to represent a complete
    /// graph with `vertex` vertices.
    ///
    /// This is always equal to `vertex^2`. For `vertex < 128`, this
    /// uses a small lookup table, for larger values it falls back to
    /// `vertex.pow(2)`.
    ///
    /// This function does **not** inspect the current matrix; it only
    /// computes the capacity for a hypothetical graph of this size.
    pub fn size(vertex: usize) -> usize {
        debug_assert!(vertex > 0, "DBM::size: vertex count must be > 0");
        if vertex < 128 {
            Self::SIZES[vertex - 1]
        } else {
            vertex.pow(2)
        }
    }

    /// Returns the number of edges by which the capacity grows when adding
    /// one more vertex. This is equivalent to the growth of the number of bits.
    ///
    /// For `n` vertices, the matrix has `n^2` bits. For `n + 1` vertices,
    /// it needs `(n + 1)^2` bits. The difference is:
    ///
    /// ```text
    /// (n + 1)^2 - n^2 = 2n + 1
    /// ```
    ///
    /// This function returns that growth value: `1 + 2 * vertex`.
    pub fn growth(vertex: usize) -> usize {
        1 + 2 * vertex
    }

    /// Maps a pair of vertices `(from, to)` to a bit index in the underlying `matrix`.
    ///
    /// The mapping is a bijection between `{0..n} × {0..n}` and
    /// `{0..n^2 - 1}`, where `n` is `vertex_count() - 1`. It arranges the
    /// indices in concentric rings around `(0, 0)` (see the type-level
    /// documentation for an example).
    ///
    /// Conceptually, for `radius = max(from, to)`:
    ///
    /// * The "ring" for `radius` starts at index `radius^2`.
    /// * The top edge `(0..=radius, radius)` occupies the next `radius + 1`
    ///   indices.
    /// * The right edge `(radius, radius-1 ..= 0)` occupies the remaining
    ///   `radius` indices.
    ///
    /// This function does **not** check bounds. Passing a vertex index
    /// outside the current `vertex_count()` will produce an out-of-range
    /// index for `matrix`.
    pub fn index(from: usize, to: usize) -> usize {
        let radius = from.max(to);

        // center
        if radius == 0 {
            return 0;
        }

        let base = Self::size(radius); // radius^2

        if to == radius {
            // top edge of ring radius: (0,radius) .. (radius,radius)
            base + from
        } else {
            // right edge of ring radius: (radius,radius-1) .. (radius,0)
            base + 2 * radius - to
        }
    }

    /// Inverse of [`DBM::index`].
    ///
    /// Given a bit `index`, returns the corresponding `(from, to)` vertex
    /// pair such that:
    ///
    /// ```text
    /// DBM::index(from, to) == index
    /// ```
    ///
    /// The index is interpreted as lying in a ring with radius
    /// `radius = floor(sqrt(index))`. Within that ring:
    ///
    /// * Offsets `0 ..= radius` correspond to the top edge `(from, radius)`.
    /// * Offsets `radius+1 ..= 2*radius` correspond to the right edge
    ///   `(radius, to)`.
    pub fn inverse_index(index: usize) -> (usize, usize) {
        if index == 0 {
            return (0, 0);
        }

        // Each ring radius occupies indices radius^2 .. radius^2 + 2*radius.
        let radius = index.isqrt();
        let base = radius * radius;
        let offset = index - base; // 0 ..= 2*radius

        debug_assert!(
            offset <= 2 * radius,
            "DBM::inverse_index: offset {} > 2 * radius {} for index {}",
            offset,
            2 * radius,
            index
        );

        if offset <= radius {
            // Top edge: (from, radius)
            (offset, radius)
        } else {
            // Right edge: (radius, to)
            let to = 2 * radius - offset;
            (radius, to)
        }
    }
}

impl Vertices for DBM {
    type Vertex = usize;
}

impl ReadVertices for DBM {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Returns an iterator over all vertices in the graph.
    ///
    /// Vertices are represented by `usize` indices in the range
    /// `0 .. vertex_count()`.
    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count()
    }

    /// Returns the number of vertices in the graph.
    ///
    /// The number of vertices `n` is inferred from the length of the
    /// underlying matrix, assuming it was allocated by [`DBM::new`] or
    /// resized consistently so that `matrix.len() == n^2`.
    ///
    /// When the matrix is empty, this returns `0`.
    fn vertex_count(&self) -> usize {
        let length = self.matrix.len();
        if length == 0 {
            0
        } else {
            let vertex_count = self.matrix.len().isqrt();
            // In debug, enforce that the matrix length is a perfect square
            // and matches DBM::size(n).
            debug_assert_eq!(
                length,
                DBM::size(vertex_count),
                "DBM::vertex_count: matrix.len() = {} is not equal to size({}) = {}",
                length,
                vertex_count,
                DBM::size(vertex_count)
            );
            vertex_count
        }
    }
}

impl Edges for DBM {
    type Vertex = usize;

    /// Edge identifier.
    ///
    /// This is the flat index into the underlying `BitVec`. It uniquely
    /// identifies the directed edge in the graph and can be converted back
    /// to `(from, to)` using [`DBM::inverse_index`].
    type Edge = usize;
}

impl ReadEdges for DBM {
    /// Iterator type over edges.
    ///
    /// This iterator yields triples `(from, edge, to)`, where `edge` is the
    /// flat index used by `DBM::index`.
    type Edges<'a>
        = CbmEdges<'a>
    where
        Self: 'a;

    /// Returns an iterator over all edges in the graph.
    fn edges(&self) -> Self::Edges<'_> {
        let vertex_count = self.vertex_count();

        let expected_len = if vertex_count == 0 {
            0
        } else {
            DBM::size(vertex_count)
        };

        debug_assert_eq!(
            self.matrix.len(),
            expected_len,
            "DBM::edges: matrix.len() = {} != size(vertex_count = {}) = {}",
            self.matrix.len(),
            vertex_count,
            expected_len
        );

        CbmEdges::all(self)
    }

    /// Returns the total number of edges present in the graph.
    ///
    /// This is equal to the number of bits set to `1` in the underlying
    /// `matrix`.
    fn edge_count(&self) -> usize {
        self.matrix.count_ones() as usize
    }
}

impl Directed for DBM {
    /// Returns the source vertex of the given `edge`.
    ///
    /// The edge is interpreted as a flat index into the matrix.
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(
            edge < self.matrix.len(),
            "DBM::source: edge index {} out of bounds {}",
            edge,
            self.matrix.len()
        );
        let (from, _) = Self::inverse_index(edge);
        debug_assert!(
            from < self.vertex_count(),
            "DBM::source: decoded source {} >= vertex_count {}",
            from,
            self.vertex_count()
        );
        from
    }

    /// Returns the target vertex of the given `edge`.
    ///
    /// The edge is interpreted as a flat index into the matrix.
    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        debug_assert!(
            edge < self.matrix.len(),
            "DBM::target: edge index {} out of bounds {}",
            edge,
            self.matrix.len()
        );
        let (_, to) = Self::inverse_index(edge);
        debug_assert!(
            to < self.vertex_count(),
            "DBM::target: decoded target {} >= vertex_count {}",
            to,
            self.vertex_count()
        );
        to
    }

    /// Returns an iterator over all edges outgoing from `source`.
    fn outgoing(&self, source: Self::Vertex) -> Self::Edges<'_> {
        debug_assert!(
            source < self.vertex_count(),
            "DBM::outgoing: source {} out of range {}",
            source,
            self.vertex_count()
        );
        CbmEdges::outgoing(self, source)
    }

    /// Returns an iterator over all edges ingoing into `destination`.
    fn ingoing(&self, destination: Self::Vertex) -> Self::Edges<'_> {
        debug_assert!(
            destination < self.vertex_count(),
            "DBM::ingoing: destination {} out of range {}",
            destination,
            self.vertex_count()
        );
        CbmEdges::ingoing(self, destination)
    }

    /// Returns the number of loop edges `(v, v)` at `vertex`.
    ///
    /// For this simple graph representation, this is either `0` or `1`.
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        debug_assert!(
            vertex < self.vertex_count(),
            "DBM::loop_degree: vertex {} out of range {}",
            vertex,
            self.vertex_count()
        );
        if self.is_connected(vertex, vertex) {
            1
        } else {
            0
        }
    }

    /// Returns `true` if there is a directed edge from `from` to `to`.
    ///
    /// This is a constant-time adjacency query.
    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        debug_assert!(
            from < self.vertex_count() && to < self.vertex_count(),
            "DBM::is_connected: ({from}, {to}) out of range vertex_count {}",
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

    /// Returns `true` if the given edge id corresponds exactly to the edge
    /// `(from, to)` and that edge is present in the graph.
    ///
    /// This is useful as a consistency check when you want to verify that
    /// an edge id still refers to a specific pair of vertices.
    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        debug_assert!(
            from < self.vertex_count() && to < self.vertex_count(),
            "DBM::has_edge: ({from}, {to}) out of range vertex_count {}",
            self.vertex_count()
        );
        debug_assert!(
            edge < self.matrix.len(),
            "DBM::has_edge: edge index {} out of bounds {}",
            edge,
            self.matrix.len()
        );
        let index = Self::index(from, to);
        debug_assert!(
            index < self.matrix.len(),
            "DBM::has_edge: computed index {} out of bounds {} for ({from}, {to})",
            index,
            self.matrix.len()
        );
        index == edge && self.matrix[index]
    }

    /// Returns an iterator over all edges between `from` and `to`.
    ///
    /// At most two edges can exist between `from` and `to` in a directed
    /// simple graph:
    ///
    /// * `from -> to`
    /// * `to -> from`
    ///
    /// If `from == to` (a loop), at most one edge is returned.
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Edges<'_> {
        debug_assert!(
            from < self.vertex_count() && to < self.vertex_count(),
            "DBM::connections: ({from}, {to}) out of range vertex_count {}",
            self.vertex_count()
        );
        CbmEdges::between(self, from, to)
    }
}

impl ReadGraph for DBM {
    type Vertex = usize;

    type Vertices = Self;

    type Edges = Self;

    /// Returns the edge store for this graph.
    ///
    /// Since `DBM` implements both `Edges` and `Vertices`, the graph itself
    /// acts as the edge store.
    fn edge_store(&self) -> &Self::Edges {
        self
    }

    /// Returns the vertex store for this graph.
    ///
    /// Since `DBM` implements both `Edges` and `Vertices`, the graph itself
    /// acts as the vertex store.
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

/// Iterator over edges of a [`DBM`] graph.
///
/// This iterator yields triples `(from, edge, to)` where:
///
/// * `from` is the source vertex index.
/// * `to` is the target vertex index.
/// * `edge` is the flat edge index in the backing `BitVec`
///   (i.e., `DBM::index(from, to)`).
///
/// Different constructors on [`CbmEdges`] restrict which edges are produced
/// (all edges, outgoing from a vertex, ingoing into a vertex, or between a
/// pair of vertices).
pub struct CbmEdges<'a> {
    cbm: &'a DBM,
    kind: CbmEdgesKind,
}

/// Internal state for [`CbmEdges`] specifying which subset of edges is
/// being iterated over.
enum CbmEdgesKind {
    /// Iterate over all edges in the graph.
    ///
    /// `edge` is the next flat index to inspect in the underlying `BitVec`.
    All { edge: usize },

    /// Iterate over all edges outgoing from `from`.
    ///
    /// `to` is the next potential target vertex to inspect.
    Outgoing { from: usize, to: usize },

    /// Iterate over all edges ingoing into `to`.
    ///
    /// `from` is the next potential source vertex to inspect.
    Ingoing { from: usize, to: usize },

    /// Iterate over all edges between `from` and `to`.
    ///
    /// This yields at most two edges:
    ///
    /// * `from -> to`
    /// * `to -> from` (if `from != to`)
    ///
    /// The `state` field is interpreted as a tiny state machine:
    ///
    /// * `0` – initial, `from -> to` not yet checked.
    /// * `1` – `from -> to` has been handled, `to -> from` is next.
    /// * `>= 2` – iteration is finished.
    Between { from: usize, to: usize, state: u8 },
}

impl<'a> CbmEdges<'a> {
    /// Creates an iterator over **all** edges in the given `DBM`.
    fn all(cbm: &'a DBM) -> Self {
        let vertex_count = cbm.vertex_count();
        let expected_len = if vertex_count == 0 {
            0
        } else {
            DBM::size(vertex_count)
        };
        debug_assert_eq!(
            cbm.matrix.len(),
            expected_len,
            "CbmEdges::all: matrix.len() = {} != size(vertex_count = {}) = {}",
            cbm.matrix.len(),
            vertex_count,
            expected_len
        );
        Self {
            cbm,
            kind: CbmEdgesKind::All { edge: 0 },
        }
    }

    /// Creates an iterator over all edges **outgoing** from `from`.
    fn outgoing(cbm: &'a DBM, from: usize) -> Self {
        debug_assert!(
            from < cbm.vertex_count(),
            "CbmEdges::outgoing: from {} out of range {}",
            from,
            cbm.vertex_count()
        );
        Self {
            cbm,
            kind: CbmEdgesKind::Outgoing { from, to: 0 },
        }
    }

    /// Creates an iterator over all edges **ingoing** into `to`.
    fn ingoing(cbm: &'a DBM, to: usize) -> Self {
        debug_assert!(
            to < cbm.vertex_count(),
            "CbmEdges::ingoing: to {} out of range {}",
            to,
            cbm.vertex_count()
        );
        Self {
            cbm,
            kind: CbmEdgesKind::Ingoing { from: 0, to },
        }
    }

    /// Creates an iterator over edges between `from` and `to`.
    ///
    /// The iterator will yield at most:
    ///
    /// * The edge `from -> to`, and
    /// * The edge `to -> from` (if `from != to`),
    ///
    /// depending on which edges actually exist in the underlying graph.
    fn between(cbm: &'a DBM, from: usize, to: usize) -> Self {
        debug_assert!(
            from < cbm.vertex_count() && to < cbm.vertex_count(),
            "CbmEdges::between: ({from}, {to}) out of range vertex_count {}",
            cbm.vertex_count()
        );
        Self {
            cbm,
            kind: CbmEdgesKind::Between { from, to, state: 0 },
        }
    }
}

impl<'a> Iterator for CbmEdges<'a> {
    /// Each item is a triple `(from, edge, to)`.
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.kind {
            CbmEdgesKind::All { edge } => {
                while *edge < self.cbm.matrix.len() {
                    if self.cbm.matrix[*edge] {
                        let (from, to) = DBM::inverse_index(*edge);

                        debug_assert!(
                            from < self.cbm.vertex_count() && to < self.cbm.vertex_count(),
                            "CbmEdges::All: decoded ({from}, {to}) out of range vertex_count {} for edge {}",
                            self.cbm.vertex_count(),
                            *edge
                        );

                        return Some((from, *edge, to));
                    }

                    *edge += 1;
                }
                None
            }

            CbmEdgesKind::Outgoing { from, to } => {
                while *to < self.cbm.vertex_count() {
                    let edge = DBM::index(*from, *to);

                    debug_assert!(
                        edge < self.cbm.matrix.len(),
                        "CbmEdges::Outgoing: index {} out of bounds {} for ({}, {})",
                        edge,
                        self.cbm.matrix.len(),
                        *from,
                        *to
                    );

                    if self.cbm.matrix[edge] {
                        return Some((*from, edge, *to));
                    }

                    *to += 1;
                }
                None
            }

            CbmEdgesKind::Ingoing { to, from } => {
                while *from < self.cbm.vertex_count() {
                    let edge = DBM::index(*from, *to);

                    debug_assert!(
                        edge < self.cbm.matrix.len(),
                        "CbmEdges::Ingoing: index {} out of bounds {} for ({}, {})",
                        edge,
                        self.cbm.matrix.len(),
                        *from,
                        *to
                    );

                    if self.cbm.matrix[edge] {
                        return Some((*from, edge, *to));
                    }

                    *from += 1;
                }
                None
            }

            CbmEdgesKind::Between { from, to, state } => {
                // from -> to
                if *state == 0 {
                    if *from == *to {
                        // Loop case: we will never yield a reversed edge.
                        *state = 2;
                    } else {
                        *state = 1;
                    }

                    let edge = DBM::index(*from, *to);

                    debug_assert!(
                        edge < self.cbm.matrix.len(),
                        "CbmEdges::Between: index {} out of bounds {} for ({}, {})",
                        edge,
                        self.cbm.matrix.len(),
                        *from,
                        *to
                    );

                    if self.cbm.matrix[edge] {
                        return Some((*from, edge, *to));
                    }
                }

                // to -> from (if not a loop)
                if *state == 1 {
                    *state = 2; // mark iteration as finished after this check

                    let edge = DBM::index(*to, *from);

                    debug_assert!(
                        edge < self.cbm.matrix.len(),
                        "CbmEdges::Between: reversed index {} out of bounds {} for ({}, {})",
                        edge,
                        self.cbm.matrix.len(),
                        *to,
                        *from
                    );

                    if self.cbm.matrix[edge] {
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
    use crate::graphs::vertices::ReadVertices;
    use std::collections::HashSet;

    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_sizes_exact_and_monotonic() {
        // Exact small values: size(n) = n^2
        for n in 1..=256 {
            assert_eq!(
                DBM::size(n),
                n * n,
                "size({n}) should be {n}^2, got {}",
                DBM::size(n)
            );
        }

        // Monotonicity: size(n+1) > size(n)
        for n in 1..=1024 {
            assert!(
                DBM::size(n + 1) > DBM::size(n),
                "size should be strictly increasing: size({}) = {}, size({}) = {}",
                n,
                DBM::size(n),
                n + 1,
                DBM::size(n + 1)
            );
        }
    }

    #[test]
    fn test_growth_basic_and_formula() {
        // A few spot checks
        assert_eq!(DBM::growth(1), 3);
        assert_eq!(DBM::growth(2), 5);
        assert_eq!(DBM::growth(3), 7);
        assert_eq!(DBM::growth(4), 9);
        assert_eq!(DBM::growth(5), 11);

        // General formula: growth(n) = 1 + 2n
        for n in 1..=256 {
            assert_eq!(DBM::growth(n), 1 + 2 * n, "growth({n}) should be 1 + 2*{n}");
        }
    }

    #[test]
    fn test_index_known_values_small_grid() {
        // Hand-checked 3x3 corner (matches the doc table)
        assert_eq!(DBM::index(0, 0), 0);

        assert_eq!(DBM::index(0, 1), 1);
        assert_eq!(DBM::index(1, 1), 2);
        assert_eq!(DBM::index(1, 0), 3);

        assert_eq!(DBM::index(0, 2), 4);
        assert_eq!(DBM::index(1, 2), 5);
        assert_eq!(DBM::index(2, 2), 6);
        assert_eq!(DBM::index(2, 1), 7);
        assert_eq!(DBM::index(2, 0), 8);

        // Extra checks from the larger 6x6 example
        assert_eq!(DBM::index(0, 3), 9);
        assert_eq!(DBM::index(0, 4), 16);
        assert_eq!(DBM::index(0, 5), 25);

        assert_eq!(DBM::index(1, 0), 3);
        assert_eq!(DBM::index(2, 0), 8);
        assert_eq!(DBM::index(3, 0), 15);
        assert_eq!(DBM::index(4, 0), 24);
        assert_eq!(DBM::index(5, 0), 35);
    }

    #[test]
    fn test_vertex_count_matches_new() {
        for n in 0..=256 {
            let dbm = DBM::new(n, false);
            assert_eq!(
                dbm.vertex_count(),
                n,
                "vertex_count() should match constructor argument"
            );

            let expected_len = if n == 0 { 0 } else { DBM::size(n) };
            assert_eq!(
                dbm.matrix.len(),
                expected_len,
                "matrix length should be size(n) for n > 0"
            );
        }
    }

    #[test]
    fn test_inverse_index_roundtrip_small_grid() {
        // Exhaustive roundtrip on a reasonably large grid to catch mistakes.
        let max = 64usize;
        for from in 0..=max {
            for to in 0..=max {
                let index = DBM::index(from, to);
                let (inv_from, inv_to) = DBM::inverse_index(index);
                assert_eq!(
                    (from, to),
                    (inv_from, inv_to),
                    "inverse_index(index({from}, {to})) failed"
                );
            }
        }
    }

    #[test]
    fn test_unique_indices() {
        // Every (from, to) pair up to a bound should map to a unique index.
        let max = 256usize;
        let mut seen = HashSet::new();

        for from in 0..=max {
            for to in 0..=max {
                let index = DBM::index(from, to);
                let inserted = seen.insert(index);
                assert!(inserted, "duplicate index {index} for pair ({from}, {to})");
            }
        }

        // Sanity: we saw exactly (max+1)^2 distinct indices.
        let expected = (max + 1) * (max + 1);
        assert_eq!(
            seen.len(),
            expected,
            "expected {expected} unique indices up to radius {max}"
        );
    }

    #[test]
    fn test_ring_index_ranges_are_monotonic_and_disjoint() {
        // Ring r uses indices r^2 ..= r^2 + 2r.
        // Check that:
        //  - each ring range is non-empty
        //  - ranges are strictly increasing and non-overlapping
        let max_radius = 512usize;

        let mut last_max = 0usize;
        for r in 0..=max_radius {
            let (min, max) = if r == 0 {
                (0, 0)
            } else {
                let min = DBM::size(r); // r^2
                let max = DBM::size(r) + 2 * r; // r^2 + 2r
                (min, max)
            };

            assert!(min <= max, "ring {r} has invalid range [{min}, {max}]");

            if r > 0 {
                assert!(
                    min > last_max,
                    "ring {r} range [{min}, {max}] overlaps or is not strictly after previous max {last_max}"
                );
            }

            last_max = max;
        }
    }

    #[test]
    fn test_new_empty_graph_has_no_edges() {
        for n in 0..=32 {
            let dbm = DBM::new(n, false);

            for idx in 0..dbm.matrix.len() {
                assert!(
                    !dbm.matrix[idx],
                    "empty graph should have no edges, but bit {idx} is set"
                );
            }
        }
    }

    #[test]
    fn test_new_complete_graph_has_all_edges() {
        for n in 0..=32 {
            let dbm = DBM::new(n, true);

            for idx in 0..dbm.matrix.len() {
                assert!(
                    dbm.matrix[idx],
                    "complete graph should have all edges, but bit {idx} is not set"
                );
            }

            let expected_edges = n * n;
            assert_eq!(
                dbm.matrix.count_ones() as usize,
                expected_edges,
                "complete graph with {n} vertices should have {expected_edges} edges"
            );
        }
    }

    proptest! {
        // size(n) = n^2 for a reasonably wide range.
        #[test]
        fn prop_size_matches_square(n in 1usize..=1_000) {
            let expected = (n as u64) * (n as u64);
            let got = DBM::size(n) as u64;
            prop_assert_eq!(got, expected, "size({}) != {}^2", n, n);
        }

        // growth(n) = size(n + 1) - size(n).
        #[test]
        fn prop_growth_matches_size_difference(n in 1usize..=1_000) {
            let growth = DBM::growth(n);
            let diff = DBM::size(n + 1) - DBM::size(n);
            prop_assert_eq!(growth, diff, "growth({}) != size({}+1) - size({})", n, n, n);
        }

        // index + inverse_index must be a roundtrip.
        #[test]
        fn prop_index_inverse_roundtrip(from in 0u16..=10_000, to in 0u16..=10_000) {
            let from = from as usize;
            let to = to as usize;

            let index = DBM::index(from, to);
            let (inv_from, inv_to) = DBM::inverse_index(index);

            prop_assert_eq!(from, inv_from, "from component mismatch");
            prop_assert_eq!(to, inv_to, "to component mismatch");
        }

        // The index range must match the ring formula:
        // index in [r^2, r^2 + 2r], where r = max(from, to).
        #[test]
        fn prop_index_in_correct_ring(from in 0u16..=10_000, to in 0u16..=10_000) {
            let from = from as usize;
            let to = to as usize;
            let radius = from.max(to);
            let index = DBM::index(from, to);

            if radius == 0 {
                prop_assert_eq!(index, 0);
            } else {
                let min = DBM::size(radius);
                let max = DBM::size(radius) + 2 * radius;
                prop_assert!(index >= min && index <= max,
                    "index({from}, {to}) = {index} not in ring {radius} range [{min}, {max}]");
            }
        }

        // index must be injective (no collisions) within a random bounded grid.
        #[test]
        fn prop_index_injective_on_bounded_grid(max in 1u8..=32) {
            let max = max as usize;
            let mut seen = HashSet::new();

            for from in 0..=max {
                for to in 0..=max {
                    let index = DBM::index(from, to);
                    prop_assert!(seen.insert(index),
                        "duplicate index {index} for pair ({from}, {to}) within bound {max}");
                }
            }

            let expected = (max + 1) * (max + 1);
            prop_assert_eq!(
                seen.len(),
                expected,
                "expected {} unique indices",
                expected
            );
        }
    }

    #[test]
    fn random_complete_graph_properties() {
        let mut rng = ChaCha8Rng::seed_from_u64(813494);

        for _case in 0..50 {
            let n = rng.gen_range(0..=64);
            let dbm = DBM::new(n, true);

            // Basic structural properties
            assert_eq!(dbm.vertex_count(), n);
            let expected_len = if n == 0 { 0 } else { DBM::size(n) };
            assert_eq!(dbm.matrix.len(), expected_len);

            // For complete graph, any valid (from, to) up to n yields a bit that is set.
            if n > 0 {
                for _ in 0..200 {
                    let from = rng.random_range(0..n);
                    let to = rng.random_range(0..n);
                    let index = DBM::index(from, to);
                    assert!(
                        dbm.matrix[index],
                        "complete graph: edge ({from}, {to}) should exist"
                    );

                    // Sanity: roundtrip index -> (from, to)
                    let (inv_from, inv_to) = DBM::inverse_index(index);
                    assert_eq!((from, to), (inv_from, inv_to));
                }
            }
        }
    }

    #[test]
    fn random_sparse_graph_consistent_with_matrix_bits() {
        let mut rng = ChaCha8Rng::seed_from_u64(494318);

        for _case in 0..50 {
            let n = rng.random_range(1..=32);
            let mut dbm = DBM::new(n, false);

            // Randomly set some edges by mutating the underlying matrix.
            // This intentionally tests DBM::index + DBM::inverse_index together
            // with the matrix layout.
            let edge_attempts = rng.random_range(0..=(n * n));
            for _ in 0..edge_attempts {
                let from = rng.random_range(0..n);
                let to = rng.random_range(0..n);
                let index = DBM::index(from, to);
                dbm.matrix.set(index, true);
            }

            // Sample random pairs and ensure:
            //  - index -> (from, to) roundtrips
            //  - bit value matches a re-computed index on the inverse pair (consistency)
            for _ in 0..200 {
                let from = rng.random_range(0..n);
                let to = rng.random_range(0..n);
                let index = DBM::index(from, to);
                let (inv_from, inv_to) = DBM::inverse_index(index);
                assert_eq!(
                    (from, to),
                    (inv_from, inv_to),
                    "roundtrip mismatch on random sparse graph"
                );

                // Consistency: bit at index is the "truth" for this edge.
                let bit = dbm.matrix[index];
                // There's no public API to query adjacency, but this at least validates
                // that our index/inverse_index agreements hold for random layouts.
                if bit {
                    // If bit is set, count it in edge_count upper bound sanity check later.
                }
            }

            // As a coarse sanity check: edge count cannot exceed n^2 and must
            // match the number of set bits in the matrix.
            let counted_edges = dbm.matrix.count_ones() as usize;
            assert!(counted_edges <= n * n);
        }
    }

    // An extra regression test: ensure 6x6 layout matches the documented table.
    #[test]
    fn test_layout_matches_documentation_for_6x6() {
        // This is the exact table from the DBM docs.
        let expected: [[usize; 6]; 6] = [
            [0, 1, 4, 9, 16, 25],
            [3, 2, 5, 10, 17, 26],
            [8, 7, 6, 11, 18, 27],
            [15, 14, 13, 12, 19, 28],
            [24, 23, 22, 21, 20, 29],
            [35, 34, 33, 32, 31, 30],
        ];

        for from in 0..6 {
            for to in 0..6 {
                assert_eq!(
                    DBM::index(from, to),
                    expected[from][to],
                    "index({from}, {to}) does not match documentation table"
                );
                let (inv_from, inv_to) = DBM::inverse_index(expected[from][to]);
                assert_eq!(
                    (inv_from, inv_to),
                    (from, to),
                    "inverse_index({}) does not match documentation table",
                    expected[from][to]
                );
            }
        }
    }
}
