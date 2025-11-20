use std::ops::Range;

use bit_vec::BitVec;

use crate::graphs::{directed::Directed, edges::Edges, graph::Graph, vertices::Vertices};

/// A Dense Bit Matrix (DBM) representation of a directed graph.
/// 
/// This is not a multi graph so no parallel edges are allowed.
/// 
/// Example: 6 vertices.
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
#[derive(Default)]
pub struct DBM {
    matrix: BitVec,
}

impl DBM {
    /// The number of bits used by vertex count.
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

    /// If complete is true then every vertex is connected and you create a complete graph.
    /// Otherwise, you start from scratch and no vertex is connected.
    pub fn new(vertices: usize, complete: bool) -> Self {
        if vertices == 0 {
            return Self {
                matrix: BitVec::default(),
            };
        } else {
            Self {
                matrix: BitVec::from_elem(Self::size(vertices), complete),
            }
        }
    }

    pub fn size(vertex: usize) -> usize {
        if vertex < 128 {
            Self::SIZES[vertex - 1]
        } else {
            vertex.pow(2)
        }
    }

    pub fn growth(vertex: usize) -> usize {
        1 + 2 * vertex
    }

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
}

impl Vertices for DBM {
    type Vertex = usize;

    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    /// Iterator over all vertices in the graph.
    ///
    /// Vertices are the integers from zero up to `vertex_count`.
    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count()
    }

    /// Number of vertices.
    fn vertex_count(&self) -> usize {
        let length = self.matrix.len();
        if length == 0 {
            0
        } else {
            self.matrix.len().isqrt()
        }
    }
}

impl Edges for DBM {
    type Vertex = usize;

    type Edge = usize;

    type Edges<'a>
        = CbmEdges<'a>
    where
        Self: 'a;

    fn edges(&self) -> Self::Edges<'_> {
        todo!()
    }

    fn edge_count(&self) -> usize {
        self.matrix.count_ones() as usize
    }
}

impl Directed for DBM {
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        todo!()
    }

    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        todo!()
    }

    fn outgoing(&self, source: Self::Vertex) -> Self::Edges<'_> {
        todo!()
    }

    fn ingoing(&self, destination: Self::Vertex) -> Self::Edges<'_> {
        todo!()
    }

    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        if self.is_connected(vertex, vertex) {
            1
        } else {
            0
        }
    }

    fn is_connected(&self, from: Self::Vertex, to: Self::Vertex) -> bool {
        self.matrix[Self::index(from, to)]
    }

    fn has_edge(&self, from: Self::Vertex, edge: Self::Edge, to: Self::Vertex) -> bool {
        let index = Self::index(from, to);
        index == edge && self.matrix[index]
    }
    
    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Edges<'_> {
        todo!()
    }
}

impl Graph for DBM {
    type Vertices = Self;

    type Edges = Self;

    fn edge_store(&self) -> &Self::Edges {
        self
    }

    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

pub struct CbmEdges<'a> {
    cbm: &'a DBM,
}

impl<'a> Iterator for CbmEdges<'a> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::graphs::{dbm::DBM, vertices::Vertices};

    #[test]
    fn test_sizes() {
        for i in 1..=256 {
            assert_eq!(DBM::size(i), i.pow(2))
        }
    }

    #[test]
    fn test_growth() {
        assert_eq!(DBM::growth(1), 3);
        assert_eq!(DBM::growth(2), 5);
        assert_eq!(DBM::growth(3), 7);
        assert_eq!(DBM::growth(4), 9);
        assert_eq!(DBM::growth(5), 11);
        for i in 1..=256 {
            assert_eq!(DBM::growth(i), 1 + i * 2);
        }
    }

    #[test]
    fn test_index() {
        assert_eq!(DBM::index(0, 0), 0);

        assert_eq!(DBM::index(0, 1), 1);
        assert_eq!(DBM::index(0, 2), 4);
        assert_eq!(DBM::index(0, 3), 9);
        assert_eq!(DBM::index(0, 4), 16);
        assert_eq!(DBM::index(0, 5), 25);

        assert_eq!(DBM::index(1, 0), 3);
        assert_eq!(DBM::index(2, 0), 8);
        assert_eq!(DBM::index(3, 0), 15);
        assert_eq!(DBM::index(4, 0), 24);
        assert_eq!(DBM::index(5, 0), 35);

        assert_eq!(DBM::index(0, 0), 0);
        assert_eq!(DBM::index(0, 1), 1);
        assert_eq!(DBM::index(1, 1), 2);
        assert_eq!(DBM::index(1, 0), 3);

        assert_eq!(DBM::index(0, 2), 4);
        assert_eq!(DBM::index(1, 2), 5);
        assert_eq!(DBM::index(2, 2), 6);
        assert_eq!(DBM::index(2, 1), 7);
        assert_eq!(DBM::index(2, 0), 8);
    }

    #[test]
    fn test_vertex_count() {
        for i in 0..=256 {
            assert_eq!(DBM::new(i, false).vertex_count(), i);
        }
    }
}
