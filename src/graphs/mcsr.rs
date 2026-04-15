use std::ops::Range;

use crate::graphs::graph::{Directed, EdgeType, Edges, Graph, VertexType, Vertices};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MCSR {
    offsets: Vec<usize>,
    indices: Vec<usize>,
}

impl MCSR {
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            indices: Vec::new(),
        }
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn row_range(&self, vertex: usize) -> Option<(usize, usize)> {
        if vertex >= self.vertex_count() {
            return None;
        }

        let start = self.offsets[vertex];
        let end = self.offsets[vertex + 1];
        Some((start, end))
    }

    #[inline]
    fn locate_source(&self, edge: usize) -> usize {
        debug_assert!(edge < self.edge_count());

        let i = self.offsets.partition_point(|&off| off <= edge);
        debug_assert!(i > 0);
        i - 1
    }

    pub fn insert_vertex(&mut self) -> usize {
        let v = self.vertex_count();
        let last = *self.offsets.last().unwrap();
        self.offsets.push(last);
        v
    }

    pub fn insert_edge(&mut self, from: usize, to: usize) -> Option<usize> {
        if from >= self.vertex_count() || to >= self.vertex_count() {
            return None;
        }

        let edge = self.offsets[from + 1];
        self.indices.insert(edge, to);

        for off in &mut self.offsets[from + 1..] {
            *off += 1;
        }

        Some(edge)
    }
}

impl VertexType for MCSR {
    type Vertex = usize;
}

impl Vertices for MCSR {
    type Vertices<'a>
        = Range<usize>
    where
        Self: 'a;

    fn vertices(&self) -> Self::Vertices<'_> {
        0..self.vertex_count()
    }
}

impl EdgeType for MCSR {
    type Edge = usize;
}

impl Edges for MCSR {
    type Edges<'a>
        = Box<dyn Iterator<Item = (Self::Vertex, Self::Edge, Self::Vertex)> + 'a>
    where
        Self: 'a;

    fn edges(&self) -> Self::Edges<'_> {
        Box::new((0..self.edge_count()).map(|edge| {
            let source = self.locate_source(edge);
            let target = self.indices[edge];
            (source, edge, target)
        }))
    }
}

impl Directed for MCSR {
    type Outgoing<'a>
        = <Self as Edges>::Edges<'a>
    where
        Self: 'a;

    type Ingoing<'a>
        = <Self as Edges>::Edges<'a>
    where
        Self: 'a;

    type Connections<'a>
        = <Self as Edges>::Edges<'a>
    where
        Self: 'a;

    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        assert!(edge < self.edge_count(), "edge out of bounds");
        self.locate_source(edge)
    }

    fn target(&self, edge: Self::Edge) -> Self::Vertex {
        assert!(edge < self.edge_count(), "edge out of bounds");
        self.indices[edge]
    }

    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        assert!(source < self.vertex_count(), "vertex out of bounds");

        let (start, end) = self.row_range(source).unwrap();

        Box::new((start..end).map(move |edge| (source, edge, self.indices[edge])))
    }

    fn ingoing(&self, destination: Self::Vertex) -> Self::Ingoing<'_> {
        assert!(destination < self.vertex_count(), "vertex out of bounds");

        Box::new((0..self.edge_count()).filter_map(move |edge| {
            let target = self.indices[edge];
            (target == destination).then(|| (self.locate_source(edge), edge, target))
        }))
    }

    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        assert!(vertex < self.vertex_count(), "vertex out of bounds");

        let (start, end) = self.row_range(vertex).unwrap();

        self.indices[start..end]
            .iter()
            .filter(|&&target| target == vertex)
            .count()
    }

    fn connections(&self, from: Self::Vertex, to: Self::Vertex) -> Self::Connections<'_> {
        assert!(from < self.vertex_count(), "source vertex out of bounds");
        assert!(to < self.vertex_count(), "destination vertex out of bounds");

        let (start, end) = self.row_range(from).unwrap();

        Box::new((start..end).filter_map(move |edge| {
            let target = self.indices[edge];
            (target == to).then_some((from, edge, target))
        }))
    }
}

impl Graph for MCSR {
    type Vertices = Self;
    type Edges = Self;

    /// Access to the edge store.
    fn edge_store(&self) -> &Self::Edges {
        self
    }

    /// Access to the vertex store.
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}
