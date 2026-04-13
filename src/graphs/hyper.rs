use crate::graphs::vertices::ReadVertices;

pub trait Hyperedges {
    type Vertex: Eq + Copy;
    type Hyperedge: Eq + Copy;
}

pub trait ReadHyperedges: Hyperedges {
    type Hyperedges<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    fn hyperedges(&self) -> Self::Hyperedges<'_>;

    fn hyperedge_count(&self) -> usize {
        self.hyperedges().count()
    }
}

pub trait ReadHypergraph {
    type Vertex: Eq + Copy;
    type Hyperedge: Eq + Copy;

    type Vertices: ReadVertices<Vertex = Self::Vertex>;
    type Hyperedges: ReadHyperedges<Vertex = Self::Vertex, Hyperedge = Self::Hyperedge>;

    fn vertex_store(&self) -> &Self::Vertices;
    fn hyperedge_store(&self) -> &Self::Hyperedges;

    fn size(&self) -> usize {
        self.vertex_store().vertex_count() + self.hyperedge_store().hyperedge_count()
    }

    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

pub trait UndirectedHypergraph: ReadHyperedges {
    type Members<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    type Incident<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    fn members(&self, e: Self::Hyperedge) -> Self::Members<'_>;
    fn incident(&self, v: Self::Vertex) -> Self::Incident<'_>;

    fn cardinality(&self, e: Self::Hyperedge) -> usize {
        self.members(e).count()
    }

    fn degree(&self, v: Self::Vertex) -> usize {
        self.incident(v).count()
    }

    fn contains(&self, e: Self::Hyperedge, v: Self::Vertex) -> bool {
        self.members(e).any(|u| u == v)
    }
}

pub trait DirectedHypergraph: ReadHyperedges {
    type Tail<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    type Head<'a>: Iterator<Item = Self::Vertex>
    where
        Self: 'a;

    type Outgoing<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    type Ingoing<'a>: Iterator<Item = Self::Hyperedge>
    where
        Self: 'a;

    fn tail(&self, e: Self::Hyperedge) -> Self::Tail<'_>;
    fn head(&self, e: Self::Hyperedge) -> Self::Head<'_>;
    fn outgoing(&self, v: Self::Vertex) -> Self::Outgoing<'_>;
    fn ingoing(&self, v: Self::Vertex) -> Self::Ingoing<'_>;

    fn tail_cardinality(&self, e: Self::Hyperedge) -> usize {
        self.tail(e).count()
    }

    fn head_cardinality(&self, e: Self::Hyperedge) -> usize {
        self.head(e).count()
    }

    fn outgoing_degree(&self, v: Self::Vertex) -> usize {
        self.outgoing(v).count()
    }

    fn ingoing_degree(&self, v: Self::Vertex) -> usize {
        self.ingoing(v).count()
    }
}

pub trait InsertUndirectedHyperedge: Hyperedges {
    fn insert_hyperedge<I>(&mut self, members: I) -> Option<Self::Hyperedge>
    where
        I: IntoIterator<Item = Self::Vertex>;
}

pub trait InsertDirectedHyperedge: Hyperedges {
    fn insert_hyperedge<T, H>(&mut self, tail: T, head: H) -> Option<Self::Hyperedge>
    where
        T: IntoIterator<Item = Self::Vertex>,
        H: IntoIterator<Item = Self::Vertex>;
}

pub trait RemoveHyperedge: Hyperedges {
    fn remove_hyperedge(&mut self, e: Self::Hyperedge) -> bool;
}
