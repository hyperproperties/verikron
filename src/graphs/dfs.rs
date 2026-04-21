use std::hash::Hash;

use crate::graphs::{
    backward::Backward, forward::Forward, frontier::StackFrontier, search::Search,
    sequential_backward_search::SequentialBackwardSearch,
    sequential_forward_search::SequentialGraphSearch, structure::VertexOf, visited::Visited,
};

/// Depth-first search marker.
pub trait DFS: Search {}

impl<'g, G, V> DFS for SequentialGraphSearch<'g, G, V, StackFrontier<VertexOf<G>>>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
{
}

impl<'g, G, V> DFS for SequentialBackwardSearch<'g, G, V, StackFrontier<VertexOf<G>>>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
{
}
