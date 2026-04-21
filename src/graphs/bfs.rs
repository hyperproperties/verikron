use std::hash::Hash;

use crate::graphs::{
    backward::Backward, forward::Forward, frontier::QueueFrontier, search::Search,
    sequential_backward_search::SequentialBackwardSearch,
    sequential_forward_search::SequentialGraphSearch, structure::VertexOf, visited::Visited,
};

/// Breadth-first search marker.
pub trait BFS: Search {}

impl<'g, G, V> BFS for SequentialGraphSearch<'g, G, V, QueueFrontier<VertexOf<G>>>
where
    G: Forward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
{
}

impl<'g, G, V> BFS for SequentialBackwardSearch<'g, G, V, QueueFrontier<VertexOf<G>>>
where
    G: Backward,
    VertexOf<G>: Eq + Hash + Copy,
    V: Visited<VertexOf<G>>,
{
}
