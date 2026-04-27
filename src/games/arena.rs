use std::hash::Hash;

use crate::graphs::{
    graph::{Directed, FiniteGraph},
    structure::{FiniteEdges, FiniteVertices, Structure},
};

/// A turn-based arena.
///
/// An arena is a directed graph whose positions are owned by players.
/// The owner of a position chooses the next move from that position.
pub trait Arena: Directed<Vertex = Self::Position> {
    /// Player type.
    type Player: Copy + Eq;

    /// Position type.
    type Position: Eq + Hash + Copy;

    /// Returns the owner of `position`.
    fn owner(&self, position: Self::Position) -> Self::Player;
}

/// A finite arena.
///
/// This is a finite directed arena with explicit position and edge stores.
pub trait FiniteArena: Arena + FiniteGraph<Vertex = Self::Position>
where
    <Self as Structure>::Vertices: FiniteVertices,
    <Self as Structure>::Edges: FiniteEdges,
{
}
