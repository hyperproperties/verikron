use crate::{
    games::{
        arena::Arena,
        region::{DenseRegion, Region},
    },
    graphs::structure::{FiniteVertices, Vertices},
    lattices::monotone::{Monotone, StatefulMonotone},
};

/// Monotone-framework implementation of a player attractor.
///
/// The target region contains the positions the player wants to force the game
/// into. The attracted region is the mutable fixed-point state computed by the
/// solver.
pub struct AttractorAnalysis<'a, A: Arena, R: Region<A::Position>, Storage: Region<A::Position>> {
    arena: &'a A,
    player: A::Player,
    target: R,
    attracted: Option<Storage>,
}

impl<'a, A, R, Storage> AttractorAnalysis<'a, A, R, Storage>
where
    A: Arena,
    R: Region<A::Position>,
    Storage: Region<A::Position>,
{
    pub fn new(arena: &'a A, player: A::Player, target: R) -> Self {
        Self {
            arena,
            player,
            target,
            attracted: None,
        }
    }

    pub fn player(&self) -> &A::Player {
        &self.player
    }

    pub fn arena(&self) -> &A {
        &self.arena
    }

    pub fn target(&self) -> &R {
        &self.target
    }

    pub fn attracted(&self) -> Option<&Storage> {
        self.attracted.as_ref()
    }
}

impl<'a, A, Target, Storage> Monotone<A> for AttractorAnalysis<'a, A, Target, Storage>
where
    A: Arena,
    A::Vertex: Copy,
    A::Player: PartialEq,
    Target: Region<A::Position>,
    Storage: Region<A::Position>,
{
    /// Whether a position is currently known to be attracted.
    ///
    /// `false` means not attracted; `true` means attracted.
    type Fact = bool;

    /// Starts the fixed-point computation with no non-target positions attracted.
    fn initial_fact(&self) -> Self::Fact {
        false
    }

    /// Forces target positions to be attracted.
    ///
    /// Returning `None` leaves the position to be computed normally by the solver.
    fn boundary_fact(&self, node: &A::Vertex) -> Option<bool> {
        self.target.includes(node).then_some(true)
    }

    /// Preserves target positions and otherwise forwards the merged input fact.
    fn transfer(&self, node: &A::Vertex, input: &Self::Fact) -> Self::Fact {
        *input || self.target.includes(node)
    }

    /// Merges successor facts according to the attractor rule.
    ///
    /// Player-owned positions need at least one attracted successor. Opponent-owned
    /// positions need all successors attracted. Dead ends are treated as not attracted.
    fn merge(&self, node: &<A>::Vertex, mut facts: impl Iterator<Item = Self::Fact>) -> Self::Fact {
        if self.arena.owner(*node) == self.player {
            // The player can choose a successor that reaches the attractor.
            facts.any(|fact| fact)
        } else {
            // The opponent must be unable to avoid the attractor.
            match facts.next() {
                None => false,
                Some(first) => first && facts.all(|fact| fact),
            }
        }
    }
}

impl<'a, A, Target> StatefulMonotone<A> for AttractorAnalysis<'a, A, Target, DenseRegion>
where
    A: Arena<Position = usize>,
    A::Vertex: Copy,
    A::Player: PartialEq,
    A::Vertices: FiniteVertices,
    Target: Region<A::Position>,
{
    /// The computed attractor region.
    type Output = DenseRegion;

    /// Reads the current stored fact for a position.
    fn fact(&self, node: &A::Vertex) -> Self::Fact {
        self.attracted
            .as_ref()
            .expect("attractor analysis must be initialized before reading facts")
            .includes(node)
    }

    /// Initializes the mutable attractor state from the target region.
    fn initialize(&mut self, graph: &A) {
        let mut attracted = DenseRegion::new(graph.vertex_store().vertex_count());

        for node in graph.vertex_store().vertices() {
            if self.target.includes(&node) {
                attracted.expand(node);
            }
        }

        self.attracted = Some(attracted);
    }

    /// Updates the mutable attractor state.
    ///
    /// Attractor computation is monotone-growing, so positions are only added.
    fn set(&mut self, node: &A::Vertex, fact: &Self::Fact) -> bool {
        let attracted = self
            .attracted
            .as_mut()
            .expect("attractor analysis must be initialized before setting facts");

        *fact && attracted.expand(*node)
    }

    /// Returns the computed attractor region.
    fn finish(self) -> Self::Output {
        self.attracted
            .expect("attractor analysis must be initialized before finishing")
    }
}
