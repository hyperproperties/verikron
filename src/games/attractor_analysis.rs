use crate::{
    games::{
        arena::Arena,
        region::{DenseStaticRegion, Region},
    },
    graphs::structure::{FiniteVertices, Vertices},
    lattices::monotone::{Monotone, StatefulMonotone},
};

/// Monotone-framework implementation of a player attractor.
///
/// The target region contains the positions the player wants to force the game
/// into. The universe region restricts where the attractor may grow. The
/// attracted region is the mutable fixed-point state computed by the solver.
///
/// For ordinary reachability games, the universe is usually all positions.
/// For nested objectives, such as Büchi games, the universe can be a smaller
/// candidate region.
pub struct AttractorAnalysis<
    'a,
    A: Arena,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
    Storage: Region<A::Position>,
> {
    arena: &'a A,
    player: A::Player,
    target: Target,
    universe: Universe,
    attracted: Option<Storage>,
}

impl<'a, A, Target, Universe, Storage> AttractorAnalysis<'a, A, Target, Universe, Storage>
where
    A: Arena,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    /// Creates an attractor analysis restricted to `universe`.
    #[must_use]
    #[inline]
    pub fn new(arena: &'a A, player: A::Player, target: Target, universe: Universe) -> Self {
        Self {
            arena,
            player,
            target,
            universe,
            attracted: None,
        }
    }

    #[must_use]
    #[inline]
    pub fn player(&self) -> &A::Player {
        &self.player
    }

    #[must_use]
    #[inline]
    pub fn arena(&self) -> &'a A {
        self.arena
    }

    #[must_use]
    #[inline]
    pub fn target(&self) -> &Target {
        &self.target
    }

    #[must_use]
    #[inline]
    pub fn universe(&self) -> &Universe {
        &self.universe
    }

    #[must_use]
    #[inline]
    pub fn attracted(&self) -> Option<&Storage> {
        self.attracted.as_ref()
    }
}

impl<'a, A, Target> AttractorAnalysis<'a, A, Target, DenseStaticRegion, DenseStaticRegion>
where
    A: Arena<Position = usize>,
    A::Vertices: FiniteVertices<Vertex = usize>,
    Target: Region<A::Position>,
{
    /// Creates an unrestricted attractor analysis over all positions.
    #[must_use]
    #[inline]
    pub fn unrestricted(arena: &'a A, player: A::Player, target: Target) -> Self {
        let mut universe = DenseStaticRegion::new(arena.vertex_store().vertex_count());

        for node in arena.vertex_store().vertices() {
            universe.expand(node);
        }

        Self::new(arena, player, target, universe)
    }
}

impl<'a, A, Target, Universe, Storage> Monotone<A>
    for AttractorAnalysis<'a, A, Target, Universe, Storage>
where
    A: Arena,
    A::Vertex: Copy,
    A::Player: PartialEq,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
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

    /// Forces target positions inside the universe to be attracted.
    ///
    /// Positions outside the universe are forced to `false`. Returning `None`
    /// leaves the position to be computed normally by the solver.
    fn boundary_fact(&self, node: &A::Vertex) -> Option<bool> {
        if !self.universe.includes(node) {
            Some(false)
        } else if self.target.includes(node) {
            Some(true)
        } else {
            None
        }
    }

    /// Preserves target positions and prevents attraction outside the universe.
    fn transfer(&self, node: &A::Vertex, input: &Self::Fact) -> Self::Fact {
        self.universe.includes(node) && (*input || self.target.includes(node))
    }

    /// Merges successor facts according to the attractor rule.
    ///
    /// Player-owned positions need at least one attracted successor. Opponent-owned
    /// positions need all successors attracted. Non-target dead ends are treated as
    /// not attracted.
    fn merge(&self, node: &A::Vertex, mut facts: impl Iterator<Item = Self::Fact>) -> Self::Fact {
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

impl<'a, A, Target, Universe> StatefulMonotone<A>
    for AttractorAnalysis<'a, A, Target, Universe, DenseStaticRegion>
where
    A: Arena<Position = usize>,
    A::Vertex: Copy,
    A::Player: PartialEq,
    A::Vertices: FiniteVertices<Vertex = usize>,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
{
    /// The computed attractor region.
    type Output = DenseStaticRegion;

    /// Reads the current stored fact for a position.
    fn fact(&self, node: &A::Vertex) -> Self::Fact {
        self.attracted
            .as_ref()
            .expect("attractor analysis must be initialized before reading facts")
            .includes(node)
    }

    /// Initializes the mutable attractor state from the target region.
    fn initialize(&mut self, graph: &A) {
        let mut attracted = DenseStaticRegion::new(graph.vertex_store().vertex_count());

        for node in graph.vertex_store().vertices() {
            if self.universe.includes(&node) && self.target.includes(&node) {
                attracted.expand(node);
            }
        }

        self.attracted = Some(attracted);
    }

    /// Updates the mutable attractor state.
    ///
    /// Attractor computation is monotone-growing, so positions are only added.
    /// Positions outside the universe are never added.
    fn set(&mut self, node: &A::Vertex, fact: &Self::Fact) -> bool {
        let attracted = self
            .attracted
            .as_mut()
            .expect("attractor analysis must be initialized before setting facts");

        *fact && self.universe.includes(node) && attracted.expand(*node)
    }

    /// Returns the computed attractor region.
    fn finish(self) -> Self::Output {
        self.attracted
            .expect("attractor analysis must be initialized before finishing")
    }
}
