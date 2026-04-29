use crate::{
    games::{
        arena::Arena,
        region::{DenseDynamicRegion, Region},
    },
    graphs::structure::{FiniteVertices, Vertices},
    lattices::monotone::{Monotone, StatefulMonotone},
};

/// Monotone-framework implementation of a safety objective.
///
/// The safe region contains the positions the player wants to stay inside
/// forever. The winning region is the mutable greatest-fixed-point state
/// computed by the solver.
pub struct SafetyAnalysis<'a, A, Safe, Storage>
where
    A: Arena,
    Safe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    arena: &'a A,
    player: A::Player,
    safe: Safe,
    winning: Option<Storage>,
}

impl<'a, A, Safe, Storage> SafetyAnalysis<'a, A, Safe, Storage>
where
    A: Arena,
    Safe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    #[must_use]
    #[inline]
    pub fn new(arena: &'a A, player: A::Player, safe: Safe) -> Self {
        Self {
            arena,
            player,
            safe,
            winning: None,
        }
    }

    #[must_use]
    #[inline]
    pub fn player(&self) -> &A::Player {
        &self.player
    }

    #[must_use]
    #[inline]
    pub fn arena(&self) -> &A {
        self.arena
    }

    #[must_use]
    #[inline]
    pub fn safe(&self) -> &Safe {
        &self.safe
    }

    #[must_use]
    #[inline]
    pub fn winning(&self) -> Option<&Storage> {
        self.winning.as_ref()
    }
}

impl<'a, A, Safe, Storage> Monotone<A> for SafetyAnalysis<'a, A, Safe, Storage>
where
    A: Arena,
    A::Vertex: Copy,
    A::Player: PartialEq,
    Safe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    /// Whether a position is currently considered safety-winning.
    ///
    /// `false` means losing; `true` means still considered winning.
    type Fact = bool;

    /// Starts the greatest-fixed-point computation with positions winning.
    fn initial_fact(&self) -> Self::Fact {
        true
    }

    /// Forces unsafe positions to be losing.
    ///
    /// Returning `None` leaves safe positions to be computed normally by the solver.
    fn boundary_fact(&self, node: &A::Vertex) -> Option<Self::Fact> {
        (!self.safe.includes(node)).then_some(false)
    }

    /// Preserves only positions that are safe and can continue safely.
    fn transfer(&self, node: &A::Vertex, input: &Self::Fact) -> Self::Fact {
        self.safe.includes(node) && *input
    }

    /// Merges successor facts according to the safety rule.
    ///
    /// Player-owned positions need at least one winning successor. Opponent-owned
    /// positions need all successors winning. Dead ends are treated as losing,
    /// so winning positions must be able to continue safely.
    fn merge(&self, node: &A::Vertex, mut facts: impl Iterator<Item = Self::Fact>) -> Self::Fact {
        if self.arena.owner(*node) == self.player {
            match facts.next() {
                None => false,
                Some(first) => first || facts.any(|fact| fact),
            }
        } else {
            match facts.next() {
                None => false,
                Some(first) => first && facts.all(|fact| fact),
            }
        }
    }
}

impl<'a, A, Safe> StatefulMonotone<A> for SafetyAnalysis<'a, A, Safe, DenseDynamicRegion>
where
    A: Arena<Position = usize>,
    A::Vertex: Copy,
    A::Player: PartialEq,
    A::Vertices: FiniteVertices,
    Safe: Region<A::Position>,
{
    /// The computed safety-winning region.
    type Output = DenseDynamicRegion;

    /// Reads the current stored fact for a position.
    fn fact(&self, node: &A::Vertex) -> Self::Fact {
        self.winning
            .as_ref()
            .expect("safety analysis must be initialized before reading facts")
            .includes(node)
    }

    /// Initializes the mutable winning region from the safe region.
    fn initialize(&mut self, graph: &A) {
        let mut winning = DenseDynamicRegion::new(graph.vertex_store().vertex_count());

        for node in graph.vertex_store().vertices() {
            if self.safe.includes(&node) {
                winning.expand(node);
            }
        }

        self.winning = Some(winning);
    }

    /// Updates the mutable winning region.
    ///
    /// Safety computation is greatest-fixed-point based, so positions are only
    /// removed when they can no longer guarantee staying safe.
    fn set(&mut self, node: &A::Vertex, fact: &Self::Fact) -> bool {
        let winning = self
            .winning
            .as_mut()
            .expect("safety analysis must be initialized before setting facts");

        !*fact && winning.contract(*node)
    }

    /// Returns the computed safety-winning region.
    fn finish(self) -> Self::Output {
        self.winning
            .expect("safety analysis must be initialized before finishing")
    }
}
