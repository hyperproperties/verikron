use std::hash::Hash;

use crate::{
    games::{
        arena::{Arena, FiniteArena},
        game::{Game, RegionSolvableGame, SolvableGame},
        lasso_play::LassoPlay,
        play::VisitedPlay,
        positional_map_strategy::PositionalMapStrategy,
        region::DenseDynamicRegion,
        safety_analysis::SafetyAnalysis,
        safety_strategy_synthesis::{SafetyStrategyResult, SafetyStrategySynthesis},
        strategic_play::StrategicPlay,
        strategy::SynthesisResult,
    },
    graphs::{
        graph::FiniteDirected,
        structure::{FiniteEdges, FiniteVertices, Structure},
    },
    lattices::{
        fixpoint::Fixpoint, lattice::MembershipLattice, monotone::BackwardDirection,
        worklist::Worklist,
    },
};

/// Arena requirements needed by dense safety-game solvers.
pub trait SafetyArena: FiniteArena<Position = usize> + FiniteDirected
where
    Self: Structure<
            Vertices: FiniteVertices<Vertex = usize>,
            Edges: FiniteEdges<Vertex = usize, Edge = Self::Edge>,
        >,
{
}

impl<A> SafetyArena for A where
    A: FiniteArena<Position = usize>
        + FiniteDirected
        + Structure<
            Vertices: FiniteVertices<Vertex = usize>,
            Edges: FiniteEdges<Vertex = usize, Edge = A::Edge>,
        >
{
}

/// A safety game with one or more safe positions.
///
/// The objective is to keep the play inside the safe region forever.
/// A play is winning for a player iff every visited position is safe.
///
/// The solver uses infinite-play safety semantics: a winning position must be
/// able to continue safely. In particular, safe dead ends are treated as losing
/// by the underlying safety analysis, because the play cannot continue.
#[derive(Clone, Debug)]
pub struct SafetyGame<'a, A>
where
    A: Arena,
{
    arena: &'a A,
    safe: Vec<A::Position>,
}

impl<'a, A> SafetyGame<'a, A>
where
    A: Arena,
{
    /// Creates a safety game with the given safe positions.
    #[must_use]
    #[inline]
    pub fn new<I>(arena: &'a A, safe: I) -> Self
    where
        I: IntoIterator<Item = A::Position>,
    {
        Self {
            arena,
            safe: safe.into_iter().collect(),
        }
    }

    /// Creates a safety game with a single safe position.
    #[must_use]
    #[inline]
    pub fn singleton(arena: &'a A, safe: A::Position) -> Self {
        Self::new(arena, [safe])
    }

    /// Returns the safe positions.
    #[must_use]
    #[inline]
    pub fn safe_positions(&self) -> &[A::Position] {
        &self.safe
    }

    /// Checks whether `position` is safe.
    #[must_use]
    #[inline]
    pub fn is_safe(&self, position: A::Position) -> bool
    where
        A::Position: PartialEq,
    {
        self.safe.contains(&position)
    }
}

impl<'a, A> SafetyGame<'a, A>
where
    A: SafetyArena,
{
    /// Builds the dense safe region used by the safety analysis.
    #[inline]
    fn safe_region(&self) -> DenseDynamicRegion {
        let mut region = DenseDynamicRegion::new(self.arena.vertex_store().vertex_count());

        for &position in &self.safe {
            region.insert(position);
        }

        region
    }

    /// Creates the backward worklist solver used for safety computations.
    #[inline]
    fn worklist(&self) -> Worklist<'_, A, BackwardDirection<'_, A>> {
        Worklist::new(self.arena, BackwardDirection::new(self.arena))
    }

    /// Creates the safety analysis from the safe region.
    #[inline]
    fn safety_analysis(
        &self,
        player: A::Player,
    ) -> SafetyAnalysis<'_, A, DenseDynamicRegion, DenseDynamicRegion> {
        SafetyAnalysis::new(self.arena, player, self.safe_region())
    }

    /// Computes the player's safety-winning region.
    #[inline]
    fn safety_region(&self, player: A::Player) -> DenseDynamicRegion {
        self.worklist().solve(self.safety_analysis(player))
    }

    /// Computes the player's safety-winning region together with a positional strategy.
    #[inline]
    fn safety_strategy(&self, player: A::Player) -> SafetyStrategyResult<A> {
        self.worklist()
            .solve(SafetyStrategySynthesis::new(self.safety_analysis(player)))
    }
}

impl<'a, A> Game for SafetyGame<'a, A>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
{
    type Arena = A;
    type Play = LassoPlay<A::Position>;

    #[inline]
    fn arena(&self) -> &Self::Arena {
        self.arena
    }

    /// Checks whether the observed play stays inside the safe region.
    #[inline]
    fn is_winning(&self, _player: A::Player, play: &Self::Play) -> bool {
        play.visited().all(|position| self.is_safe(position))
    }
}

impl<'a, A> RegionSolvableGame for SafetyGame<'a, A>
where
    A: SafetyArena,
{
    type Region = DenseDynamicRegion;

    /// Computes the safety-winning region as the greatest fixed point of
    /// positions from which the player can keep the play inside the safe region
    /// forever.
    #[inline]
    fn winning_region(&self, player: A::Player) -> Self::Region {
        self.safety_region(player)
    }
}

impl<'a, A> SolvableGame for SafetyGame<'a, A>
where
    A: SafetyArena,
{
    type Strategy = PositionalMapStrategy<A>;

    /// Synthesizes a positional winning strategy from `start`, if one exists.
    #[inline]
    fn winning_strategy_from(
        &self,
        player: A::Player,
        start: A::Position,
    ) -> SynthesisResult<Self::Strategy> {
        let result = self.safety_strategy(player);

        if result.region.contains(&start) {
            SynthesisResult::winning(result.strategy)
        } else {
            SynthesisResult::losing()
        }
    }

    /// Checks whether `player` has a strategy to keep the play inside the safe
    /// region forever from `start`.
    #[inline]
    fn has_winning_strategy_from(&self, player: A::Player, start: A::Position) -> bool {
        self.winning_region(player).contains(&start)
    }

    /// Checks whether following `strategy` from `start` keeps the observed play
    /// inside the safe region.
    #[inline]
    fn is_winning_strategy_from(
        &self,
        _player: A::Player,
        strategy: &Self::Strategy,
        start: A::Position,
    ) -> bool {
        StrategicPlay::new(strategy, start)
            .visited()
            .all(|position| self.is_safe(position))
    }
}
