use std::hash::Hash;

use crate::{
    games::{
        arena::{Arena, FiniteArena},
        attractor_analysis::AttractorAnalysis,
        attractor_strategy_synthesis::{AttractorStrategyResult, AttractorStrategySynthesis},
        game::{Game, RegionSolvableGame, SolvableGame},
        play::VisitedPlay,
        play_sequence::PlaySequence,
        positional_map_strategy::PositionalMapStrategy,
        region::DenseRegion,
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

/// Arena requirements needed by dense reachability-game solvers.
pub trait ReachabilityArena: FiniteArena<Position = usize> + FiniteDirected
where
    Self: Structure<
            Vertices: FiniteVertices<Vertex = usize>,
            Edges: FiniteEdges<Vertex = usize, Edge = Self::Edge>,
        >,
{
}

impl<A> ReachabilityArena for A where
    A: FiniteArena<Position = usize>
        + FiniteDirected
        + Structure<
            Vertices: FiniteVertices<Vertex = usize>,
            Edges: FiniteEdges<Vertex = usize, Edge = A::Edge>,
        >
{
}

/// A reachability game with one or more target positions.
///
/// A play is winning for a player iff it eventually visits one of the goals.
#[derive(Clone, Debug)]
pub struct ReachabilityGame<'a, A>
where
    A: Arena,
{
    arena: &'a A,
    goals: Vec<A::Position>,
}

impl<'a, A> ReachabilityGame<'a, A>
where
    A: Arena,
{
    /// Creates a reachability game with the given goal positions.
    #[must_use]
    #[inline]
    pub fn new<I>(arena: &'a A, goals: I) -> Self
    where
        I: IntoIterator<Item = A::Position>,
    {
        Self {
            arena,
            goals: goals.into_iter().collect(),
        }
    }

    /// Creates a reachability game with a single goal position.
    #[must_use]
    #[inline]
    pub fn singleton(arena: &'a A, goal: A::Position) -> Self {
        Self::new(arena, [goal])
    }

    /// Returns the goal positions.
    #[must_use]
    #[inline]
    pub fn goals(&self) -> &[A::Position] {
        &self.goals
    }

    /// Checks whether `position` is a goal.
    #[must_use]
    #[inline]
    pub fn is_goal(&self, position: A::Position) -> bool
    where
        A::Position: PartialEq,
    {
        self.goals.contains(&position)
    }
}

impl<'a, A> ReachabilityGame<'a, A>
where
    A: ReachabilityArena,
{
    /// Builds the dense target region used as the attractor seed.
    #[inline]
    fn goal_region(&self) -> DenseRegion {
        let mut region = DenseRegion::new(self.arena.vertex_store().vertex_count());

        for &goal in &self.goals {
            region.insert(goal);
        }

        region
    }

    /// Creates the backward worklist solver used for attractor computations.
    #[inline]
    fn worklist(&self) -> Worklist<'_, A, BackwardDirection<'_, A>> {
        Worklist::new(self.arena, BackwardDirection::new(self.arena))
    }

    /// Creates the attractor analysis from the goal region.
    #[inline]
    fn attractor_analysis(
        &self,
        player: A::Player,
    ) -> AttractorAnalysis<'_, A, DenseRegion, DenseRegion> {
        AttractorAnalysis::new(self.arena, player, self.goal_region())
    }

    /// Computes the player's reachability winning region.
    #[inline]
    fn attractor(&self, player: A::Player) -> DenseRegion {
        self.worklist().solve(self.attractor_analysis(player))
    }

    /// Computes the player's attractor together with a positional strategy.
    #[inline]
    fn attractor_strategy(&self, player: A::Player) -> AttractorStrategyResult<A> {
        self.worklist().solve(AttractorStrategySynthesis::new(
            self.attractor_analysis(player),
        ))
    }
}

impl<'a, A> Game for ReachabilityGame<'a, A>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
{
    type Arena = A;
    type Play = PlaySequence<A::Position>;

    #[inline]
    fn arena(&self) -> &Self::Arena {
        self.arena
    }

    #[inline]
    fn is_winning(&self, _player: A::Player, play: &Self::Play) -> bool {
        play.visited().any(|position| self.is_goal(position))
    }
}

impl<'a, A> RegionSolvableGame for ReachabilityGame<'a, A>
where
    A: ReachabilityArena,
{
    type Region = DenseRegion;

    /// Computes the reachability winning region as the player's attractor to
    /// the goal region.
    #[inline]
    fn winning_region(&self, player: A::Player) -> Self::Region {
        self.attractor(player)
    }
}

impl<'a, A> SolvableGame for ReachabilityGame<'a, A>
where
    A: ReachabilityArena,
{
    type Strategy = PositionalMapStrategy<A>;

    /// Synthesizes a positional winning strategy from `start`, if one exists.
    #[inline]
    fn winning_strategy_from(
        &self,
        player: A::Player,
        start: A::Position,
    ) -> SynthesisResult<Self::Strategy> {
        let result = self.attractor_strategy(player);

        if result.region.contains(&start) {
            SynthesisResult::winning(result.strategy)
        } else {
            SynthesisResult::losing()
        }
    }

    /// Checks whether `player` can force a visit to a goal from `start`.
    #[inline]
    fn has_winning_strategy_from(&self, player: A::Player, start: A::Position) -> bool {
        self.winning_region(player).contains(&start)
    }

    /// Checks whether following `strategy` from `start` reaches a goal.
    #[inline]
    fn is_winning_strategy_from(
        &self,
        _player: A::Player,
        strategy: &Self::Strategy,
        start: A::Position,
    ) -> bool {
        StrategicPlay::new(strategy, start)
            .visited()
            .any(|position| self.is_goal(position))
    }
}
