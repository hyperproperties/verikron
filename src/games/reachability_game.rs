use std::{fmt::Debug, hash::Hash};

use crate::{
    games::{
        arena::{Arena, FiniteArena},
        attractor::{Attractor, AttractorStrategySynthesis},
        controllable_predecessors::ControllablePredecessors,
        game::{Game, RegionSolvableGame, SolvableGame},
        play::VisitedPlay,
        play_sequence::PlaySequence,
        positional_map_strategy::PositionalMapStrategy,
        region::{DenseRegion, Region},
        strategic_play::StrategicPlay,
        strategy::SynthesisResult,
        worklist_attractor::WorklistAttractor,
        worklist_attractor_strategy_synthesis::WorklistAttractorStrategySynthesis,
    },
    graphs::{
        expansion::{BackwardExpansion, Expansion, ForwardExpansion},
        graph::FiniteDirected,
        structure::{FiniteEdges, FiniteVertices, Structure},
    },
    lattices::lattice::MembershipLattice,
};

/// Arena requirements needed by dense reachability-game solvers.
pub trait ReachabilityArena:
    FiniteArena<Position = usize>
    + FiniteDirected
    + Structure<
        Vertices: FiniteVertices<Vertex = usize>,
        Edges: FiniteEdges<Vertex = usize, Edge = Self::Edge>,
    >
where
    for<'g> BackwardExpansion<'g, Self>: Expansion<Vertex = usize>,
    for<'g> ForwardExpansion<'g, Self>: Expansion<Vertex = usize>,
{
}

impl<A> ReachabilityArena for A
where
    A: FiniteArena<Position = usize>
        + FiniteDirected
        + Structure<
            Vertices: FiniteVertices<Vertex = usize>,
            Edges: FiniteEdges<Vertex = usize, Edge = A::Edge>,
        >,
    for<'g> BackwardExpansion<'g, A>: Expansion<Vertex = usize>,
    for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = usize>,
{
}

/// A reachability game with a single target position.
///
/// A play is winning for a player iff it eventually visits `goal`.
#[derive(Clone, Copy, Debug)]
pub struct ReachabilityGame<'a, A>
where
    A: Arena,
{
    arena: &'a A,
    goal: A::Position,
}

impl<'a, A> ReachabilityGame<'a, A>
where
    A: Arena,
{
    #[must_use]
    #[inline]
    pub const fn new(arena: &'a A, goal: A::Position) -> Self {
        Self { arena, goal }
    }

    #[must_use]
    #[inline]
    pub const fn goal(&self) -> A::Position {
        self.goal
    }
}

impl<'a, A> ReachabilityGame<'a, A>
where
    A: ReachabilityArena + 'a,
{
    #[inline]
    fn goal_region(&self) -> DenseRegion {
        let vertex_count = self.arena.vertex_store().vertex_count();

        let mut region = DenseRegion::new(vertex_count);
        region.insert(self.goal);

        region
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
        play.visited().any(|position| position == self.goal)
    }
}

impl<'a, A> RegionSolvableGame for ReachabilityGame<'a, A>
where
    A: ReachabilityArena + 'a,
    A::Player: ControllablePredecessors<A, DenseRegion> + Debug,
{
    type Region = DenseRegion;

    #[inline]
    fn winning_region(&self, player: A::Player) -> Self::Region {
        WorklistAttractor::new().attractor(self.arena, player, self.goal_region())
    }
}

impl<'a, A> SolvableGame for ReachabilityGame<'a, A>
where
    A: ReachabilityArena + 'a,
    A::Player: ControllablePredecessors<A, DenseRegion> + Debug,
{
    type Strategy = PositionalMapStrategy<A, PlaySequence<usize>>;

    #[inline]
    fn winning_strategy_from(
        &self,
        player: A::Player,
        start: A::Position,
    ) -> SynthesisResult<Self::Strategy> {
        WorklistAttractorStrategySynthesis::new()
            .synthesize(self.arena, player, start, self.goal_region())
            .into()
    }

    #[inline]
    fn has_winning_strategy_from(&self, player: A::Player, start: A::Position) -> bool {
        self.winning_region(player).includes(start)
    }

    #[inline]
    fn is_winning_strategy_from(
        &self,
        _player: A::Player,
        strategy: &Self::Strategy,
        start: A::Position,
    ) -> bool {
        StrategicPlay::new(strategy, start)
            .visited()
            .any(|position| position == self.goal)
    }
}
