use std::{fmt::Debug, hash::Hash};

use crate::{
    games::{
        arena::{Arena, FiniteArena},
        attractor::Attractor,
        controllable_predecessors::ControllablePredecessors,
        game::{Game, RegionSolvableGame, SolvableGame},
        play::VisitedPlay,
        play_sequence::PlaySequence,
        players::OpposedPlayer,
        positional_map_strategy::PositionalMapStrategy,
        region::{DenseRegion, Region},
        strategic_play::StrategicPlay,
        strategy::{PositionalStrategy, Strategy, SynthesisResult},
        worklist_attractor::WorklistAttractor,
    },
    graphs::{
        expansion::{BackwardExpansion, Expansion, ForwardExpansion},
        graph::FiniteDirected,
        structure::{FiniteEdges, FiniteVertices, Structure},
    },
    lattices::lattice::MembershipLattice,
};

/// Arena requirements needed by dense safety-game solvers.
pub trait SafetyArena:
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

impl<A> SafetyArena for A
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

/// A safety game with one or more bad positions.
///
/// A play is winning for a player iff it never visits a bad position.
#[derive(Clone, Debug)]
pub struct SafetyGame<'a, A>
where
    A: Arena,
{
    arena: &'a A,
    bad_positions: Vec<A::Position>,
}

impl<'a, A> SafetyGame<'a, A>
where
    A: Arena,
{
    #[must_use]
    #[inline]
    pub fn new<I>(arena: &'a A, bad_positions: I) -> Self
    where
        I: IntoIterator<Item = A::Position>,
    {
        Self {
            arena,
            bad_positions: bad_positions.into_iter().collect(),
        }
    }

    #[must_use]
    #[inline]
    pub fn singleton(arena: &'a A, bad_position: A::Position) -> Self {
        Self {
            arena,
            bad_positions: vec![bad_position],
        }
    }

    #[must_use]
    #[inline]
    pub fn bad_positions(&self) -> &[A::Position] {
        &self.bad_positions
    }

    #[must_use]
    #[inline]
    pub fn is_bad(&self, position: A::Position) -> bool
    where
        A::Position: PartialEq,
    {
        self.bad_positions.contains(&position)
    }
}

impl<'a, A> SafetyGame<'a, A>
where
    A: SafetyArena + 'a,
{
    #[inline]
    fn bad_region(&self) -> DenseRegion {
        let vertex_count = self.arena.vertex_store().vertex_count();

        let mut region = DenseRegion::new(vertex_count);

        for &bad_position in &self.bad_positions {
            region.insert(bad_position);
        }

        region
    }

    #[inline]
    fn complement_region(&self, region: &DenseRegion) -> DenseRegion {
        let vertex_count = self.arena.vertex_store().vertex_count();

        let mut complement = DenseRegion::new(vertex_count);

        for position in 0..vertex_count {
            if !region.includes(position) {
                complement.insert(position);
            }
        }

        complement
    }
}

impl<'a, A> Game for SafetyGame<'a, A>
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
        play.visited().all(|position| !self.is_bad(position))
    }
}

impl<'a, A> RegionSolvableGame for SafetyGame<'a, A>
where
    A: SafetyArena + 'a,
    A::Player: ControllablePredecessors<A, DenseRegion> + OpposedPlayer + Debug,
{
    type Region = DenseRegion;

    #[inline]
    fn winning_region(&self, player: A::Player) -> Self::Region {
        let losing_region =
            WorklistAttractor::new().attractor(self.arena, player.opponent(), self.bad_region());

        self.complement_region(&losing_region)
    }
}

impl<'a, A> SolvableGame for SafetyGame<'a, A>
where
    A: SafetyArena + 'a,
    A::Player: ControllablePredecessors<A, DenseRegion> + OpposedPlayer + Debug,
{
    type Strategy = PositionalMapStrategy<A, PlaySequence<usize>>;

    #[inline]
    fn winning_strategy_from(
        &self,
        player: A::Player,
        start: A::Position,
    ) -> SynthesisResult<Self::Strategy> {
        let winning_region = self.winning_region(player);

        if !winning_region.includes(start) {
            return SynthesisResult::losing();
        }

        let mut strategy = Self::Strategy::empty(player);

        for position in winning_region.positions() {
            if let Some(successor) =
                player.strategy_successor(self.arena, &winning_region, position)
            {
                strategy.insert_choice(position, successor);
            }
        }

        SynthesisResult::winning(strategy)
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
            .all(|position| !self.is_bad(position))
    }
}
