use std::{fmt::Debug, hash::Hash};

use crate::{
    games::{
        arena::{Arena, FiniteArena},
        attractor::Attractor,
        controllable_predecessors::ControllablePredecessors,
        game::{Game, RegionSolvableGame, SolvableGame},
        play::VisitedPlay,
        play_sequence::PlaySequence,
        positional_map_strategy::PositionalMapStrategy,
        region::{DenseRegion, Region},
        strategic_play::StrategicPlay,
        worklist_attractor::WorklistAttractor,
    },
    graphs::{
        expansion::{BackwardExpansion, Expansion, ForwardExpansion},
        graph::FiniteDirected,
        structure::{FiniteEdges, FiniteVertices, Structure},
    }, lattices::lattice::MembershipLattice,
};

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
    pub fn new(arena: &'a A, goal: A::Position) -> Self {
        Self { arena, goal }
    }

    #[must_use]
    #[inline]
    pub fn goal(&self) -> A::Position {
        self.goal
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
    A: FiniteArena<Position = usize> + FiniteDirected + 'a,
    A::Player: ControllablePredecessors<A, DenseRegion> + Debug,
    <A as Structure>::Vertices: FiniteVertices<Vertex = usize>,
    <A as Structure>::Edges: FiniteEdges<Vertex = usize, Edge = A::Edge>,
    for<'g> BackwardExpansion<'g, A>: Expansion<Vertex = usize>,
    for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = usize>,
{
    type Region = DenseRegion;

    #[inline]
    fn winning_region(&self, player: A::Player) -> Self::Region {
        let vertex_count = self.arena.vertex_store().vertex_count();
        let mut region = DenseRegion::new(vertex_count);
        region.insert(self.goal);
        let attractor = WorklistAttractor::new();
        attractor.attractor_closure_from(self.arena, player, region)
    }
}

impl<'a, A> SolvableGame for ReachabilityGame<'a, A>
where
    A: FiniteArena<Position = usize> + FiniteDirected + 'a,
    A::Player: ControllablePredecessors<A, DenseRegion> + Debug,
    <A as Structure>::Vertices: FiniteVertices<Vertex = usize>,
    <A as Structure>::Edges: FiniteEdges<Vertex = usize, Edge = A::Edge>,
    for<'g> BackwardExpansion<'g, A>: Expansion<Vertex = usize>,
    for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = usize>,
{
    type Strategy = PositionalMapStrategy<A, PlaySequence<usize>>;

    fn winning_strategy_from(
        &self,
        player: A::Player,
        start: A::Position,
    ) -> Option<Self::Strategy> {
        let vertex_count = self.arena.vertex_store().vertex_count();

        let mut region = DenseRegion::new(vertex_count);
        region.expand(self.goal);

        let mut strategy = PositionalMapStrategy::empty(player);

        if start == self.goal {
            return Some(strategy);
        }

        let backward = BackwardExpansion::new(self.arena);
        let mut queue = vec![self.goal];

        while let Some(position) = queue.pop() {
            for predecessor in backward.successors(position) {
                if region.includes(predecessor) {
                    continue;
                }

                if !player.is_controllable_predecessor(self.arena, &region, predecessor) {
                    continue;
                }

                let choice = player.strategy_successor(self.arena, &region, predecessor);

                if region.expand(predecessor) {
                    if let Some(successor) = choice {
                        strategy.insert_choice(predecessor, successor);
                    }

                    if predecessor == start {
                        return Some(strategy);
                    }

                    queue.push(predecessor);
                }
            }
        }

        if region.includes(start) {
            Some(strategy)
        } else {
            None
        }
    }

    fn has_winning_strategy_from(&self, player: A::Player, start: A::Position) -> bool {
        self.winning_region(player).includes(start)
    }

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
