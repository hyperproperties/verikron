use std::{fmt::Debug, hash::Hash};

use crate::{
    games::{
        arena::{Arena, FiniteArena},
        attractor::{Attractor, AttractorPredecessor},
        dense_region::DenseRegion,
        game::{Game, RegionSolvableGame, SolvableGame},
        play::Play,
        play_sequence::PlaySequence,
        positional_map_strategy::PositionalMapStrategy,
        region::{Region, RegionInsertion},
        strategic_play::StrategicPlay,
    },
    graphs::{
        expansion::{BackwardExpansion, Expansion, ForwardExpansion},
        graph::FiniteDirected,
        structure::{FiniteEdges, FiniteVertices, Structure},
    },
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

impl<'a, A> SolvableGame for ReachabilityGame<'a, A>
where
    A: FiniteArena<Position = usize> + FiniteDirected + 'a,
    A::Player: AttractorPredecessor + Debug,
    <A as Structure>::Vertices: FiniteVertices<Vertex = A::Position>,
    <A as Structure>::Edges: FiniteEdges<Vertex = A::Position, Edge = A::Edge>,
    for<'g> BackwardExpansion<'g, A>: Expansion<Vertex = usize>,
    for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = usize>,
{
    type Strategy = PositionalMapStrategy<A, PlaySequence<usize>>;

    fn winning_strategy_from(
        &self,
        player: <Self::Arena as Arena>::Player,
        start: <Self::Arena as Arena>::Position,
    ) -> Option<Self::Strategy> {
        let mut region = DenseRegion::new(player, self.arena);
        region.insert(self.goal);

        let mut strategy = PositionalMapStrategy::empty(player);

        let backward = BackwardExpansion::new(self.arena);
        let mut queue: Vec<_> = region.positions().collect();

        while let Some(position) = queue.pop() {
            for predecessor in backward.successors(position) {
                if region.contains(&predecessor) {
                    continue;
                }

                let Some(successor) = player.attractor_successor(self.arena, &region, predecessor)
                else {
                    continue;
                };

                if region.insert(predecessor) {
                    if self.arena.owner(predecessor) == player {
                        strategy.insert_choice(predecessor, successor);
                    }

                    if predecessor == start {
                        return Some(strategy);
                    }

                    queue.push(predecessor);
                }
            }
        }

        if region.contains(&start) {
            Some(strategy)
        } else {
            None
        }
    }

    fn has_winning_strategy_from(
        &self,
        player: <Self::Arena as Arena>::Player,
        start: <Self::Arena as Arena>::Position,
    ) -> bool {
        let region = self.winning_region(player);
        region.contains(&start)
    }

    fn is_winning_strategy_from(
        &self,
        _player: <Self::Arena as Arena>::Player,
        strategy: &Self::Strategy,
        start: <Self::Arena as Arena>::Position,
    ) -> bool {
        StrategicPlay::new(strategy, start)
            .visited()
            .any(|position| position == self.goal)
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
    fn is_winning(&self, _player: <Self::Arena as Arena>::Player, play: &Self::Play) -> bool {
        play.visited().any(|position| position == self.goal)
    }
}

impl<'a, A> RegionSolvableGame for ReachabilityGame<'a, A>
where
    A: FiniteArena<Position = usize> + FiniteDirected + 'a,
    A::Player: AttractorPredecessor + Debug,
    <A as Structure>::Vertices: FiniteVertices<Vertex = A::Position>,
    <A as Structure>::Edges: FiniteEdges<Vertex = A::Position, Edge = A::Edge>,
    for<'g> BackwardExpansion<'g, A>: Expansion<Vertex = usize>,
    for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = usize>,
{
    type Region = DenseRegion<'a, A>;

    #[inline]
    fn winning_region(&self, player: <Self::Arena as Arena>::Player) -> Self::Region {
        let mut region = DenseRegion::new(player, self.arena);

        region.insert(self.goal);
        region.attractor_closure();

        region
    }
}
