use crate::{
    games::{
        arena::Arena,
        players::Player,
        region::{ArenaRegion, Region, RegionInsertion},
    },
    graphs::expansion::{BackwardExpansion, Expansion, ForwardExpansion},
};

/// A region that can compute its own attractor using its attached arena.
pub trait Attractor<'arena>: ArenaRegion<'arena> + RegionInsertion {
    /// Performs one attractor expansion step:
    ///
    /// self := self ∪ Pre_attr(self)
    ///
    /// Returns whether `self` changed.
    fn expand_attractor(&mut self) -> bool
    where
        Self::Player: AttractorPredecessor,
        for<'g> BackwardExpansion<'g, Self::Arena>: Expansion<Vertex = Self::Position>,
        for<'g> ForwardExpansion<'g, Self::Arena>: Expansion<Vertex = Self::Position>,
    {
        let arena = self.arena();
        let player = self.owner();
        let backward = BackwardExpansion::new(arena);

        let mut predecessors = Vec::new();

        for position in self.positions() {
            for predecessor in backward.successors(position) {
                if self.contains(&predecessor) {
                    continue;
                }

                if player.is_attractor_predecessor(arena, self, predecessor) {
                    predecessors.push(predecessor);
                }
            }
        }

        let mut changed = false;

        for predecessor in predecessors {
            changed |= self.insert(predecessor);
        }

        changed
    }

    /// Computes the full reachability attractor:
    ///
    /// self := μY. self ∪ Pre_attr(Y)
    ///
    /// Returns whether `self` changed.
    fn attractor_closure(&mut self) -> bool
    where
        Self::Player: AttractorPredecessor,
        for<'g> BackwardExpansion<'g, Self::Arena>: Expansion<Vertex = Self::Position>,
        for<'g> ForwardExpansion<'g, Self::Arena>: Expansion<Vertex = Self::Position>,
    {
        let arena = self.arena();
        let player = self.owner();
        let backward = BackwardExpansion::new(arena);

        let mut queue: Vec<_> = self.positions().collect();
        let mut changed = false;

        while let Some(position) = queue.pop() {
            for predecessor in backward.successors(position) {
                if self.contains(&predecessor) {
                    continue;
                }

                if player.is_attractor_predecessor(arena, self, predecessor)
                    && self.insert(predecessor)
                {
                    changed = true;
                    queue.push(predecessor);
                }
            }
        }

        changed
    }
}

impl<'arena, R> Attractor<'arena> for R where R: ArenaRegion<'arena> + RegionInsertion {}

pub trait AttractorPredecessor: Player {
    fn attractor_successor<A, R>(
        self,
        arena: &A,
        target: &R,
        position: R::Position,
    ) -> Option<R::Position>
    where
        A: Arena<Player = Self, Position = R::Position>,
        R: Region<Player = Self>,
        for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = R::Position>;

    #[inline]
    fn is_attractor_predecessor<A, R>(self, arena: &A, target: &R, position: R::Position) -> bool
    where
        A: Arena<Player = Self, Position = R::Position>,
        R: Region<Player = Self>,
        for<'g> ForwardExpansion<'g, A>: Expansion<Vertex = R::Position>,
    {
        self.attractor_successor(arena, target, position).is_some()
    }
}
