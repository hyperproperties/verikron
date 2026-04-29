use crate::{
    games::{
        arena::Arena,
        attractor_analysis::AttractorAnalysis,
        players::OpposedPlayer,
        region::{DenseDynamicRegion, Region},
    },
    graphs::{
        forward::Forward,
        structure::{FiniteVertices, Vertices},
    },
    lattices::{fixpoint::Fixpoint, monotone::BackwardDirection, worklist::Worklist},
};

/// Monotone-style analysis for a two-player Büchi objective.
///
/// The accepting region contains the positions the player wants to visit
/// infinitely often. The winning region is computed by an outer greatest
/// fixed point that repeatedly solves restricted attractor subproblems.
///
/// This analysis is for adversarial two-player games, so the player type must
/// provide a unique opponent.
pub struct BuchiAnalysis<'a, A, Accepting>
where
    A: Arena<Position = usize, Vertex = usize>,
    Accepting: Region<usize>,
{
    arena: &'a A,
    player: A::Player,
    accepting: Accepting,
}

impl<'a, A, Accepting> BuchiAnalysis<'a, A, Accepting>
where
    A: Arena<Position = usize, Vertex = usize>,
    Accepting: Region<usize>,
{
    #[must_use]
    #[inline]
    pub fn new(arena: &'a A, player: A::Player, accepting: Accepting) -> Self {
        Self {
            arena,
            player,
            accepting,
        }
    }

    #[must_use]
    #[inline]
    pub fn arena(&self) -> &'a A {
        self.arena
    }

    #[must_use]
    #[inline]
    pub fn player(&self) -> A::Player
    where
        A::Player: Copy,
    {
        self.player
    }

    #[must_use]
    #[inline]
    pub fn accepting(&self) -> &Accepting {
        &self.accepting
    }
}

/// Fixed-point solver for two-player Büchi analyses.
///
/// The solver maintains a candidate winning region. In each iteration it:
///
/// 1. computes the player's restricted attractor to accepting positions,
/// 2. finds the positions that cannot force such a visit,
/// 3. removes the opponent's restricted attractor to those bad positions.
///
/// When no bad positions remain, the candidate is the Büchi-winning region.
pub struct Buchi;

impl Buchi {
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Builds the accepting target used by the current Büchi iteration.
    ///
    /// An accepting position is a valid target only if it is still inside the
    /// candidate region and the play can continue inside that region.
    #[inline]
    fn accepting_target<A, Accepting>(
        analysis: &BuchiAnalysis<'_, A, Accepting>,
        candidate: &DenseDynamicRegion,
    ) -> DenseDynamicRegion
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Player: OpposedPlayer,
        A::Vertices: FiniteVertices<Vertex = usize>,
        Accepting: Region<usize>,
    {
        let mut target = DenseDynamicRegion::new(analysis.arena.vertex_store().vertex_count());

        for node in analysis.arena.vertex_store().vertices() {
            if candidate.includes(&node)
                && analysis.accepting.includes(&node)
                && Self::can_continue_inside(analysis.arena, analysis.player, node, candidate)
            {
                target.expand(node);
            }
        }

        target
    }

    /// Checks whether `player` can keep the next step inside `region`.
    ///
    /// Player-owned positions need at least one successor inside the region.
    /// Opponent-owned positions need all successors inside the region. Dead
    /// ends are treated as losing because Büchi uses infinite-play semantics.
    #[inline]
    fn can_continue_inside<A>(
        arena: &A,
        player: A::Player,
        node: usize,
        region: &DenseDynamicRegion,
    ) -> bool
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Player: OpposedPlayer,
    {
        let mut successors = arena
            .successors(node)
            .map(|edge| arena.destination(edge))
            .map(|successor| region.includes(&successor));

        if arena.owner(node) == player {
            successors.any(|inside| inside)
        } else {
            match successors.next() {
                None => false,
                Some(first) => first && successors.all(|inside| inside),
            }
        }
    }

    /// Builds `left \ right` over the arena vertices.
    #[inline]
    fn difference<A, L, R>(arena: &A, left: &L, right: &R) -> DenseDynamicRegion
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Vertices: FiniteVertices<Vertex = usize>,
        L: Region<usize>,
        R: Region<usize>,
    {
        let mut region = DenseDynamicRegion::new(arena.vertex_store().vertex_count());

        for node in arena.vertex_store().vertices() {
            if left.includes(&node) && !right.includes(&node) {
                region.expand(node);
            }
        }

        region
    }

    /// Checks whether `region` contains no arena vertices.
    #[inline]
    fn is_empty<A, R>(arena: &A, region: &R) -> bool
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Vertices: FiniteVertices<Vertex = usize>,
        R: Region<usize>,
    {
        arena
            .vertex_store()
            .vertices()
            .all(|node| !region.includes(&node))
    }

    /// Computes the player's attractor to `target`, restricted to `universe`.
    #[inline]
    fn restricted_attractor<A>(
        arena: &A,
        player: A::Player,
        target: DenseDynamicRegion,
        universe: DenseDynamicRegion,
    ) -> DenseDynamicRegion
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Player: OpposedPlayer,
        A::Vertices: FiniteVertices<Vertex = usize>,
    {
        let analysis = AttractorAnalysis::new(arena, player, target, universe);
        let direction = BackwardDirection::new(arena);
        let worklist = Worklist::new(arena, direction);

        worklist.solve(analysis)
    }
}

impl Default for Buchi {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, A, Accepting> Fixpoint<BuchiAnalysis<'a, A, Accepting>> for Buchi
where
    A: Arena<Position = usize, Vertex = usize>,
    A::Player: OpposedPlayer,
    A::Vertices: FiniteVertices<Vertex = usize>,
    Accepting: Region<usize>,
{
    type Solution = DenseDynamicRegion;

    fn solve(&self, analysis: BuchiAnalysis<'a, A, Accepting>) -> Self::Solution {
        let mut candidate = DenseDynamicRegion::ones(analysis.arena.vertex_store().vertex_count());

        loop {
            let target = Self::accepting_target(&analysis, &candidate);

            let player_attractor = Self::restricted_attractor(
                analysis.arena,
                analysis.player,
                target,
                candidate.clone(),
            );

            let bad = Self::difference(analysis.arena, &candidate, &player_attractor);

            if Self::is_empty(analysis.arena, &bad) {
                return candidate;
            }

            let opponent_attractor = Self::restricted_attractor(
                analysis.arena,
                analysis.player.opponent(),
                bad,
                candidate.clone(),
            );

            candidate = Self::difference(analysis.arena, &candidate, &opponent_attractor);
        }
    }
}
