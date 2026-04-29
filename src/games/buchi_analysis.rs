use crate::{
    games::{
        arena::Arena,
        attractor_analysis::AttractorAnalysis,
        region::{DenseRegion, Region},
    },
    graphs::{
        forward::Forward,
        structure::{FiniteVertices, Vertices},
    },
    lattices::{fixpoint::Fixpoint, monotone::BackwardDirection, worklist::Worklist},
};

/// Monotone-style analysis for a Büchi objective.
///
/// The accepting region contains the positions the player wants to visit
/// infinitely often. The winning region is computed by an outer greatest
/// fixed point that repeatedly solves restricted attractor subproblems.
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
    pub fn player(&self) -> &A::Player {
        &self.player
    }

    #[must_use]
    #[inline]
    pub fn accepting(&self) -> &Accepting {
        &self.accepting
    }
}

/// Fixed-point solver for Büchi analyses.
///
/// The solver maintains a candidate winning region. In each iteration it
/// computes the player's attractor, restricted to the current candidate, to
/// accepting positions that can continue inside the candidate.
pub struct Buchi;

impl Buchi {
    /// Builds the region containing every position.
    #[inline]
    fn full_region<A>(arena: &A) -> DenseRegion
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Vertices: FiniteVertices<Vertex = usize>,
    {
        let mut region = DenseRegion::new(arena.vertex_store().vertex_count());

        for node in arena.vertex_store().vertices() {
            region.expand(node);
        }

        region
    }

    /// Copies `source` into a dense region over the arena vertices.
    #[inline]
    fn copy_region<A, R>(arena: &A, source: &R) -> DenseRegion
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Vertices: FiniteVertices<Vertex = usize>,
        R: Region<usize>,
    {
        let mut region = DenseRegion::new(arena.vertex_store().vertex_count());

        for node in arena.vertex_store().vertices() {
            if source.includes(&node) {
                region.expand(node);
            }
        }

        region
    }

    /// Builds the accepting target used by the current Büchi iteration.
    ///
    /// An accepting position is a valid target only if it is still inside the
    /// candidate region and the player can continue the play inside that region.
    #[inline]
    fn accepting_target<A, Accepting>(
        analysis: &BuchiAnalysis<'_, A, Accepting>,
        candidate: &DenseRegion,
    ) -> DenseRegion
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Player: PartialEq,
        A::Vertices: FiniteVertices<Vertex = usize>,
        Accepting: Region<usize>,
    {
        let mut target = DenseRegion::new(analysis.arena.vertex_store().vertex_count());

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
        region: &DenseRegion,
    ) -> bool
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Player: PartialEq,
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

    /// Compares two regions over the arena vertices.
    #[inline]
    fn same_region<A, L, R>(arena: &A, left: &L, right: &R) -> bool
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Vertices: FiniteVertices<Vertex = usize>,
        L: Region<usize>,
        R: Region<usize>,
    {
        arena
            .vertex_store()
            .vertices()
            .all(|node| left.includes(&node) == right.includes(&node))
    }

    /// Computes the player's attractor to `target`, restricted to `universe`.
    #[inline]
    fn restricted_attractor<A>(
        arena: &A,
        player: A::Player,
        target: DenseRegion,
        universe: DenseRegion,
    ) -> DenseRegion
    where
        A: Arena<Position = usize, Vertex = usize>,
        A::Player: PartialEq + Copy,
        A::Vertices: FiniteVertices<Vertex = usize>,
    {
        let analysis = AttractorAnalysis::new(arena, player, target, universe);
        let direction = BackwardDirection::new(arena);
        let worklist = Worklist::new(arena, direction);

        worklist.solve(analysis)
    }
}

impl<'a, A, Accepting> Fixpoint<BuchiAnalysis<'a, A, Accepting>> for Buchi
where
    A: Arena<Position = usize, Vertex = usize>,
    A::Player: PartialEq + Copy,
    A::Vertices: FiniteVertices<Vertex = usize>,
    Accepting: Region<usize>,
{
    type Solution = DenseRegion;

    fn solve(&self, analysis: BuchiAnalysis<'a, A, Accepting>) -> Self::Solution {
        let mut candidate = Self::full_region(analysis.arena);

        loop {
            let target = Self::accepting_target(&analysis, &candidate);
            let universe = Self::copy_region(analysis.arena, &candidate);

            let next =
                Self::restricted_attractor(analysis.arena, analysis.player, target, universe);

            if Self::same_region(analysis.arena, &candidate, &next) {
                return next;
            }

            candidate = next;
        }
    }
}
