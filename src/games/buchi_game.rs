use crate::{
    games::{
        arena::{Arena, FiniteArena},
        buchi_analysis::{Buchi, BuchiAnalysis},
        game::{Game, RegionSolvableGame},
        lasso_play::LassoPlay,
        players::OpposedPlayer,
        region::DenseStaticRegion,
    },
    graphs::{
        graph::FiniteDirected,
        structure::{FiniteEdges, FiniteVertices, Structure},
    },
    lattices::{fixpoint::Fixpoint, lattice::MembershipLattice},
};

/// Arena requirements needed by dense Büchi-game solvers.
///
/// Büchi solvers in this module use dense `usize` positions and vertices, so
/// regions can be represented efficiently by dense bit-vector-like structures.
pub trait BuchiArena: FiniteArena<Position = usize> + FiniteDirected
where
    Self: Structure<
            Vertices: FiniteVertices<Vertex = usize>,
            Edges: FiniteEdges<Vertex = usize, Edge = Self::Edge>,
        >,
{
}

impl<A> BuchiArena for A where
    A: FiniteArena<Position = usize>
        + FiniteDirected
        + Structure<
            Vertices: FiniteVertices<Vertex = usize>,
            Edges: FiniteEdges<Vertex = usize, Edge = A::Edge>,
        >
{
}

/// A Büchi game with one or more accepting positions.
///
/// A play is winning for a protagonist iff it visits the accepting region infinitely
/// often. Since Büchi is an infinite-play objective, plays are represented as
/// lassos: a finite stem followed by a non-empty cycle repeated forever.
///
/// For a lasso, the positions visited infinitely often are exactly the distinct
/// positions in the cycle. Therefore, a lasso is winning iff its cycle contains
/// at least one accepting position.
#[derive(Clone, Debug)]
pub struct BuchiGame<'a, A>
where
    A: Arena,
{
    arena: &'a A,
    accepting: Vec<A::Position>,
}

impl<'a, A> BuchiGame<'a, A>
where
    A: Arena,
{
    /// Creates a Büchi game with the given accepting positions.
    #[must_use]
    #[inline]
    pub fn new<I>(arena: &'a A, accepting: I) -> Self
    where
        I: IntoIterator<Item = A::Position>,
    {
        Self {
            arena,
            accepting: accepting.into_iter().collect(),
        }
    }

    /// Creates a Büchi game with a single accepting position.
    #[must_use]
    #[inline]
    pub fn singleton(arena: &'a A, accepting: A::Position) -> Self {
        Self::new(arena, [accepting])
    }

    /// Returns the accepting positions.
    #[must_use]
    #[inline]
    pub fn accepting_positions(&self) -> &[A::Position] {
        &self.accepting
    }

    /// Checks whether `position` is accepting.
    #[must_use]
    #[inline]
    pub fn is_accepting(&self, position: A::Position) -> bool {
        self.accepting.contains(&position)
    }
}

impl<'a, A> BuchiGame<'a, A>
where
    A: BuchiArena,
    A::Player: OpposedPlayer,
{
    /// Builds the dense accepting region used by the Büchi analysis.
    #[inline]
    fn accepting_region(&self) -> DenseStaticRegion {
        let mut region = DenseStaticRegion::new(self.arena.vertex_store().vertex_count());

        for &position in &self.accepting {
            region.insert(position);
        }

        region
    }

    /// Creates the Büchi analysis from the accepting region.
    #[inline]
    fn buchi_analysis(&self, protagonist: A::Player) -> BuchiAnalysis<'_, A, DenseStaticRegion> {
        BuchiAnalysis::new(self.arena, protagonist, self.accepting_region())
    }

    /// Computes the protagonist's Büchi-winning region.
    #[inline]
    fn buchi_region(&self, protagonist: A::Player) -> DenseStaticRegion {
        let buchi = Buchi::new();
        buchi.solve(self.buchi_analysis(protagonist))
    }
}

impl<'a, A> Game for BuchiGame<'a, A>
where
    A: Arena,
{
    type Arena = A;
    type Play = LassoPlay<A::Position>;

    #[inline]
    fn arena(&self) -> &Self::Arena {
        self.arena
    }

    /// Checks whether the lasso visits an accepting position infinitely often.
    #[inline]
    fn is_winning(&self, _protagonist: A::Player, play: &Self::Play) -> bool {
        play.infinitely_often()
            .any(|position| self.is_accepting(position))
    }
}

impl<'a, A> RegionSolvableGame for BuchiGame<'a, A>
where
    A: BuchiArena,
    A::Player: OpposedPlayer,
{
    type Region = DenseStaticRegion;

    /// Computes the Büchi-winning region for `protagonist`.
    ///
    /// The region is the greatest fixed point of positions from which `protagonist`
    /// can force infinitely many visits to the accepting region.
    #[inline]
    fn winning_region(&self, protagonist: A::Player) -> Self::Region {
        self.buchi_region(protagonist)
    }
}
