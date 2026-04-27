use std::fmt::Debug;
use std::{cmp::Ordering, ptr};

use crate::{
    games::{
        arena::{Arena, FiniteArena},
        region::{ArenaRegion, Region, RegionInsertion, RegionRemoval},
    },
    graphs::structure::{FiniteEdges, FiniteVertices, Structure},
    lattices::{
        bit_vector::BitVector,
        lattice::{JoinSemiLattice, MeetSemiLattice},
    },
};

/// Dense bit-vector-backed region over a finite arena with `usize` positions.
#[derive(Clone, Debug)]
pub struct DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    owner: A::Player,
    arena: &'a A,
    mask: BitVector,
}

impl<'a, A> DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    /// Creates an empty dense region for `owner` over `arena`.
    #[must_use]
    #[inline]
    pub fn new(owner: A::Player, arena: &'a A) -> Self {
        Self::empty(owner, arena)
    }

    /// Creates a dense region from an existing mask.
    ///
    /// Panics if the mask length does not match the arena's vertex count.
    #[must_use]
    #[inline]
    pub fn new_with_mask(owner: A::Player, arena: &'a A, mask: BitVector) -> Self {
        assert_eq!(arena.vertex_store().vertex_count(), mask.len());
        Self { owner, arena, mask }
    }

    /// Returns the owner/controller of this region.
    #[must_use]
    #[inline]
    pub fn owner(&self) -> A::Player {
        self.owner
    }

    /// Returns the arena this region belongs to.
    #[must_use]
    #[inline]
    pub fn arena(&self) -> &'a A {
        self.arena
    }

    /// Returns the underlying bit mask.
    #[must_use]
    #[inline]
    pub fn mask(&self) -> &BitVector {
        &self.mask
    }

    /// Returns the underlying bit mask mutably.
    #[must_use]
    #[inline]
    pub fn mask_mut(&mut self) -> &mut BitVector {
        &mut self.mask
    }

    /// Creates an empty region.
    #[must_use]
    #[inline]
    pub fn empty(owner: A::Player, arena: &'a A) -> Self {
        Self::new_with_mask(
            owner,
            arena,
            BitVector::zeros(arena.vertex_store().vertex_count()),
        )
    }

    /// Creates a full region.
    #[must_use]
    #[inline]
    pub fn full(owner: A::Player, arena: &'a A) -> Self {
        Self::new_with_mask(
            owner,
            arena,
            BitVector::ones(arena.vertex_store().vertex_count()),
        )
    }

    /// Returns the size of the arena universe for this dense region.
    #[must_use]
    #[inline]
    pub fn universe_len(&self) -> usize {
        self.mask.len()
    }
}

impl<'a, A> PartialEq for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.arena, other.arena) && self.owner == other.owner && self.mask == other.mask
    }
}

impl<'a, A> Eq for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
}

impl<'a, A> PartialOrd for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !ptr::eq(self.arena, other.arena) || self.owner != other.owner {
            return None;
        }

        self.mask.partial_cmp(&other.mask)
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        ptr::eq(self.arena, other.arena) && self.owner == other.owner && self.mask <= other.mask
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        ptr::eq(self.arena, other.arena) && self.owner == other.owner && self.mask >= other.mask
    }
}

impl<'a, A> JoinSemiLattice for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Arena>::Player: Debug,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    #[inline]
    fn join(&self, other: &Self) -> Self {
        assert!(ptr::eq(self.arena, other.arena));
        assert_eq!(self.owner, other.owner);

        Self::new_with_mask(self.owner, self.arena, self.mask.join(&other.mask))
    }
}

impl<'a, A> MeetSemiLattice for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Arena>::Player: Debug,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        assert!(ptr::eq(self.arena, other.arena));
        assert_eq!(self.owner, other.owner);

        Self::new_with_mask(self.owner, self.arena, self.mask.meet(&other.mask))
    }
}

/// Iterator over the positions contained in a dense region.
#[derive(Clone, Debug)]
pub struct DenseRegionPositions<'a> {
    mask: &'a BitVector,
    index: usize,
}

impl<'a> DenseRegionPositions<'a> {
    #[must_use]
    #[inline]
    pub fn new(mask: &'a BitVector) -> Self {
        Self { mask, index: 0 }
    }
}

impl Iterator for DenseRegionPositions<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.mask.len() {
            let index = self.index;
            self.index += 1;

            if self.mask.get(index).unwrap_or(false) {
                return Some(index);
            }
        }

        None
    }
}

impl<'a, A> Region for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Arena>::Player: Debug,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    type Player = A::Player;
    type Position = usize;

    type Positions<'b>
        = DenseRegionPositions<'b>
    where
        Self: 'b;

    #[inline]
    fn owner(&self) -> Self::Player {
        self.owner
    }

    #[inline]
    fn positions(&self) -> Self::Positions<'_> {
        DenseRegionPositions::new(&self.mask)
    }

    #[inline]
    fn contains(&self, position: &Self::Position) -> bool {
        self.mask.get(*position).unwrap_or(false)
    }

    #[inline]
    fn len(&self) -> usize {
        self.mask.count_ones()
    }
}

impl<'a, A> ArenaRegion<'a> for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize> + 'a,
    <A as Arena>::Player: Debug,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    type Arena = A;

    #[inline]
    fn arena(&self) -> &'a Self::Arena {
        self.arena
    }
}

impl<'a, A> RegionInsertion for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Arena>::Player: Debug,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    #[inline]
    fn insert(&mut self, position: Self::Position) -> bool {
        debug_assert!(position < self.mask.len());

        if self.mask.get(position).unwrap_or(false) {
            false
        } else {
            self.mask.set(position, true);
            true
        }
    }
}

impl<'a, A> RegionRemoval for DenseRegion<'a, A>
where
    A: FiniteArena<Position = usize>,
    <A as Arena>::Player: Debug,
    <A as Structure>::Vertices: FiniteVertices,
    <A as Structure>::Edges: FiniteEdges,
{
    #[inline]
    fn remove(&mut self, position: &Self::Position) -> bool {
        debug_assert!(*position < self.mask.len());

        if self.mask.get(*position).unwrap_or(false) {
            self.mask.set(*position, false);
            true
        } else {
            false
        }
    }

    #[inline]
    fn clear(&mut self) {
        for position in 0..self.mask.len() {
            self.mask.set(position, false);
        }
    }
}
