use crate::lattices::partial_order::PartialOrder;

pub trait JoinSemiLattice: PartialOrder + Sized {
    fn join(&self, other: &Self) -> Self;
}

pub trait MeetSemiLattice: PartialOrder + Sized {
    fn meet(&self, other: &Self) -> Self;
}

pub trait Lattice: JoinSemiLattice + MeetSemiLattice {}

impl<T: JoinSemiLattice + MeetSemiLattice> Lattice for T {}

pub trait BoundedLattice: Lattice {
    fn bottom() -> Self;
    fn top() -> Self;
}

pub trait CompleteLattice: BoundedLattice {
    fn join_all<I: IntoIterator<Item = Self>>(iteratot: I) -> Self;
    fn meet_all<I: IntoIterator<Item = Self>>(iteratot: I) -> Self;
}
