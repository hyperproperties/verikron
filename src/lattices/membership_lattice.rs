use crate::graphs::visited::Visited;
use crate::lattices::lattice::JoinSemiLattice;

pub trait MembershipLattice<V>: JoinSemiLattice + Clone + PartialEq {
    fn insert(&mut self, value: V) -> bool;
    fn contains(&self, value: &V) -> bool;
}

impl<V, L> Visited<V> for L
where
    V: Copy,
    L: MembershipLattice<V> + Default,
{
    #[inline]
    fn visit(&mut self, value: V) -> bool {
        self.insert(value)
    }

    #[inline]
    fn is_visited(&self, value: &V) -> bool {
        self.contains(value)
    }
}
