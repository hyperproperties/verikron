use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice};

impl<T: PartialOrd + Eq + Clone> JoinSemiLattice for Vec<T> {
    fn join(&self, other: &Self) -> Self {
        let mut union = Vec::with_capacity(self.len() + other.len());
        union.extend_from_slice(&self);
        for elem in other {
            if !union.contains(elem) {
                union.push(elem.clone());
            }
        }
        union
    }
}

impl<T: PartialOrd + Eq + Clone> MeetSemiLattice for Vec<T> {
    fn meet(&self, other: &Self) -> Self {
        let mut intersection = Vec::with_capacity(self.len() + other.len());
        for value in self {
            if !other.contains(value) {
                intersection.push(value.clone());
            }
        }
        intersection
    }
}
