use std::{cmp::Ordering, hash::Hash, iter::FromIterator};

use rustc_hash::FxHashSet;

use crate::lattices::lattice::{
    Bottom, JoinSemiLattice, MeetSemiLattice, MembershipLattice, Semilattice,
};

/// Finite hash set ordered by subset inclusion.
///
/// Join is union. Meet is intersection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Set<T: Eq + Hash>(FxHashSet<T>);

impl<T: Eq + Hash> Set<T> {
    /// Creates an empty set.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self(FxHashSet::default())
    }

    /// Creates an empty set with capacity.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(FxHashSet::with_capacity_and_hasher(
            capacity,
            Default::default(),
        ))
    }

    /// Returns the number of elements.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true iff the set is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns true iff the set contains `value`.
    #[must_use]
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }

    /// Inserts `value`; returns true iff it was new.
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }

    /// Removes `value`; returns true iff it was present.
    #[inline]
    pub fn remove(&mut self, value: &T) -> bool {
        self.0.remove(value)
    }

    /// Returns true iff `self ⊆ other`.
    #[must_use]
    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Returns true iff `self ⊇ other`.
    #[must_use]
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }

    /// Returns true iff the sets have no elements in common.
    #[must_use]
    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.0.is_disjoint(&other.0)
    }

    /// Iterates over elements in arbitrary order.
    #[must_use]
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Returns the union of two sets.
    #[must_use]
    #[inline]
    pub fn union(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        self.0.union(&other.0).cloned().collect()
    }

    /// Returns the intersection of two sets.
    #[must_use]
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        self.0.intersection(&other.0).cloned().collect()
    }

    /// Consumes the wrapper.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> FxHashSet<T> {
        self.0
    }
}

impl<T: Eq + Hash> Default for Set<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash> Bottom for Set<T> {
    type Context = ();

    fn bottom_with(_: &Self::Context) -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash + Clone> Semilattice for Set<T> {}

impl<T: Eq + Hash + Clone> JoinSemiLattice for Set<T> {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        self.union(other)
    }
}

impl<T: Eq + Hash + Clone> MeetSemiLattice for Set<T> {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        self.intersection(other)
    }
}

impl<T: Eq + Hash + Clone> MembershipLattice<T> for Set<T> {
    #[inline]
    fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }

    #[inline]
    fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }
}

impl<T: Eq + Hash> PartialOrd for Set<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.is_subset(other), self.is_superset(other)) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.is_subset(other)
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.is_superset(other)
    }
}

impl<T: Eq + Hash> IntoIterator for Set<T> {
    type Item = T;
    type IntoIter = <FxHashSet<T> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Eq + Hash> FromIterator<T> for Set<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(FxHashSet::from_iter(iter))
    }
}

impl<T: Eq + Hash> Extend<T> for Set<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

impl<T: Eq + Hash> From<FxHashSet<T>> for Set<T> {
    #[inline]
    fn from(value: FxHashSet<T>) -> Self {
        Self(value)
    }
}

impl<T: Eq + Hash> From<Set<T>> for FxHashSet<T> {
    #[inline]
    fn from(value: Set<T>) -> Self {
        value.0
    }
}

impl<T: Eq + Hash> From<Vec<T>> for Set<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        value.into_iter().collect()
    }
}

impl<T: Eq + Hash, const N: usize> From<[T; N]> for Set<T> {
    #[inline]
    fn from(value: [T; N]) -> Self {
        value.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Set;

    use std::{cmp::Ordering, collections::HashSet, hash::Hash};

    use proptest::prelude::*;

    use crate::lattices::lattice::{Bottom, JoinSemiLattice, MeetSemiLattice};

    fn std_set<T: Eq + Hash + Clone>(set: &Set<T>) -> HashSet<T> {
        set.iter().cloned().collect()
    }

    prop_compose! {
        fn arb_set()(values in proptest::collection::vec(any::<i32>(), 0..20)) -> Set<i32> {
            values.into_iter().collect()
        }
    }

    #[test]
    fn empty_set_is_bottom() {
        let set = Set::<i32>::bottom();

        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn insert_remove_contains() {
        let mut set = Set::new();

        assert!(set.insert(1));
        assert!(set.contains(&1));
        assert!(!set.insert(1));

        assert!(set.remove(&1));
        assert!(!set.contains(&1));
        assert!(!set.remove(&1));
    }

    #[test]
    fn subset_order() {
        let a: Set<_> = [1, 2].into();
        let b: Set<_> = [1, 2, 3].into();
        let c: Set<_> = [2, 4].into();

        assert!(a < b);
        assert!(a <= b);
        assert!(b > a);
        assert!(b >= a);

        assert_eq!(a.partial_cmp(&c), None);
    }

    #[test]
    fn join_is_union() {
        let a: Set<_> = [1, 2].into();
        let b: Set<_> = [2, 3].into();

        assert_eq!(std_set(&a.join(&b)), HashSet::from([1, 2, 3]));
    }

    #[test]
    fn meet_is_intersection() {
        let a: Set<_> = [1, 2, 3].into();
        let b: Set<_> = [2, 3, 4].into();

        assert_eq!(std_set(&a.meet(&b)), HashSet::from([2, 3]));
    }

    proptest! {
        #[test]
        fn join_laws(a in arb_set(), b in arb_set(), c in arb_set()) {
            prop_assert_eq!(std_set(&a.join(&a)), std_set(&a));
            prop_assert_eq!(std_set(&a.join(&b)), std_set(&b.join(&a)));
            prop_assert_eq!(
                std_set(&a.join(&b).join(&c)),
                std_set(&a.join(&b.join(&c))),
            );
        }

        #[test]
        fn meet_laws(a in arb_set(), b in arb_set(), c in arb_set()) {
            prop_assert_eq!(std_set(&a.meet(&a)), std_set(&a));
            prop_assert_eq!(std_set(&a.meet(&b)), std_set(&b.meet(&a)));
            prop_assert_eq!(
                std_set(&a.meet(&b).meet(&c)),
                std_set(&a.meet(&b.meet(&c))),
            );
        }

        #[test]
        fn absorption(a in arb_set(), b in arb_set()) {
            prop_assert_eq!(std_set(&a.join(&a.meet(&b))), std_set(&a));
            prop_assert_eq!(std_set(&a.meet(&a.join(&b))), std_set(&a));
        }

        #[test]
        fn join_is_least_upper_bound(a in arb_set(), b in arb_set(), c in arb_set()) {
            let join = a.join(&b);

            prop_assert!(a <= join);
            prop_assert!(b <= join);

            if a <= c && b <= c {
                prop_assert!(join <= c);
            }
        }

        #[test]
        fn meet_is_greatest_lower_bound(a in arb_set(), b in arb_set(), c in arb_set()) {
            let meet = a.meet(&b);

            prop_assert!(meet <= a);
            prop_assert!(meet <= b);

            if c <= a && c <= b {
                prop_assert!(c <= meet);
            }
        }

        #[test]
        fn partial_order_matches_subset(a in arb_set(), b in arb_set()) {
            match a.partial_cmp(&b) {
                Some(Ordering::Less) => {
                    prop_assert!(a.is_subset(&b));
                    prop_assert!(!b.is_subset(&a));
                }
                Some(Ordering::Greater) => {
                    prop_assert!(b.is_subset(&a));
                    prop_assert!(!a.is_subset(&b));
                }
                Some(Ordering::Equal) => {
                    prop_assert!(a.is_subset(&b));
                    prop_assert!(b.is_subset(&a));
                }
                None => {
                    prop_assert!(!a.is_subset(&b));
                    prop_assert!(!b.is_subset(&a));
                }
            }
        }
    }
}
