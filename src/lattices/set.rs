use std::{
    cmp::Ordering,
    collections::hash_set::{Drain, IntoIter, Iter},
    hash::Hash,
    iter::FromIterator,
};

use rustc_hash::{FxBuildHasher, FxHashSet};

use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice};

/// A hash-based finite set of values, backed by [`FxHashSet`].
///
/// This type is a thin, lattice-friendly wrapper around `FxHashSet<T>`
/// that also implements the standard set operations and the lattice traits
/// [`JoinSemiLattice`] and [`MeetSemiLattice`] using the powerset order.
///
/// The partial order on `Set<T>` is **subset** (`⊆`):
///
/// * `a <= b` iff `a` is a subset of `b` (`a.is_subset(&b)`).
///
/// Under this order:
///
/// * `join` corresponds to **union** of sets.
/// * `meet` corresponds to **intersection** of sets.
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct Set<T: Eq + Hash>(FxHashSet<T>);

impl<T: Eq + Hash> Set<T> {
    /// Creates an empty [`Set`] with the default [`FxBuildHasher`].
    ///
    /// This is equivalent to `Set::default()` but is more explicit.
    pub fn new(hash_set: FxHashSet<T>) -> Self {
        Self(hash_set)
    }

    /// Creates an empty [`Set`] with the specified capacity.
    ///
    /// The set will be able to hold at least `capacity` elements
    /// without reallocating.
    pub fn with_capacity(capacity: usize) -> Self {
        Self(FxHashSet::with_capacity_and_hasher(
            capacity,
            FxBuildHasher::default(),
        ))
    }

    /// Returns the number of elements the set can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns an iterator over the elements of the set (in arbitrary order).
    pub fn iter(&self) -> Iter<'_, T> {
        self.0.iter()
    }

    /// Clears the set, returning all elements as a draining iterator.
    ///
    /// After calling this, the set is empty.
    pub fn drain(&mut self) -> Drain<'_, T> {
        self.0.drain()
    }

    /// Clears all elements from the set.
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `x` such that `f(&x)` returns `false`.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.0.retain(f)
    }

    /// Returns a shared reference to the underlying [`FxHashSet`].
    ///
    /// This can be used when you need access to APIs that are not
    /// re-exposed by [`Set`].
    pub fn inner(&self) -> &FxHashSet<T> {
        &self.0
    }

    /// Inserts a value into the set.
    ///
    /// Returns `true` if the value was not already present in the set.
    pub fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }

    /// Removes a value from the set.
    ///
    /// Returns `true` if the value was present in the set.
    pub fn remove(&mut self, value: &T) -> bool {
        self.0.remove(value)
    }

    /// Returns `true` if the set contains the specified value.
    pub fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }

    /// Returns `true` if `self` is a subset of `other`.
    ///
    /// That is, returns `true` if every element of `self` is also an element of `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Returns `true` if `self` is a superset of `other`.
    ///
    /// That is, returns `true` if every element of `other` is also an element of `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }

    /// Returns `true` if `self` and `other` have no elements in common.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.0.is_disjoint(&other.0)
    }

    /// Returns an iterator over the union of `self` and `other` (borrowed view).
    ///
    /// The iterator yields each element that is in `self` or `other` (or both),
    /// without duplicates.
    pub fn union<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.union(&other.0)
    }

    /// Returns an iterator over the intersection of `self` and `other` (borrowed view).
    ///
    /// The iterator yields each element that is in both `self` and `other`.
    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.intersection(&other.0)
    }

    /// Returns an iterator over the difference of `self` and `other` (borrowed view).
    ///
    /// The iterator yields each element that is in `self` but **not** in `other`.
    pub fn difference<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.difference(&other.0)
    }

    /// Returns an iterator over the symmetric difference of `self` and `other` (borrowed view).
    ///
    /// The iterator yields each element that is in exactly one of `self` or `other`,
    /// but not in both.
    pub fn symmetric_difference<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.symmetric_difference(&other.0)
    }

    /// Returns a new [`Set`] containing the union of `self` and `other`.
    ///
    /// This is an owning version of [`Set::union`], and is equivalent to
    /// the lattice `join` operation on the powerset lattice.
    pub fn union_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        // Quick trivial cases
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        // Cheap superset checks that may avoid allocation if sets are nested
        if self.len() >= other.len() && self.is_superset(other) {
            return self.clone();
        }
        if other.len() >= self.len() && other.is_superset(self) {
            return other.clone();
        }

        // General case
        let mut set = Set::with_capacity(self.len() + other.len());
        set.extend(self.iter().cloned());
        set.extend(other.iter().cloned());
        set
    }

    /// Returns a new [`Set`] containing the intersection of `self` and `other`.
    ///
    /// This is an owning version of [`Set::intersection`], and is equivalent to
    /// the lattice `meet` operation on the powerset lattice.
    pub fn intersection_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        if self.is_empty() || other.is_empty() {
            return Set::with_capacity(0); // or Set::new()
        }

        // Iterate the smaller set, check membership in the larger one.
        let (small, big) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };

        let mut out = Set::with_capacity(small.len());
        for v in small.iter() {
            if big.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }

    /// Returns a new [`Set`] containing the difference of `self` and `other`.
    ///
    /// The result contains all elements that are in `self` but **not** in `other`.
    pub fn difference_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        let mut out = Set::with_capacity(self.len());
        for v in self.iter() {
            if !other.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }

    /// Returns a new [`Set`] containing the symmetric difference of `self` and `other`.
    ///
    /// The result contains all elements that are in exactly one of the sets,
    /// but not in both.
    pub fn symmetric_difference_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        let mut out = Set::with_capacity(self.len() + other.len());
        for v in self.iter() {
            if !other.contains(v) {
                out.insert(v.clone());
            }
        }
        for v in other.iter() {
            if !self.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }
}

/// Consumes the [`Set`] and returns an owning iterator over its elements.
///
/// This forwards to the underlying `FxHashSet`'s `IntoIterator` implementation.
impl<T: Eq + Hash> IntoIterator for Set<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Extends the set with the contents of an iterator.
///
/// This is equivalent to calling [`insert`](Set::insert) on each element
/// of the iterator.
impl<T: Eq + Hash> Extend<T> for Set<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

/// Builds a [`Set`] from an iterator.
///
/// This collects all items from the iterator into a new set, discarding
/// any duplicates according to `Eq`/`Hash`.
impl<T: Eq + Hash> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(FxHashSet::from_iter(iter))
    }
}

/// Partial order on sets given by subset inclusion.
///
/// * `self <= other` iff `self` is a subset of `other`.
/// * Incomparable sets (neither subset nor superset) result in `None` from
///   [`partial_cmp`](PartialOrd::partial_cmp).
impl<T: Eq + Hash> PartialOrd for Set<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let a = &self.0;
        let b = &other.0;

        let a_sub_b = a.is_subset(b);
        let b_sub_a = b.is_subset(a);

        match (a_sub_b, b_sub_a) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }

    /// Returns `true` if `self` is a subset of `other`.
    fn le(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Returns `true` if `self` is a superset of `other`.
    fn ge(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }
}

/// Join-semilattice instance for [`Set`], with join = set union.
///
/// The join of two sets is their union, which is the least upper bound
/// under the subset order.
impl<T: Eq + Hash + Clone> JoinSemiLattice for Set<T> {
    fn join(&self, other: &Self) -> Self {
        self.union_owned(other)
    }
}

/// Meet-semilattice instance for [`Set`], with meet = set intersection.
///
/// The meet of two sets is their intersection, which is the greatest
/// lower bound under the subset order.
impl<T: Eq + Hash + Clone> MeetSemiLattice for Set<T> {
    fn meet(&self, other: &Self) -> Self {
        self.intersection_owned(other)
    }
}

#[cfg(test)]
mod tests {
    use super::Set;
    use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice};

    use std::cmp::Ordering;
    use std::collections::HashSet;
    use std::hash::Hash;

    use proptest::prelude::*;
    use rand::{Rng, rng};

    fn to_std_set<T: Eq + Hash + Clone>(s: &Set<T>) -> HashSet<T> {
        s.iter().cloned().collect()
    }

    #[test]
    fn unit_empty_set_basic_ops() {
        let s: Set<i32> = Set::default();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(!s.contains(&1));
    }

    #[test]
    fn unit_insert_remove_contains() {
        let mut s = Set::default();
        assert!(s.insert(1));
        assert!(s.contains(&1));
        assert!(!s.insert(1)); // already present
        assert!(s.remove(&1));
        assert!(!s.contains(&1));
        assert!(!s.remove(&1)); // already removed
    }

    #[test]
    fn unit_subset_superset_disjoint() {
        let a: Set<_> = [1, 2].into_iter().collect();
        let b: Set<_> = [1, 2, 3].into_iter().collect();
        let c: Set<_> = [4].into_iter().collect();

        assert!(a.is_subset(&b));
        assert!(b.is_superset(&a));
        assert!(!b.is_subset(&a));
        assert!(a.is_disjoint(&c));
        assert!(b.is_disjoint(&c));
    }

    #[test]
    fn unit_union_intersection_difference_views() {
        let a: Set<_> = [1, 2].into_iter().collect();
        let b: Set<_> = [2, 3].into_iter().collect();

        let u: HashSet<_> = a.union(&b).cloned().collect();
        let i: HashSet<_> = a.intersection(&b).cloned().collect();
        let d: HashSet<_> = a.difference(&b).cloned().collect();
        let sd: HashSet<_> = a.symmetric_difference(&b).cloned().collect();

        assert_eq!(u, HashSet::from([1, 2, 3]));
        assert_eq!(i, HashSet::from([2]));
        assert_eq!(d, HashSet::from([1]));
        assert_eq!(sd, HashSet::from([1, 3]));
    }

    #[test]
    fn unit_owned_set_algebra_matches_views() {
        let a: Set<_> = [1, 2].into_iter().collect();
        let b: Set<_> = [2, 3].into_iter().collect();

        assert_eq!(
            to_std_set(&a.union_owned(&b)),
            a.union(&b).cloned().collect()
        );
        assert_eq!(
            to_std_set(&a.intersection_owned(&b)),
            a.intersection(&b).cloned().collect()
        );
        assert_eq!(
            to_std_set(&a.difference_owned(&b)),
            a.difference(&b).cloned().collect()
        );
        assert_eq!(
            to_std_set(&a.symmetric_difference_owned(&b)),
            a.symmetric_difference(&b).cloned().collect()
        );
    }

    #[test]
    fn unit_partial_ord_subset_semantics() {
        let a: Set<_> = [1, 2].into_iter().collect();
        let b: Set<_> = [1, 2, 3].into_iter().collect();
        let c: Set<_> = [3, 4].into_iter().collect();

        // a ⊂ b
        assert!(a < b);
        assert!(a <= b);
        assert!(b > a);
        assert!(b >= a);

        // a and c are incomparable
        assert_eq!(a.partial_cmp(&c), None);
        assert_eq!(c.partial_cmp(&a), None);
    }

    #[test]
    fn unit_join_is_union() {
        let a: Set<_> = [1, 2].into_iter().collect();
        let b: Set<_> = [2, 3].into_iter().collect();

        let j = a.join(&b);
        assert_eq!(to_std_set(&j), HashSet::from([1, 2, 3]));
    }

    #[test]
    fn unit_meet_is_intersection() {
        let a: Set<_> = [1, 2, 3].into_iter().collect();
        let b: Set<_> = [2, 3, 4].into_iter().collect();

        let m = a.meet(&b);
        assert_eq!(to_std_set(&m), HashSet::from([2, 3]));
    }

    #[test]
    fn random_join_is_superset_of_both() {
        let mut rng = rng();

        for _ in 0..200 {
            let len_a = rng.random_range(0..20);
            let len_b = rng.random_range(0..20);

            let a_vec: Vec<_> = (0..len_a).map(|_| rng.random_range(0..50)).collect();
            let b_vec: Vec<_> = (0..len_b).map(|_| rng.random_range(0..50)).collect();

            let a: Set<_> = a_vec.into_iter().collect();
            let b: Set<_> = b_vec.into_iter().collect();

            let j = a.join(&b);

            // a ⊑ j and b ⊑ j
            assert!(a.is_subset(&j));
            assert!(b.is_subset(&j));
        }
    }

    #[test]
    fn random_meet_is_subset_of_both() {
        let mut rng = rng();

        for _ in 0..200 {
            let len_a = rng.random_range(0..20);
            let len_b = rng.random_range(0..20);

            let a_vec: Vec<_> = (0..len_a).map(|_| rng.random_range(0..50)).collect();
            let b_vec: Vec<_> = (0..len_b).map(|_| rng.random_range(0..50)).collect();

            let a: Set<_> = a_vec.into_iter().collect();
            let b: Set<_> = b_vec.into_iter().collect();

            let m = a.meet(&b);

            // m ⊑ a and m ⊑ b
            assert!(m.is_subset(&a));
            assert!(m.is_subset(&b));
        }
    }

    // Strategy: generate small vectors of i32, then collect into sets.
    prop_compose! {
        fn arb_set()(v in proptest::collection::vec(any::<i32>(), 0..20)) -> Set<i32> {
            v.into_iter().collect()
        }
    }

    proptest! {
        // Idempotence: a ⊔ a = a, a ⊓ a = a
        #[test]
        fn prop_idempotence(a in arb_set()) {
            let j = a.join(&a);
            let m = a.meet(&a);
            prop_assert_eq!(to_std_set(&j), to_std_set(&a));
            prop_assert_eq!(to_std_set(&m), to_std_set(&a));
        }

        // Commutativity: a ⊔ b = b ⊔ a, a ⊓ b = b ⊓ a
        #[test]
        fn prop_commutativity(a in arb_set(), b in arb_set()) {
            let j1 = a.join(&b);
            let j2 = b.join(&a);
            let m1 = a.meet(&b);
            let m2 = b.meet(&a);

            prop_assert_eq!(to_std_set(&j1), to_std_set(&j2));
            prop_assert_eq!(to_std_set(&m1), to_std_set(&m2));
        }

        // Associativity: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c), and similarly for meet.
        #[test]
        fn prop_associativity(a in arb_set(), b in arb_set(), c in arb_set()) {
            let j_left  = a.join(&b).join(&c);
            let j_right = a.join(&b.join(&c));
            prop_assert_eq!(to_std_set(&j_left), to_std_set(&j_right));

            let m_left  = a.meet(&b).meet(&c);
            let m_right = a.meet(&b.meet(&c));
            prop_assert_eq!(to_std_set(&m_left), to_std_set(&m_right));
        }

        // Absorption: a ⊔ (a ⊓ b) = a and a ⊓ (a ⊔ b) = a
        #[test]
        fn prop_absorption(a in arb_set(), b in arb_set()) {
            let lhs1 = a.join(&a.meet(&b));
            let lhs2 = a.meet(&a.join(&b));

            prop_assert_eq!(to_std_set(&lhs1), to_std_set(&a));
            prop_assert_eq!(to_std_set(&lhs2), to_std_set(&a));
        }

        // Order vs join: a ⊑ a ⊔ b and b ⊑ a ⊔ b;
        // and if a ⊑ c and b ⊑ c then a ⊔ b ⊑ c.
        #[test]
        fn prop_join_is_least_upper_bound(a in arb_set(), b in arb_set(), c in arb_set()) {
            let j = a.join(&b);

            // a ⊑ j and b ⊑ j
            prop_assert!(a.is_subset(&j));
            prop_assert!(b.is_subset(&j));

            // If a ⊑ c and b ⊑ c then j ⊑ c
            if a.is_subset(&c) && b.is_subset(&c) {
                prop_assert!(j.is_subset(&c));
            }
        }

        // Order vs meet: a ⊓ b ⊑ a and a ⊓ b ⊑ b;
        // and if c ⊑ a and c ⊑ b then c ⊑ a ⊓ b.
        #[test]
        fn prop_meet_is_greatest_lower_bound(a in arb_set(), b in arb_set(), c in arb_set()) {
            let m = a.meet(&b);

            // m ⊑ a and m ⊑ b
            prop_assert!(m.is_subset(&a));
            prop_assert!(m.is_subset(&b));

            // If c ⊑ a and c ⊑ b then c ⊑ m
            if c.is_subset(&a) && c.is_subset(&b) {
                prop_assert!(c.is_subset(&m));
            }
        }

        // PartialOrd consistency: when partial_cmp returns Some,
        // it should match subset / superset semantics.
        #[test]
        fn prop_partial_ord_matches_subset(a in arb_set(), b in arb_set()) {
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
                    // incomparable: not subset of each other
                    prop_assert!(!a.is_subset(&b));
                    prop_assert!(!b.is_subset(&a));
                }
            }
        }

        // Join and meet agree with view-based union/intersection.
        #[test]
        fn prop_join_meet_match_std_union_intersection(a in arb_set(), b in arb_set()) {
            let j = a.join(&b);
            let m = a.meet(&b);

            let u_std: HashSet<_> = a.union(&b).cloned().collect();
            let i_std: HashSet<_> = a.intersection(&b).cloned().collect();

            prop_assert_eq!(to_std_set(&j), u_std);
            prop_assert_eq!(to_std_set(&m), i_std);
        }
    }
}
