use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice};

/// Join-semilattice instance for `Vec<T>`, treating vectors as
/// **order-preserving sets** under element equality.
///
/// Semantics:
///
/// * The “order” this is aiming for is: `self ⊑ other` iff every element of
///   `self` appears in `other` (ignoring multiplicity), i.e. subset-like.
/// * `join` corresponds to a **union** of elements, preserving the order
///   of the first vector and then appending new elements from the second.
///
/// # Behaviour
///
/// * All elements of `self` are copied into the result in their original order.
/// * Each element of `other` is added **only if** it is not already present
///   (according to `Eq`), preserving the order in which it appears in `other`.
///
/// This gives you an order-preserving, set-like union:
impl<T: PartialOrd + Eq + Clone> JoinSemiLattice for Vec<T> {
    fn join(&self, other: &Self) -> Self {
        let mut union = Vec::with_capacity(self.len() + other.len());
        union.extend_from_slice(self);
        for elem in other {
            if !union.contains(elem) {
                union.push(elem.clone());
            }
        }
        union
    }
}

/// Meet-semilattice instance for `Vec<T>`, treating vectors as
/// **order-preserving sets** under element equality.
///
/// Semantics:
///
/// * The intended meet is **intersection**: elements common to both vectors,
///   preserving the order in which they appear in `self`.
///
/// # Behaviour
///
/// * Iterates over `self` and selects elements that are also in `other`.
/// * Each common element is added at most once, in the order seen in `self`.
///
/// Note: this implementation is **O(n²)** in the worst case because it uses
/// linear `contains` checks. For small vectors or lattice prototyping this is
/// often fine; for large data you may want a set-backed lattice instead.
impl<T: PartialOrd + Eq + Clone> MeetSemiLattice for Vec<T> {
    fn meet(&self, other: &Self) -> Self {
        // At most min(len(self), len(other)) common elements.
        let mut intersection = Vec::with_capacity(self.len().min(other.len()));
        for value in self {
            if other.contains(value) && !intersection.contains(value) {
                intersection.push(value.clone());
            }
        }
        intersection
    }
}
#[cfg(test)]
mod tests {
    use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice};

    use std::collections::HashSet;

    use proptest::prelude::*;
    use rand::{Rng, rng};

    fn vec_to_set<T: Eq + std::hash::Hash + Clone>(v: &Vec<T>) -> HashSet<T> {
        v.iter().cloned().collect()
    }

    // Generate a Vec<i32> with *no duplicates* (so semilattice laws make sense
    // under the “vector as set” interpretation).
    prop_compose! {
        fn arb_vec_set()(s in proptest::collection::btree_set(any::<i32>(), 0..20)) -> Vec<i32> {
            // BTreeSet removes duplicates and sorts; order is deterministic but
            // irrelevant for our set-based comparisons.
            s.into_iter().collect()
        }
    }

    #[test]
    fn unit_join_basic() {
        let a = vec![1, 2];
        let b = vec![2, 3];

        let j = a.join(&b);
        assert_eq!(vec_to_set(&j), HashSet::from([1, 2, 3]));
    }

    #[test]
    fn unit_meet_basic() {
        let a = vec![1, 2, 3];
        let b = vec![2, 3, 4];

        let m = a.meet(&b);
        assert_eq!(vec_to_set(&m), HashSet::from([2, 3]));
    }

    #[test]
    fn unit_join_preserves_order_of_first() {
        let a = vec![10, 20, 30];
        let b = vec![30, 40, 50];

        let j = a.join(&b);

        // Order: first all of `a` in order...
        assert_eq!(&j[0..3], &[10, 20, 30]);
        // ...then only new elements from `b` in order
        assert!(j.contains(&40));
        assert!(j.contains(&50));
        assert_eq!(vec_to_set(&j), HashSet::from([10, 20, 30, 40, 50]));
    }

    #[test]
    fn unit_meet_preserves_order_from_self() {
        let a = vec![3, 1, 2];
        let b = vec![2, 3, 4];

        let m = a.meet(&b);
        // Intersection is {3, 2}, in the order they appear in `a`.
        assert_eq!(m, vec![3, 2]);
        assert_eq!(vec_to_set(&m), HashSet::from([2, 3]));
    }

    #[test]
    fn unit_join_idempotent_no_duplicates() {
        let a = vec![1, 2, 3];
        let j = a.join(&a);
        // For duplicate-free inputs, join is idempotent as vectors too.
        assert_eq!(j, a);
    }

    #[test]
    fn unit_meet_idempotent_no_duplicates() {
        let a = vec![4, 5, 6];
        let m = a.meet(&a);
        assert_eq!(m, a);
    }

    #[test]
    fn random_join_is_superset_of_both_as_sets() {
        let mut rng = rng();

        for _ in 0..200 {
            let len_a = rng.random_range(0..10);
            let len_b = rng.random_range(0..10);

            // generate lists with possible duplicates, then dedup into sets
            let mut a_raw = Vec::with_capacity(len_a);
            let mut b_raw = Vec::with_capacity(len_b);

            for _ in 0..len_a {
                a_raw.push(rng.random_range(0..20));
            }
            for _ in 0..len_b {
                b_raw.push(rng.random_range(0..20));
            }

            let a_set: HashSet<_> = a_raw.iter().cloned().collect();
            let b_set: HashSet<_> = b_raw.iter().cloned().collect();

            let a: Vec<_> = a_set.iter().cloned().collect();
            let b: Vec<_> = b_set.iter().cloned().collect();

            let j = a.join(&b);
            let j_set = vec_to_set(&j);

            assert!(a_set.is_subset(&j_set));
            assert!(b_set.is_subset(&j_set));
        }
    }

    #[test]
    fn random_meet_is_subset_of_both_as_sets() {
        let mut rng = rng();

        for _ in 0..200 {
            let len_a = rng.random_range(0..10);
            let len_b = rng.random_range(0..10);

            let mut a_raw = Vec::with_capacity(len_a);
            let mut b_raw = Vec::with_capacity(len_b);

            for _ in 0..len_a {
                a_raw.push(rng.random_range(0..20));
            }
            for _ in 0..len_b {
                b_raw.push(rng.random_range(0..20));
            }

            let a_set: HashSet<_> = a_raw.iter().cloned().collect();
            let b_set: HashSet<_> = b_raw.iter().cloned().collect();

            let a: Vec<_> = a_set.iter().cloned().collect();
            let b: Vec<_> = b_set.iter().cloned().collect();

            let m = a.meet(&b);
            let m_set = vec_to_set(&m);

            assert!(m_set.is_subset(&a_set));
            assert!(m_set.is_subset(&b_set));
        }
    }

    proptest! {
        // Idempotence (set semantics): a ⊔ a = a, a ⊓ a = a
        #[test]
        fn prop_idempotence_as_sets(a in arb_vec_set()) {
            let j = a.join(&a);
            let m = a.meet(&a);

            prop_assert_eq!(vec_to_set(&j), vec_to_set(&a));
            prop_assert_eq!(vec_to_set(&m), vec_to_set(&a));
        }

        // Commutativity (set semantics): a ⊔ b = b ⊔ a, a ⊓ b = b ⊓ a
        #[test]
        fn prop_commutativity_as_sets(a in arb_vec_set(), b in arb_vec_set()) {
            let j1 = a.join(&b);
            let j2 = b.join(&a);
            let m1 = a.meet(&b);
            let m2 = b.meet(&a);

            prop_assert_eq!(vec_to_set(&j1), vec_to_set(&j2));
            prop_assert_eq!(vec_to_set(&m1), vec_to_set(&m2));
        }

        // Associativity (set semantics): (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c), similarly for ⊓
        #[test]
        fn prop_associativity_as_sets(a in arb_vec_set(), b in arb_vec_set(), c in arb_vec_set()) {
            let j_left  = a.join(&b).join(&c);
            let j_right = a.join(&b.join(&c));

            let m_left  = a.meet(&b).meet(&c);
            let m_right = a.meet(&b.meet(&c));

            prop_assert_eq!(vec_to_set(&j_left), vec_to_set(&j_right));
            prop_assert_eq!(vec_to_set(&m_left), vec_to_set(&m_right));
        }

        // Absorption (set semantics): a ⊔ (a ⊓ b) = a and a ⊓ (a ⊔ b) = a
        #[test]
        fn prop_absorption_as_sets(a in arb_vec_set(), b in arb_vec_set()) {
            let lhs1 = a.join(&a.meet(&b));
            let lhs2 = a.meet(&a.join(&b));

            prop_assert_eq!(vec_to_set(&lhs1), vec_to_set(&a));
            prop_assert_eq!(vec_to_set(&lhs2), vec_to_set(&a));
        }

        // Join is least upper bound (set semantics).
        #[test]
        fn prop_join_is_lub_as_sets(a in arb_vec_set(), b in arb_vec_set(), c in arb_vec_set()) {
            let j = a.join(&b);

            let a_set = vec_to_set(&a);
            let b_set = vec_to_set(&b);
            let j_set = vec_to_set(&j);
            let c_set = vec_to_set(&c);

            // a ⊆ j and b ⊆ j
            prop_assert!(a_set.is_subset(&j_set));
            prop_assert!(b_set.is_subset(&j_set));

            // If a ⊆ c and b ⊆ c then j ⊆ c
            if a_set.is_subset(&c_set) && b_set.is_subset(&c_set) {
                prop_assert!(j_set.is_subset(&c_set));
            }
        }

        // Meet is greatest lower bound (set semantics).
        #[test]
        fn prop_meet_is_glb_as_sets(a in arb_vec_set(), b in arb_vec_set(), c in arb_vec_set()) {
            let m = a.meet(&b);

            let a_set = vec_to_set(&a);
            let b_set = vec_to_set(&b);
            let m_set = vec_to_set(&m);
            let c_set = vec_to_set(&c);

            // m ⊆ a and m ⊆ b
            prop_assert!(m_set.is_subset(&a_set));
            prop_assert!(m_set.is_subset(&b_set));

            // If c ⊆ a and c ⊆ b then c ⊆ m
            if c_set.is_subset(&a_set) && c_set.is_subset(&b_set) {
                prop_assert!(c_set.is_subset(&m_set));
            }
        }
    }
}
