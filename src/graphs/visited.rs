use bit_vec::BitVec;
use std::hash::Hash;

use crate::lattices::set::Set;

// TODO: Create a BitVec Lattice type. Maybe even a generic lattice driven visited set.
// TODO: Atomic boolean Lattice for threadsafe visited.

pub trait Visited<V>: Default {
    fn visit(&mut self, value: V) -> bool;

    fn is_visited(&self, value: &V) -> bool;
}

impl<V> Visited<V> for Set<V>
where
    V: Eq + Hash + Copy,
{
    #[inline]
    fn visit(&mut self, value: V) -> bool {
        self.insert(value)
    }

    #[inline]
    fn is_visited(&self, value: &V) -> bool {
        self.contains(&value)
    }
}

impl Visited<usize> for BitVec {
    #[inline]
    fn visit(&mut self, value: usize) -> bool {
        let len = self.len();
        if value >= len {
            let grow_by = value + 1 - len;
            self.grow(grow_by, false);
        }

        if !self[value] {
            self.set(value, true);
            true
        } else {
            false
        }
    }

    #[inline]
    fn is_visited(&self, value: &usize) -> bool {
        match self.get(*value) {
            Some(value) => value,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn set_visited_default_is_empty() {
        let visited = Set::<usize>::default();
        assert!(visited.is_empty());
        assert!(!visited.is_visited(&0));
        assert!(!visited.is_visited(&42));
    }

    #[test]
    fn set_visit_returns_true_only_first_time() {
        let mut visited = Set::<usize>::default();

        assert!(visited.visit(10)); // first time: true
        assert!(visited.is_visited(&10));
        assert!(!visited.visit(10)); // second time: false
        assert!(visited.is_visited(&10));
    }

    #[test]
    fn set_visited_behaves_like_hashset() {
        let mut visited = Set::<usize>::default();
        let mut reference = HashSet::new();

        for v in [1, 2, 1, 3, 2, 4] {
            let got = visited.visit(v);
            let was_new = reference.insert(v);
            assert_eq!(
                got, was_new,
                "visit({}) should match HashSet::insert({})",
                v, v
            );
            assert_eq!(
                visited.is_visited(&v),
                reference.contains(&v),
                "is_visited({}) mismatch",
                v
            );
        }

        assert_eq!(visited.len(), reference.len());
        for v in &reference {
            assert!(visited.is_visited(v));
        }
    }

    #[test]
    fn bitvec_visited_within_initial_capacity() {
        let mut visited = BitVec::from_elem(8, false);
        for i in 0..8 {
            assert!(!visited.is_visited(&i));
        }

        assert!(visited.visit(3));
        assert!(visited.is_visited(&3));
        assert!(!visited.visit(3)); // second time: false
        assert!(visited.is_visited(&3));

        // Other bits unchanged
        for i in 0..8 {
            if i != 3 {
                assert!(!visited.is_visited(&i));
            }
        }
    }

    #[test]
    fn bitvec_visited_grows_on_out_of_range_visit() {
        let mut visited = BitVec::from_elem(4, false);
        assert_eq!(visited.len(), 4);

        // Visit an out-of-range index -> should grow.
        let idx = 10usize;
        let first = visited.visit(idx);
        assert!(first);
        assert!(visited.len() > idx);
        assert!(visited.is_visited(&idx));

        // Second visit: must return false.
        let second = visited.visit(idx);
        assert!(!second);
        assert!(visited.is_visited(&idx));
    }

    #[test]
    fn bitvec_visit_multiple_indices() {
        let mut visited = BitVec::default();
        let mut reference = HashSet::new();

        let indices = [0usize, 5, 2, 5, 100, 2, 100];

        for &i in &indices {
            let got = visited.visit(i);
            let was_new = reference.insert(i);
            assert_eq!(
                got, was_new,
                "visit({}) should match HashSet::insert({})",
                i, i
            );
            assert!(visited.len() > i);
            assert_eq!(visited.is_visited(&i), reference.contains(&i));
        }

        for i in &reference {
            assert!(visited.is_visited(i));
        }
    }

    // Random vector of small usize values to exercise both implementations.
    prop_compose! {
        fn small_usize_vec()
            (values in proptest::collection::vec(0usize..200, 0..200))
            -> Vec<usize>
        {
            values
        }
    }

    proptest! {
        // Set<V> as Visited<V> must behave like a HashSet in terms of first-time insertion
        // and membership.
        #[test]
        fn prop_set_visited_matches_hashset(values in small_usize_vec()) {
            let mut visited = Set::<usize>::default();
            let mut reference = HashSet::new();

            for v in &values {
                let got_first = visited.visit(*v);
                let was_new = reference.insert(*v);
                prop_assert_eq!(
                    got_first,
                    was_new,
                    "visit({}) mismatch",
                    v
                );
                prop_assert_eq!(
                    visited.is_visited(v),
                    reference.contains(v),
                    "is_visited({}) mismatch",
                    v
                );
            }

            prop_assert_eq!(visited.len(), reference.len());
            for v in &reference {
                prop_assert!(visited.is_visited(v));
            }
        }

        // BitVec as Visited<usize> must behave like a set of visited indices.
        // We start from default (empty) so growth behavior is also exercised.
        #[test]
        fn prop_bitvec_visited_behaves_like_set(values in small_usize_vec()) {
            let mut visited = BitVec::default();
            let mut reference = HashSet::new();

            for v in &values {
                let got_first = visited.visit(*v);
                let was_new = reference.insert(*v);

                prop_assert_eq!(
                    got_first,
                    was_new,
                    "visit({}) mismatch",
                    v
                );

                // After visit, we require the bitvec to be large enough.
                prop_assert!(
                    visited.len() > *v,
                    "BitVec must grow so that len() > value ({})",
                    v
                );

                prop_assert_eq!(
                    visited.is_visited(v),
                    reference.contains(v),
                    "is_visited({}) mismatch",
                    v
                );
            }

            for v in &reference {
                prop_assert!(visited.is_visited(v));
            }
        }
    }

    #[test]
    fn random_stress_set_visited() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x_5649_5349_5445_445F);

        for _case in 0..100 {
            let mut visited = Set::<u32>::default();
            let mut reference = HashSet::new();

            let len = rng.random_range(0..500);
            for _ in 0..len {
                let v: u32 = rng.random_range(0..1_000);
                let got_first = visited.visit(v);
                let was_new = reference.insert(v);
                assert_eq!(got_first, was_new);
                assert_eq!(visited.is_visited(&v), reference.contains(&v));
            }

            assert_eq!(visited.len(), reference.len());
            for v in &reference {
                assert!(visited.is_visited(v));
            }
        }
    }

    #[test]
    fn random_stress_bitvec_visited() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x_5649_5349_5445_445F ^ 0xDEAD_BEEF);

        for _case in 0..100 {
            // Random initial capacity (including 0) to exercise grow() paths.
            let initial_len = rng.random_range(0..64);
            let mut visited = BitVec::from_elem(initial_len, false);
            let mut reference = HashSet::new();

            let steps = rng.random_range(0..500);
            for _ in 0..steps {
                let v: usize = rng.random_range(0..256);
                let got_first = visited.visit(v);
                let was_new = reference.insert(v);
                assert_eq!(got_first, was_new, "visit({}) mismatch in random test", v);
                assert!(visited.len() > v, "BitVec must grow so len() > {}", v);
                assert_eq!(visited.is_visited(&v), reference.contains(&v));
            }

            for v in &reference {
                assert!(visited.is_visited(v));
            }
        }
    }
}
