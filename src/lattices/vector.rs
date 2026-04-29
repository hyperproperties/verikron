use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice, Semilattice};

impl<T: PartialOrd> Semilattice for Vec<T> {}

/// Order-preserving set-like lattice operations for `Vec<T>`.
///
/// Duplicates are ignored by the operations:
/// - join = union
/// - meet = intersection
impl<T> JoinSemiLattice for Vec<T>
where
    T: PartialOrd + Eq + Clone,
{
    fn join(&self, other: &Self) -> Self {
        let mut out = self.clone();

        for value in other {
            if !out.contains(value) {
                out.push(value.clone());
            }
        }

        out
    }
}

impl<T> MeetSemiLattice for Vec<T>
where
    T: PartialOrd + Eq + Clone,
{
    fn meet(&self, other: &Self) -> Self {
        let mut out = Vec::new();

        for value in self {
            if other.contains(value) && !out.contains(value) {
                out.push(value.clone());
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{collections::HashSet, hash::Hash};

    use proptest::prelude::*;

    fn as_set<T>(values: &[T]) -> HashSet<T>
    where
        T: Eq + Hash + Clone,
    {
        values.iter().cloned().collect()
    }

    prop_compose! {
        fn vec_set()(values in proptest::collection::btree_set(any::<i32>(), 0..20)) -> Vec<i32> {
            values.into_iter().collect()
        }
    }

    #[test]
    fn join_is_order_preserving_union() {
        let a = vec![1, 2];
        let b = vec![2, 3, 4];

        assert_eq!(a.join(&b), vec![1, 2, 3, 4]);
    }

    #[test]
    fn meet_is_order_preserving_intersection() {
        let a = vec![3, 1, 2];
        let b = vec![2, 3, 4];

        assert_eq!(a.meet(&b), vec![3, 2]);
    }

    proptest! {
        #[test]
        fn join_is_union(a in vec_set(), b in vec_set()) {
            let join = a.join(&b);

            let expected: HashSet<_> = as_set(&a)
                .union(&as_set(&b))
                .cloned()
                .collect();

            prop_assert_eq!(as_set(&join), expected);
        }

        #[test]
        fn meet_is_intersection(a in vec_set(), b in vec_set()) {
            let meet = a.meet(&b);

            let expected: HashSet<_> = as_set(&a)
                .intersection(&as_set(&b))
                .cloned()
                .collect();

            prop_assert_eq!(as_set(&meet), expected);
        }

        #[test]
        fn join_laws_as_sets(a in vec_set(), b in vec_set(), c in vec_set()) {
            prop_assert_eq!(as_set(&a.join(&a)), as_set(&a));
            prop_assert_eq!(as_set(&a.join(&b)), as_set(&b.join(&a)));
            prop_assert_eq!(
                as_set(&a.join(&b).join(&c)),
                as_set(&a.join(&b.join(&c))),
            );
        }

        #[test]
        fn meet_laws_as_sets(a in vec_set(), b in vec_set(), c in vec_set()) {
            prop_assert_eq!(as_set(&a.meet(&a)), as_set(&a));
            prop_assert_eq!(as_set(&a.meet(&b)), as_set(&b.meet(&a)));
            prop_assert_eq!(
                as_set(&a.meet(&b).meet(&c)),
                as_set(&a.meet(&b.meet(&c))),
            );
        }

        #[test]
        fn absorption_as_sets(a in vec_set(), b in vec_set()) {
            prop_assert_eq!(as_set(&a.join(&a.meet(&b))), as_set(&a));
            prop_assert_eq!(as_set(&a.meet(&a.join(&b))), as_set(&a));
        }

        #[test]
        fn join_is_least_upper_bound_as_sets(a in vec_set(), b in vec_set(), c in vec_set()) {
            let join = a.join(&b);

            let a = as_set(&a);
            let b = as_set(&b);
            let c = as_set(&c);
            let join = as_set(&join);

            prop_assert!(a.is_subset(&join));
            prop_assert!(b.is_subset(&join));

            if a.is_subset(&c) && b.is_subset(&c) {
                prop_assert!(join.is_subset(&c));
            }
        }

        #[test]
        fn meet_is_greatest_lower_bound_as_sets(a in vec_set(), b in vec_set(), c in vec_set()) {
            let meet = a.meet(&b);

            let a = as_set(&a);
            let b = as_set(&b);
            let c = as_set(&c);
            let meet = as_set(&meet);

            prop_assert!(meet.is_subset(&a));
            prop_assert!(meet.is_subset(&b));

            if c.is_subset(&a) && c.is_subset(&b) {
                prop_assert!(c.is_subset(&meet));
            }
        }
    }
}
