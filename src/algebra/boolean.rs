use crate::{
    algebra::positive_zero_order::PositiveZeroOrder,
    lattices::lattice::{
        Bottom, BoundedLattice, DistributiveLattice, JoinSemiLattice, MeetSemiLattice, Semilattice,
        Top,
    },
};

pub type PositiveBooleanFormula<A> = PositiveZeroOrder<bool, A>;

impl Semilattice for bool {}

impl JoinSemiLattice for bool {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        *self || *other
    }
}

impl MeetSemiLattice for bool {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        *self && *other
    }
}

impl Bottom for bool {
    type Context = ();

    #[inline]
    fn bottom_with(_: &Self::Context) -> Self {
        false
    }
}

impl Top for bool {
    type Context = ();

    #[inline]
    fn top_with(_: &Self::Context) -> Self {
        true
    }
}

impl BoundedLattice for bool {}

impl DistributiveLattice for bool {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice};

    #[test]
    fn join_truth_table() {
        assert_eq!(false.join(&false), false);
        assert_eq!(false.join(&true), true);
        assert_eq!(true.join(&false), true);
        assert_eq!(true.join(&true), true);
    }

    #[test]
    fn meet_truth_table() {
        assert_eq!(false.meet(&false), false);
        assert_eq!(false.meet(&true), false);
        assert_eq!(true.meet(&false), false);
        assert_eq!(true.meet(&true), true);
    }

    #[test]
    fn bounded_lattice_top_and_bottom() {
        assert_eq!(bool::bottom(), false);
        assert_eq!(bool::top(), true);
    }

    #[test]
    fn join_is_commutative() {
        for a in [false, true] {
            for b in [false, true] {
                assert_eq!(a.join(&b), b.join(&a));
            }
        }
    }

    #[test]
    fn meet_is_commutative() {
        for a in [false, true] {
            for b in [false, true] {
                assert_eq!(a.meet(&b), b.meet(&a));
            }
        }
    }

    #[test]
    fn join_is_associative() {
        for a in [false, true] {
            for b in [false, true] {
                for c in [false, true] {
                    assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
                }
            }
        }
    }

    #[test]
    fn meet_is_associative() {
        for a in [false, true] {
            for b in [false, true] {
                for c in [false, true] {
                    assert_eq!(a.meet(&b).meet(&c), a.meet(&b.meet(&c)));
                }
            }
        }
    }

    #[test]
    fn join_and_meet_are_idempotent() {
        for a in [false, true] {
            assert_eq!(a.join(&a), a);
            assert_eq!(a.meet(&a), a);
        }
    }

    #[test]
    fn absorption_laws_hold() {
        for a in [false, true] {
            for b in [false, true] {
                assert_eq!(a.join(&a.meet(&b)), a);
                assert_eq!(a.meet(&a.join(&b)), a);
            }
        }
    }

    #[test]
    fn top_and_bottom_behave_correctly() {
        for a in [false, true] {
            assert_eq!(a.join(&bool::bottom()), a);
            assert_eq!(a.meet(&bool::top()), a);
            assert_eq!(a.join(&bool::top()), bool::top());
            assert_eq!(a.meet(&bool::bottom()), bool::bottom());
        }
    }

    #[test]
    fn boolean_positive_formula_alias_is_usable() {
        fn accepts_formula<A>(_formula: Option<PositiveBooleanFormula<A>>) {}

        let formula1: Option<PositiveBooleanFormula<&'static str>> = None;
        let formula2: Option<PositiveBooleanFormula<u32>> = None;

        accepts_formula(formula1);
        accepts_formula(formula2);
    }

    #[test]
    fn boolean_positive_formula_alias_preserves_bool_lattice_choice() {
        fn uses_boolean_positive_formula<A>() -> Option<PositiveBooleanFormula<A>> {
            None
        }

        let _: Option<PositiveBooleanFormula<&'static str>> = uses_boolean_positive_formula();
    }
}
