use crate::lattices::{lattice::{Bottom, Top}, partial_order::PartialOrder};

/// A monotone endomorphism on a carrier `T`.
///
/// This trait does not try to prove monotonicity; that is a semantic contract
/// of the implementer.
pub trait MonotoneOperator<T> {
    /// Applies the operator to the current approximation.
    fn apply(&self, value: &T) -> T;
}

/// Least-fixpoint computation for monotone operators.
///
/// Intended for ascending iteration on a finite-height carrier.
pub trait LeastFixpoint<T>: MonotoneOperator<T>
where
    T: Clone + PartialOrder,
{
    /// Computes the least fixpoint starting from `current`.
    #[must_use]
    fn least_fixpoint_from(&self, mut current: T) -> T {
        loop {
            let next = self.apply(&current);

            if next == current {
                return current;
            }

            debug_assert!(
                current <= next,
                "least-fixpoint iteration must be ascending",
            );

            current = next;
        }
    }

    /// Computes the least fixpoint from the least element.
    #[must_use]
    fn least_fixpoint(&self) -> T
    where
        T: Bottom,
    {
        self.least_fixpoint_from(T::bottom())
    }
}

impl<T, O> LeastFixpoint<T> for O
where
    O: MonotoneOperator<T> + ?Sized,
    T: Clone + PartialOrder,
{
}

/// Greatest-fixpoint computation for monotone operators.
///
/// Intended for descending iteration on a finite-height carrier.
pub trait GreatestFixpoint<T>: MonotoneOperator<T>
where
    T: Clone + PartialOrder,
{
    /// Computes the greatest fixpoint starting from `current`.
    #[must_use]
    fn greatest_fixpoint_from(&self, mut current: T) -> T {
        loop {
            let next = self.apply(&current);

            if next == current {
                return current;
            }

            debug_assert!(
                next <= current,
                "greatest-fixpoint iteration must be descending",
            );

            current = next;
        }
    }

    /// Computes the greatest fixpoint from the greatest element.
    #[must_use]
    fn greatest_fixpoint(&self) -> T
    where
        T: Top,
    {
        self.greatest_fixpoint_from(T::top())
    }
}

impl<T, O> GreatestFixpoint<T> for O
where
    O: MonotoneOperator<T> + ?Sized,
    T: Clone + PartialOrder,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::lattices::lattice::{Bottom, Top};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Counter(u8);

    impl Bottom for Counter {
        #[inline]
        fn bottom() -> Self {
            Self(0)
        }
    }

    impl Top for Counter {
        #[inline]
        fn top() -> Self {
            Self(10)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct GrowTo {
        limit: u8,
    }

    impl MonotoneOperator<Counter> for GrowTo {
        #[inline]
        fn apply(&self, value: &Counter) -> Counter {
            Counter(value.0.max(self.limit).min(self.limit))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct StepUpTo {
        limit: u8,
    }

    impl MonotoneOperator<Counter> for StepUpTo {
        #[inline]
        fn apply(&self, value: &Counter) -> Counter {
            Counter(value.0.saturating_add(1).min(self.limit))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct DropTo {
        limit: u8,
    }

    impl MonotoneOperator<Counter> for DropTo {
        #[inline]
        fn apply(&self, value: &Counter) -> Counter {
            Counter(value.0.min(self.limit))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct StepDownTo {
        limit: u8,
    }

    impl MonotoneOperator<Counter> for StepDownTo {
        #[inline]
        fn apply(&self, value: &Counter) -> Counter {
            Counter(value.0.saturating_sub(1).max(self.limit))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Identity;

    impl MonotoneOperator<Counter> for Identity {
        #[inline]
        fn apply(&self, value: &Counter) -> Counter {
            *value
        }
    }

    #[test]
    fn least_fixpoint_from_returns_immediate_fixpoint() {
        let op = GrowTo { limit: 4 };

        assert_eq!(op.least_fixpoint_from(Counter(4)), Counter(4));
    }

    #[test]
    fn least_fixpoint_starts_from_bottom() {
        let op = GrowTo { limit: 4 };

        assert_eq!(op.least_fixpoint(), Counter(4));
    }

    #[test]
    fn least_fixpoint_from_iterates_until_stable() {
        let op = StepUpTo { limit: 4 };

        assert_eq!(op.least_fixpoint_from(Counter(1)), Counter(4));
    }

    #[test]
    fn least_fixpoint_from_bottom_iterates_until_stable() {
        let op = StepUpTo { limit: 3 };

        assert_eq!(op.least_fixpoint(), Counter(3));
    }

    #[test]
    fn greatest_fixpoint_from_returns_immediate_fixpoint() {
        let op = DropTo { limit: 4 };

        assert_eq!(op.greatest_fixpoint_from(Counter(4)), Counter(4));
    }

    #[test]
    fn greatest_fixpoint_starts_from_top() {
        let op = DropTo { limit: 4 };

        assert_eq!(op.greatest_fixpoint(), Counter(4));
    }

    #[test]
    fn greatest_fixpoint_from_iterates_until_stable() {
        let op = StepDownTo { limit: 4 };

        assert_eq!(op.greatest_fixpoint_from(Counter(8)), Counter(4));
    }

    #[test]
    fn greatest_fixpoint_from_top_iterates_until_stable() {
        let op = StepDownTo { limit: 7 };

        assert_eq!(op.greatest_fixpoint(), Counter(7));
    }

    #[test]
    fn identity_operator_is_both_least_and_greatest_fixpoint() {
        let op = Identity;

        assert_eq!(op.least_fixpoint_from(Counter(6)), Counter(6));
        assert_eq!(op.greatest_fixpoint_from(Counter(6)), Counter(6));
        assert_eq!(op.least_fixpoint(), Counter(0));
        assert_eq!(op.greatest_fixpoint(), Counter(10));
    }

    #[test]
    fn blanket_impls_make_monotone_operators_fixpoint_operators() {
        let op = StepUpTo { limit: 2 };

        let least = LeastFixpoint::least_fixpoint(&op);
        let greatest = GreatestFixpoint::greatest_fixpoint_from(&op, Counter(2));

        assert_eq!(least, Counter(2));
        assert_eq!(greatest, Counter(2));
    }
}
