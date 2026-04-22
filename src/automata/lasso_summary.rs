use std::{collections::HashSet, hash::Hash};

use crate::automata::{finite_summary::FiniteStateSummary, infinite_summary::InfiniteStateSummary};

/// A summary that exposes an ultimately periodic run.
///
/// The represented run is
/// `stem · cycle^ω`, where `cycle` must be non-empty.
pub trait LassoSummary {
    type State;

    fn stem(&self) -> &[Self::State];
    fn cycle(&self) -> &[Self::State];
}

/// A concrete lasso-shaped run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateLasso<S> {
    stem: Vec<S>,
    cycle: Vec<S>,
}

impl<S> StateLasso<S> {
    #[must_use]
    #[inline]
    pub fn new(stem: Vec<S>, cycle: Vec<S>) -> Self {
        assert!(!cycle.is_empty(), "lasso cycle must be non-empty");
        Self { stem, cycle }
    }

    #[must_use]
    #[inline]
    pub fn stem(&self) -> &[S] {
        &self.stem
    }

    #[must_use]
    #[inline]
    pub fn cycle(&self) -> &[S] {
        &self.cycle
    }

    #[must_use]
    #[inline]
    pub fn into_parts(self) -> (Vec<S>, Vec<S>) {
        (self.stem, self.cycle)
    }
}

impl<S> LassoSummary for StateLasso<S> {
    type State = S;

    #[inline]
    fn stem(&self) -> &[Self::State] {
        &self.stem
    }

    #[inline]
    fn cycle(&self) -> &[Self::State] {
        &self.cycle
    }
}

impl<S> InfiniteStateSummary for StateLasso<S>
where
    S: Eq + Hash,
{
    type State = S;

    type InfinitelyOften<'a>
        = std::collections::hash_set::IntoIter<&'a S>
    where
        Self: 'a,
        S: 'a;

    #[inline]
    fn infinitely_often(&self) -> Self::InfinitelyOften<'_> {
        self.cycle.iter().collect::<HashSet<_>>().into_iter()
    }
}

impl<S> FiniteStateSummary for StateLasso<S>
where
    S: Eq + Hash,
{
    type State = S;

    type FinitelyOften<'a>
        = std::collections::hash_set::IntoIter<&'a S>
    where
        Self: 'a,
        S: 'a;

    #[inline]
    fn finitely_often(&self) -> Self::FinitelyOften<'_> {
        let cycle_states: HashSet<_> = self.cycle.iter().collect();

        self.stem
            .iter()
            .filter(|state| !cycle_states.contains(state))
            .collect::<HashSet<_>>()
            .into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_lasso_exposes_stem_and_cycle() {
        let lasso = StateLasso::new(vec![0, 1], vec![2, 3]);

        assert_eq!(lasso.stem(), &[0, 1]);
        assert_eq!(lasso.cycle(), &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "lasso cycle must be non-empty")]
    fn state_lasso_new_panics_on_empty_cycle() {
        let _ = StateLasso::<u32>::new(vec![0, 1], vec![]);
    }

    #[test]
    fn lasso_summary_trait_exposes_stem_and_cycle() {
        let lasso = StateLasso::new(vec![1, 2], vec![3, 4]);

        let summary: &dyn LassoSummary<State = u32> = &lasso;
        assert_eq!(summary.stem(), &[1, 2]);
        assert_eq!(summary.cycle(), &[3, 4]);
    }

    #[test]
    fn infinite_state_summary_is_given_by_cycle_states() {
        let lasso = StateLasso::new(vec![0, 1, 2], vec![2, 3, 2]);

        let states: HashSet<_> = lasso.infinitely_often().into_iter().copied().collect();
        assert_eq!(states, HashSet::from([2, 3]));
    }

    #[test]
    fn finite_state_summary_is_given_by_stem_states_not_in_cycle() {
        let lasso = StateLasso::new(vec![0, 1, 2, 1], vec![2, 3, 2]);

        let states: HashSet<_> = lasso.finitely_often().into_iter().copied().collect();
        assert_eq!(states, HashSet::from([0, 1]));
    }

    #[test]
    fn state_lasso_into_parts_returns_original_vectors() {
        let lasso = StateLasso::new(vec![1, 2], vec![3, 4]);
        let (stem, cycle) = lasso.into_parts();

        assert_eq!(stem, vec![1, 2]);
        assert_eq!(cycle, vec![3, 4]);
    }
}
