use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::automata::{
    acceptors::{Acceptor, OmegaAcceptor},
    automaton::Automaton,
    emptiness::Emptiness,
    infinite_summary::InfiniteStateSummary,
    transition_relation::{BackwardExplicitTransitionRelation, FiniteExplicitTransitionRelation},
};

/// Parity acceptance convention.
///
/// The convention chooses whether the minimum or maximum priority seen
/// infinitely often is relevant, and whether acceptance requires that
/// priority to be even or odd.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityConvention {
    MinEven,
    MinOdd,
    MaxEven,
    MaxOdd,
}

impl ParityConvention {
    /// Returns whether this convention uses the minimum infinitely-often
    /// priority rather than the maximum.
    #[must_use]
    #[inline]
    pub fn uses_min(self) -> bool {
        matches!(self, Self::MinEven | Self::MinOdd)
    }

    /// Returns whether `priority` is accepting under this convention.
    #[must_use]
    #[inline]
    pub fn accepts_priority(self, priority: usize) -> bool {
        match self {
            Self::MinEven | Self::MaxEven => priority.is_multiple_of(2),
            Self::MinOdd | Self::MaxOdd => !priority.is_multiple_of(2),
        }
    }
}

/// A summary that is sufficient to evaluate parity acceptance.
pub trait ParitySummary {
    type State: Eq + Hash;

    /// Returns the extremal priority seen infinitely often under `convention`.
    ///
    /// Returns `None` if no state is seen infinitely often or if some
    /// infinitely-often state has no assigned priority.
    fn extremal_priority(
        &self,
        priorities: &FxHashMap<Self::State, usize>,
        convention: ParityConvention,
    ) -> Option<usize>;
}

impl<T> ParitySummary for T
where
    T: InfiniteStateSummary,
{
    type State = T::State;

    #[inline]
    fn extremal_priority(
        &self,
        priorities: &FxHashMap<Self::State, usize>,
        convention: ParityConvention,
    ) -> Option<usize> {
        let mut inf = self
            .infinitely_often()
            .into_iter()
            .map(|state| priorities.get(state).copied());

        let first = inf.next()??;

        if convention.uses_min() {
            inf.try_fold(first, |best, priority| Some(best.min(priority?)))
        } else {
            inf.try_fold(first, |best, priority| Some(best.max(priority?)))
        }
    }
}

/// Parity acceptance condition.
///
/// A run is accepting iff it is infinite and the extremal priority among the
/// states visited infinitely often is accepting under the chosen convention.
///
/// If some infinitely-often state has no assigned priority, the run is
/// rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parity<S: Eq + Hash> {
    priorities: FxHashMap<S, usize>,
    convention: ParityConvention,
}

impl<S> Parity<S>
where
    S: Eq + Hash,
{
    /// Creates a parity condition from a priority map and convention.
    #[must_use]
    #[inline]
    pub fn new(priorities: FxHashMap<S, usize>, convention: ParityConvention) -> Self {
        Self {
            priorities,
            convention,
        }
    }

    /// Returns the priority map.
    #[must_use]
    #[inline]
    pub fn priorities(&self) -> &FxHashMap<S, usize> {
        &self.priorities
    }

    /// Returns the parity convention.
    #[must_use]
    #[inline]
    pub fn convention(&self) -> ParityConvention {
        self.convention
    }

    #[must_use]
    pub fn accepts_states<'a>(&self, states: impl IntoIterator<Item = &'a S>) -> bool
    where
        S: 'a,
    {
        let mut priorities = states
            .into_iter()
            .map(|state| self.priorities.get(state).copied());

        let Some(first) = priorities.next().flatten() else {
            return false;
        };

        let extremal = if self.convention.uses_min() {
            priorities.try_fold(first, |best, p| Some(best.min(p?)))
        } else {
            priorities.try_fold(first, |best, p| Some(best.max(p?)))
        };

        extremal.is_some_and(|p| self.convention.accepts_priority(p))
    }
}

impl<S> Acceptor for Parity<S>
where
    S: Eq + Hash,
{
    type Summary = dyn ParitySummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        summary
            .extremal_priority(&self.priorities, self.convention)
            .is_some_and(|priority| self.convention.accepts_priority(priority))
    }
}

impl<S> OmegaAcceptor for Parity<S> where S: Eq + Hash {}

impl<R> Emptiness for Automaton<R, Parity<usize>>
where
    R: BackwardExplicitTransitionRelation + FiniteExplicitTransitionRelation<State = usize>,
{
    fn is_empty(&self) -> bool {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        automata::{
            acceptors::Acceptor, alphabet::Alphabet, automaton::Automaton,
            infinite_summary::InfiniteStateSummary,
        },
        graphs::{
            arc::FromArcs, attributed::AttributedGraph, csr::CSR, properties::IndexedProperties,
        },
    };

    fn priorities_of(items: impl IntoIterator<Item = (usize, usize)>) -> FxHashMap<usize, usize> {
        items.into_iter().collect()
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct InfStates<S> {
        states: Vec<S>,
    }

    impl<S> InfStates<S> {
        fn new(states: Vec<S>) -> Self {
            Self { states }
        }
    }

    impl<S> InfiniteStateSummary for InfStates<S>
    where
        S: Eq + Hash,
    {
        type State = S;

        type InfinitelyOften<'a>
            = std::slice::Iter<'a, S>
        where
            Self: 'a,
            S: 'a;

        fn infinitely_often(&self) -> Self::InfinitelyOften<'_> {
            self.states.iter()
        }
    }

    fn sample_automaton()
    -> Automaton<AttributedGraph<CSR, (), IndexedProperties<char>>, Parity<usize>> {
        let graph = CSR::from_arcs([(0, 1), (1, 2), (2, 2)]);
        let relation = AttributedGraph::with_edge_properties(
            graph,
            IndexedProperties::<char>::from(vec!['a', 'b', 'a']),
        );
        let alphabet: Alphabet<char> = ['a', 'b'].into();
        let acceptor = Parity::new(
            priorities_of([(0, 3), (1, 2), (2, 0)]),
            ParityConvention::MinEven,
        );

        Automaton::new(0, relation, alphabet, acceptor)
    }

    #[test]
    fn convention_uses_min_matches_variants() {
        assert!(ParityConvention::MinEven.uses_min());
        assert!(ParityConvention::MinOdd.uses_min());
        assert!(!ParityConvention::MaxEven.uses_min());
        assert!(!ParityConvention::MaxOdd.uses_min());
    }

    #[test]
    fn convention_accepts_priority_matches_parity() {
        assert!(ParityConvention::MinEven.accepts_priority(2));
        assert!(!ParityConvention::MinEven.accepts_priority(3));

        assert!(ParityConvention::MinOdd.accepts_priority(3));
        assert!(!ParityConvention::MinOdd.accepts_priority(2));

        assert!(ParityConvention::MaxEven.accepts_priority(4));
        assert!(!ParityConvention::MaxEven.accepts_priority(5));

        assert!(ParityConvention::MaxOdd.accepts_priority(5));
        assert!(!ParityConvention::MaxOdd.accepts_priority(4));
    }

    #[test]
    fn new_exposes_priorities_and_convention() {
        let priorities = priorities_of([(1, 2), (2, 5)]);
        let parity = Parity::new(priorities.clone(), ParityConvention::MaxOdd);

        assert_eq!(parity.priorities(), &priorities);
        assert_eq!(parity.convention(), ParityConvention::MaxOdd);
    }

    #[test]
    fn parity_summary_uses_minimum_priority_for_min_conventions() {
        let summary = InfStates::new(vec![1usize, 2, 3]);
        let priorities = priorities_of([(1, 4), (2, 1), (3, 6)]);

        assert_eq!(
            summary.extremal_priority(&priorities, ParityConvention::MinEven),
            Some(1)
        );
        assert_eq!(
            summary.extremal_priority(&priorities, ParityConvention::MinOdd),
            Some(1)
        );
    }

    #[test]
    fn parity_summary_uses_maximum_priority_for_max_conventions() {
        let summary = InfStates::new(vec![1usize, 2, 3]);
        let priorities = priorities_of([(1, 4), (2, 1), (3, 6)]);

        assert_eq!(
            summary.extremal_priority(&priorities, ParityConvention::MaxEven),
            Some(6)
        );
        assert_eq!(
            summary.extremal_priority(&priorities, ParityConvention::MaxOdd),
            Some(6)
        );
    }

    #[test]
    fn parity_summary_rejects_when_some_infinitely_often_state_has_no_priority() {
        let summary = InfStates::new(vec![1usize, 2, 3]);
        let priorities = priorities_of([(1, 4), (3, 6)]);

        assert_eq!(
            summary.extremal_priority(&priorities, ParityConvention::MinEven),
            None
        );
        assert_eq!(
            summary.extremal_priority(&priorities, ParityConvention::MaxOdd),
            None
        );
    }

    #[test]
    fn parity_accepts_when_extremal_priority_is_accepting() {
        let parity = Parity::new(
            priorities_of([(1, 4), (2, 1), (3, 6)]),
            ParityConvention::MinOdd,
        );
        let summary = InfStates::new(vec![1usize, 2, 3]);

        assert!(Acceptor::accept(&parity, &summary));
    }

    #[test]
    fn parity_rejects_when_extremal_priority_is_rejecting() {
        let parity = Parity::new(
            priorities_of([(1, 4), (2, 1), (3, 6)]),
            ParityConvention::MinEven,
        );
        let summary = InfStates::new(vec![1usize, 2, 3]);

        assert!(!Acceptor::accept(&parity, &summary));
    }

    #[test]
    fn accepts_states_checks_min_even_correctly() {
        let parity = Parity::new(
            priorities_of([(0, 3), (1, 2), (2, 4)]),
            ParityConvention::MinEven,
        );

        assert!(parity.accepts_states([&1usize, &2usize]));
        assert!(parity.accepts_states([&0usize, &1usize]));
        assert!(!parity.accepts_states([&0usize]));
    }

    #[test]
    fn accepts_states_checks_max_odd_correctly() {
        let parity = Parity::new(
            priorities_of([(0, 3), (1, 2), (2, 4)]),
            ParityConvention::MaxOdd,
        );

        assert!(!parity.accepts_states([&1usize, &2usize]));
        assert!(parity.accepts_states([&0usize, &1usize]));
    }

    #[test]
    fn accepts_states_rejects_empty_state_set() {
        let parity = Parity::new(priorities_of([(0, 2)]), ParityConvention::MinEven);

        assert!(!parity.accepts_states(std::iter::empty::<&usize>()));
    }

    #[test]
    fn accepts_states_rejects_missing_priority() {
        let parity = Parity::new(priorities_of([(0, 2)]), ParityConvention::MinEven);

        assert!(!parity.accepts_states([&0usize, &1usize]));
    }

    #[test]
    fn automaton_with_attributed_graph_transition_relation_accepts_accepting_summary() {
        let automaton = sample_automaton();
        let summary = InfStates::new(vec![2usize]);

        assert!(automaton.accepts(&summary));
    }

    #[test]
    fn automaton_with_attributed_graph_transition_relation_rejects_rejecting_summary() {
        let automaton = sample_automaton();
        let summary = InfStates::new(vec![0usize]);

        assert!(!automaton.accepts(&summary));
    }
}
