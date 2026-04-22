use std::hash::Hash;

use crate::{
    automata::{
        acceptors::{Acceptor, OmegaAcceptor},
        infinite_summary::InfiniteStateSummary,
    },
    lattices::set::Set,
};

/// A summary that is sufficient to evaluate Büchi acceptance.
pub trait BuchiSummary {
    type State: Eq + Hash;

    fn visits_accepting_infinitely_often(&self, accepting: &Set<Self::State>) -> bool;
}

impl<T> BuchiSummary for T
where
    T: InfiniteStateSummary,
{
    type State = T::State;

    #[inline]
    fn visits_accepting_infinitely_often(&self, accepting: &Set<Self::State>) -> bool {
        self.infinitely_often()
            .into_iter()
            .any(|state| accepting.contains(state))
    }
}

/// Büchi acceptance condition.
///
/// A run is accepting iff it is infinite and visits at least one accepting
/// state infinitely often.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buchi<S: Eq + Hash> {
    accepting: Set<S>,
}

impl<S> Buchi<S>
where
    S: Eq + Hash,
{
    /// Creates a Büchi condition with the given accepting states.
    #[must_use]
    #[inline]
    pub fn new(accepting: Set<S>) -> Self {
        Self { accepting }
    }

    /// Returns the accepting states.
    #[must_use]
    #[inline]
    pub fn accepting(&self) -> &Set<S> {
        &self.accepting
    }
}

impl<S> From<Set<S>> for Buchi<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(accepting: Set<S>) -> Self {
        Self::new(accepting)
    }
}

impl<S> Acceptor for Buchi<S>
where
    S: Eq + Hash,
{
    type Summary = dyn BuchiSummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        summary.visits_accepting_infinitely_often(&self.accepting)
    }
}

impl<S> OmegaAcceptor for Buchi<S> where S: Eq + Hash {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        automata::{
            acceptors::Acceptor, alphabet::Alphabet, automaton::Automaton,
            infinite_summary::InfiniteStateSummary,
        },
        graphs::{
            arc::FromArcs,
            attributed::AttributedGraph,
            csr::CSR,
            properties::IndexedProperties,
        },
        lattices::set::Set,
    };

    fn set_of<T: Eq + Hash>(items: impl IntoIterator<Item = T>) -> Set<T> {
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
    -> Automaton<AttributedGraph<CSR, (), IndexedProperties<char>>, Buchi<usize>> {
        let graph = CSR::from_arcs([(0, 1), (0, 2), (1, 2), (2, 2)]);

        let relation =
            AttributedGraph::with_edge_properties(graph, vec!['a', 'b', 'a', 'b'].into());

        let alphabet: Alphabet<char> = ['a', 'b'].into();
        let acceptor = Buchi::new([2].into());

        Automaton::new(0, relation, alphabet, acceptor)
    }

    #[test]
    fn new_exposes_accepting_states() {
        let accepting = set_of([1, 3, 5]);
        let buchi = Buchi::new(accepting.clone());

        assert_eq!(buchi.accepting(), &accepting);
    }

    #[test]
    fn from_set_builds_same_acceptor() {
        let accepting = set_of([2, 4, 6]);

        let from_new = Buchi::new(accepting.clone());
        let from_from: Buchi<u32> = accepting.into();

        assert_eq!(from_new, from_from);
    }

    #[test]
    fn accepts_when_infinitely_often_states_contain_an_accepting_state() {
        let buchi = Buchi::new(set_of([2, 4]));
        let summary = InfStates::new(vec![1, 2, 3]);

        assert!(Acceptor::accept(&buchi, &summary));
    }

    #[test]
    fn rejects_when_infinitely_often_states_are_disjoint_from_accepting_states() {
        let buchi = Buchi::new(set_of([4, 5]));
        let summary = InfStates::new(vec![1, 2, 3]);

        assert!(!Acceptor::accept(&buchi, &summary));
    }

    #[test]
    fn accepts_summary_through_trait_object() {
        let buchi = Buchi::new(set_of([2]));
        let summary = InfStates::new(vec![2, 3]);

        let witness: &dyn BuchiSummary<State = usize> = &summary;
        assert!(Acceptor::accept(&buchi, witness));
    }

    #[test]
    fn automaton_with_attributed_graph_transition_relation_accepts_accepting_summary() {
        let automaton = sample_automaton();
        let summary = InfStates::new(vec![2]);

        assert!(automaton.accepts(&summary));
    }

    #[test]
    fn automaton_with_attributed_graph_transition_relation_rejects_non_accepting_summary() {
        let automaton = sample_automaton();
        let summary = InfStates::new(vec![0, 1]);

        assert!(!automaton.accepts(&summary));
    }

    #[test]
    fn automaton_acceptance_is_independent_of_transition_letters() {
        let automaton = sample_automaton();

        let accepting_summary = InfStates::new(vec![2]);
        let rejecting_summary = InfStates::new(vec![0, 1]);

        assert!(automaton.accepts(&accepting_summary));
        assert!(!automaton.accepts(&rejecting_summary));
    }
}
