use std::hash::Hash;

use crate::{
    automata::{
        acceptors::{Acceptor, FiniteAcceptor},
        terminal_summary::TerminalSummary,
    },
    lattices::set::Set,
};

/// Final-state acceptance condition.
///
/// A run is accepting iff it is finite and its terminal state is accepting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Final<S: Eq + Hash> {
    accepting: Set<S>,
}

impl<S> Final<S>
where
    S: Eq + Hash,
{
    #[must_use]
    #[inline]
    pub fn new(accepting: Set<S>) -> Self {
        Self { accepting }
    }

    #[must_use]
    #[inline]
    pub fn accepting(&self) -> &Set<S> {
        &self.accepting
    }
}

impl<S> From<Set<S>> for Final<S>
where
    S: Eq + Hash,
{
    #[inline]
    fn from(accepting: Set<S>) -> Self {
        Self::new(accepting)
    }
}

impl<S> Acceptor for Final<S>
where
    S: Eq + Hash,
{
    type Summary = dyn TerminalSummary<State = S>;

    #[inline]
    fn accept(&self, summary: &Self::Summary) -> bool {
        self.accepting.contains(summary.terminal())
    }
}

impl<S> FiniteAcceptor for Final<S> where S: Eq + Hash {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::{
        acceptors::Acceptor, finite_state_run::FiniteStateRun, terminal_state::TerminalState,
    };

    fn set_of<T: Eq + Hash>(items: impl IntoIterator<Item = T>) -> Set<T> {
        items.into_iter().collect()
    }

    #[test]
    fn finite_state_run_terminal_is_last_state() {
        let run = FiniteStateRun::new(vec![0, 1, 2, 3]);
        assert_eq!(run.terminal(), &3);
    }

    #[test]
    #[should_panic(expected = "finite run must be non-empty")]
    fn finite_state_run_new_panics_on_empty_run() {
        let _ = FiniteStateRun::<u32>::new(vec![]);
    }

    #[test]
    fn terminal_state_summary_returns_terminal_state() {
        let summary = TerminalState::new(7);
        assert_eq!(summary.terminal(), &7);
    }

    #[test]
    fn terminal_state_from_state() {
        let summary: TerminalState<u32> = 11.into();
        assert_eq!(summary.terminal(), &11);
    }

    #[test]
    fn final_new_exposes_accepting_states() {
        let accepting = set_of([1, 3, 5]);
        let final_acceptor = Final::new(accepting.clone());

        assert_eq!(final_acceptor.accepting(), &accepting);
    }

    #[test]
    fn final_from_set_builds_same_acceptor() {
        let accepting = set_of([2, 4, 6]);

        let from_new = Final::new(accepting.clone());
        let from_from: Final<u32> = accepting.into();

        assert_eq!(from_new, from_from);
    }

    #[test]
    fn final_accepts_finite_run_when_terminal_state_is_accepting() {
        let final_acceptor = Final::new(set_of([2, 4, 6]));
        let run = FiniteStateRun::new(vec![0, 1, 2]);

        assert!(final_acceptor.accept(&run));
    }

    #[test]
    fn final_rejects_finite_run_when_terminal_state_is_not_accepting() {
        let final_acceptor = Final::new(set_of([2, 4, 6]));
        let run = FiniteStateRun::new(vec![0, 1, 3]);

        assert!(!final_acceptor.accept(&run));
    }

    #[test]
    fn final_accepts_terminal_state_summary_when_accepting() {
        let final_acceptor = Final::new(set_of([10, 20, 30]));
        let summary = TerminalState::new(20);

        assert!(final_acceptor.accept(&summary));
    }

    #[test]
    fn final_rejects_terminal_state_summary_when_not_accepting() {
        let final_acceptor = Final::new(set_of([10, 20, 30]));
        let summary = TerminalState::new(25);

        assert!(!final_acceptor.accept(&summary));
    }

    #[test]
    fn final_accepts_through_trait_object_summary() {
        let final_acceptor = Final::new(set_of([3]));
        let run = FiniteStateRun::new(vec![1, 2, 3]);

        let summary: &dyn TerminalSummary<State = u32> = &run;
        assert!(final_acceptor.accept(summary));
    }

    #[test]
    fn final_works_with_multiple_summary_types() {
        let final_acceptor = Final::new(set_of(["accept"]));

        let run = FiniteStateRun::new(vec!["start", "middle", "accept"]);
        let terminal = TerminalState::new("accept");

        assert!(final_acceptor.accept(&run));
        assert!(final_acceptor.accept(&terminal));
    }
}
