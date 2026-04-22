use crate::automata::terminal_summary::TerminalSummary;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteStateRun<S> {
    states: Vec<S>,
}

impl<S> FiniteStateRun<S> {
    #[must_use]
    #[inline]
    pub fn new(states: Vec<S>) -> Self {
        assert!(!states.is_empty(), "finite run must be non-empty");
        Self { states }
    }

    #[must_use]
    #[inline]
    pub fn states(&self) -> &[S] {
        &self.states
    }

    #[must_use]
    #[inline]
    pub fn into_states(self) -> Vec<S> {
        self.states
    }
}

impl<S> TerminalSummary for FiniteStateRun<S> {
    type State = S;

    #[inline]
    fn terminal(&self) -> &Self::State {
        self.states.last().expect("finite run must be non-empty")
    }
}
