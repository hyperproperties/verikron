use crate::automata::terminal_summary::TerminalSummary;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalState<S> {
    terminal: S,
}

impl<S> TerminalState<S> {
    #[must_use]
    #[inline]
    pub fn new(terminal: S) -> Self {
        Self { terminal }
    }

    #[must_use]
    #[inline]
    pub fn into_terminal(self) -> S {
        self.terminal
    }
}

impl<S> From<S> for TerminalState<S> {
    #[inline]
    fn from(terminal: S) -> Self {
        Self::new(terminal)
    }
}

impl<S> TerminalSummary for TerminalState<S> {
    type State = S;

    #[inline]
    fn terminal(&self) -> &Self::State {
        &self.terminal
    }
}
