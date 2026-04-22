pub trait TerminalSummary {
    type State;

    fn terminal(&self) -> &Self::State;
}
