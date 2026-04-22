use std::hash::Hash;

use crate::automata::{
    acceptors::{Acceptor, StateSummary},
    alphabet::Alphabet,
};

pub trait OmegaAutomaton {
    type State: Eq + Hash + Clone;
    type Label: Eq + Hash + Clone;
    type Acceptor: Acceptor<Summary = StateSummary<Self::State>>;

    fn initial(&self) -> Self::State;
    fn alphabet(&self) -> &Alphabet<Self::Label>;
    fn acceptor(&self) -> &Self::Acceptor;

    #[must_use]
    #[inline]
    fn accepts_summary(&self, summary: &StateSummary<Self::State>) -> bool {
        self.acceptor().accept(summary)
    }
}
