use std::hash::Hash;

use crate::automata::{
    acceptors::{Acceptor, OmegaAcceptor},
    alphabet::Alphabet,
};

pub trait OmegaAutomaton {
    type State: Eq + Hash + Clone;
    type Label: Eq + Hash + Clone;
    type Acceptor: OmegaAcceptor;

    fn initial(&self) -> Self::State;
    fn alphabet(&self) -> &Alphabet<Self::Label>;
    fn acceptor(&self) -> &Self::Acceptor;

    #[must_use]
    #[inline]
    fn accepts_summary(&self, summary: &<Self::Acceptor as Acceptor>::Summary) -> bool {
        self.acceptor().accept(summary)
    }
}
