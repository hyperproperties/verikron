use crate::automata::{
    acceptors::{Acceptor, StateSummary},
    alphabet::Alphabet,
    omega::OmegaAutomaton,
    transition_relation::TransitionRelation,
};

#[derive(Clone, PartialEq, Eq)]
pub struct Automaton<R, A>
where
    R: TransitionRelation,
    A: Acceptor<Summary = StateSummary<R::State>>,
{
    initial: R::State,
    transition_relation: R,
    alphabet: Alphabet<R::Label>,
    acceptor: A,
}

impl<R, A> Automaton<R, A>
where
    R: TransitionRelation,
    A: Acceptor<Summary = StateSummary<R::State>>,
{
    #[must_use]
    #[inline]
    pub fn new(
        initial: R::State,
        transition_relation: R,
        alphabet: Alphabet<R::Label>,
        acceptor: A,
    ) -> Self {
        Self {
            initial,
            transition_relation,
            alphabet,
            acceptor,
        }
    }

    #[must_use]
    #[inline]
    pub fn transition_relation(&self) -> &R {
        &self.transition_relation
    }

    #[must_use]
    #[inline]
    pub fn into_transition_relation(self) -> R {
        self.transition_relation
    }
}

impl<R, A> OmegaAutomaton for Automaton<R, A>
where
    R: TransitionRelation,
    A: Acceptor<Summary = StateSummary<R::State>>,
{
    type State = R::State;
    type Label = R::Label;
    type Acceptor = A;

    #[inline]
    fn initial(&self) -> Self::State {
        self.initial.clone()
    }

    #[inline]
    fn alphabet(&self) -> &Alphabet<Self::Label> {
        &self.alphabet
    }

    #[inline]
    fn acceptor(&self) -> &Self::Acceptor {
        &self.acceptor
    }
}
