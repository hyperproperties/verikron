use crate::automata::{
    acceptors::{Acceptor, OmegaAcceptor},
    alphabet::Alphabet,
    omega::OmegaAutomaton,
    transition_relation::{BackwardTransitionRelation, TransitionRelation},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Automaton<R, A>
where
    R: TransitionRelation,
    A: OmegaAcceptor,
{
    initial: R::State,
    transition_relation: R,
    alphabet: Alphabet<R::Label>,
    acceptor: A,
}

impl<R, A> Automaton<R, A>
where
    R: TransitionRelation,
    A: OmegaAcceptor,
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

    #[must_use]
    #[inline]
    pub fn accepts(&self, summary: &A::Summary) -> bool {
        self.acceptor.accept(summary)
    }
}

impl<R, A> Automaton<R, A>
where
    R: BackwardTransitionRelation,
    A: OmegaAcceptor,
{
    #[must_use]
    #[inline]
    pub fn incoming_transitions(&self, target: &R::State) -> R::IncomingTransitions<'_> {
        self.transition_relation.incoming_transitions(target)
    }

    #[must_use]
    #[inline]
    pub fn incoming_transitions_under_label(
        &self,
        target: &R::State,
        label: &R::Label,
    ) -> R::IncomingTransitionsUnderLabel<'_> {
        self.transition_relation
            .incoming_transitions_under_label(target, label)
    }
}

impl<R, A> OmegaAutomaton for Automaton<R, A>
where
    R: TransitionRelation,
    A: OmegaAcceptor,
    <R as TransitionRelation>::State: Clone,
    <R as TransitionRelation>::Label: Clone,
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
