use std::hash::Hash;

use crate::automata::{
    acceptors::OmegaAcceptor, alphabet::Alphabet, omega::OmegaAutomaton,
    transition_relation::TransitionRelation,
};

pub trait AlternatingTransitionRelation {
    type State: Eq + Hash;
    type Label: Eq + Hash;

    type Clause: Copy + Eq;
    type Clauses<'a>: Iterator<Item = Self::Clause>
    where
        Self: 'a;

    type Successors<'a>: Iterator<Item = Self::State>
    where
        Self: 'a;

    fn clauses(&self, source: &Self::State, label: &Self::Label) -> Self::Clauses<'_>;

    fn successors(&self, clause: Self::Clause) -> Self::Successors<'_>;
}

impl<T> AlternatingTransitionRelation for T
where
    T: TransitionRelation,
{
    type State = T::State;
    type Label = T::Label;

    type Clause = T::Transition;
    type Clauses<'a>
        = T::Transitions<'a>
    where
        Self: 'a;

    type Successors<'a>
        = std::iter::Once<Self::State>
    where
        Self: 'a;

    #[inline]
    fn clauses(&self, source: &Self::State, label: &Self::Label) -> Self::Clauses<'_> {
        self.transitions(source, label)
    }

    #[inline]
    fn successors(&self, clause: Self::Clause) -> Self::Successors<'_> {
        std::iter::once(self.target(clause))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlternatingAutomaton<R, A>
where
    R: AlternatingTransitionRelation,
    A: OmegaAcceptor,
{
    initial: R::State,
    transition_relation: R,
    alphabet: Alphabet<R::Label>,
    acceptor: A,
}

impl<R, A> AlternatingAutomaton<R, A>
where
    R: AlternatingTransitionRelation,
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

impl<R, A> OmegaAutomaton for AlternatingAutomaton<R, A>
where
    R: AlternatingTransitionRelation,
    A: OmegaAcceptor,
    <R as AlternatingTransitionRelation>::State: Clone,
    <R as AlternatingTransitionRelation>::Label: Clone,
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
