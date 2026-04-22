use std::hash::Hash;

pub trait FiniteStateSummary {
    type State: Eq + Hash;

    type FinitelyOften<'a>: IntoIterator<Item = &'a Self::State>
    where
        Self: 'a,
        Self::State: 'a;

    fn finitely_often(&self) -> Self::FinitelyOften<'_>;
}

pub trait FiniteTransitionSummary {
    type Transition: Eq + Hash;

    type FinitelyOftenTransitions<'a>: IntoIterator<Item = &'a Self::Transition>
    where
        Self: 'a,
        Self::Transition: 'a;

    fn finitely_often_transitions(&self) -> Self::FinitelyOftenTransitions<'_>;
}
