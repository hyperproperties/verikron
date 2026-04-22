use std::hash::Hash;

pub trait InfiniteStateSummary {
    type State: Eq + Hash;

    type InfinitelyOften<'a>: IntoIterator<Item = &'a Self::State>
    where
        Self: 'a,
        Self::State: 'a;

    fn infinitely_often(&self) -> Self::InfinitelyOften<'_>;
}

pub trait InfiniteTransitionSummary {
    type Transition: Eq + Hash;

    type InfinitelyOftenTransitions<'a>: IntoIterator<Item = &'a Self::Transition>
    where
        Self: 'a,
        Self::Transition: 'a;

    fn infinitely_often_transitions(&self) -> Self::InfinitelyOftenTransitions<'_>;
}
