use std::hash::Hash;

pub trait TransitionRelation {
    type State: Eq + Hash + Clone;
    type Label: Eq + Hash + Clone;

    type Transition: Copy + Eq;
    type Transitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    fn transitions(&self, source: Self::State, label: &Self::Label) -> Self::Transitions<'_>;
    fn target(&self, transition: Self::Transition) -> Self::State;
}

pub trait BackwardTransitionRelation: TransitionRelation {
    type IncomingTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    fn incoming_transitions(
        &self,
        target: Self::State,
        label: &Self::Label,
    ) -> Self::IncomingTransitions<'_>;

    fn source(&self, transition: Self::Transition) -> Self::State;
}
