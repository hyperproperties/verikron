use std::hash::Hash;

/// A labeled transition system.
///
/// This is the natural operational semantics for ordinary automata.
/// States are configurations, transitions are one-step moves, and each move
/// carries a label.
pub trait LabeledTransitionSystem {
    type State: Eq + Hash;
    type Label: Eq + Hash;

    /// An explicit transition identifier.
    type Transition: Copy + Eq;

    type Transitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// Returns the outgoing transitions of `source`.
    fn outgoing(&self, source: &Self::State) -> Self::Transitions<'_>;

    /// Returns the source of `transition`.
    fn source(&self, transition: Self::Transition) -> Self::State;

    /// Returns the destination of `transition`.
    fn destination(&self, transition: Self::Transition) -> Self::State;

    /// Returns the label of `transition`.
    fn label(&self, transition: Self::Transition) -> &Self::Label;
}

/// A labeled transition system that also supports backward exploration.
pub trait BackwardLabeledTransitionSystem: LabeledTransitionSystem {
    type IncomingTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// Returns the incoming transitions of `target`.
    fn incoming(&self, target: &Self::State) -> Self::IncomingTransitions<'_>;
}
