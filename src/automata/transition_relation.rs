use std::hash::Hash;

/// A labeled transition relation for ordinary automata.
///
/// This trait exposes both:
/// - label-indexed transitions, useful for automata-theoretic constructions,
/// - all outgoing transitions, useful for graph-style exploration such as
///   reachability and SCC decompositions.
pub trait TransitionRelation {
    type State: Eq + Hash;
    type Label: Eq + Hash;

    /// An explicit transition identifier.
    type Transition: Copy + Eq;

    /// The transitions from a source state under a fixed label.
    type Transitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// All outgoing transitions from a source state, regardless of label.
    type OutgoingTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// Returns the transitions from `source` under `label`.
    fn transitions(&self, source: &Self::State, label: &Self::Label) -> Self::Transitions<'_>;

    /// Returns all outgoing transitions from `source`.
    fn outgoing_transitions(&self, source: &Self::State) -> Self::OutgoingTransitions<'_>;

    /// Returns the source of `transition`.
    fn source(&self, transition: Self::Transition) -> Self::State;

    /// Returns the target of `transition`.
    fn target(&self, transition: Self::Transition) -> Self::State;

    /// Returns the label of `transition`.
    fn label(&self, transition: Self::Transition) -> &Self::Label;
}

/// A transition relation that also supports backward exploration.
pub trait BackwardTransitionRelation: TransitionRelation {
    /// All incoming transitions to a target state, regardless of label.
    type IncomingTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// The incoming transitions to a target state under a fixed label.
    type IncomingTransitionsUnderLabel<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// Returns all incoming transitions to `target`.
    fn incoming_transitions(&self, target: &Self::State) -> Self::IncomingTransitions<'_>;

    /// Returns the incoming transitions to `target` under `label`.
    fn incoming_transitions_under_label(
        &self,
        target: &Self::State,
        label: &Self::Label,
    ) -> Self::IncomingTransitionsUnderLabel<'_>;
}
