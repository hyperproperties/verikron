use std::hash::Hash;

use crate::algebra::boolean::PositiveBooleanFormula;

/// A formula-based transition system.
///
/// The one-step behavior from `source` under `label` is given by a positive
/// Boolean formula over successor states.
pub trait FormulaTransitionSystem {
    type State: Eq + Hash;
    type Label: Eq + Hash;

    /// Returns the transition formula from `source` under `label`.
    fn formula(
        &self,
        source: &Self::State,
        label: &Self::Label,
    ) -> PositiveBooleanFormula<Self::State>;
}
