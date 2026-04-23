use std::hash::Hash;

/// A branching transition system in DNF-style operational form.
///
/// One step consists of:
/// - existentially choosing one branch, and then
/// - universally continuing from all targets in that branch.
///
/// This is the operational view corresponding to DNF or clause/hypergraph
/// presentations of alternating automata.
pub trait BranchingTransitionSystem {
    type State: Eq + Hash;
    type Label: Eq + Hash;

    /// A branch represents one conjunctive obligation.
    type Branch: Copy + Eq;

    type Branches<'a>: Iterator<Item = Self::Branch>
    where
        Self: 'a;

    type Targets<'a>: Iterator<Item = Self::State>
    where
        Self: 'a;

    /// Returns the available branches from `source` under `label`.
    fn branches(&self, source: &Self::State, label: &Self::Label) -> Self::Branches<'_>;

    /// Returns the target states required by `branch`.
    fn targets(&self, branch: Self::Branch) -> Self::Targets<'_>;
}
