use std::hash::Hash;

use crate::lattices::set::Set;

/// Summary for state-based acceptance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateSummary<S: Eq + Hash> {
    Finite { terminal: S },
    Infinite { states: Set<S> },
}

/// Summary for transition-based acceptance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransitionSummary<T: Eq + Hash> {
    Finite { last: Option<T> },
    Infinite { transitions: Set<T> },
}

/// Summary for history-sensitive acceptance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonitorSummary<M> {
    pub monitor: M,
}

/// Acceptance condition over a finite summary.
///
/// The summary type is chosen by the acceptance condition.
/// This supports state-based, transition-based, and history-sensitive
/// acceptance without forcing them into one universal summary format.
pub trait Acceptor {
    type Summary;

    #[must_use]
    fn accept(&self, summary: &Self::Summary) -> bool;
}
