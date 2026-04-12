use std::hash::Hash;

use nonempty::NonEmpty;

use crate::lattices::set::Set;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transition<L, S> {
    pub label: L,
    pub state: S,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trace<S, L> {
    Finite {
        initial: S,
        steps: Vec<Transition<L, S>>,
    },
    Lasso {
        initial: S,
        prefix: Vec<Transition<L, S>>,
        cycle: NonEmpty<Transition<L, S>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Summary<S: Eq + Hash> {
    Terminal { terminal: S },
    Infinite { states: Set<S> },
}

impl<S: Eq + Hash> Summary<S> {
    #[inline]
    pub fn terminal(&self) -> Option<&S> {
        match self {
            Self::Terminal { terminal } => Some(terminal),
            Self::Infinite { .. } => None,
        }
    }

    #[inline]
    pub fn infinite_states(&self) -> Option<&Set<S>> {
        match self {
            Self::Infinite { states } => Some(states),
            Self::Terminal { .. } => None,
        }
    }
}

impl<S: Eq + Hash, L> Trace<S, L> {
    #[must_use]
    pub fn summarize(self) -> Summary<S> {
        match self {
            Trace::Finite {
                initial: start,
                mut steps,
            } => Summary::Terminal {
                terminal: steps.pop().map(|step| step.state).unwrap_or(start),
            },
            Trace::Lasso { cycle, .. } => Summary::Infinite {
                states: cycle.into_iter().map(|step| step.state).collect(),
            },
        }
    }
}
