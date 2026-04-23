use std::hash::Hash;

/// An unlabeled transition system.
///
/// This is the most basic operational semantics used by generic exploration
/// algorithms such as reachability, worklists, BFS, and DFS.
pub trait TransitionSystem {
    type State: Eq + Hash;

    type Successors<'a>: Iterator<Item = Self::State>
    where
        Self: 'a;

    /// Returns the immediate successors of `state`.
    fn successors(&self, state: &Self::State) -> Self::Successors<'_>;
}

/// A transition system with a distinguished initial state.
pub trait RootedTransitionSystem: TransitionSystem {
    /// Returns the initial state.
    fn initial(&self) -> Self::State;
}

/// A transition system that also supports predecessor exploration.
pub trait BackwardTransitionSystem: TransitionSystem {
    type Predecessors<'a>: Iterator<Item = Self::State>
    where
        Self: 'a;

    /// Returns the immediate predecessors of `state`.
    fn predecessors(&self, state: &Self::State) -> Self::Predecessors<'_>;
}

/// A transition system with an explicit finite state space.
///
/// This is useful for global graph algorithms such as SCC decompositions and
/// explicit fixed-point computations.
pub trait FiniteStateSpace {
    type State: Eq + Hash;

    type States<'a>: Iterator<Item = &'a Self::State>
    where
        Self: 'a,
        Self::State: 'a;

    /// Returns all states of the transition system.
    fn states(&self) -> Self::States<'_>;
}
