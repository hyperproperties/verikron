use crate::lattices::lattice::{BoundedLattice, JoinSemiLattice, MeetSemiLattice};

/// Generic fixpoint iterator parameterized by a combination operator.
///
/// Repeatedly does:
///
///   next = op(state, f(state))
///
/// and stops when `next == state` or `max_iters` is reached.
#[inline]
fn fixpoint_with_op<L, F, Op>(mut state: L, f: F, mut op: Op, max_iters: usize) -> L
where
    L: Clone + PartialEq,
    F: Fn(&L) -> L,
    Op: FnMut(&L, &L) -> L,
{
    for _ in 0..max_iters {
        let fx = f(&state);
        let next = op(&state, &fx);
        if next == state {
            return state;
        }
        state = next;
    }
    state
}

/// Ascending Kleene-style fixpoint iteration:
///
///   x_{n+1} = x_n ⊔ f(x_n)
pub fn fixpoint_increasing<L, F>(state: L, f: F, max_iters: usize) -> L
where
    L: JoinSemiLattice + Clone + PartialEq,
    F: Fn(&L) -> L,
{
    fixpoint_with_op(state, f, |x, fx| x.join(fx), max_iters)
}

/// Descending Kleene-style fixpoint iteration:
///
///   x_{n+1} = x_n ⊓ f(x_n)
pub fn fixpoint_decreasing<L, F>(state: L, f: F, max_iters: usize) -> L
where
    L: MeetSemiLattice + Clone + PartialEq,
    F: Fn(&L) -> L,
{
    fixpoint_with_op(state, f, |x, fx| x.meet(fx), max_iters)
}

/// Least fixpoint: start from ⊥ and iterate upwards.
///
/// Useful when you know `f` is monotone and you want the least solution
/// of the equation x = f(x).
pub fn lfp<L, F>(f: F, max_iters: usize) -> L
where
    L: BoundedLattice + Clone + PartialEq,
    F: Fn(&L) -> L,
{
    fixpoint_increasing(L::bottom(), f, max_iters)
}

/// Greatest fixpoint: start from ⊤ and iterate downwards.
///
/// Useful for coinductive-style definitions, invariants, etc.
pub fn gfp<L, F>(f: F, max_iters: usize) -> L
where
    L: BoundedLattice + Clone + PartialEq,
    F: Fn(&L) -> L,
{
    fixpoint_decreasing(L::top(), f, max_iters)
}
