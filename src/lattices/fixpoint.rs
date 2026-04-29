pub trait Fixpoint<A> {
    /// Result returned by the solver.
    type Solution;

    /// Solves `analysis` to a fixed point.
    fn solve(&self, analysis: A) -> Self::Solution;
}
