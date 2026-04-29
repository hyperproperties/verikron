use crate::{
    graphs::graph::Directed,
    lattices::lattice::{Bottom, JoinSemiLattice, MeetSemiLattice, Top},
};

/// A monotone dataflow-analysis framework over a directed graph.
///
/// `Fact` is the abstract information tracked at each node, such as live
/// variables, reaching definitions, available expressions, or constants.
pub trait Monotone<G: Directed> {
    /// The abstract state tracked by the analysis.
    ///
    /// Examples:
    /// - live variables: `HashSet<Variable>`
    /// - reaching definitions: `HashSet<Definition>`
    /// - available expressions: `HashSet<Expression>`
    /// - constant propagation: `HashMap<Variable, ConstantValue>`
    type Fact: Clone + PartialEq;

    /// Default fact assigned to each node before solving starts.
    fn initial_fact(&self) -> Self::Fact;

    /// Optional special fact for boundary nodes.
    ///
    /// For forward analyses, this is usually an entry fact.
    /// For backward analyses, this is usually an exit fact.
    fn boundary_fact(&self, node: G::Vertex) -> Option<Self::Fact>;

    /// Applies the node's transfer function to an input fact.
    fn transfer(&self, node: G::Vertex, input: &Self::Fact) -> Self::Fact;
}

/// A monotone framework that merges facts using lattice join.
///
/// Common for may analyses, such as reaching definitions or live variables.
pub trait JoinMonotone<G: Directed>: Monotone<G>
where
    Self::Fact: JoinSemiLattice + Bottom,
{
    /// Merges facts by folding from `bottom` with `join`.
    fn merge<'a>(&self, facts: impl Iterator<Item = &'a Self::Fact>) -> Self::Fact
    where
        Self::Fact: 'a,
    {
        facts.fold(Self::Fact::bottom(), |acc, fact| acc.join(fact))
    }
}

/// A monotone framework that merges facts using lattice meet.
///
/// Common for must analyses, such as available expressions.
pub trait MeetMonotone<G: Directed>: Monotone<G>
where
    Self::Fact: MeetSemiLattice + Top,
{
    /// Merges facts by folding from `top` with `meet`.
    fn merge<'a>(&self, facts: impl Iterator<Item = &'a Self::Fact>) -> Self::Fact
    where
        Self::Fact: 'a,
    {
        facts.fold(Self::Fact::top(), |acc, fact| acc.meet(fact))
    }
}