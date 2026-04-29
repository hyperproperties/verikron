use crate::graphs::{backward::Backward, forward::Forward, graph::Directed};

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

    /// Optional special fact for boundary or seed nodes.
    ///
    /// For forward analyses, this is usually an entry fact.
    /// For backward analyses, this is usually an exit fact.
    /// For attractor computation, this is the target region.
    fn boundary_fact(&self, node: &G::Vertex) -> Option<Self::Fact>;

    /// Applies the node's transfer function to an input fact.
    fn transfer(&self, node: &G::Vertex, input: &Self::Fact) -> Self::Fact;

    /// Merges neighboring facts into one input fact for a node.
    ///
    /// For forward analyses, the facts usually come from predecessors.
    /// For backward analyses, the facts usually come from successors.
    ///
    /// The `node` parameter allows analyses whose merge operation depends on
    /// the current node, such as game-theoretic attractor computation.
    fn merge(&self, node: &G::Vertex, facts: impl Iterator<Item = Self::Fact>) -> Self::Fact;
}

/// A monotone analysis with analysis-owned fact storage.
///
/// This lets each analysis choose its own internal representation, such as a
/// dense map, sparse map, bit vector, or region.
pub trait StatefulMonotone<G: Directed>: Monotone<G> {
    /// Final result produced after solving.
    type Output;

    /// Reads the current fact for a node.
    fn fact(&self, node: &G::Vertex) -> Self::Fact;

    /// Initializes the analysis state before solving.
    fn initialize(&mut self, graph: &G);

    /// Called when the solver has computed a new fact.
    ///
    /// Returns `true` if the stored fact changed.
    fn set(&mut self, node: &G::Vertex, fact: &Self::Fact) -> bool;

    /// Converts the final internal state into the analysis-specific result.
    fn finish(self) -> Self::Output;
}

/// Selects which neighboring nodes provide input facts and which nodes depend
/// on the current node.
pub trait Direction {
    /// Vertex type traversed by this direction.
    type Vertex: Copy;

    /// Neighbors whose facts are merged to compute `node`.
    fn dependencies(&self, node: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_;

    /// Neighbors that may need recomputation after `node` changes.
    fn dependents(&self, node: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_;
}

/// Direction for forward analyses.
///
/// A node depends on its predecessors, and changes propagate to successors.
pub struct ForwardDirection<'g, G> {
    graph: &'g G,
}

impl<'g, G> ForwardDirection<'g, G> {
    /// Creates a forward direction over `graph`.
    pub fn new(graph: &'g G) -> Self {
        Self { graph }
    }
}

impl<'g, G> Direction for ForwardDirection<'g, G>
where
    G: Directed,
{
    type Vertex = G::Vertex;

    fn dependencies(&self, node: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_ {
        self.graph
            .predecessors(node)
            .map(|edge| self.graph.source(edge))
    }

    fn dependents(&self, node: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_ {
        self.graph
            .successors(node)
            .map(|edge| self.graph.destination(edge))
    }
}

/// Direction for backward analyses.
///
/// A node depends on its successors, and changes propagate to predecessors.
pub struct BackwardDirection<'g, G> {
    graph: &'g G,
}

impl<'g, G> BackwardDirection<'g, G> {
    /// Creates a backward direction over `graph`.
    pub fn new(graph: &'g G) -> Self {
        Self { graph }
    }
}

impl<'g, G> Direction for BackwardDirection<'g, G>
where
    G: Directed,
{
    type Vertex = G::Vertex;

    fn dependencies(&self, node: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_ {
        self.graph
            .successors(node)
            .map(|edge| self.graph.destination(edge))
    }

    fn dependents(&self, node: Self::Vertex) -> impl Iterator<Item = Self::Vertex> + '_ {
        self.graph
            .predecessors(node)
            .map(|edge| self.graph.source(edge))
    }
}
