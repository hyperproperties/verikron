use std::collections::HashMap;

use crate::{
    games::{
        arena::Arena,
        positional_map_strategy::PositionalMapStrategy,
        region::{DenseDynamicRegion, Region},
        safety_analysis::SafetyAnalysis,
    },
    graphs::{
        forward::Forward,
        structure::{FiniteVertices, Vertices},
    },
    lattices::monotone::{Monotone, StatefulMonotone},
};

/// Result of computing a safety-winning region together with a positional strategy.
pub struct SafetyStrategyResult<A>
where
    A: Arena<Position = usize, Vertex = usize>,
{
    pub region: DenseDynamicRegion,
    pub strategy: PositionalMapStrategy<A>,
}

/// Monotone-framework implementation of a safety objective with strategy synthesis.
///
/// The wrapped safety analysis computes the greatest fixed point of positions
/// from which the protagonist can stay inside the safe region. The strategy is built
/// from the final winning region by choosing, at each protagonist-owned winning
/// position, a successor that remains winning.
pub struct SafetyStrategySynthesis<
    'a,
    A: Arena,
    Safe: Region<A::Position>,
    Storage: Region<A::Position>,
> {
    analysis: SafetyAnalysis<'a, A, Safe, Storage>,
}

impl<'a, A, Safe, Storage> SafetyStrategySynthesis<'a, A, Safe, Storage>
where
    A: Arena,
    Safe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    #[must_use]
    #[inline]
    pub fn new(analysis: SafetyAnalysis<'a, A, Safe, Storage>) -> Self
    where
        A::Player: Copy,
    {
        Self { analysis }
    }
}

impl<'a, A, Safe, Storage> Monotone<A> for SafetyStrategySynthesis<'a, A, Safe, Storage>
where
    A: Arena,
    A::Vertex: Copy,
    A::Player: PartialEq,
    Safe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    /// Whether a position is currently considered safety-winning.
    ///
    /// `false` means losing; `true` means still considered winning.
    type Fact = bool;

    /// Starts the greatest-fixed-point computation with positions winning.
    fn initial_fact(&self) -> Self::Fact {
        self.analysis.initial_fact()
    }

    /// Forces unsafe positions to be losing.
    ///
    /// Returning `None` leaves safe positions to be computed normally by the solver.
    fn boundary_fact(&self, node: &A::Vertex) -> Option<Self::Fact> {
        self.analysis.boundary_fact(node)
    }

    /// Preserves only positions that are safe and can continue safely.
    fn transfer(&self, node: &A::Vertex, input: &Self::Fact) -> Self::Fact {
        self.analysis.transfer(node, input)
    }

    /// Merges successor facts according to the safety rule.
    ///
    /// Player-owned positions need at least one winning successor. Opponent-owned
    /// positions need all successors winning. Dead ends follow the wrapped
    /// safety analysis semantics.
    fn merge(&self, node: &A::Vertex, facts: impl Iterator<Item = Self::Fact>) -> Self::Fact {
        self.analysis.merge(node, facts)
    }
}

impl<'a, A, Safe> StatefulMonotone<A> for SafetyStrategySynthesis<'a, A, Safe, DenseDynamicRegion>
where
    A: Arena<Position = usize, Vertex = usize>,
    A::Player: PartialEq + Copy,
    A::Vertices: FiniteVertices<Vertex = usize>,
    Safe: Region<A::Position>,
{
    /// The computed safety-winning region together with the synthesized strategy.
    type Output = SafetyStrategyResult<A>;

    /// Reads the current stored fact for a position.
    fn fact(&self, node: &A::Vertex) -> Self::Fact {
        self.analysis.fact(node)
    }

    /// Initializes the wrapped safety analysis.
    fn initialize(&mut self, graph: &A) {
        self.analysis.initialize(graph);
    }

    /// Updates the wrapped safety analysis.
    ///
    /// Safety computation is greatest-fixed-point based, so positions may be
    /// removed when they can no longer guarantee staying safe.
    fn set(&mut self, node: &A::Vertex, fact: &Self::Fact) -> bool {
        self.analysis.set(node, fact)
    }

    /// Returns the computed safety-winning region and synthesized strategy.
    fn finish(self) -> Self::Output {
        let arena = self.analysis.arena();
        let protagonist = *self.analysis.protagonist();
        let winning = self
            .analysis
            .winning()
            .expect("safety strategy synthesis must be initialized before finishing");

        let mut strategy = PositionalMapStrategy::new(protagonist, HashMap::new());

        for node in arena.vertex_store().vertices() {
            if arena.owner(node) != protagonist || !winning.includes(&node) {
                continue;
            }

            let successor = arena
                .successors(node)
                .map(|edge| arena.destination(edge))
                .find(|successor| winning.includes(successor));

            if let Some(successor) = successor {
                strategy.insert_choice(node, successor);
            }
        }

        let region = self.analysis.finish();

        SafetyStrategyResult { region, strategy }
    }
}
