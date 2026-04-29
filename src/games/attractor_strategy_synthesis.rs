use std::collections::HashMap;

use crate::{
    games::{
        arena::Arena,
        attractor_analysis::AttractorAnalysis,
        positional_map_strategy::PositionalMapStrategy,
        region::{DenseStaticRegion, Region},
    },
    graphs::{forward::Forward, structure::FiniteVertices},
    lattices::monotone::{Monotone, StatefulMonotone},
};

/// Result of computing an attractor together with a positional strategy.
pub struct AttractorStrategyResult<A>
where
    A: Arena<Position = usize, Vertex = usize>,
{
    pub region: DenseStaticRegion,
    pub strategy: PositionalMapStrategy<A>,
}

/// Monotone-framework implementation of an attractor with strategy synthesis.
///
/// The wrapped attractor analysis computes the attracted region. The strategy
/// records, for each protagonist-owned position that enters the attractor, a
/// successor that was already attracted.
pub struct AttractorStrategySynthesis<
    'a,
    A: Arena,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
    Storage: Region<A::Position>,
> {
    analysis: AttractorAnalysis<'a, A, Target, Universe, Storage>,
    strategy: Option<PositionalMapStrategy<A>>,
}

impl<'a, A, Target, Universe, Storage> AttractorStrategySynthesis<'a, A, Target, Universe, Storage>
where
    A: Arena,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    #[must_use]
    #[inline]
    pub fn new(analysis: AttractorAnalysis<'a, A, Target, Universe, Storage>) -> Self {
        Self {
            analysis,
            strategy: None,
        }
    }
}

impl<'a, A, Target, Universe, Storage> Monotone<A>
    for AttractorStrategySynthesis<'a, A, Target, Universe, Storage>
where
    A: Arena,
    A::Vertex: Copy,
    A::Player: PartialEq,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
    Storage: Region<A::Position>,
{
    /// Whether a position is currently known to be attracted.
    ///
    /// `false` means not attracted; `true` means attracted.
    type Fact = bool;

    /// Starts the fixed-point computation with no non-target positions attracted.
    fn initial_fact(&self) -> Self::Fact {
        self.analysis.initial_fact()
    }

    /// Forces target positions inside the universe to be attracted.
    ///
    /// Returning `None` leaves the position to be computed normally by the solver.
    fn boundary_fact(&self, node: &A::Vertex) -> Option<bool> {
        self.analysis.boundary_fact(node)
    }

    /// Preserves target positions and prevents attraction outside the universe.
    fn transfer(&self, node: &A::Vertex, input: &Self::Fact) -> Self::Fact {
        self.analysis.transfer(node, input)
    }

    /// Merges successor facts according to the attractor rule.
    ///
    /// Player-owned positions need at least one attracted successor. Opponent-owned
    /// positions need all successors attracted. Non-target dead ends are treated as
    /// not attracted.
    fn merge(&self, node: &A::Vertex, facts: impl Iterator<Item = Self::Fact>) -> Self::Fact {
        self.analysis.merge(node, facts)
    }
}

impl<'a, A, Target, Universe> StatefulMonotone<A>
    for AttractorStrategySynthesis<'a, A, Target, Universe, DenseStaticRegion>
where
    A: Arena<Position = usize, Vertex = usize>,
    A::Player: PartialEq + Copy,
    A::Vertices: FiniteVertices<Vertex = usize>,
    Target: Region<A::Position>,
    Universe: Region<A::Position>,
{
    /// The computed attractor region together with the synthesized strategy.
    type Output = AttractorStrategyResult<A>;

    /// Reads the current stored fact for a position.
    fn fact(&self, node: &A::Vertex) -> Self::Fact {
        self.analysis.fact(node)
    }

    /// Initializes the wrapped attractor analysis and an empty strategy.
    fn initialize(&mut self, graph: &A) {
        self.analysis.initialize(graph);

        self.strategy = Some(PositionalMapStrategy::new(
            *self.analysis.protagonist(),
            HashMap::new(),
        ));
    }

    /// Updates the mutable attractor state and records strategy choices.
    ///
    /// Attractor computation is monotone-growing, so positions are only added.
    /// The witness successor is chosen before the current position is inserted,
    /// ensuring that the strategy moves to an already attracted position.
    fn set(&mut self, node: &A::Vertex, fact: &Self::Fact) -> bool {
        if !*fact {
            return false;
        }

        let arena = self.analysis.arena();
        let protagonist = *self.analysis.protagonist();
        let is_protagonist_position = arena.owner(*node) == protagonist;
        let is_target_position = self.analysis.target().includes(node);

        let witness = if is_protagonist_position && !is_target_position {
            let attracted = self.analysis.attracted().expect(
                "attractor strategy synthesis must be initialized before reading attracted region",
            );

            arena
                .successors(*node)
                .map(|edge| arena.destination(edge))
                .find(|successor| attracted.includes(successor))
        } else {
            None
        };

        let changed = self.analysis.set(node, fact);

        if changed && is_protagonist_position && !is_target_position {
            let successor = witness
                .expect("newly attracted protagonist-owned position should have an attracted successor");

            self.strategy
                .as_mut()
                .expect("attractor strategy synthesis must be initialized before setting strategy choices")
                .insert_choice(*node, successor);
        }

        changed
    }

    /// Returns the computed attractor region and synthesized strategy.
    fn finish(self) -> Self::Output {
        let Self { analysis, strategy } = self;

        AttractorStrategyResult {
            region: analysis.finish(),
            strategy: strategy
                .expect("attractor strategy synthesis must be initialized before finishing"),
        }
    }
}
