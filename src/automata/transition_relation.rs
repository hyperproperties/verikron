use std::hash::Hash;

use crate::graphs::{
    attributed::AttributedGraph,
    graph::{Directed, FiniteDirected, Graph, IndexedDirected},
    properties::Properties,
    structure::{
        EdgeOf, EdgeType, Edges, FiniteEdges, FiniteVertices, Structure, VertexOf, VertexType,
        Vertices,
    },
};

/// A labeled transition relation for ordinary automata.
pub trait TransitionRelation {
    type State: Eq + Hash;
    type Alphabet: Eq + Hash;

    /// An explicit transition identifier.
    type Transition: Copy + Eq;

    /// The transitions from a source state under a fixed letter.
    type ReadTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// All outgoing transitions from a source state, regardless of letter.
    type OutgoingTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// Returns the transitions from `source` under `letter`.
    fn transitions_under_letter(
        &self,
        source: &Self::State,
        letter: &Self::Alphabet,
    ) -> Self::ReadTransitions<'_>;

    /// Returns all outgoing transitions from `source`.
    fn outgoing_transitions(&self, source: &Self::State) -> Self::OutgoingTransitions<'_>;

    /// Returns the source of `transition`.
    fn source(&self, transition: Self::Transition) -> Self::State;

    /// Returns the destination of `transition`.
    fn destination(&self, transition: Self::Transition) -> Self::State;

    /// Returns the letter of `transition`.
    fn letter(&self, transition: Self::Transition) -> &Self::Alphabet;
}

/// An explicit, enumerable transition relation.
pub trait ExplicitTransitionRelation: TransitionRelation {
    /// All states of the transition relation.
    type States<'a>: Iterator<Item = Self::State>
    where
        Self: 'a;

    /// All transitions of the transition relation.
    type Transitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// Returns all states.
    fn states(&self) -> Self::States<'_>;

    /// Returns all transitions.
    fn transitions(&self) -> Self::Transitions<'_>;
}

/// A finite explicit transition relation.
pub trait FiniteExplicitTransitionRelation: ExplicitTransitionRelation {
    /// Returns the number of states.
    fn state_count(&self) -> usize;

    /// Returns the number of transitions.
    fn transition_count(&self) -> usize;
}

/// A transition relation that also supports backward exploration.
pub trait BackwardTransitionRelation: TransitionRelation {
    /// All incoming transitions to a target state, regardless of letter.
    type IncomingTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// The incoming transitions to a target state under a fixed letter.
    type WrittenTransitions<'a>: Iterator<Item = Self::Transition>
    where
        Self: 'a;

    /// Returns all incoming transitions to `target`.
    fn incoming_transitions(&self, target: &Self::State) -> Self::IncomingTransitions<'_>;

    /// Returns the incoming transitions to `target` under `letter`.
    fn incoming_transitions_under_letter(
        &self,
        target: &Self::State,
        letter: &Self::Alphabet,
    ) -> Self::WrittenTransitions<'_>;
}

/// A transition relation that is both explicit and backward-explorable.
pub trait BackwardExplicitTransitionRelation:
    ExplicitTransitionRelation + BackwardTransitionRelation
{
}

impl<T> BackwardExplicitTransitionRelation for T where
    T: ExplicitTransitionRelation + BackwardTransitionRelation
{
}

/// Iterator over edges from `source` to `destination`.
#[derive(Clone, Debug)]
pub struct Connections<'a, R>
where
    R: TransitionRelation,
    R::State: Copy,
{
    relation: &'a R,
    destination: R::State,
    outgoing: R::OutgoingTransitions<'a>,
}

impl<'a, R> Iterator for Connections<'a, R>
where
    R: TransitionRelation,
    R::State: Copy,
{
    type Item = R::Transition;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.outgoing
            .find(|edge| self.relation.destination(*edge) == self.destination)
    }
}

impl<R> VertexType for R
where
    R: ExplicitTransitionRelation,
    R::State: Copy,
{
    type Vertex = R::State;
}

impl<R> Vertices for R
where
    R: ExplicitTransitionRelation,
    R::State: Copy,
{
    type Vertices<'a>
        = R::States<'a>
    where
        Self: 'a;

    #[inline]
    fn vertices(&self) -> Self::Vertices<'_> {
        self.states()
    }
}

impl<R> EdgeType for R
where
    R: ExplicitTransitionRelation,
    R::State: Copy,
{
    type Edge = R::Transition;
}

impl<R> Edges for R
where
    R: ExplicitTransitionRelation,
    R::State: Copy,
{
    type Edges<'a>
        = R::Transitions<'a>
    where
        Self: 'a;

    #[inline]
    fn edges(&self) -> Self::Edges<'_> {
        self.transitions()
    }
}

impl<R> Structure for R
where
    R: ExplicitTransitionRelation,
    R::State: Copy,
{
    type Vertices = Self;
    type Edges = Self;

    #[inline]
    fn edge_store(&self) -> &Self::Edges {
        self
    }

    #[inline]
    fn vertex_store(&self) -> &Self::Vertices {
        self
    }
}

impl<R> FiniteVertices for R
where
    R: FiniteExplicitTransitionRelation,
    R::State: Copy,
{
    #[inline]
    fn vertex_count(&self) -> usize {
        FiniteExplicitTransitionRelation::state_count(self)
    }
}

impl<R> FiniteEdges for R
where
    R: FiniteExplicitTransitionRelation,
    R::State: Copy,
{
    #[inline]
    fn edge_count(&self) -> usize {
        FiniteExplicitTransitionRelation::transition_count(self)
    }
}

impl<R> Graph for R
where
    R: BackwardTransitionRelation + ExplicitTransitionRelation,
    R::State: Copy,
{
}

impl<R> Directed for R
where
    R: BackwardTransitionRelation + ExplicitTransitionRelation,
    R::State: Copy,
{
    type Outgoing<'a>
        = R::OutgoingTransitions<'a>
    where
        Self: 'a;

    type Incoming<'a>
        = R::IncomingTransitions<'a>
    where
        Self: 'a;

    type Connections<'a>
        = Connections<'a, R>
    where
        Self: 'a;

    #[inline]
    fn source(&self, edge: Self::Edge) -> Self::Vertex {
        TransitionRelation::source(self, edge)
    }

    #[inline]
    fn destination(&self, edge: Self::Edge) -> Self::Vertex {
        TransitionRelation::destination(self, edge)
    }

    #[inline]
    fn outgoing(&self, source: Self::Vertex) -> Self::Outgoing<'_> {
        self.outgoing_transitions(&source)
    }

    #[inline]
    fn incoming(&self, destination: Self::Vertex) -> Self::Incoming<'_> {
        self.incoming_transitions(&destination)
    }

    #[inline]
    fn connections(
        &self,
        source: Self::Vertex,
        destination: Self::Vertex,
    ) -> Self::Connections<'_> {
        Connections {
            relation: self,
            destination,
            outgoing: self.outgoing_transitions(&source),
        }
    }
}

impl<R> FiniteDirected for R
where
    R: BackwardTransitionRelation + FiniteExplicitTransitionRelation,
    R::State: Copy,
{
    #[inline]
    fn loop_degree(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex)
            .filter(|edge| self.destination(*edge) == vertex)
            .count()
    }
}

impl<R> IndexedDirected for R
where
    R: BackwardTransitionRelation + ExplicitTransitionRelation,
    R::State: Copy,
{
    #[inline]
    fn outgoing_count(&self, vertex: Self::Vertex) -> usize {
        self.outgoing(vertex).count()
    }

    #[inline]
    fn outgoing_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex> {
        self.outgoing(vertex)
            .nth(index)
            .map(|edge| self.destination(edge))
    }

    #[inline]
    fn incoming_count(&self, vertex: Self::Vertex) -> usize {
        self.incoming(vertex).count()
    }

    #[inline]
    fn incoming_at(&self, vertex: Self::Vertex, index: usize) -> Option<Self::Vertex> {
        self.incoming(vertex)
            .nth(index)
            .map(|edge| self.source(edge))
    }
}

impl<G, VP, LP> TransitionRelation for AttributedGraph<G, VP, LP>
where
    G: Directed,
    VertexOf<G>: Eq + Hash,
    LP: Properties<Key = EdgeOf<G>>,
    LP::Property: Eq + Hash,
{
    type State = VertexOf<G>;
    type Alphabet = LP::Property;
    type Transition = EdgeOf<G>;

    type ReadTransitions<'a>
        = std::vec::IntoIter<EdgeOf<G>>
    where
        Self: 'a;

    type OutgoingTransitions<'a>
        = G::Outgoing<'a>
    where
        Self: 'a;

    #[inline]
    fn transitions_under_letter(
        &self,
        source: &Self::State,
        letter: &Self::Alphabet,
    ) -> Self::ReadTransitions<'_> {
        self.graph()
            .outgoing(*source)
            .filter(|edge| {
                self.edge_properties()
                    .property(*edge)
                    .is_some_and(|l| l == letter)
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[inline]
    fn outgoing_transitions(&self, source: &Self::State) -> Self::OutgoingTransitions<'_> {
        self.graph().outgoing(*source)
    }

    #[inline]
    fn source(&self, transition: Self::Transition) -> Self::State {
        self.graph().source(transition)
    }

    #[inline]
    fn destination(&self, transition: Self::Transition) -> Self::State {
        self.graph().destination(transition)
    }

    #[inline]
    fn letter(&self, transition: Self::Transition) -> &Self::Alphabet {
        self.edge_properties()
            .property(transition)
            .expect("every transition must have a letter")
    }
}

impl<G, VP, LP> BackwardTransitionRelation for AttributedGraph<G, VP, LP>
where
    G: Directed,
    VertexOf<G>: Eq + Hash,
    LP: Properties<Key = EdgeOf<G>>,
    LP::Property: Eq + Hash,
{
    type IncomingTransitions<'a>
        = G::Incoming<'a>
    where
        Self: 'a;

    type WrittenTransitions<'a>
        = std::vec::IntoIter<EdgeOf<G>>
    where
        Self: 'a;

    #[inline]
    fn incoming_transitions(&self, target: &Self::State) -> Self::IncomingTransitions<'_> {
        self.graph().incoming(*target)
    }

    #[inline]
    fn incoming_transitions_under_letter(
        &self,
        target: &Self::State,
        letter: &Self::Alphabet,
    ) -> Self::WrittenTransitions<'_> {
        self.graph()
            .incoming(*target)
            .filter(|edge| {
                self.edge_properties()
                    .property(*edge)
                    .is_some_and(|l| l == letter)
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<G, VP, LP> ExplicitTransitionRelation for AttributedGraph<G, VP, LP>
where
    G: Directed + Structure,
    VertexOf<G>: Eq + Hash,
    LP: Properties<Key = EdgeOf<G>>,
    LP::Property: Eq + Hash,
    G::Vertices: Vertices<Vertex = VertexOf<G>>,
    G::Edges: Edges<Vertex = VertexOf<G>, Edge = EdgeOf<G>>,
{
    type States<'a>
        = <G::Vertices as Vertices>::Vertices<'a>
    where
        Self: 'a;

    type Transitions<'a>
        = <G::Edges as Edges>::Edges<'a>
    where
        Self: 'a;

    #[inline]
    fn states(&self) -> Self::States<'_> {
        self.vertex_store().vertices()
    }

    #[inline]
    fn transitions(&self) -> Self::Transitions<'_> {
        self.edge_store().edges()
    }
}

impl<G, VP, LP> FiniteExplicitTransitionRelation for AttributedGraph<G, VP, LP>
where
    G: Directed + Structure,
    VertexOf<G>: Eq + Hash,
    LP: Properties<Key = EdgeOf<G>>,
    LP::Property: Eq + Hash,
    G::Vertices: FiniteVertices<Vertex = VertexOf<G>>,
    G::Edges: FiniteEdges<Vertex = VertexOf<G>, Edge = EdgeOf<G>>,
{
    #[inline]
    fn state_count(&self) -> usize {
        self.vertex_store().vertex_count()
    }

    #[inline]
    fn transition_count(&self) -> usize {
        self.edge_store().edge_count()
    }
}
