use std::{collections::HashMap, marker::PhantomData};

use crate::games::{arena::Arena, play::Play, strategy::Strategy};

/// A positional strategy represented by a partial position-to-successor map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionalMapStrategy<A, P>
where
    A: Arena,
    P: Play<Position = A::Position>,
{
    player: A::Player,
    choices: HashMap<A::Position, A::Position>,
    marker: PhantomData<P>,
}

impl<A, P> PositionalMapStrategy<A, P>
where
    A: Arena,
    P: Play<Position = A::Position>,
{
    /// Creates a positional strategy from a choice map.
    #[must_use]
    #[inline]
    pub fn new(player: A::Player, choices: HashMap<A::Position, A::Position>) -> Self {
        Self {
            player,
            choices,
            marker: PhantomData,
        }
    }

    /// Returns the empty positional strategy.
    #[must_use]
    #[inline]
    pub fn empty(player: A::Player) -> Self {
        Self::new(player, HashMap::new())
    }

    /// Returns the player following the strategy.
    #[must_use]
    #[inline]
    pub fn player(&self) -> A::Player {
        self.player
    }

    /// Returns the underlying choice map.
    #[must_use]
    #[inline]
    pub fn choices(&self) -> &HashMap<A::Position, A::Position> {
        &self.choices
    }

    /// Consumes the strategy and returns the choice map.
    #[must_use]
    #[inline]
    pub fn into_choices(self) -> HashMap<A::Position, A::Position> {
        self.choices
    }

    /// Inserts or replaces the choice at `position`.
    #[inline]
    pub fn insert_choice(
        &mut self,
        position: A::Position,
        successor: A::Position,
    ) -> Option<A::Position> {
        self.choices.insert(position, successor)
    }

    /// Removes the choice at `position`.
    #[inline]
    pub fn remove_choice(&mut self, position: &A::Position) -> Option<A::Position> {
        self.choices.remove(position)
    }

    /// Returns the chosen successor at `position`, if defined.
    #[must_use]
    #[inline]
    pub fn choice_of(&self, position: A::Position) -> Option<A::Position> {
        self.choices.get(&position).copied()
    }

    /// Returns whether the strategy is defined at `position`.
    #[must_use]
    #[inline]
    pub fn is_defined_at(&self, position: A::Position) -> bool {
        self.choices.contains_key(&position)
    }
}

impl<A, P> Strategy for PositionalMapStrategy<A, P>
where
    A: Arena,
    P: Play<Position = A::Position>,
{
    type Arena = A;
    type Memory = ();

    #[inline]
    fn player(&self) -> <Self::Arena as Arena>::Player {
        self.player
    }

    #[inline]
    fn initial_memory(&self) -> Self::Memory {}

    #[inline]
    fn update(
        &self,
        _memory: &Self::Memory,
        _position: <Self::Arena as Arena>::Position,
    ) -> Self::Memory {
    }

    #[inline]
    fn choice(
        &self,
        _memory: &Self::Memory,
        position: <Self::Arena as Arena>::Position,
    ) -> Option<<Self::Arena as Arena>::Position> {
        self.choice_of(position)
    }
}
