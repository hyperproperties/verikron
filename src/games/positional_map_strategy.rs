use std::{collections::HashMap, hash::Hash};

use crate::games::{
    arena::Arena,
    strategy::{PositionalStrategy, Strategy},
};

/// A positional strategy represented by a partial position-to-successor map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionalMapStrategy<A>
where
    A: Arena,
{
    player: A::Player,
    choices: HashMap<A::Position, A::Position>,
}

impl<A> PositionalMapStrategy<A>
where
    A: Arena,
{
    #[must_use]
    #[inline]
    pub fn new(player: A::Player, choices: HashMap<A::Position, A::Position>) -> Self {
        Self { player, choices }
    }

    #[must_use]
    #[inline]
    pub fn player(&self) -> A::Player
    where
        A::Player: Copy,
    {
        self.player
    }

    #[must_use]
    #[inline]
    pub fn choices(&self) -> &HashMap<A::Position, A::Position> {
        &self.choices
    }

    #[must_use]
    #[inline]
    pub fn into_choices(self) -> HashMap<A::Position, A::Position> {
        self.choices
    }
}

impl<A> PositionalMapStrategy<A>
where
    A: Arena,
    A::Position: Eq + Hash + Copy,
{
    #[inline]
    pub fn insert_choice(&mut self, position: A::Position, successor: A::Position) {
        self.choices.insert(position, successor);
    }

    #[inline]
    pub fn remove_choice(&mut self, position: A::Position) -> Option<A::Position> {
        self.choices.remove(&position)
    }

    #[must_use]
    #[inline]
    pub fn choice_of(&self, position: A::Position) -> Option<A::Position> {
        self.choices.get(&position).copied()
    }

    #[must_use]
    #[inline]
    pub fn is_defined_at(&self, position: A::Position) -> bool {
        self.choices.contains_key(&position)
    }
}

impl<A> Strategy for PositionalMapStrategy<A>
where
    A: Arena,
    A::Player: Copy,
    A::Position: Eq + Hash + Copy,
{
    type Arena = A;
    type Memory = ();

    #[inline]
    fn player(&self) -> A::Player {
        self.player
    }

    #[inline]
    fn initial_memory(&self) -> Self::Memory {
        ()
    }

    #[inline]
    fn choice(&self, _memory: Self::Memory, position: A::Position) -> Option<A::Position> {
        self.choice_of(position)
    }

    #[inline]
    fn empty(player: <Self::Arena as Arena>::Player) -> Self {
        Self::new(player, HashMap::new())
    }
}

impl<A> PositionalStrategy for PositionalMapStrategy<A>
where
    A: Arena,
    A::Player: Copy,
    A::Position: Eq + Hash + Copy,
{
    #[inline]
    fn insert_choice(
        &mut self,
        position: <Self::Arena as Arena>::Position,
        successor: <Self::Arena as Arena>::Position,
    ) {
        self.choices.insert(position, successor);
    }
}
