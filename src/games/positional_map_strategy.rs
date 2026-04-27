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
    #[must_use]
    #[inline]
    pub fn new(player: A::Player, choices: HashMap<A::Position, A::Position>) -> Self {
        Self {
            player,
            choices,
            marker: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn empty(player: A::Player) -> Self {
        Self::new(player, HashMap::new())
    }

    #[must_use]
    #[inline]
    pub fn player(&self) -> A::Player {
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

    #[inline]
    pub fn insert_choice(
        &mut self,
        position: A::Position,
        successor: A::Position,
    ) -> Option<A::Position> {
        self.choices.insert(position, successor)
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

impl<A, P> Strategy for PositionalMapStrategy<A, P>
where
    A: Arena,
    P: Play<Position = A::Position>,
{
    type Arena = A;
    type Memory = ();

    #[inline]
    fn player(&self) -> A::Player {
        self.player
    }

    #[inline]
    fn initial_memory(&self) -> Self::Memory {}

    #[inline]
    fn choice(&self, _memory: Self::Memory, position: A::Position) -> Option<A::Position> {
        self.choice_of(position)
    }
}
