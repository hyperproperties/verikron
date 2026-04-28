use std::collections::VecDeque;

pub trait Frontier {
    type Item;

    /// Pushes a value onto the frontier.
    fn push(&mut self, value: Self::Item);

    /// Pops the next value from the frontier.
    fn pop(&mut self) -> Option<Self::Item>;

    /// Returns the number of stored values.
    fn len(&self) -> usize;

    /// Returns whether the frontier is empty.
    #[must_use]
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// LIFO frontier.
pub type StackFrontier<T> = Vec<T>;

/// FIFO frontier.
pub type QueueFrontier<T> = VecDeque<T>;

impl<T> Frontier for Vec<T> {
    type Item = T;

    #[inline]
    fn push(&mut self, value: Self::Item) {
        Vec::push(self, value);
    }

    #[inline]
    fn pop(&mut self) -> Option<Self::Item> {
        Vec::pop(self)
    }

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T> Frontier for VecDeque<T> {
    type Item = T;

    #[inline]
    fn push(&mut self, value: Self::Item) {
        self.push_back(value);
    }

    #[inline]
    fn pop(&mut self) -> Option<Self::Item> {
        self.pop_front()
    }

    #[inline]
    fn len(&self) -> usize {
        VecDeque::len(self)
    }
}
