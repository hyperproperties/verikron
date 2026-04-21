use std::collections::VecDeque;

use crate::graphs::frontier::{Frontier, SearchFrontier};

/// FIFO frontier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueFrontier<T>(VecDeque<T>);

impl<T> Default for QueueFrontier<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> QueueFrontier<T> {
    /// Creates an empty queue frontier.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self(VecDeque::new())
    }

    /// Creates an empty queue frontier with the given capacity.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(VecDeque::with_capacity(capacity))
    }

    /// Returns an iterator over the queued values in dequeue order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.0.iter()
    }

    /// Consumes the frontier and returns the underlying deque.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> VecDeque<T> {
        self.0
    }
}

impl<T> FromIterator<T> for QueueFrontier<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> Extend<T> for QueueFrontier<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

impl<T> Frontier<T> for QueueFrontier<T> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T> SearchFrontier<T> for QueueFrontier<T> {
    #[inline]
    fn push(&mut self, value: T) {
        self.0.push_back(value);
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.0.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn new_is_empty() {
        let frontier = QueueFrontier::<usize>::new();

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert_eq!(
            frontier.iter().copied().collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn with_capacity_is_empty() {
        let frontier = QueueFrontier::<usize>::with_capacity(8);

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert!(frontier.into_inner().capacity() >= 8);
    }

    #[test]
    fn behaves_like_queue() {
        let mut frontier = QueueFrontier::new();

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), None);
        assert!(frontier.is_empty());
    }

    #[test]
    fn clear_removes_all_values() {
        let mut frontier = QueueFrontier::new();
        frontier.extend([1, 2, 3]);

        frontier.clear();

        assert!(frontier.is_empty());
        assert_eq!(frontier.pop(), None);
    }

    #[test]
    fn from_iter_preserves_queue_order() {
        let frontier: QueueFrontier<_> = [1, 2, 3].into_iter().collect();
        assert_eq!(frontier.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    proptest! {
        #[test]
        fn prop_pops_in_push_order(values in prop::collection::vec(any::<i32>(), 0..64)) {
            let mut frontier = QueueFrontier::new();

            for &value in &values {
                frontier.push(value);
            }

            let mut popped = Vec::new();
            while let Some(value) = frontier.pop() {
                popped.push(value);
            }

            prop_assert_eq!(popped, values);
            prop_assert!(frontier.is_empty());
        }

        #[test]
        fn prop_extend_matches_vec(values in prop::collection::vec(any::<i32>(), 0..64)) {
            let mut frontier = QueueFrontier::new();
            frontier.extend(values.clone());

            prop_assert_eq!(
                frontier.iter().copied().collect::<Vec<_>>(),
                values,
            );
        }
    }
}
