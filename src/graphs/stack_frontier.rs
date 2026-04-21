use crate::graphs::frontier::{Frontier, SearchFrontier};

/// LIFO frontier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackFrontier<T>(Vec<T>);

impl<T> Default for StackFrontier<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StackFrontier<T> {
    /// Creates an empty stack frontier.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Creates an empty stack frontier with the given capacity.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Returns the stored values as a slice.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Consumes the frontier and returns the underlying vector.
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T> FromIterator<T> for StackFrontier<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> Extend<T> for StackFrontier<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

impl<T> Frontier<T> for StackFrontier<T> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T> SearchFrontier<T> for StackFrontier<T> {
    #[inline]
    fn push(&mut self, value: T) {
        self.0.push(value);
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn new_is_empty() {
        let frontier = StackFrontier::<usize>::new();

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert_eq!(frontier.as_slice(), &[]);
    }

    #[test]
    fn with_capacity_is_empty() {
        let frontier = StackFrontier::<usize>::with_capacity(8);

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert!(frontier.into_inner().capacity() >= 8);
    }

    #[test]
    fn behaves_like_stack() {
        let mut frontier = StackFrontier::new();

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), None);
        assert!(frontier.is_empty());
    }

    #[test]
    fn clear_removes_all_values() {
        let mut frontier = StackFrontier::new();
        frontier.extend([1, 2, 3]);

        frontier.clear();

        assert!(frontier.is_empty());
        assert_eq!(frontier.pop(), None);
    }

    #[test]
    fn from_iter_preserves_push_order() {
        let frontier: StackFrontier<_> = [1, 2, 3].into_iter().collect();
        assert_eq!(frontier.as_slice(), &[1, 2, 3]);
    }

    proptest! {
        #[test]
        fn prop_pops_reverse_of_push_order(values in prop::collection::vec(any::<i32>(), 0..64)) {
            let mut frontier = StackFrontier::new();

            for &value in &values {
                frontier.push(value);
            }

            let mut popped = Vec::new();
            while let Some(value) = frontier.pop() {
                popped.push(value);
            }

            let expected: Vec<_> = values.into_iter().rev().collect();
            prop_assert_eq!(popped, expected);
            prop_assert!(frontier.is_empty());
        }

        #[test]
        fn prop_extend_matches_vec(values in prop::collection::vec(any::<i32>(), 0..64)) {
            let mut frontier = StackFrontier::new();
            frontier.extend(values.clone());

            prop_assert_eq!(frontier.into_inner(), values);
        }
    }
}
