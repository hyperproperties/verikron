use std::{mem, ops::Index};

use crate::graphs::frontier::{Frontier, IncrementalFrontier};

/// Layered frontier with separate current and next layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayeredFrontier<T> {
    current: Vec<T>,
    next: Vec<T>,
}

impl<T> Default for LayeredFrontier<T> {
    #[inline]
    fn default() -> Self {
        Self {
            current: Vec::new(),
            next: Vec::new(),
        }
    }
}

impl<T> LayeredFrontier<T> {
    /// Creates a frontier from an initial layer.
    #[must_use]
    pub fn new<I>(initial: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self {
            current: initial.into_iter().collect(),
            next: Vec::new(),
        }
    }

    /// Creates an empty frontier with explicit capacities for the current and
    /// next layers.
    #[must_use]
    #[inline]
    pub fn with_capacity(current_capacity: usize, next_capacity: usize) -> Self {
        Self {
            current: Vec::with_capacity(current_capacity),
            next: Vec::with_capacity(next_capacity),
        }
    }

    /// Returns the current layer.
    #[must_use]
    #[inline]
    pub fn layer(&self) -> &[T] {
        &self.current
    }

    /// Returns the size of the current layer.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.current.len()
    }

    /// Returns whether the current layer is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.current.is_empty()
    }

    /// Returns the capacity of the current-layer buffer.
    #[must_use]
    #[inline]
    pub fn current_capacity(&self) -> usize {
        self.current.capacity()
    }

    /// Returns the capacity of the next-layer buffer.
    #[must_use]
    #[inline]
    pub fn next_capacity(&self) -> usize {
        self.next.capacity()
    }

    /// Clears both the current and next layers.
    #[inline]
    pub fn clear(&mut self) {
        self.current.clear();
        self.next.clear();
    }
}

impl<T> FromIterator<T> for LayeredFrontier<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter)
    }
}

impl<T> Index<usize> for LayeredFrontier<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.current[index]
    }
}

impl<T> Frontier<T> for LayeredFrontier<T> {
    #[inline]
    fn len(&self) -> usize {
        self.current.len()
    }

    #[inline]
    fn clear(&mut self) {
        LayeredFrontier::clear(self);
    }
}

impl<T> IncrementalFrontier<T> for LayeredFrontier<T> {
    #[inline]
    fn layer(&self) -> &[T] {
        &self.current
    }

    fn increment<F>(&mut self, mut expand: F) -> Option<Vec<T>>
    where
        F: FnMut(&[T], &mut Vec<T>),
    {
        if self.current.is_empty() {
            return None;
        }

        let current = mem::take(&mut self.current);
        self.next.clear();

        expand(&current, &mut self.next);

        mem::swap(&mut self.current, &mut self.next);
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn new_stores_initial_layer() {
        let frontier = LayeredFrontier::new([1, 2, 3]);

        assert!(!frontier.is_empty());
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.layer(), &[1, 2, 3]);
        assert_eq!(frontier[0], 1);
        assert_eq!(frontier[1], 2);
        assert_eq!(frontier[2], 3);
    }

    #[test]
    fn default_is_empty() {
        let frontier = LayeredFrontier::<usize>::default();

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert_eq!(frontier.layer(), &[]);
    }

    #[test]
    fn with_capacity_is_empty() {
        let frontier = LayeredFrontier::<usize>::with_capacity(8, 16);

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert!(frontier.current_capacity() >= 8);
        assert!(frontier.next_capacity() >= 16);
    }

    #[test]
    fn from_iter_matches_new() {
        let a = LayeredFrontier::new([1, 2, 3]);
        let b: LayeredFrontier<_> = [1, 2, 3].into_iter().collect();

        assert_eq!(a, b);
    }

    #[test]
    fn step_returns_current_layer_and_sets_next_layer() {
        let mut frontier = LayeredFrontier::new([1, 2, 3]);

        let current = frontier.increment(|current, next| {
            assert_eq!(current, &[1, 2, 3]);
            next.extend(current.iter().map(|x| x * 10));
        });

        assert_eq!(current, Some(vec![1, 2, 3]));
        assert_eq!(frontier.layer(), &[10, 20, 30]);
        assert_eq!(frontier.len(), 3);
        assert!(!frontier.is_empty());
    }

    #[test]
    fn step_returns_none_when_empty() {
        let mut frontier = LayeredFrontier::<usize>::default();

        let current = frontier.increment(|_, _| panic!("expand must not be called"));

        assert_eq!(current, None);
        assert!(frontier.is_empty());
        assert_eq!(frontier.layer(), &[]);
    }

    #[test]
    fn multiple_steps_form_layers() {
        let mut frontier = LayeredFrontier::new([0]);

        let first = frontier.increment(|current, next| {
            for &x in current {
                next.push(x + 1);
                next.push(x + 2);
            }
        });
        assert_eq!(first, Some(vec![0]));
        assert_eq!(frontier.layer(), &[1, 2]);

        let second = frontier.increment(|current, next| {
            for &x in current {
                next.push(x + 10);
            }
        });
        assert_eq!(second, Some(vec![1, 2]));
        assert_eq!(frontier.layer(), &[11, 12]);

        let third = frontier.increment(|_, _| {});
        assert_eq!(third, Some(vec![11, 12]));
        assert!(frontier.is_empty());

        let fourth = frontier.increment(|_, _| {});
        assert_eq!(fourth, None);
    }

    #[test]
    fn step_clears_previous_next_contents() {
        let mut frontier = LayeredFrontier::new([1, 2]);

        let _ = frontier.increment(|_, next| {
            next.extend([10, 20, 30]);
        });
        assert_eq!(frontier.layer(), &[10, 20, 30]);

        let _ = frontier.increment(|_, next| {
            next.push(99);
        });
        assert_eq!(frontier.layer(), &[99]);
    }

    #[test]
    fn clear_removes_all_state() {
        let mut frontier = LayeredFrontier::new([1, 2, 3]);

        let _ = frontier.increment(|_, next| next.extend([4, 5]));
        frontier.clear();

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert_eq!(frontier.layer(), &[]);
    }

    proptest! {
        #[test]
        fn prop_step_returns_previous_layer(
            initial in prop::collection::vec(any::<i32>(), 0..32),
            produced in prop::collection::vec(any::<i32>(), 0..32),
        ) {
            let mut frontier = LayeredFrontier::new(initial.clone());

            let result = frontier.increment(|_, next| {
                next.extend(produced.iter().copied());
            });

            if initial.is_empty() {
                prop_assert_eq!(result, None);
                prop_assert!(frontier.is_empty());
                prop_assert_eq!(frontier.layer(), &[]);
            } else {
                prop_assert_eq!(result, Some(initial));
                prop_assert_eq!(frontier.layer(), produced.as_slice());
            }
        }

        #[test]
        fn prop_clear_leaves_frontier_empty(
            initial in prop::collection::vec(any::<i32>(), 0..32),
        ) {
            let mut frontier = LayeredFrontier::new(initial);
            frontier.clear();

            prop_assert!(frontier.is_empty());
            prop_assert_eq!(frontier.len(), 0);
            prop_assert_eq!(frontier.layer(), &[]);
        }
    }
}
