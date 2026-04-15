use std::{collections::VecDeque, mem, ops::Index};

/// Layer-by-layer frontier interface.
pub trait Frontier<T> {
    /// Advances one layer and returns the previous layer.
    ///
    /// The callback receives the current layer and fills the next layer.
    fn step<F>(&mut self, expand: F) -> Option<Vec<T>>
    where
        F: FnMut(&[T], &mut Vec<T>);
}

/// Layered frontier with separate current and next layers.
#[derive(Default)]
pub struct LayeredFrontier<T> {
    frontier: Vec<T>,
    next: Vec<T>,
}

impl<T> LayeredFrontier<T> {
    /// Creates a frontier from an initial layer.
    pub fn new<I: IntoIterator<Item = T>>(initial: I) -> Self {
        let mut frontier = Vec::new();
        frontier.extend(initial);
        Self {
            frontier,
            next: Vec::new(),
        }
    }

    /// Returns whether the current layer is empty.
    pub fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }

    /// Returns the current layer.
    pub fn layer(&self) -> &[T] {
        &self.frontier
    }

    /// Returns the size of the current layer.
    pub fn len(&self) -> usize {
        self.frontier.len()
    }
}

impl<T> Index<usize> for LayeredFrontier<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.frontier[index]
    }
}

impl<T> Frontier<T> for LayeredFrontier<T> {
    fn step<F>(&mut self, mut expand: F) -> Option<Vec<T>>
    where
        F: FnMut(&[T], &mut Vec<T>),
    {
        if self.frontier.is_empty() {
            return None;
        }

        let current = mem::take(&mut self.frontier);
        self.next.clear();

        expand(&current, &mut self.next);

        self.frontier = mem::take(&mut self.next);
        Some(current)
    }
}

/// Push/pop frontier interface for graph search.
pub trait SearchFrontier<T> {
    /// Pushes a value.
    fn push(&mut self, v: T);

    /// Pops the next value.
    fn pop(&mut self) -> Option<T>;

    /// Returns whether the frontier is empty.
    fn is_empty(&self) -> bool;
}

/// LIFO search frontier.
pub struct StackFrontier<T>(Vec<T>);

/// FIFO search frontier.
pub struct QueueFrontier<T>(VecDeque<T>);

impl<T> Default for StackFrontier<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StackFrontier<T> {
    /// Creates an empty stack frontier.
    pub fn new() -> Self {
        StackFrontier(Vec::new())
    }
}

impl<T> Default for QueueFrontier<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> QueueFrontier<T> {
    /// Creates an empty queue frontier.
    pub fn new() -> Self {
        QueueFrontier(VecDeque::new())
    }
}

impl<T> SearchFrontier<T> for StackFrontier<T> {
    #[inline]
    fn push(&mut self, v: T) {
        self.0.push(v)
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T> SearchFrontier<T> for QueueFrontier<T> {
    #[inline]
    fn push(&mut self, v: T) {
        self.0.push_back(v)
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.0.pop_front()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn layered_frontier_new_stores_initial_layer() {
        let frontier = LayeredFrontier::new([1, 2, 3]);

        assert!(!frontier.is_empty());
        assert_eq!(frontier.len(), 3);
        assert_eq!(frontier.layer(), &[1, 2, 3]);
        assert_eq!(frontier[0], 1);
        assert_eq!(frontier[1], 2);
        assert_eq!(frontier[2], 3);
    }

    #[test]
    fn layered_frontier_empty_is_empty() {
        let frontier = LayeredFrontier::<usize>::new([]);

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert_eq!(frontier.layer(), &[]);
    }

    #[test]
    fn layered_frontier_step_returns_current_layer_and_sets_next_layer() {
        let mut frontier = LayeredFrontier::new([1, 2, 3]);

        let current = frontier.step(|current, next| {
            assert_eq!(current, &[1, 2, 3]);
            next.extend(current.iter().map(|x| x * 10));
        });

        assert_eq!(current, Some(vec![1, 2, 3]));
        assert_eq!(frontier.layer(), &[10, 20, 30]);
        assert_eq!(frontier.len(), 3);
        assert!(!frontier.is_empty());
    }

    #[test]
    fn layered_frontier_step_returns_none_when_empty() {
        let mut frontier = LayeredFrontier::<usize>::new([]);

        let current = frontier.step(|_, _| panic!("expand must not be called"));

        assert_eq!(current, None);
        assert!(frontier.is_empty());
        assert_eq!(frontier.layer(), &[]);
    }

    #[test]
    fn layered_frontier_multiple_steps_form_layers() {
        let mut frontier = LayeredFrontier::new([0]);

        let first = frontier.step(|current, next| {
            for &x in current {
                next.push(x + 1);
                next.push(x + 2);
            }
        });
        assert_eq!(first, Some(vec![0]));
        assert_eq!(frontier.layer(), &[1, 2]);

        let second = frontier.step(|current, next| {
            for &x in current {
                next.push(x + 10);
            }
        });
        assert_eq!(second, Some(vec![1, 2]));
        assert_eq!(frontier.layer(), &[11, 12]);

        let third = frontier.step(|_, _| {});
        assert_eq!(third, Some(vec![11, 12]));
        assert!(frontier.is_empty());

        let fourth = frontier.step(|_, _| {});
        assert_eq!(fourth, None);
    }

    #[test]
    fn layered_frontier_clears_previous_next_layer_contents() {
        let mut frontier = LayeredFrontier::new([1, 2]);

        let _ = frontier.step(|_, next| {
            next.extend([10, 20, 30]);
        });
        assert_eq!(frontier.layer(), &[10, 20, 30]);

        let _ = frontier.step(|_, next| {
            next.push(99);
        });
        assert_eq!(frontier.layer(), &[99]);
    }

    #[test]
    fn stack_frontier_behaves_like_stack() {
        let mut frontier = StackFrontier::new();

        assert!(frontier.is_empty());

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), None);
        assert!(frontier.is_empty());
    }

    #[test]
    fn queue_frontier_behaves_like_queue() {
        let mut frontier = QueueFrontier::new();

        assert!(frontier.is_empty());

        frontier.push(1);
        frontier.push(2);
        frontier.push(3);

        assert_eq!(frontier.pop(), Some(1));
        assert_eq!(frontier.pop(), Some(2));
        assert_eq!(frontier.pop(), Some(3));
        assert_eq!(frontier.pop(), None);
        assert!(frontier.is_empty());
    }

    proptest! {
        #[test]
        fn prop_stack_frontier_pops_reverse_of_push_order(values in prop::collection::vec(any::<i32>(), 0..64)) {
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
        fn prop_queue_frontier_pops_in_push_order(values in prop::collection::vec(any::<i32>(), 0..64)) {
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
        fn prop_layered_frontier_step_returns_previous_layer(
            initial in prop::collection::vec(any::<i32>(), 0..32),
            produced in prop::collection::vec(any::<i32>(), 0..32),
        ) {
            let mut frontier = LayeredFrontier::new(initial.clone());

            let result = frontier.step(|_, next| {
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
    }
}
