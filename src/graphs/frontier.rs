use std::{collections::VecDeque, mem, ops::Index};

/// Frontier that advances one whole layer at a time.
pub trait IncrementalFrontier<T> {
    /// Advances to the next layer and returns the previous one.
    ///
    /// The callback receives the current layer and appends the next layer to
    /// the provided buffer.
    fn step<F>(&mut self, expand: F) -> Option<Vec<T>>
    where
        F: FnMut(&[T], &mut Vec<T>);
}

/// Layered frontier with separate current and next layers.
#[derive(Debug, Clone)]
pub struct LayeredFrontier<T> {
    current: Vec<T>,
    next: Vec<T>,
}

impl<T> LayeredFrontier<T> {
    /// Creates a frontier from an initial layer.
    pub fn new<I>(initial: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut current = Vec::new();
        current.extend(initial);

        Self {
            current,
            next: Vec::new(),
        }
    }

    /// Returns whether the current layer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.current.is_empty()
    }

    /// Returns the current layer.
    #[inline]
    pub fn layer(&self) -> &[T] {
        &self.current
    }

    /// Returns the size of the current layer.
    #[inline]
    pub fn len(&self) -> usize {
        self.current.len()
    }

    /// Clears both the current and next layers.
    #[inline]
    pub fn clear(&mut self) {
        self.current.clear();
        self.next.clear();
    }
}

impl<T> Index<usize> for LayeredFrontier<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.current[index]
    }
}

impl<T> IncrementalFrontier<T> for LayeredFrontier<T> {
    fn step<F>(&mut self, mut expand: F) -> Option<Vec<T>>
    where
        F: FnMut(&[T], &mut Vec<T>),
    {
        if self.current.is_empty() {
            return None;
        }

        let current = mem::take(&mut self.current);
        self.next.clear();

        expand(&current, &mut self.next);

        self.current = mem::take(&mut self.next);
        Some(current)
    }
}

/// Push/pop frontier for worklist-based search.
pub trait SearchFrontier<T> {
    /// Pushes a value onto the frontier.
    fn push(&mut self, value: T);

    /// Pops the next value from the frontier.
    fn pop(&mut self) -> Option<T>;

    /// Returns the number of stored values.
    fn len(&self) -> usize;

    /// Returns whether the frontier is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// LIFO frontier.
pub struct StackFrontier<T>(Vec<T>);

/// FIFO frontier.
pub struct QueueFrontier<T>(VecDeque<T>);

impl<T> Default for StackFrontier<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StackFrontier<T> {
    /// Creates an empty stack frontier.
    #[inline]
    pub fn new() -> Self {
        Self(Vec::new())
    }
}

impl<T> Default for QueueFrontier<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> QueueFrontier<T> {
    /// Creates an empty queue frontier.
    #[inline]
    pub fn new() -> Self {
        Self(VecDeque::new())
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

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
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

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
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
    fn layered_frontier_clears_previous_next_contents() {
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
    fn layered_frontier_clear_removes_all_state() {
        let mut frontier = LayeredFrontier::new([1, 2, 3]);

        let _ = frontier.step(|_, next| next.extend([4, 5]));
        frontier.clear();

        assert!(frontier.is_empty());
        assert_eq!(frontier.len(), 0);
        assert_eq!(frontier.layer(), &[]);
    }

    #[test]
    fn stack_frontier_behaves_like_stack() {
        let mut frontier = StackFrontier::new();

        assert!(frontier.is_empty());

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
    fn queue_frontier_behaves_like_queue() {
        let mut frontier = QueueFrontier::new();

        assert!(frontier.is_empty());

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
