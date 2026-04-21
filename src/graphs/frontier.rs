/// Common functionality shared by all frontiers.
pub trait Frontier<T> {
    /// Returns the number of stored values.
    fn len(&self) -> usize;

    /// Removes all stored values.
    fn clear(&mut self);

    /// Returns whether the frontier is empty.
    #[must_use]
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Push/pop frontier for worklist-based search.
pub trait SearchFrontier<T>: Frontier<T> {
    /// Pushes a value onto the frontier.
    fn push(&mut self, value: T);

    /// Pops the next value from the frontier.
    fn pop(&mut self) -> Option<T>;
}

/// Frontier that advances one whole layer at a time.
pub trait IncrementalFrontier<T>: Frontier<T> {
    /// Returns the current layer.
    fn layer(&self) -> &[T];

    /// Advances to the next layer and returns the previous one.
    ///
    /// The callback receives the current layer and appends the next layer to
    /// the provided buffer.
    fn increment<F>(&mut self, expand: F) -> Option<Vec<T>>
    where
        F: FnMut(&[T], &mut Vec<T>);
}
