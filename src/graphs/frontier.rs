use std::{collections::VecDeque, mem, ops::Index};

pub trait Frontier<T> {
    fn step<F>(&mut self, expand: F) -> Option<Vec<T>>
    where
        F: FnMut(&[T], &mut Vec<T>);
}

#[derive(Default)]
pub struct LayeredFrontier<T> {
    frontier: Vec<T>,
    next: Vec<T>,
}

impl<T> LayeredFrontier<T> {
    pub fn new<I: IntoIterator<Item = T>>(initial: I) -> Self {
        let mut frontier = Vec::new();
        frontier.extend(initial);
        Self {
            frontier,
            next: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.frontier.is_empty()
    }

    pub fn layer(&self) -> &[T] {
        &self.frontier
    }

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

pub trait SearchFrontier<T> {
    fn push(&mut self, v: T);
    fn pop(&mut self) -> Option<T>;
    fn is_empty(&self) -> bool;
}

pub struct StackFrontier<T>(Vec<T>);
pub struct QueueFrontier<T>(VecDeque<T>);

impl<T> Default for StackFrontier<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StackFrontier<T> {
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
