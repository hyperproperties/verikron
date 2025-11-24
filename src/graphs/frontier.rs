use std::mem;

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
