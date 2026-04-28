use rayon::prelude::*;

use crate::lattices::{
    fixpoint::{Fixpoint, IncrementalMonotoneOperator},
    frontier::Frontier,
    partial_order::PartialOrder,
};

#[derive(Clone, Debug)]
pub struct ParallelWorklist<O, F> {
    operator: O,
    frontier: F,
}

impl<O, F> ParallelWorklist<O, F> {
    #[must_use]
    #[inline]
    pub fn new(operator: O, frontier: F) -> Self {
        Self { operator, frontier }
    }
}

impl<T, O, F> Fixpoint<T> for ParallelWorklist<O, F>
where
    T: PartialOrder + Sync,
    O: IncrementalMonotoneOperator<T, Item = F::Item> + Sync,
    F: Frontier,
    F::Item: Send,
{
    fn least_fixpoint_from(self, current: &mut T) {
        let current: &T = current;
        let operator = &self.operator;
        let mut frontier = self.frontier;

        loop {
            let mut batch = Vec::with_capacity(frontier.len());

            while let Some(item) = frontier.pop() {
                batch.push(item);
            }

            if batch.is_empty() {
                break;
            }

            let affected: Vec<F::Item> = batch
                .into_par_iter()
                .filter_map(|item| operator.apply_increment(current, item))
                .flatten()
                .collect();

            for item in affected {
                frontier.push(item);
            }
        }
    }
}
