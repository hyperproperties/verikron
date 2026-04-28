use crate::lattices::{
    fixpoint::{Fixpoint, IncrementalMonotoneOperator},
    frontier::Frontier,
    partial_order::PartialOrder,
};

#[derive(Clone, Debug)]
pub struct Worklist<O, F> {
    operator: O,
    frontier: F,
}

impl<O, F> Worklist<O, F> {
    #[must_use]
    #[inline]
    pub fn new(operator: O, frontier: F) -> Self {
        Self { operator, frontier }
    }

    #[must_use]
    #[inline]
    pub fn operator(&self) -> &O {
        &self.operator
    }

    #[must_use]
    #[inline]
    pub fn frontier(&self) -> &F {
        &self.frontier
    }
}

impl<T, O, F> Fixpoint<T> for Worklist<O, F>
where
    T: PartialOrder,
    F: Frontier + Clone,
    F::Item: Copy,
    O: IncrementalMonotoneOperator<T, Item = F::Item>,
{
    #[inline]
    fn least_fixpoint_from(&self, current: &mut T) {
        let mut frontier = self.frontier.clone();

        while let Some(item) = frontier.pop() {
            if let Some(affected) = self.operator.apply_increment(current, item) {
                for item in affected {
                    frontier.push(item);
                }
            }
        }
    }
}
