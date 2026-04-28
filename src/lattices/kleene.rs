use crate::lattices::{
    fixpoint::{Fixpoint, MonotoneOperator},
    partial_order::PartialOrder,
};

#[derive(Clone, Copy, Debug)]
pub struct Kleene<O> {
    operator: O,
}

impl<O> Kleene<O> {
    #[must_use]
    #[inline]
    pub fn new(operator: O) -> Self {
        Self { operator }
    }

    #[must_use]
    #[inline]
    pub fn operator(&self) -> &O {
        &self.operator
    }
}

impl<O> Default for Kleene<O>
where
    O: Default,
{
    #[inline]
    fn default() -> Self {
        Self {
            operator: O::default(),
        }
    }
}

impl<T, O> Fixpoint<T> for Kleene<O>
where
    T: PartialOrder,
    O: MonotoneOperator<T>,
{
    #[inline]
    fn least_fixpoint_from(&self, current: &mut T) {
        while self.operator.apply(current) {}
    }
}
