use crate::lattices::{lattice::Bottom, partial_order::PartialOrder};

pub trait MonotoneOperator<T>
where
    T: PartialOrder,
{
    fn apply(&self, current: &mut T) -> bool;
}

pub trait IncrementalMonotoneOperator<T>
where
    T: PartialOrder,
{
    type Item;

    fn apply_increment(&self, current: &T, item: Self::Item) -> Option<Vec<Self::Item>>;
}

pub trait Fixpoint<T>
where
    Self: Sized,
    T: PartialOrder,
{
    fn least_fixpoint_from(self, current: &mut T);

    #[must_use]
    fn least_fixpoint(self) -> T
    where
        T: Bottom,
    {
        let mut current = T::bottom();
        self.least_fixpoint_from(&mut current);
        current
    }
}
