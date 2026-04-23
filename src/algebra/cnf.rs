use crate::algebra::{
    algebra::{Addition, Additive, Distributive, Embed, Multiplication, Multiplicative},
    nnf::NNF,
};
use crate::lattices::lattice::{Bottom, Top};

pub trait CNF
where
    Self: NNF + Additive + Multiplicative + Embed + NNF + Distributive<Addition, Multiplication>,
{
    fn cnf(self) -> Self;
}

impl<T> CNF for T
where
    T: NNF + Additive + Multiplicative + Embed + NNF + Distributive<Addition, Multiplication>,
{
    fn cnf(self) -> Self {
        match self.nnf().into_multiplicative() {
            Ok(xs) => xs
                .into_iter()
                .map(Self::cnf)
                .reduce(Self::multiply)
                .unwrap_or_else(|| Self::embed(<Self as Embed>::Value::top())),

            Err(this) => match this.into_additive() {
                Ok(xs) => xs
                    .into_iter()
                    .map(Self::cnf)
                    .reduce(<Self as Distributive<Addition, Multiplication>>::distribute)
                    .unwrap_or_else(|| Self::embed(<Self as Embed>::Value::bottom())),

                Err(this) => this,
            },
        }
    }
}
