use crate::algebra::{
    algebra::{Addition, Additive, Distributive, Embed, Multiplication, Multiplicative},
    nnf::NNF,
    positive::Positive,
};
use crate::lattices::lattice::{Bottom, Top};

pub trait DNF
where
    Self: Additive + Multiplicative + Embed + Distributive<Multiplication, Addition>,
{
    fn dnf(self) -> Self;
}

impl<T> DNF for T
where
    T: NNF + Positive + Additive + Multiplicative + Embed + Distributive<Multiplication, Addition>,
{
    fn dnf(self) -> Self {
        match self.into_additive() {
            Ok(xs) => xs
                .into_iter()
                .map(Self::dnf)
                .reduce(Self::sum)
                .unwrap_or_else(|| Self::embed(<Self as Embed>::Value::bottom())),

            Err(this) => match this.into_multiplicative() {
                Ok(xs) => xs
                    .into_iter()
                    .map(Self::dnf)
                    .reduce(<Self as Distributive<Multiplication, Addition>>::distribute)
                    .unwrap_or_else(|| Self::embed(<Self as Embed>::Value::top())),

                Err(this) => this,
            },
        }
    }
}
