use crate::algebra::algebra::{
    Addition, Additive, Complemented, DeMorgan, Embed, Multiplication, Multiplicative,
};
use crate::lattices::lattice::{Bottom, Top};

pub trait NNF {
    fn nnf(self) -> Self;
}

impl<T> NNF for T
where
    T: Additive
        + Multiplicative
        + Complemented
        + Embed
        + DeMorgan<Addition, Multiplication>
        + DeMorgan<Multiplication, Addition>
        + Clone,
    <<T as Embed>::Value as Top>::Context: Default,
    <<T as Embed>::Value as Bottom>::Context: Default,
{
    fn nnf(self) -> Self {
        match self.into_complement() {
            // Handle a negated formula.
            Ok(inner) => match inner.into_complement() {
                // ¬¬x => x
                Ok(x) => x.nnf(),

                Err(inner) => match inner.clone().into_additive() {
                    // ¬(x₁ ∨ ... ∨ xₙ) => ¬x₁ ∧ ... ∧ ¬xₙ
                    Ok(_) => {
                        <T as DeMorgan<Addition, Multiplication>>::demorgan(Self::complement(inner))
                            .nnf()
                    }

                    Err(inner) => match inner.clone().into_multiplicative() {
                        // ¬(x₁ ∧ ... ∧ xₙ) => ¬x₁ ∨ ... ∨ ¬xₙ
                        Ok(_) => <T as DeMorgan<Multiplication, Addition>>::demorgan(
                            Self::complement(inner),
                        )
                        .nnf(),

                        // Leaves keep their negation in NNF.
                        Err(inner) => Self::complement(inner),
                    },
                },
            },

            // Recurse through positive structure.
            Err(this) => match this.into_additive() {
                Ok(xs) => xs
                    .into_iter()
                    .map(Self::nnf)
                    .reduce(Self::sum)
                    .unwrap_or_else(|| Self::embed(<T as Embed>::Value::bottom())),

                Err(this) => match this.into_multiplicative() {
                    Ok(xs) => xs
                        .into_iter()
                        .map(Self::nnf)
                        .reduce(Self::multiply)
                        .unwrap_or_else(|| Self::embed(<T as Embed>::Value::top())),

                    // Leaves are already in NNF.
                    Err(this) => this,
                },
            },
        }
    }
}
