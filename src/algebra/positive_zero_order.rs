use crate::{
    algebra::{
        algebra::{
            Addition, Additive, Distributive, Embed, Generate, Multiplication, Multiplicative,
        },
        nnf::NNF,
        positive::Positive,
        zero_order::ZeroOrder,
    },
    lattices::lattice::{BoundedLattice, DistributiveLattice},
};

/// Positive formulas over embedded lattice values and generators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositiveZeroOrder<V: BoundedLattice + DistributiveLattice, A> {
    Embedding(V),
    Generator(A),
    Conjunction(Vec<Self>),
    Disjunction(Vec<Self>),
}

impl<V, A> PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    #[must_use]
    pub fn conjunction(xs: Vec<Self>) -> Self {
        match xs.len() {
            0 => Self::Embedding(V::top()),
            1 => xs.into_iter().next().unwrap(),
            _ => Self::Conjunction(xs),
        }
    }

    #[must_use]
    pub fn disjunction(xs: Vec<Self>) -> Self {
        match xs.len() {
            0 => Self::Embedding(V::bottom()),
            1 => xs.into_iter().next().unwrap(),
            _ => Self::Disjunction(xs),
        }
    }
}

impl<V, A> NNF for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
    fn nnf(self) -> Self {
        self
    }
}

impl<V, A> ZeroOrder for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
}

impl<V, A> Positive for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
}

impl<V, A> Embed for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    type Value = V;

    fn embed(value: Self::Value) -> Self {
        Self::Embedding(value)
    }
}

impl<V, A> Generate for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    type Symbol = A;

    fn generate(generator: Self::Symbol) -> Self {
        Self::Generator(generator)
    }
}

/// Disjunction is the additive structure.
impl<V, A> Additive for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    fn sum(self, other: Self) -> Self {
        match (self, other) {
            // Flatten nested disjunctions when combining.
            (Self::Disjunction(mut xs), Self::Disjunction(ys)) => {
                xs.extend(ys);
                Self::Disjunction(xs)
            }
            (Self::Disjunction(mut xs), y) => {
                xs.push(y);
                Self::Disjunction(xs)
            }
            (x, Self::Disjunction(mut ys)) => {
                ys.insert(0, x);
                Self::Disjunction(ys)
            }
            (x, y) => Self::Disjunction(vec![x, y]),
        }
    }

    fn into_additive(self) -> Result<Vec<Self>, Self> {
        match self {
            Self::Disjunction(xs) => Ok(xs),
            other => Err(other),
        }
    }
}

/// Conjunction is the multiplicative structure.
impl<V, A> Multiplicative for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    fn multiply(self, other: Self) -> Self {
        match (self, other) {
            // Flatten nested conjunctions when combining.
            (Self::Conjunction(mut xs), Self::Conjunction(ys)) => {
                xs.extend(ys);
                Self::Conjunction(xs)
            }
            (Self::Conjunction(mut xs), y) => {
                xs.push(y);
                Self::Conjunction(xs)
            }
            (x, Self::Conjunction(mut ys)) => {
                ys.insert(0, x);
                Self::Conjunction(ys)
            }
            (x, y) => Self::Conjunction(vec![x, y]),
        }
    }

    fn into_multiplicative(self) -> Result<Vec<Self>, Self> {
        match self {
            Self::Conjunction(xs) => Ok(xs),
            other => Err(other),
        }
    }
}

/// Conjunction distributes over disjunction.
impl<V, A> Distributive<Multiplication, Addition> for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
    fn distribute(self, other: Self) -> Self {
        match (self, other) {
            // (x₁ ∨ ... ∨ xₙ) ∧ y = (x₁ ∧ y) ∨ ... ∨ (xₙ ∧ y)
            (Self::Disjunction(xs), y) => Self::Disjunction(
                xs.into_iter()
                    .map(|x| {
                        <Self as Distributive<Multiplication, Addition>>::distribute(x, y.clone())
                    })
                    .collect(),
            ),
            // x ∧ (y₁ ∨ ... ∨ yₙ) = (x ∧ y₁) ∨ ... ∨ (x ∧ yₙ)
            (x, Self::Disjunction(ys)) => Self::Disjunction(
                ys.into_iter()
                    .map(|y| {
                        <Self as Distributive<Multiplication, Addition>>::distribute(x.clone(), y)
                    })
                    .collect(),
            ),
            (x, y) => x.multiply(y),
        }
    }
}

/// Disjunction distributes over conjunction.
impl<V, A> Distributive<Addition, Multiplication> for PositiveZeroOrder<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
    fn distribute(self, other: Self) -> Self {
        match (self, other) {
            // (x₁ ∧ ... ∧ xₙ) ∨ y = (x₁ ∨ y) ∧ ... ∧ (xₙ ∨ y)
            (Self::Conjunction(xs), y) => Self::Conjunction(
                xs.into_iter()
                    .map(|x| {
                        <Self as Distributive<Addition, Multiplication>>::distribute(x, y.clone())
                    })
                    .collect(),
            ),
            // x ∨ (y₁ ∧ ... ∧ yₙ) = (x ∨ y₁) ∧ ... ∧ (x ∨ yₙ)
            (x, Self::Conjunction(ys)) => Self::Conjunction(
                ys.into_iter()
                    .map(|y| {
                        <Self as Distributive<Addition, Multiplication>>::distribute(x.clone(), y)
                    })
                    .collect(),
            ),
            (x, y) => x.sum(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::algebra::{cnf::CNF, dnf::DNF};

    use super::*;

    type F = PositiveZeroOrder<bool, &'static str>;

    fn atom(name: &'static str) -> F {
        F::generate(name)
    }

    fn top() -> F {
        F::embed(true)
    }

    fn bottom() -> F {
        F::embed(false)
    }

    fn and(xs: Vec<F>) -> F {
        F::conjunction(xs)
    }

    fn or(xs: Vec<F>) -> F {
        F::disjunction(xs)
    }

    #[test]
    fn dnf_of_atom_is_itself() {
        let f = atom("a");
        assert_eq!(f.clone().dnf(), f);
    }

    #[test]
    fn dnf_of_embedded_value_is_itself() {
        let f = top();
        assert_eq!(f.clone().dnf(), f);
    }

    #[test]
    fn dnf_of_disjunction_is_disjunction_of_dnfs() {
        let f = or(vec![atom("a"), atom("b")]);
        assert_eq!(f.clone().dnf(), f);
    }

    #[test]
    fn distributes_conjunction_over_disjunction() {
        let f = and(vec![atom("a"), or(vec![atom("b"), atom("c")])]);

        let expected = or(vec![
            and(vec![atom("a"), atom("b")]),
            and(vec![atom("a"), atom("c")]),
        ]);

        assert_eq!(f.dnf(), expected);
    }

    #[test]
    fn distributes_single_disjunction_in_longer_conjunction() {
        let f = and(vec![atom("a"), atom("b"), or(vec![atom("c"), atom("d")])]);

        let expected = or(vec![
            and(vec![atom("a"), atom("b"), atom("c")]),
            and(vec![atom("a"), atom("b"), atom("d")]),
        ]);

        assert_eq!(f.dnf(), expected);
    }

    #[test]
    fn empty_conjunction_becomes_top() {
        let f = and(vec![]);
        assert_eq!(f.dnf(), top());
    }

    #[test]
    fn distributes_disjunction_over_conjunction() {
        let f = or(vec![atom("a"), and(vec![atom("b"), atom("c")])]);

        let expected = and(vec![
            or(vec![atom("a"), atom("b")]),
            or(vec![atom("a"), atom("c")]),
        ]);

        assert_eq!(f.cnf(), expected);
    }

    #[test]
    fn distributes_single_conjunction_in_longer_disjunction() {
        let f = or(vec![atom("a"), atom("b"), and(vec![atom("c"), atom("d")])]);

        let expected = and(vec![
            or(vec![atom("a"), atom("b"), atom("c")]),
            or(vec![atom("a"), atom("b"), atom("d")]),
        ]);

        assert_eq!(f.cnf(), expected);
    }

    #[test]
    fn empty_disjunction_becomes_bottom() {
        let f = or(vec![]);
        assert_eq!(f.cnf(), bottom());
    }
}
