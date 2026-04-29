use crate::{
    algebra::algebra::{
        Addition, Additive, Complemented, DeMorgan, Distributive, Embed, Generate, Multiplication,
        Multiplicative,
    },
    lattices::lattice::{Bottom, BoundedLattice, DistributiveLattice, Top},
};

/// Marker trait for zero-order formulas.
pub trait ZeroOrder {}

/// Zero-order formulas with negation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZeroOrderFormula<V: BoundedLattice + DistributiveLattice, A> {
    Embedding(V),
    Generator(A),
    Negation(Box<Self>),
    Conjunction(Vec<Self>),
    Disjunction(Vec<Self>),
}

impl<V, A> ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
    <V as Top>::Context: Default,
    <V as Bottom>::Context: Default,
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

    #[must_use]
    pub fn negation(x: Self) -> Self {
        Self::Negation(Box::new(x))
    }
}

impl<V, A> ZeroOrder for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
}

impl<V, A> Embed for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    type Value = V;

    fn embed(value: Self::Value) -> Self {
        Self::Embedding(value)
    }
}

impl<V, A> Generate for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    type Symbol = A;

    fn generate(symbol: Self::Symbol) -> Self {
        Self::Generator(symbol)
    }
}

impl<V, A> Additive for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    fn sum(self, other: Self) -> Self {
        match (self, other) {
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

impl<V, A> Multiplicative for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    fn multiply(self, other: Self) -> Self {
        match (self, other) {
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

impl<V, A> Complemented for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    fn complement(self) -> Self {
        Self::Negation(Box::new(self))
    }

    fn into_complement(self) -> Result<Self, Self> {
        match self {
            Self::Negation(x) => Ok(*x),
            other => Err(other),
        }
    }
}

/// Conjunction distributes over disjunction.
///
/// Negations are treated as leaves by the additive/multiplicative view.
impl<V, A> Distributive<Multiplication, Addition> for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
    fn distribute(self, other: Self) -> Self {
        match (self, other) {
            (Self::Disjunction(xs), y) => Self::Disjunction(
                xs.into_iter()
                    .map(|x| {
                        <Self as Distributive<Multiplication, Addition>>::distribute(x, y.clone())
                    })
                    .collect(),
            ),
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
///
/// Negations are treated as leaves by the additive/multiplicative view.
impl<V, A> Distributive<Addition, Multiplication> for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice + Clone,
    A: Clone,
{
    fn distribute(self, other: Self) -> Self {
        match (self, other) {
            (Self::Conjunction(xs), y) => Self::Conjunction(
                xs.into_iter()
                    .map(|x| {
                        <Self as Distributive<Addition, Multiplication>>::distribute(x, y.clone())
                    })
                    .collect(),
            ),
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

/// De Morgan for negated disjunctions:
///
/// `¬(x₁ ∨ ... ∨ xₙ) = ¬x₁ ∧ ... ∧ ¬xₙ`
impl<V, A> DeMorgan<Addition, Multiplication> for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    fn demorgan(self) -> Self {
        match self {
            Self::Negation(inner) => match *inner {
                Self::Disjunction(xs) => {
                    Self::Conjunction(xs.into_iter().map(Self::complement).collect())
                }
                other => Self::Negation(Box::new(other)),
            },
            other => other,
        }
    }
}

/// De Morgan for negated conjunctions:
///
/// `¬(x₁ ∧ ... ∧ xₙ) = ¬x₁ ∨ ... ∨ ¬xₙ`
impl<V, A> DeMorgan<Multiplication, Addition> for ZeroOrderFormula<V, A>
where
    V: BoundedLattice + DistributiveLattice,
{
    fn demorgan(self) -> Self {
        match self {
            Self::Negation(inner) => match *inner {
                Self::Conjunction(xs) => {
                    Self::Disjunction(xs.into_iter().map(Self::complement).collect())
                }
                other => Self::Negation(Box::new(other)),
            },
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::nnf::NNF;

    type F = ZeroOrderFormula<bool, &'static str>;

    fn atom(name: &'static str) -> F {
        F::generate(name)
    }

    fn top() -> F {
        F::embed(true)
    }

    fn bottom() -> F {
        F::embed(false)
    }

    fn not(x: F) -> F {
        F::negation(x)
    }

    fn and(xs: Vec<F>) -> F {
        F::conjunction(xs)
    }

    fn or(xs: Vec<F>) -> F {
        F::disjunction(xs)
    }

    #[test]
    fn embed_constructs_embedding() {
        assert_eq!(F::embed(true), F::Embedding(true));
        assert_eq!(F::embed(false), F::Embedding(false));
    }

    #[test]
    fn generate_constructs_generator() {
        assert_eq!(F::generate("a"), F::Generator("a"));
    }

    #[test]
    fn negation_constructor_wraps_formula() {
        let f = atom("a");
        assert_eq!(F::negation(f.clone()), F::Negation(Box::new(f)));
    }

    #[test]
    fn complement_wraps_formula_in_negation() {
        let f = atom("a");
        assert_eq!(f.clone().complement(), not(f));
    }

    #[test]
    fn into_complement_extracts_inner_formula() {
        let f = atom("a");
        assert_eq!(not(f.clone()).into_complement(), Ok(f));
    }

    #[test]
    fn into_complement_rejects_non_negated_formula() {
        let f = atom("a");
        assert_eq!(f.clone().into_complement(), Err(f));
    }

    #[test]
    fn sum_flattens_disjunctions() {
        let left = or(vec![atom("a"), atom("b")]);
        let right = or(vec![atom("c"), atom("d")]);

        let expected = or(vec![atom("a"), atom("b"), atom("c"), atom("d")]);

        assert_eq!(left.sum(right), expected);
    }

    #[test]
    fn multiply_flattens_conjunctions() {
        let left = and(vec![atom("a"), atom("b")]);
        let right = and(vec![atom("c"), atom("d")]);

        let expected = and(vec![atom("a"), atom("b"), atom("c"), atom("d")]);

        assert_eq!(left.multiply(right), expected);
    }

    #[test]
    fn into_additive_extracts_disjunction_children() {
        let f = or(vec![atom("a"), atom("b")]);
        assert_eq!(f.into_additive(), Ok(vec![atom("a"), atom("b")]));
    }

    #[test]
    fn into_additive_rejects_non_disjunction() {
        let f = atom("a");
        assert_eq!(f.clone().into_additive(), Err(f));
    }

    #[test]
    fn into_multiplicative_extracts_conjunction_children() {
        let f = and(vec![atom("a"), atom("b")]);
        assert_eq!(f.into_multiplicative(), Ok(vec![atom("a"), atom("b")]));
    }

    #[test]
    fn into_multiplicative_rejects_non_conjunction() {
        let f = atom("a");
        assert_eq!(f.clone().into_multiplicative(), Err(f));
    }

    #[test]
    fn multiplication_distributes_over_addition() {
        let f = and(vec![atom("a"), or(vec![atom("b"), atom("c")])]);

        let expected = or(vec![
            and(vec![atom("a"), atom("b")]),
            and(vec![atom("a"), atom("c")]),
        ]);

        let actual = <F as Distributive<Multiplication, Addition>>::distribute(
            atom("a"),
            or(vec![atom("b"), atom("c")]),
        );

        assert_eq!(actual, expected);
        assert_eq!(
            <F as Distributive<Multiplication, Addition>>::distribute(
                or(vec![atom("b"), atom("c")]),
                atom("a")
            ),
            or(vec![
                and(vec![atom("b"), atom("a")]),
                and(vec![atom("c"), atom("a")]),
            ])
        );

        assert_eq!(f, and(vec![atom("a"), or(vec![atom("b"), atom("c")])]));
    }

    #[test]
    fn addition_distributes_over_multiplication() {
        let expected = and(vec![
            or(vec![atom("a"), atom("b")]),
            or(vec![atom("a"), atom("c")]),
        ]);

        let actual = <F as Distributive<Addition, Multiplication>>::distribute(
            atom("a"),
            and(vec![atom("b"), atom("c")]),
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn negations_are_treated_as_leaves_for_distribution() {
        let expected = or(vec![
            and(vec![not(atom("a")), atom("b")]),
            and(vec![not(atom("a")), atom("c")]),
        ]);

        let actual = <F as Distributive<Multiplication, Addition>>::distribute(
            not(atom("a")),
            or(vec![atom("b"), atom("c")]),
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn top_and_bottom_embed_correctly() {
        assert_eq!(top(), F::Embedding(true));
        assert_eq!(bottom(), F::Embedding(false));
    }

    #[test]
    fn nnf_of_atom_is_itself() {
        let f = atom("a");
        assert_eq!(f.clone().nnf(), f);
    }

    #[test]
    fn nnf_of_embedding_is_itself() {
        let f = top();
        assert_eq!(f.clone().nnf(), f);
    }

    #[test]
    fn nnf_of_positive_formula_is_itself() {
        let f = and(vec![atom("a"), or(vec![atom("b"), atom("c")])]);
        assert_eq!(f.clone().nnf(), f);
    }

    #[test]
    fn nnf_keeps_negated_leaf_as_is() {
        let f = not(atom("a"));
        assert_eq!(f.clone().nnf(), f);
    }

    #[test]
    fn nnf_eliminates_double_negation() {
        let f = not(not(atom("a")));
        assert_eq!(f.nnf(), atom("a"));
    }

    #[test]
    fn nnf_pushes_negation_through_disjunction() {
        let f = not(or(vec![atom("a"), atom("b")]));

        let expected = and(vec![not(atom("a")), not(atom("b"))]);

        assert_eq!(f.nnf(), expected);
    }

    #[test]
    fn nnf_pushes_negation_through_conjunction() {
        let f = not(and(vec![atom("a"), atom("b")]));

        let expected = or(vec![not(atom("a")), not(atom("b"))]);

        assert_eq!(f.nnf(), expected);
    }

    #[test]
    fn nnf_recurses_under_positive_connectives() {
        let f = and(vec![
            not(not(atom("a"))),
            or(vec![not(not(atom("b"))), atom("c")]),
        ]);

        let expected = and(vec![atom("a"), or(vec![atom("b"), atom("c")])]);

        assert_eq!(f.nnf(), expected);
    }

    #[test]
    fn nnf_of_negated_nested_formula() {
        let f = not(or(vec![atom("a"), and(vec![atom("b"), atom("c")])]));

        let expected = and(vec![
            not(atom("a")),
            or(vec![not(atom("b")), not(atom("c"))]),
        ]);

        assert_eq!(f.nnf(), expected);
    }

    #[test]
    fn nnf_of_empty_disjunction_is_bottom() {
        let f = or(vec![]);
        assert_eq!(f.nnf(), bottom());
    }

    #[test]
    fn nnf_of_empty_conjunction_is_top() {
        let f = and(vec![]);
        assert_eq!(f.nnf(), top());
    }
}
