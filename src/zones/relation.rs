use core::{
    fmt,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use crate::zones::{
    bound::{Bound, BoundError, MAX_BOUND, MIN_BOUND, ZERO_BOUND},
    strictness::Strictness,
};

/// Errors returned by checked [`Relation`] operations.
///
/// These errors are produced by the `checked_*` methods on [`Relation`].  The checked
/// operations enforce additional invariants that are useful in DBM / zone algorithms,
/// such as rejecting a computation where two finite relations would collapse to the
/// infinity sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationError {
    /// An operation required a finite bound but encountered the infinity sentinel.
    ///
    /// This is returned when an operation conceptually needs an integer constant `c`,
    /// but the relation is [`INFINITY`], which represents the absence of a finite bound.
    InfinityNotFinite,

    BoundError {
        value: BoundError,
    },

    /// A finite-only operation would produce the infinity sentinel.
    ///
    /// With this encoding, the pair `(MAX_BOUND, ≤)` is used as the canonical sentinel
    /// [`INFINITY`]. Some computations on finite relations could otherwise yield that
    /// pair (e.g. a weak + weak sum reaching `MAX_BOUND`). The checked operations treat
    /// such cases like overflow and return this error instead of producing [`INFINITY`].
    FiniteBecameInfinity,
}

/// Packed DBM bound `(c, s)` representing `x_i - x_j s c` where `s` is `<` or `≤`.
///
/// In a difference bound matrix (DBM), each cell stores a constraint of the form:
///
/// - `x_i - x_j <  c`  (strict)
/// - `x_i - x_j ≤  c`  (weak / non-strict)
///
/// This type stores the pair `(c, s)` compactly in a single `i16`, and is intended
/// to be used as the value type of DBM cells.
///
/// # Encoding
///
/// `raw = (c << 1) | weak_bit`, where:
/// - `weak_bit = 0` for `<`
/// - `weak_bit = 1` for `≤`
///
/// # Ordering
///
/// The derived ordering follows the standard DBM convention "tighter is smaller":
/// for the same constant `c`, `< c` is tighter (and therefore smaller) than `≤ c`.
///
/// # Infinity
///
/// `(MAX_BOUND, ≤)` is reserved as the canonical "no constraint" value `∞` and is
/// represented by the constant [`INFINITY`]. This is a space optimization: it uses
/// the topmost encodable bit pattern as a sentinel.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Relation(i16);

/// Canonical "no constraint" / infinity sentinel `(∞, ≤)`.
///
/// Semantically, this is the weakest constraint: it does not restrict `x_i - x_j`.
pub const INFINITY: Relation = Relation::weak(MAX_BOUND);

/// The relation `(0, ≤)`.
///
/// In DBM path composition, this acts as an additive identity for finite bounds:
/// `(0, ≤) + r = r` and `r + (0, ≤) = r` (when the result is defined).
pub const POSITIVE: Relation = Relation::weak(ZERO_BOUND);

/// The relation `(0, <)`.
///
/// This "strict zero" is sometimes used in DBM manipulations to preserve strictness
/// information at the boundary.
pub const NEGATIVE: Relation = Relation::strict(ZERO_BOUND);

impl Relation {
    /// Converts a strictness into the packed `weak_bit` representation.
    ///
    /// - `<`  → `0`
    /// - `≤`  → `1`
    #[inline]
    const fn weak_bit(s: Strictness) -> i16 {
        match s {
            Strictness::Strict => 0,
            Strictness::Weak => 1,
        }
    }

    /// Constructs a packed relation without validation.
    ///
    /// This is a low-level constructor. It assumes `bound` is within the representable
    /// range and does not treat `(MAX_BOUND, ≤)` specially.
    ///
    /// Prefer [`Relation::try_from_parts`] when building relations from arithmetic.
    #[inline]
    pub const fn new(bound: Bound, strictness: Strictness) -> Self {
        Self(bound.pack() | Self::weak_bit(strictness))
    }

    /// Constructs `(bound, ≤)` without validation.
    #[inline]
    pub const fn weak(bound: Bound) -> Self {
        Self::new(bound, Strictness::Weak)
    }

    /// Constructs `(bound, <)` without validation.
    #[inline]
    pub const fn strict(bound: Bound) -> Self {
        Self::new(bound, Strictness::Strict)
    }

    /// Returns `true` if this relation is the canonical infinity sentinel.
    #[inline]
    pub const fn is_infinity(self) -> bool {
        self.0 == INFINITY.0
    }

    /// Returns the strictness component (`<` or `≤`).
    #[inline]
    pub const fn strictness(self) -> Strictness {
        if (self.0 & 1) == 0 {
            Strictness::Strict
        } else {
            Strictness::Weak
        }
    }

    /// Returns `true` if the relation is strict (`<`).
    #[inline]
    pub const fn is_strict(self) -> bool {
        (self.0 & 1) == 0
    }

    /// Returns `true` if the relation is weak (`≤`).
    #[inline]
    pub const fn is_weak(self) -> bool {
        !self.is_strict()
    }

    /// Returns the finite bound constant if this relation is not [`INFINITY`].
    ///
    /// This is the preferred accessor when infinity is a valid value.
    #[inline]
    pub const fn finite_bound(self) -> Option<Bound> {
        if self.is_infinity() {
            None
        } else {
            Some(Bound::unpack(self.0))
        }
    }

    /// Returns the finite bound constant, or an error if this relation is [`INFINITY`].
    #[inline]
    pub fn try_bound(self) -> Result<Bound, RelationError> {
        self.finite_bound().ok_or(RelationError::InfinityNotFinite)
    }

    /// Returns the finite bound constant.
    ///
    /// # Panics
    /// Panics if `self` is [`INFINITY`].
    ///
    /// Prefer [`Relation::finite_bound`] or [`Relation::try_bound`] when infinity
    /// is a valid input.
    #[inline]
    pub const fn bound(self) -> Bound {
        match self.finite_bound() {
            Some(b) => b,
            None => panic!("no finite bound for infinity"),
        }
    }

    /// Checked DBM path composition.
    ///
    /// This is the operation used during DBM closure (e.g. Floyd–Warshall):
    ///
    /// `(c1, s1) + (c2, s2) = (c1 + c2, strict if either strict)`.
    ///
    /// # Semantics
    /// - `∞ + x = ∞`
    ///
    /// # Errors
    /// - [`RelationError::BoundOutOfRange`] on overflow/underflow.
    /// - [`RelationError::FiniteBecameInfinity`] if both operands are finite but the
    ///   result would be the infinity sentinel.
    #[inline]
    pub fn checked_add(self, rhs: Self) -> Result<Self, RelationError> {
        if self.is_infinity() || rhs.is_infinity() {
            return Ok(INFINITY);
        }

        let bound = self.bound().checked_add(rhs.bound())?;
        let strictness = if self.is_weak() && rhs.is_weak() {
            Strictness::Weak
        } else {
            Strictness::Strict
        };
        let relation = Relation::new(bound, strictness);

        // Key rule: if both operands were finite, the result must not be infinity.
        if relation.is_infinity() {
            return Err(RelationError::FiniteBecameInfinity);
        }

        Ok(relation)
    }

    /// Checked negation for use in zone/federation operations: `(-c, opposite(s))`.
    ///
    /// This is not DBM closure composition; it is a utility used when transforming
    /// constraints during zone difference / federation construction.
    ///
    /// # Errors
    /// - [`RelationError::InfinityNotFinite`] if `self` is [`INFINITY`].
    /// - [`RelationError::BoundOutOfRange`] on overflow/underflow.
    /// - [`RelationError::FiniteBecameInfinity`] if a finite input would map to the
    ///   infinity sentinel.
    #[inline]
    pub fn checked_neg(self) -> Result<Self, RelationError> {
        if self.is_infinity() {
            return Err(RelationError::InfinityNotFinite);
        }

        let bound = self.bound().checked_neg()?;
        let strictness = self.strictness().opposite();
        let relation = Relation::new(bound, strictness);

        // Finite negation must not become infinity.
        if relation.is_infinity() {
            return Err(RelationError::FiniteBecameInfinity);
        }

        Ok(relation)
    }

    /// Checked subtraction defined as `self + (-rhs)`.
    ///
    /// # Errors
    /// Propagates errors from [`Relation::checked_neg`] and [`Relation::checked_add`].
    #[inline]
    pub fn checked_sub(self, rhs: Self) -> Result<Self, RelationError> {
        self.checked_add(rhs.checked_neg()?)
    }
}

impl fmt::Display for Relation {
    /// Formats the relation as `(c, <)` / `(c, ≤)` or `(∞, ≤)` for infinity.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(f, "(∞, {})", self.strictness())
        } else {
            write!(f, "({}, {})", self.bound(), self.strictness())
        }
    }
}

/// Operator `+` delegates to [`Relation::checked_add`] and panics on error.
///
/// Use the checked methods if you want to handle errors explicitly.
impl Add for Relation {
    type Output = Relation;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs)
            .unwrap_or_else(|e| panic!("Relation::add failed: {e:?} (lhs={self}, rhs={rhs})"))
    }
}

/// Operator `+=` delegates to [`Add`].
impl AddAssign for Relation {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Unary `-` delegates to [`Relation::checked_neg`] and panics on error.
///
/// Use [`Relation::checked_neg`] if you want to handle errors explicitly.
impl Neg for Relation {
    type Output = Relation;

    #[inline]
    fn neg(self) -> Self::Output {
        self.checked_neg()
            .unwrap_or_else(|e| panic!("Relation::neg failed: {e:?} (x={self})"))
    }
}

/// Operator `-` delegates to [`Relation::checked_sub`] and panics on error.
///
/// Use [`Relation::checked_sub`] if you want to handle errors explicitly.
impl Sub for Relation {
    type Output = Relation;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs)
            .unwrap_or_else(|e| panic!("Relation::sub failed: {e:?} (lhs={self}, rhs={rhs})"))
    }
}

/// Operator `-=` delegates to [`Sub`].
impl SubAssign for Relation {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Enables `rng.random::<Relation>()`.
///
/// Note: this may generate [`INFINITY`] when sampling `(MAX_BOUND, ≤)`.
impl Distribution<Relation> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Relation {
        Relation::new(rng.random_range(MIN_BOUND..=MAX_BOUND), rng.random())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn infinity_is_reserved_pair() {
        assert!(INFINITY.is_infinity());
        assert_eq!(INFINITY.finite_bound(), None);
        assert_eq!(INFINITY.strictness(), Strictness::Weak);

        // (MAX_BOUND, <) is not infinity.
        assert!(!Relation::strict(MAX_BOUND).is_infinity());
    }

    #[test]
    fn zero_constants() {
        assert_eq!(POSITIVE, Relation::weak(ZERO_BOUND));
        assert_eq!(NEGATIVE, Relation::strict(ZERO_BOUND));
        assert!(!POSITIVE.is_infinity());
        assert!(!NEGATIVE.is_infinity());
    }

    #[test]
    fn finite_bound_and_try_bound() {
        let r = Relation::strict(Bound::try_new(7).unwrap());
        assert_eq!(r.finite_bound().unwrap().get(), 7);
        assert_eq!(r.try_bound().unwrap().get(), 7);

        assert_eq!(INFINITY.finite_bound(), None);
        assert_eq!(INFINITY.try_bound(), Err(RelationError::InfinityNotFinite));
    }

    #[test]
    #[should_panic(expected = "no finite bound for infinity")]
    fn bound_panics_on_infinity() {
        let _ = INFINITY.bound();
    }

    #[test]
    fn checked_add_infinity_absorbs() {
        let a = Relation::strict(Bound::try_new(5).unwrap());
        let b = Relation::weak(Bound::try_new(-7).unwrap());

        assert_eq!(a.checked_add(INFINITY).unwrap(), INFINITY);
        assert_eq!(INFINITY.checked_add(b).unwrap(), INFINITY);
        assert_eq!(INFINITY.checked_add(INFINITY).unwrap(), INFINITY);
    }

    #[test]
    fn checked_add_strictness_rule() {
        let a = Relation::weak(Bound::try_new(10).unwrap());
        let b = Relation::weak(Bound::try_new(2).unwrap());
        let c = Relation::strict(Bound::try_new(2).unwrap());

        let ab = a.checked_add(b).unwrap();
        assert_eq!(ab.bound().get(), 12);
        assert_eq!(ab.strictness(), Strictness::Weak);

        let ac = a.checked_add(c).unwrap();
        assert_eq!(ac.bound().get(), 12);
        assert_eq!(ac.strictness(), Strictness::Strict);

        let ca = c.checked_add(a).unwrap();
        assert_eq!(ca.bound().get(), 12);
        assert_eq!(ca.strictness(), Strictness::Strict);
    }

    #[test]
    fn checked_add_rejects_reserved_sentinel_from_finite_plus_finite() {
        // sum == MAX_BOUND and both weak => would encode infinity sentinel
        let a = Relation::weak(Bound::new(10_000));
        let b = Relation::weak(Bound::new(MAX_BOUND.get() - 10_000));

        assert_eq!(a.checked_add(b), Err(RelationError::FiniteBecameInfinity));
    }

    #[test]
    fn checked_neg_errors_on_infinity() {
        assert_eq!(
            INFINITY.checked_neg(),
            Err(RelationError::InfinityNotFinite)
        );
    }

    #[test]
    fn checked_neg_zero_is_zero() {
        assert_eq!(POSITIVE.checked_neg().unwrap(), NEGATIVE);
        // strict zero flips to weak zero
        assert_eq!(NEGATIVE.checked_neg().unwrap(), POSITIVE);
    }

    #[test]
    fn checked_neg_rejects_mapping_to_reserved_sentinel() {
        // To land on (MAX_BOUND, ≤), we need:
        // output bound = MAX_BOUND => input bound = -MAX_BOUND
        // output strictness = Weak => input strictness = Strict (since opposite(Strict)=Weak)
        let relation = Relation::strict(MAX_BOUND.checked_neg().unwrap());
        assert_eq!(
            relation.checked_neg(),
            Err(RelationError::FiniteBecameInfinity)
        );
    }

    #[test]
    fn checked_sub_matches_add_of_negation_when_defined() {
        let a = Relation::strict(Bound::new(9));
        let b = Relation::weak(Bound::new(4));

        let lhs = a.checked_sub(b).unwrap();
        let rhs = a.checked_add(b.checked_neg().unwrap()).unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn display_formats() {
        assert_eq!(Relation::strict(Bound::new(7)).to_string(), "(7, <)");
        assert_eq!(Relation::weak(Bound::new(-3)).to_string(), "(-3, ≤)");
        assert_eq!(INFINITY.to_string(), "(∞, ≤)");
    }

    #[test]
    #[should_panic]
    fn add_operator_panics_on_reserved_sentinel_case() {
        let _ = Relation::weak(Bound::new(123)) + Relation::weak(Bound::new(MAX_BOUND.get() - 123));
    }

    #[test]
    #[should_panic]
    fn neg_operator_panics_on_infinity() {
        let _ = -INFINITY;
    }

    #[test]
    fn prop_checked_add_never_returns_infinity_from_two_finite_operands() {
        let mut rng = StdRng::seed_from_u64(0xA11D_ADD);

        for _ in 0..200_000 {
            // sample finite a
            let a = loop {
                let b: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                let s: Strictness = rng.random();
                if !(b == MAX_BOUND && s == Strictness::Weak) {
                    break Relation::new(b, s);
                }
            };

            // sample finite b
            let b = loop {
                let bb: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                let ss: Strictness = rng.random();
                if !(bb == MAX_BOUND && ss == Strictness::Weak) {
                    break Relation::new(bb, ss);
                }
            };

            if let Ok(r) = a.checked_add(b) {
                assert!(
                    !r.is_infinity(),
                    "checked_add returned INFINITY from finite inputs: a={a}, b={b}"
                );
            }
        }
    }

    #[test]
    fn prop_checked_add_commutative() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..200_000 {
            let a = if rng.random_ratio(1, 20) {
                INFINITY
            } else {
                loop {
                    let b: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                    let s: Strictness = rng.random();
                    if !(b == MAX_BOUND && s == Strictness::Weak) {
                        break Relation::new(b, s);
                    }
                }
            };

            let b = if rng.random_ratio(1, 20) {
                INFINITY
            } else {
                loop {
                    let bb: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                    let ss: Strictness = rng.random();
                    if !(bb == MAX_BOUND && ss == Strictness::Weak) {
                        break Relation::new(bb, ss);
                    }
                }
            };

            assert_eq!(a.checked_add(b), b.checked_add(a), "a={a}, b={b}");
        }
    }

    #[test]
    fn prop_checked_add_associative_when_all_three_succeed() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..300_000 {
            let a = if rng.random_ratio(1, 30) {
                INFINITY
            } else {
                loop {
                    let b: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                    let s: Strictness = rng.random();
                    if !(b == MAX_BOUND && s == Strictness::Weak) {
                        break Relation::new(b, s);
                    }
                }
            };

            let b = if rng.random_ratio(1, 30) {
                INFINITY
            } else {
                loop {
                    let bb: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                    let ss: Strictness = rng.random();
                    if !(bb == MAX_BOUND && ss == Strictness::Weak) {
                        break Relation::new(bb, ss);
                    }
                }
            };

            let c = if rng.random_ratio(1, 30) {
                INFINITY
            } else {
                loop {
                    let bc: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                    let sc: Strictness = rng.random();
                    if !(bc == MAX_BOUND && sc == Strictness::Weak) {
                        break Relation::new(bc, sc);
                    }
                }
            };

            let ab = match a.checked_add(b) {
                Ok(x) => x,
                Err(_) => continue,
            };
            let left = match ab.checked_add(c) {
                Ok(x) => x,
                Err(_) => continue,
            };

            let bc = match b.checked_add(c) {
                Ok(x) => x,
                Err(_) => continue,
            };
            let right = match a.checked_add(bc) {
                Ok(x) => x,
                Err(_) => continue,
            };

            assert_eq!(left, right, "a={a}, b={b}, c={c}");
        }
    }

    #[test]
    fn prop_checked_neg_is_involution_when_defined() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..200_000 {
            // finite only (exclude reserved)
            let a = loop {
                let b: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                let s: Strictness = rng.random();
                if !(b == MAX_BOUND && s == Strictness::Weak) {
                    break Relation::new(b, s);
                }
            };

            let na = match a.checked_neg() {
                Ok(x) => x,
                Err(_) => continue,
            };
            let nna = match na.checked_neg() {
                Ok(x) => x,
                Err(_) => continue,
            };

            assert_eq!(nna, a, "a={a}, na={na}");
        }
    }

    #[test]
    fn prop_checked_sub_matches_add_of_neg_when_defined() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..200_000 {
            // sample a (sometimes infinity)
            let a = if rng.random_ratio(1, 20) {
                INFINITY
            } else {
                loop {
                    let b: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                    let s: Strictness = rng.random();
                    if !(b == MAX_BOUND && s == Strictness::Weak) {
                        break Relation::new(b, s);
                    }
                }
            };

            // sample b (sometimes infinity)
            let b = if rng.random_ratio(1, 20) {
                INFINITY
            } else {
                loop {
                    let bb: Bound = rng.random_range(MIN_BOUND..=MAX_BOUND);
                    let ss: Strictness = rng.random();
                    if !(bb == MAX_BOUND && ss == Strictness::Weak) {
                        break Relation::new(bb, ss);
                    }
                }
            };

            let sub = a.checked_sub(b);
            let alt = match b.checked_neg() {
                Ok(nb) => a.checked_add(nb),
                Err(e) => Err(e),
            };

            assert_eq!(sub, alt, "a={a}, b={b}");
        }
    }

    #[test]
    fn random_sampling_produces_well_formed_relations() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..100_000 {
            let r: Relation = rng.random();

            match r.finite_bound() {
                None => assert!(r.is_infinity()),
                Some(b) => {
                    assert!(!r.is_infinity());
                    assert!(b >= MIN_BOUND && b <= MAX_BOUND);

                    // If bound hits MAX_BOUND, it must be strict (otherwise it would be infinity).
                    if b == MAX_BOUND {
                        assert_eq!(r.strictness(), Strictness::Strict);
                    }
                }
            }
        }
    }
}
