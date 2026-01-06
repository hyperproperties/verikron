use core::fmt;

use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

/// Strictness of a clock constraint bound in a zone / DBM.
///
/// In a DBM cell a bound is typically represented as a pair `(c, s)` where `c` is
/// an integer constant and `s` encodes whether the constraint is strict (`< c`)
/// or non-strict (`≤ c`).
///
/// This enum models that strictness bit.
///
/// # Ordering
///
/// The derived ordering follows the usual DBM convention "tighter is smaller":
///
/// - `Strict` (`<`) is *smaller* than `Weak` (`≤`).
///
/// This is useful when minimizing bounds: for the same constant `c`,
/// the strict constraint is tighter and should win.
///
/// # Representation
///
/// The enum uses `#[repr(u8)]` so it has a stable numeric encoding:
///
/// - `Strict` maps to `0`
/// - `Weak` maps to `1`
///
/// This encoding is convenient for compact DBM entries.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strictness {
    /// Strict inequality `<`.
    Strict = 0,
    /// Non-strict inequality `≤`.
    Weak = 1,
}

impl Strictness {
    /// Returns the opposite strictness (`<` ↔ `≤`).
    ///
    /// # Examples
    ///
    /// ```
    /// use verikron::zones::strictness::Strictness;
    ///
    /// assert_eq!(Strictness::Strict.opposite(), Strictness::Weak);
    /// assert_eq!(Strictness::Weak.opposite(), Strictness::Strict);
    /// ```
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Strictness::Strict => Strictness::Weak,
            Strictness::Weak => Strictness::Strict,
        }
    }

    /// Converts this strictness to its stable `u8` representation.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Converts a `u8` back into a `Strictness`.
    ///
    /// # Panics
    ///
    /// Panics if `v` is not `0` or `1`.
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Strictness::Strict,
            1 => Strictness::Weak,
            _ => panic!("Strictness must be 0 or 1"),
        }
    }
}

/// Enables `rng.random::<Strictness>()`.
///
/// This samples `Strict` and `Weak` with equal probability.
impl Distribution<Strictness> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Strictness {
        if rng.random::<bool>() {
            Strictness::Strict
        } else {
            Strictness::Weak
        }
    }
}

/// Formats a strictness as the corresponding comparison operator (`<` or `≤`).
impl fmt::Display for Strictness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Strictness::Strict => "<",
            Strictness::Weak => "≤",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn opposite_is_involution() {
        assert_eq!(Strictness::Strict.opposite(), Strictness::Weak);
        assert_eq!(Strictness::Weak.opposite(), Strictness::Strict);

        for s in [Strictness::Strict, Strictness::Weak] {
            assert_eq!(s.opposite().opposite(), s);
        }
    }

    #[test]
    fn u8_roundtrip_and_values() {
        assert_eq!(Strictness::Strict.to_u8(), 0);
        assert_eq!(Strictness::Weak.to_u8(), 1);

        assert_eq!(Strictness::from_u8(0), Strictness::Strict);
        assert_eq!(Strictness::from_u8(1), Strictness::Weak);

        for s in [Strictness::Strict, Strictness::Weak] {
            assert_eq!(Strictness::from_u8(s.to_u8()), s);
        }
    }

    #[test]
    #[should_panic(expected = "Strictness must be 0 or 1")]
    fn from_u8_panics_on_invalid() {
        let _ = Strictness::from_u8(2);
    }

    #[test]
    fn ordering_matches_dbm_convention() {
        assert!(Strictness::Strict < Strictness::Weak);
    }

    #[test]
    fn display_is_correct() {
        assert_eq!(Strictness::Strict.to_string(), "<");
        assert_eq!(Strictness::Weak.to_string(), "≤");
    }

    #[test]
    fn standard_uniform_sampling_produces_only_valid_values() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        for _ in 0..10_000 {
            let s: Strictness = rng.random();
            assert!(matches!(s, Strictness::Strict | Strictness::Weak));
        }
    }
}
