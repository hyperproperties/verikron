use std::fmt::Display;

use rand::Rng;
use rand::distr::uniform::{self, SampleBorrow, SampleUniform, UniformInt, UniformSampler};

use crate::zones::relation::RelationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundError {
    BoundOutOfRange { value: i32 },
}

impl From<BoundError> for RelationError {
    fn from(value: BoundError) -> Self {
        RelationError::BoundError { value }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bound(i16);

pub const ZERO_BOUND: Bound = Bound(0);

/// Smallest finite bound (`-(1<<14)`).
pub const MIN_BOUND: Bound = Bound(-(1 << 14)); // -16384
/// Largest finite bound (`(1<<14)-1`).
pub const MAX_BOUND: Bound = Bound((1 << 14) - 1); // 16383

impl Bound {
    #[inline(always)]
    pub const fn get(self) -> i16 {
        self.0
    }

    #[inline(always)]
    pub fn new(value: i16) -> Self {
        Self::try_new(value as i32).unwrap()
    }

    #[inline(always)]
    pub const fn try_new(value: i32) -> Result<Self, BoundError> {
        if value < MIN_BOUND.0 as i32 || value > MAX_BOUND.0 as i32 {
            Err(BoundError::BoundOutOfRange { value })
        } else {
            Ok(Bound(value as i16))
        }
    }

    // (You probably want these names swapped, see note below)
    #[inline(always)]
    pub const fn pack(self) -> i16 {
        self.0 << 1
    }

    #[inline(always)]
    pub const fn unpack(packed: i16) -> Self {
        Bound(packed >> 1)
    }

    #[inline(always)]
    pub fn checked_add(self, other: Self) -> Result<Self, BoundError> {
        Self::try_new((self.0 as i32) + (other.0 as i32))
    }

    #[inline(always)]
    pub fn checked_sub(self, other: Self) -> Result<Self, BoundError> {
        Self::try_new((self.0 as i32) - (other.0 as i32))
    }

    #[inline(always)]
    pub fn checked_neg(self) -> Result<Self, BoundError> {
        Self::try_new(-(self.0 as i32))
    }
}

impl Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl SampleUniform for Bound {
    type Sampler = UniformBound;
}

/// Uniform sampler for `Bound`, implemented by sampling an `i16`
/// and re-wrapping it as `Bound`.
#[derive(Clone, Debug)]
pub struct UniformBound(UniformInt<i16>);

impl UniformSampler for UniformBound {
    type X = Bound;

    #[inline]
    fn new<B1, B2>(low: B1, high: B2) -> Result<Self, uniform::Error>
    where
        B1: SampleBorrow<Bound>,
        B2: SampleBorrow<Bound>,
    {
        let low = low.borrow().0;
        let high = high.borrow().0;

        // half-open range [low, high) must be non-empty
        if low >= high {
            return Err(uniform::Error::EmptyRange);
        }

        Ok(Self(UniformInt::new(low, high)?))
    }

    #[inline]
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Result<Self, uniform::Error>
    where
        B1: SampleBorrow<Bound>,
        B2: SampleBorrow<Bound>,
    {
        let low = low.borrow().0;
        let high = high.borrow().0;

        // inclusive range [low, high] must be non-empty
        if low > high {
            return Err(uniform::Error::EmptyRange);
        }

        Ok(Self(UniformInt::new_inclusive(low, high)?))
    }

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Bound {
        Bound(self.0.sample(rng))
    }
}
