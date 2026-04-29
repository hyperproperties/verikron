use std::cmp::Ordering;

use crate::lattices::lattice::{
    BooleanLattice, Bottom, BoundedLattice, ComplementedLattice, DistributiveLattice,
    JoinSemiLattice, MeetSemiLattice, MembershipLattice, Semilattice, Top,
};

/// Fixed-size bit-array lattice.
///
/// The universe is `{0, ..., len - 1}` where `len` is chosen when the value is
/// created. The length never changes afterwards.
///
/// - Order: `a ⊑ b` iff every bit set in `a` is also set in `b`
/// - Bottom: all bits unset
/// - Top: all bits set
/// - Join: bitwise OR
/// - Meet: bitwise AND
/// - Complement: bitwise NOT inside the fixed universe
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BitArray {
    bits: Box<[u32]>,
    len: usize,
}

impl BitArray {
    const WORD_BITS: usize = u32::BITS as usize;

    /// Creates a bit array with all bits unset.
    #[must_use]
    #[inline]
    pub fn new(len: usize) -> Self {
        Self::zeros(len)
    }

    /// Creates a bit array with all bits unset.
    #[must_use]
    #[inline]
    pub fn zeros(len: usize) -> Self {
        Self {
            bits: vec![0; Self::word_count(len)].into_boxed_slice(),
            len,
        }
    }

    /// Creates a bit array with all valid bits set.
    #[must_use]
    #[inline]
    pub fn ones(len: usize) -> Self {
        let mut array = Self {
            bits: vec![u32::MAX; Self::word_count(len)].into_boxed_slice(),
            len,
        };

        array.clear_unused_bits();
        array
    }

    /// Returns the number of valid bits in the array.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true iff the array has no valid bits.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true iff no bit is set.
    #[must_use]
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.bits.iter().all(|word| *word == 0)
    }

    /// Returns the bit at `index`, or `None` if `index` is outside the universe.
    #[must_use]
    #[inline]
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            return None;
        }

        let word = index / Self::WORD_BITS;
        let bit = index % Self::WORD_BITS;

        Some((self.bits[word] & (1 << bit)) != 0)
    }

    /// Sets the bit at `index`.
    ///
    /// Returns true iff the bit changed.
    ///
    /// Panics if `index` is outside the fixed universe.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) -> bool {
        assert!(index < self.len, "bit index out of bounds");

        let word = index / Self::WORD_BITS;
        let bit = index % Self::WORD_BITS;
        let mask = 1 << bit;

        let old = (self.bits[word] & mask) != 0;

        if value {
            self.bits[word] |= mask;
        } else {
            self.bits[word] &= !mask;
        }

        old != value
    }

    /// Returns the number of set bits.
    #[must_use]
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.bits
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Returns the number of unset bits.
    #[must_use]
    #[inline]
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Returns `self \ other`.
    ///
    /// Panics if the two arrays have different universe sizes.
    #[must_use]
    #[inline]
    pub fn difference(&self, other: &Self) -> Self {
        self.assert_same_len(other);

        let bits = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(left, right)| *left & !*right)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut array = Self {
            bits,
            len: self.len,
        };

        array.clear_unused_bits();
        array
    }

    #[must_use]
    #[inline]
    fn word_count(len: usize) -> usize {
        len.div_ceil(Self::WORD_BITS)
    }

    #[inline]
    fn assert_same_len(&self, other: &Self) {
        assert_eq!(
            self.len, other.len,
            "bit arrays must have the same universe size"
        );
    }

    #[inline]
    fn clear_unused_bits(&mut self) {
        if self.len == 0 {
            return;
        }

        let used_bits = self.len % Self::WORD_BITS;

        if used_bits == 0 {
            return;
        }

        let mask = (1u32 << used_bits) - 1;
        let last = self.bits.len() - 1;

        self.bits[last] &= mask;
    }
}

impl PartialOrd for BitArray {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.len != other.len {
            return None;
        }

        let le = self.le(other);
        let ge = self.ge(other);

        match (le, ge) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        self.bits
            .iter()
            .zip(other.bits.iter())
            .all(|(left, right)| (*left & !*right) == 0)
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        other.le(self)
    }
}

impl Semilattice for BitArray {}

impl JoinSemiLattice for BitArray {
    #[inline]
    fn join(&self, other: &Self) -> Self {
        self.assert_same_len(other);

        let bits = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(left, right)| *left | *right)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            bits,
            len: self.len,
        }
    }
}

impl MeetSemiLattice for BitArray {
    #[inline]
    fn meet(&self, other: &Self) -> Self {
        self.assert_same_len(other);

        let bits = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(left, right)| *left & *right)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            bits,
            len: self.len,
        }
    }
}

impl Bottom for BitArray {
    type Context = usize;

    #[inline]
    fn bottom_with(context: &Self::Context) -> Self {
        Self::zeros(*context)
    }
}

impl Top for BitArray {
    type Context = usize;

    #[inline]
    fn top_with(context: &Self::Context) -> Self {
        Self::ones(*context)
    }
}

impl BoundedLattice for BitArray {}

impl ComplementedLattice for BitArray {
    #[inline]
    fn complement(self) -> Self {
        let bits = self
            .bits
            .iter()
            .map(|word| !*word)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut array = Self {
            bits,
            len: self.len,
        };

        array.clear_unused_bits();
        array
    }
}

impl DistributiveLattice for BitArray {}

impl BooleanLattice for BitArray {}

impl MembershipLattice<usize> for BitArray {
    #[inline]
    fn insert(&mut self, value: usize) -> bool {
        self.set(value, true)
    }

    #[inline]
    fn contains(&self, value: &usize) -> bool {
        self.get(*value).unwrap_or(false)
    }
}