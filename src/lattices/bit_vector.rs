use bit_vec::BitVec;
use std::cmp::Ordering;

use crate::lattices::{
    lattice::{JoinSemiLattice, MeetSemiLattice},
    membership_lattice::MembershipLattice,
};

/// Lattice over finite bit-vectors (bit-sets).
///
/// - Universe: `{0, ..., len-1}`
/// - Order: `a ⊑ b` iff for all `i`, `a[i] => b[i]`
/// - Join: bitwise OR
/// - Meet: bitwise AND
///
/// In practice this is useful as a very compact lattice for visited sets,
/// reachability, dataflow masks, etc.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BitVector(BitVec);

impl BitVector {
    pub fn new(len: usize) -> Self {
        Self(BitVec::from_elem(len, false))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<bool> {
        self.0.get(idx)
    }

    #[inline]
    pub fn set(&mut self, idx: usize, value: bool) {
        let len = self.0.len();
        if idx >= len {
            let grow_by = idx + 1 - len;
            self.0.grow(grow_by, false);
        }
        self.0.set(idx, value);
    }
}

impl From<BitVec> for BitVector {
    fn from(bits: BitVec) -> Self {
        Self(bits)
    }
}

impl From<BitVector> for BitVec {
    fn from(value: BitVector) -> Self {
        value.0
    }
}

impl From<Vec<bool>> for BitVector {
    fn from(value: Vec<bool>) -> Self {
        let mut bits = BitVec::from_elem(value.len(), false);
        for (i, value) in value.iter().enumerate() {
            if *value {
                bits.set(i, *value);
            }
        }
        Self(bits)
    }
}

impl From<BitVector> for Vec<bool> {
    fn from(value: BitVector) -> Self {
        value.0.into_iter().collect()
    }
}

impl PartialOrd for BitVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Require equal lengths for an ordered comparison.
        if self.0.len() != other.0.len() {
            return None;
        }

        let mut less = false;
        let mut greater = false;

        for (a, b) in self.0.iter().zip(other.0.iter()) {
            match (a, b) {
                (false, true) => less = true,
                (true, false) => greater = true,
                _ => {}
            }
            if less && greater {
                return None;
            }
        }

        match (less, greater) {
            (false, false) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (true, true) => None,
        }
    }

    fn le(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0.iter().zip(other.0.iter()).all(|(a, b)| !a || b)
    }

    fn ge(&self, other: &Self) -> bool {
        other.le(self)
    }
}

impl JoinSemiLattice for BitVector {
    fn join(&self, other: &Self) -> Self {
        let (short, long) = if self.0.len() < other.0.len() {
            (&self.0, &other.0)
        } else {
            (&other.0, &self.0)
        };

        let mut join = short.clone();
        join.grow(long.len() - join.len(), false);
        join.or(long);
        join.into()
    }
}

impl MeetSemiLattice for BitVector {
    fn meet(&self, other: &Self) -> Self {
        let (short, long) = if self.0.len() < other.0.len() {
            (&self.0, &other.0)
        } else {
            (&other.0, &self.0)
        };

        let mut join = short.clone();
        join.grow(long.len() - join.len(), false);
        join.and(long);
        join.into()
    }
}

impl MembershipLattice<usize> for BitVector {
    fn insert(&mut self, value: usize) -> bool {
        match self.get(value) {
            Some(included) => {
                if !included {
                    self.0.set(value, true);
                }
                !included
            }
            None => {
                self.set(value, true);
                true
            }
        }
    }

    fn contains(&self, value: &usize) -> bool {
        self.get(*value).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bit_vec::BitVec;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::cmp::Ordering;

    #[test]
    fn new_len_is_empty_and_set_get() {
        let mut lat = BitVector::new(5);
        assert_eq!(lat.len(), 5);
        assert!(!lat.is_empty());

        // initial bits are false
        for i in 0..5 {
            assert_eq!(lat.get(i), Some(false));
        }
        assert_eq!(lat.get(10), None);

        // setting in range
        lat.set(2, true);
        assert_eq!(lat.get(2), Some(true));

        // setting out of range grows the bitvec
        lat.set(10, true);
        assert_eq!(lat.len(), 11);
        assert_eq!(lat.get(10), Some(true));
        // intermediate bits are still false
        assert_eq!(lat.get(7), Some(false));
    }

    #[test]
    fn from_and_into_bitvec_roundtrip() {
        let mut bv = BitVec::from_elem(8, false);
        bv.set(1, true);
        bv.set(5, true);

        let lat = BitVector::from(bv.clone());
        let bv2: BitVec = lat.into();
        assert_eq!(bv, bv2);
    }

    #[test]
    fn partial_ord_equal_less_greater_incomparable_and_len_mismatch() {
        let a: BitVector = vec![false, true, false].into(); // 010
        let b: BitVector = vec![true, true, false].into(); // 110
        let c: BitVector = vec![true, false, true].into(); // 101

        // a ⊑ b (010 -> 110)
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert!(a.le(&b));
        assert!(!b.le(&a));

        // b ⊒ a
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Greater));
        assert!(b.ge(&a));
        assert!(!a.ge(&b));

        // incomparable: 110 vs 101
        assert_eq!(b.partial_cmp(&c), None);
        assert_eq!(c.partial_cmp(&b), None);

        // equal
        let a2: BitVector = vec![false, true, false].into();
        assert_eq!(a.partial_cmp(&a2), Some(Ordering::Equal));
        assert!(a.le(&a2) && a2.le(&a));

        // length mismatch -> None / false
        let short: BitVector = vec![true, false].into();
        let long: BitVector = vec![true, false, true].into();
        assert_eq!(short.partial_cmp(&long), None);
        assert_eq!(long.partial_cmp(&short), None);
        assert!(!short.le(&long));
        assert!(!long.le(&short));
    }

    #[test]
    fn join_is_bitwise_or_with_padding() {
        let a: BitVector = vec![false, true, false].into(); // 010
        let b: BitVector = vec![true, false, true].into(); // 101

        let j = a.join(&b);
        let j_vec: Vec<bool> = j.into();
        assert_eq!(j_vec, vec![true, true, true]); // 111

        // unequal lengths
        let a: BitVector = vec![true, false].into(); // 10
        let b: BitVector = vec![false, true, true].into(); // 011

        let j = a.join(&b);
        // padded a = 100, OR 011 = 111
        let j_vec: Vec<bool> = j.into();
        assert_eq!(j_vec, vec![true, true, true]);
    }

    #[test]
    fn meet_is_bitwise_and_with_padding() {
        let a: BitVector = vec![false, true, false].into(); // 010
        let b: BitVector = vec![true, false, true].into(); // 101

        let m = a.meet(&b);
        let m_vec: Vec<bool> = m.into();
        assert_eq!(m_vec, vec![false, false, false]); // 000

        // unequal lengths
        let a: BitVector = vec![true, false].into(); // 10
        let b: BitVector = vec![false, true, true].into(); // 011

        let m = a.meet(&b);
        // padded a = 100, AND 011 = 000
        let m_vec: Vec<bool> = m.into();
        assert_eq!(m_vec, vec![false, false, false]);
    }

    #[test]
    fn join_and_meet_idempotent_commutative_same_len() {
        let a: BitVector = vec![true, false, true].into();
        let b: BitVector = vec![false, true, true].into();

        // idempotent
        assert_eq!(a.join(&a), a);
        assert_eq!(a.meet(&a), a);

        // commutative
        assert_eq!(a.join(&b), b.join(&a));
        assert_eq!(a.meet(&b), b.meet(&a));
    }

    #[test]
    fn random_equal_len_join_and_meet_match_manual() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);

        for _ in 0..100 {
            let len = rng.random_range(0..=64);
            let mut a_bits = Vec::with_capacity(len);
            let mut b_bits = Vec::with_capacity(len);

            for _ in 0..len {
                a_bits.push(rng.random());
                b_bits.push(rng.random());
            }

            let a = BitVector::from(a_bits.clone());
            let b = BitVector::from(b_bits.clone());

            let join = a.join(&b);
            let meet = a.meet(&b);

            let join_bits: Vec<bool> = join.clone().into();
            let meet_bits: Vec<bool> = meet.clone().into();

            let expected_join: Vec<bool> = (0..len).map(|i| a_bits[i] || b_bits[i]).collect();
            let expected_meet: Vec<bool> = (0..len).map(|i| a_bits[i] && b_bits[i]).collect();

            assert_eq!(join_bits, expected_join);
            assert_eq!(meet_bits, expected_meet);

            // join ≥ a and ≥ b, meet ≤ a and ≤ b, in subset sense
            for i in 0..len {
                // a[i] ⇒ join[i]
                assert!(!a_bits[i] || join_bits[i]);
                // b[i] ⇒ join[i]
                assert!(!b_bits[i] || join_bits[i]);
                // meet[i] ⇒ a[i] and meet[i] ⇒ b[i]
                assert!(!meet_bits[i] || a_bits[i]);
                assert!(!meet_bits[i] || b_bits[i]);
            }
        }
    }

    // Strategy: arbitrary bool vectors up to some reasonable size
    fn arb_bool_vec(max_len: usize) -> impl Strategy<Value = Vec<bool>> {
        prop::collection::vec(any::<bool>(), 0..=max_len)
    }

    // Join = OR with zero-padding
    proptest! {
        #[test]
        fn prop_join_is_bitwise_or_with_padding(
            a_bits in arb_bool_vec(64),
            b_bits in arb_bool_vec(64),
        ) {
            let a_lat = BitVector::from(a_bits.clone());
            let b_lat = BitVector::from(b_bits.clone());

            let join = a_lat.join(&b_lat);
            let join_bits: Vec<bool> = join.into();

            let max_len = a_bits.len().max(b_bits.len());
            let mut expected = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let av = a_bits.get(i).copied().unwrap_or(false);
                let bv = b_bits.get(i).copied().unwrap_or(false);
                expected.push(av || bv);
            }

            prop_assert_eq!(join_bits, expected);
        }

        #[test]
        fn prop_meet_is_bitwise_and_with_padding(
            a_bits in arb_bool_vec(64),
            b_bits in arb_bool_vec(64),
        ) {
            let a_lat = BitVector::from(a_bits.clone());
            let b_lat = BitVector::from(b_bits.clone());

            let meet = a_lat.meet(&b_lat);
            let meet_bits: Vec<bool> = meet.into();

            let max_len = a_bits.len().max(b_bits.len());
            let mut expected = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let av = a_bits.get(i).copied().unwrap_or(false);
                let bv = b_bits.get(i).copied().unwrap_or(false);
                expected.push(av && bv);
            }

            prop_assert_eq!(meet_bits, expected);
        }

        // For arbitrary lengths:
        // - if lengths differ, partial_cmp must be None
        // - if equal, partial_cmp matches subset relation
        #[test]
        fn prop_partial_ord_matches_subset(
            a_bits in arb_bool_vec(64),
            b_bits in arb_bool_vec(64),
        ) {
            let a_lat = BitVector::from(a_bits.clone());
            let b_lat = BitVector::from(b_bits.clone());

            let n = a_bits.len().min(b_bits.len());
            let subset_ab = (0..n).all(|i| !a_bits[i] || b_bits[i]);
            let subset_ba = (0..n).all(|i| !b_bits[i] || a_bits[i]);
            let same_len = a_bits.len() == b_bits.len();

            let cmp = a_lat.partial_cmp(&b_lat);

            let expected = if !same_len {
                None
            } else {
                match (subset_ab, subset_ba) {
                    (true,  true)  => Some(Ordering::Equal),
                    (true,  false) => Some(Ordering::Less),
                    (false, true)  => Some(Ordering::Greater),
                    (false, false) => None,
                }
            };

            prop_assert_eq!(cmp, expected);

            // le/ge shortcuts
            prop_assert_eq!(a_lat.le(&b_lat), same_len && subset_ab);
            prop_assert_eq!(a_lat.ge(&b_lat), same_len && subset_ba);
        }

        // Join/meet are idempotent and commutative for *all* lengths.
        #[test]
        fn prop_join_meet_idempotent_commutative(
            a_bits in arb_bool_vec(32),
            b_bits in arb_bool_vec(32),
        ) {
            let a_lat = BitVector::from(a_bits);
            let b_lat = BitVector::from(b_bits);

            // idempotence
            prop_assert_eq!(a_lat.join(&a_lat), a_lat.clone());
            prop_assert_eq!(a_lat.meet(&a_lat), a_lat.clone());

            // commutativity
            prop_assert_eq!(a_lat.join(&b_lat), b_lat.join(&a_lat));
            prop_assert_eq!(a_lat.meet(&b_lat), b_lat.meet(&a_lat));
        }
    }
}
