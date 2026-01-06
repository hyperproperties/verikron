use core::ops::{Index, IndexMut};

use rand::distr::{Distribution, StandardUniform};

use crate::zones::{
    bound::{Bound, ZERO_BOUND},
    clock::Clock,
    relation::{INFINITY, POSITIVE, Relation, RelationError},
    strictness::Strictness,
};

pub const REFERENCE: Clock = 0;
pub const ACTIVE: Clock = REFERENCE + 1;

#[derive(Debug, Clone)]
pub struct BaseDBM {
    dimension: Clock,      // number of clocks including reference x0
    data: Box<[Relation]>, // row-major dimension*dimension matrix
}

impl BaseDBM {
    #[inline]
    pub fn new(clocks: Clock) -> Self {
        // dimension = clocks + 1 (include reference clock x0)
        let dimension = clocks.checked_add(1).expect("clocks + 1 overflow");
        let n = dimension as usize;

        let len = n.checked_mul(n).expect("DBM size overflow");
        let mut data = vec![INFINITY; len];

        // row 0: x0 - xi <= 0  <=>  xi >= 0
        data[..n].fill(POSITIVE);

        // diagonal: xi - xi <= 0
        for i in 0..n {
            data[i * n + i] = POSITIVE;
        }

        Self {
            dimension,
            data: data.into_boxed_slice(),
        }
    }

    /// Dimension `n` (including reference clock).
    #[inline(always)]
    pub const fn dimension(&self) -> Clock {
        self.dimension
    }

    /// Row-major linear offset for cell `(i, j)`:
    /// `offset = i * n + j`.
    #[inline(always)]
    const fn offset(dimension: Clock, minuend: Clock, subtrahend: Clock) -> usize {
        debug_assert!(minuend < dimension, "minuend clock out of range");
        debug_assert!(subtrahend < dimension, "subtrahend clock out of range");

        (minuend as usize) * (dimension as usize) + (subtrahend as usize)
    }

    #[inline(always)]
    pub fn diagonal(&self, clock: Clock) -> Relation {
        debug_assert!(clock < self.dimension, "clock out of range");
        self[(clock, clock)]
    }

    /// Upper bound cell `(i, 0)` encoding `xi - x0 s c` i.e. `xi s c`.
    #[inline(always)]
    pub fn upper(&self, clock: Clock) -> Relation {
        debug_assert!(clock < self.dimension, "clock out of range");
        self[(clock, REFERENCE)]
    }

    #[inline(always)]
    pub fn set(&mut self, i: Clock, j: Clock, relation: Relation) {
        let offset = BaseDBM::offset(self.dimension, i, j);
        self.data[offset] = relation;
    }

    #[inline(always)]
    pub fn set_upper(&mut self, clock: Clock, relation: Relation) {
        debug_assert!(clock < self.dimension, "clock out of range");
        self[(clock, REFERENCE)] = relation;
    }

    /// Lower bound stored in cell `(0, i)` encoding `x0 - xi s b` i.e. `-xi s b`.
    #[inline(always)]
    pub fn lower(&self, clock: Clock) -> Relation {
        debug_assert!(clock < self.dimension, "clock out of range");
        self[(REFERENCE, clock)]
    }

    #[inline(always)]
    pub fn set_lower(&mut self, clock: Clock, relation: Relation) {
        debug_assert!(clock < self.dimension, "clock out of range");
        self[(REFERENCE, clock)] = relation;
    }

    /// Tightens the cell `(i, j)` by intersecting it with `relation`.
    ///
    /// This is the primitive "meet" on a single DBM constraint:
    /// `d[i,j] := min(d[i,j], relation)`.
    ///
    /// Returns `true` iff the cell changed.
    ///
    /// Note: this does **not** re-close the DBM. After tightening constraints,
    /// call [`DBM::close`] (or use [`DBM::intersect_constraint`]).
    #[inline(always)]
    pub fn tighten(&mut self, i: Clock, j: Clock, relation: Relation) -> bool {
        debug_assert!(i < self.dimension, "row clock out of range");
        debug_assert!(j < self.dimension, "col clock out of range");

        let offset = Self::offset(self.dimension, i, j);
        if relation < self.data[offset] {
            self.data[offset] = relation;
            true
        } else {
            false
        }
    }

    /// Intersects the current zone with a single guard constraint.
    ///
    /// Returns `true` iff tightening changed the DBM *before* closing.
    #[inline(always)]
    pub fn intersect_constraint(&mut self, i: Clock, j: Clock, relation: Relation) -> bool {
        self.tighten(i, j, relation)
    }

    /// Time elapse / delay operator ("up").
    ///
    /// This removes all upper bounds `x_i <= c` by setting `(i,0)` to infinity
    /// for `i > 0`. Lower bounds are kept.
    ///
    /// If the DBM is closed before `up()`, it remains closed after `up()`
    /// (we only *relax* constraints).
    #[inline(always)]
    pub fn up(&mut self) {
        // IMPORTANT: do NOT touch REFERENCE (0). Only i>0.
        for clock in 1..self.dimension {
            self.set_upper(clock, INFINITY);
        }
        // Keep canonical diagonal for x0 (cheap, defensive).
        self[(REFERENCE, REFERENCE)] = POSITIVE;
    }

    /// Resets clock `clock` to zero.
    ///
    /// Standard DBM reset operation:
    /// - Row copy: `d[clock, j] := d[0, j]` for all `j`
    /// - Col copy: `d[i, clock] := d[i, 0]` for all `i`
    /// - Then enforce `d[clock,0] = 0≤` and `d[0,clock] = 0≤`
    #[inline(always)]
    pub fn reset(&mut self, clock: Clock) {
        debug_assert!(clock < self.dimension, "clock out of range");

        if clock == REFERENCE {
            return;
        }

        // 1) Copy row 0 into row `clock`: d[clock, j] = d[0, j]
        //    (Constraint: x_clock - x_j <= d[0,j] since x_clock becomes x_0)
        for j in REFERENCE..self.dimension {
            self[(clock, j)] = self.lower(j);
        }

        // 2) Copy column 0 into column `clock`: d[i, clock] = d[i, 0]
        //    (Constraint: x_i - x_clock <= d[i,0])
        for i in REFERENCE..self.dimension {
            self[(i, clock)] = self.upper(i);
        }

        // 3) Enforce exact reset constraints (x_clock == 0):
        self.set_upper(clock, POSITIVE);
        self.set_lower(clock, POSITIVE);
        self[(clock, clock)] = POSITIVE;
    }

    #[inline]
    pub fn copy_clock(&mut self, dst: Clock, src: Clock) {
        debug_assert!(dst < self.dimension, "dst out of range");
        debug_assert!(src < self.dimension, "src out of range");

        if dst == src {
            return;
        }

        // 1) Row copy: d[dst, j] := d[src, j]
        for j in REFERENCE..self.dimension {
            self[(dst, j)] = self[(src, j)];
        }

        // 2) Column copy: d[i, dst] := d[i, src]
        for i in REFERENCE..self.dimension {
            self[(i, dst)] = self[(i, src)];
        }

        // 3) Enforce equality (dst == src) and canonical diagonal.
        self[(dst, src)] = POSITIVE;
        self[(src, dst)] = POSITIVE;
        self[(dst, dst)] = POSITIVE;
    }

    #[inline]
    pub fn assign_offset(
        &mut self,
        dst: Clock,
        src: Clock,
        delta: Bound,
    ) -> Result<(), RelationError> {
        debug_assert!(dst < self.dimension, "dst out of range");
        debug_assert!(src < self.dimension, "src out of range");

        if delta == ZERO_BOUND {
            self.copy_clock(dst, src);
            return Ok(());
        }

        if dst == src {
            // x := x + delta is only satisfiable if delta == 0, which we handled above.
            // If you prefer a different error kind, add a variant.
            return Err(RelationError::InfinityNotFinite);
        }

        // 1) Update row: d[dst, j] = d[src, j] + delta
        for j in REFERENCE..self.dimension {
            let relation = self[(src, j)];
            self[(dst, j)] = if relation.is_infinity() {
                INFINITY
            } else {
                let bound = relation.bound().checked_add(delta)?;
                Relation::new(bound, relation.strictness())
            };
        }

        // 2) Update column: d[i, dst] = d[i, src] - delta
        for i in REFERENCE..self.dimension {
            let relation = self[(i, src)];
            self[(i, dst)] = if relation.is_infinity() {
                INFINITY
            } else {
                let bound = relation.bound().checked_sub(delta)?;
                Relation::new(bound, relation.strictness())
            };
        }

        // 3) Enforce exact relation between dst and src.
        self[(dst, src)] = Relation::weak(delta);
        self[(src, dst)] = Relation::weak(delta.checked_neg().unwrap());

        // 4) Keep canonical diagonal.
        self[(dst, dst)] = POSITIVE;

        Ok(())
    }

    /// Returns `true` iff `self ⊆ other`, assuming both DBMs are **closed**.
    ///
    /// For closed DBMs, zone inclusion is exactly elementwise comparison:
    /// `self[i,j] <= other[i,j]` for all `i,j`.
    #[inline(always)]
    pub fn subseteq(&self, other: &BaseDBM) -> bool {
        debug_assert!(self.dimension == other.dimension, "dimension mismatch");

        if self.dimension != other.dimension {
            return false;
        }

        // Fast path: slice walk.
        let a: &[Relation] = &self.data;
        let b: &[Relation] = &other.data;

        // Same length if same dimension.
        for idx in 0..a.len() {
            if a[idx] > b[idx] {
                return false;
            }
        }
        true
    }

    /// Intersects this DBM with `other` (zone meet).
    ///
    /// Computes the elementwise minimum of the constraints.
    /// This is the standard zone intersection used for invariants.
    #[inline(always)]
    pub fn intersect(&mut self, other: &BaseDBM) -> bool {
        if self.dimension != other.dimension {
            return false;
        }

        let a: &mut [Relation] = &mut self.data;
        let b: &[Relation] = &other.data;

        for idx in 0..a.len() {
            let rhs = b[idx];
            if rhs < a[idx] {
                a[idx] = rhs;
            }
        }

        true
    }

    /// Floyd–Warshall closure.
    ///
    /// After closing, the DBM satisfies the triangle inequality:
    /// `d[i,j] <= d[i,k] + d[k,j]` for all `i,j,k`.
    #[inline(always)]
    pub fn close(&mut self) -> Result<(), RelationError> {
        let data: &mut [Relation] = &mut self.data;

        // 1) Ensure empty path: d[i,i] <= 0. (Canonical diagonal.)
        for i in REFERENCE..self.dimension {
            let ii_offset = BaseDBM::offset(self.dimension, i, i);
            if data[ii_offset] > POSITIVE {
                data[ii_offset] = POSITIVE;
            }
        }

        // 2) Floyd–Warshall relaxation.
        for k in REFERENCE..self.dimension {
            for i in REFERENCE..self.dimension {
                // If d[i,k] is ∞, paths i -> k -> * cannot improve anything.
                let ik = data[BaseDBM::offset(self.dimension, i, k)];
                if ik.is_infinity() {
                    continue;
                }

                // Tighten d[i,j] via k.
                for j in REFERENCE..self.dimension {
                    let kj = data[BaseDBM::offset(self.dimension, k, j)];
                    if kj.is_infinity() {
                        continue;
                    }

                    // Candidate constraint for i -> j via (i -> k -> j).
                    let ikj = ik.checked_add(kj)?;

                    let ij = BaseDBM::offset(self.dimension, i, j);
                    if ikj < data[ij] {
                        data[ij] = ikj;
                    }
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn close_pair(&mut self, i: Clock, j: Clock) -> Result<(), RelationError> {
        debug_assert!(i < self.dimension, "row clock out of range");
        debug_assert!(j < self.dimension, "col clock out of range");

        let data: &mut [Relation] = &mut self.data;

        // Tightened edge w = d[i,j]. If it's ∞, nothing can improve via it.
        let ij = data[Self::offset(self.dimension, i, j)];
        if ij.is_infinity() {
            return Ok(());
        }

        // Incremental closure for a single decreased edge (i -> j):
        // For every (k,l), try the path k -> i -> j -> l.
        //
        // d[k,l] := min(d[k,l], d[k,i] + w + d[j,l])
        //
        // This is correct when the DBM was closed before the tightening.
        for k in REFERENCE..self.dimension {
            let ki = data[Self::offset(self.dimension, k, i)];
            if ki.is_infinity() {
                continue;
            }

            // Prefix k -> i -> j (shared for all l).
            let kij = ki.checked_add(ij)?;

            for l in REFERENCE..self.dimension {
                let jl = data[Self::offset(self.dimension, j, l)];
                if jl.is_infinity() {
                    continue;
                }

                // Candidate k -> i -> j -> l.
                let kijl = kij.checked_add(jl)?;

                let kl_offset = Self::offset(self.dimension, k, l);
                if kijl < data[kl_offset] {
                    data[kl_offset] = kijl;
                }
            }
        }

        Ok(())
    }

    /// Returns `true` iff this DBM is consistent **assuming it is already closed**.
    ///
    /// For closed DBMs, inconsistency (negative cycle) is equivalent to `d[i,i] < 0<=`
    /// for some `i`.
    #[inline(always)]
    pub fn is_consistent(&self) -> bool {
        for i in REFERENCE..self.dimension {
            if self.diagonal(i) < POSITIVE {
                return false;
            }
        }
        true
    }

    /// `labels` must not include the reference label, e.g. `["x", "y"]`.
    pub fn fmt_geogebra_conjunctions(&self, labels: &[&str]) -> String {
        debug_assert!(
            labels.len() == (self.dimension as usize) - 1,
            "labels must cover all active clocks excluding the reference clock"
        );

        let mut parts: Vec<String> = Vec::new();

        for i in ACTIVE..self.dimension {
            // Lower bound: 0 - i R N.
            let lower = self.lower(i);
            if !lower.is_infinity() {
                parts.push(format!(
                    "-{} {} {}",
                    labels[(i - 1) as usize],
                    lower.strictness(),
                    lower.bound()
                ));
            }

            // Upper bound: i - 0 R N.
            let upper = self.upper(i);
            if !upper.is_infinity() {
                parts.push(format!(
                    "{} {} {}",
                    labels[(i - 1) as usize],
                    upper.strictness(),
                    upper.bound()
                ));
            }

            for j in ACTIVE..self.dimension {
                if i == REFERENCE || j == REFERENCE || i == j {
                    continue;
                }

                let relation = self[(i, j)];
                if relation.is_infinity() {
                    continue;
                }

                // Difference constraints: i - j R N.
                parts.push(format!(
                    "{} - {} {} {}",
                    labels[(i - 1) as usize],
                    labels[(j - 1) as usize],
                    relation.strictness(),
                    relation.bound()
                ));
            }
        }

        parts.join(" && ")
    }
}

impl Index<(Clock, Clock)> for BaseDBM {
    type Output = Relation;

    #[inline(always)]
    fn index(&self, (minuend, subtrahend): (Clock, Clock)) -> &Self::Output {
        let k = Self::offset(self.dimension, minuend, subtrahend);
        &self.data[k]
    }
}

impl IndexMut<(Clock, Clock)> for BaseDBM {
    #[inline(always)]
    fn index_mut(&mut self, (minuend, subtrahend): (Clock, Clock)) -> &mut Self::Output {
        let k = Self::offset(self.dimension, minuend, subtrahend);
        &mut self.data[k]
    }
}

/// Default random generation for **consistent** DBMs (useful for property tests).
impl Distribution<BaseDBM> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BaseDBM {
        const MAX_CLOCKS: Clock = 8;
        const MIN_C: i16 = -50;
        const MAX_C: i16 = 50;

        let clocks: Clock = rng.random_range(0..=MAX_CLOCKS);
        let mut dbm = BaseDBM::new(clocks);

        // Only generate per-clock bounds against the reference; this guarantees consistency.
        for i in ACTIVE..dbm.dimension() {
            let l: i16 = rng.random_range(MIN_C..=MAX_C);
            let u: i16 = rng.random_range(l..=MAX_C);

            let mut s_low: Strictness = rng.random();
            let mut s_up: Strictness = rng.random();

            // If l == u, only (>= c) and (<= c) is satisfiable.
            if l == u {
                s_low = Strictness::Weak;
                s_up = Strictness::Weak;
            }

            // lower: xi >= l  <=>  x0 - xi s -l
            dbm[(REFERENCE, i as Clock)] = Relation::new(Bound::new(-l), s_low);

            // upper: xi <= u  <=>  xi - x0 s u
            dbm[(i as Clock, REFERENCE)] = Relation::new(Bound::new(u), s_up);
        }

        // DBM::new already set row 0 and the diagonal; we didn't touch diagonal, so no fixup needed.
        dbm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn unit_close_dimension_one_is_trivial() {
        let mut dbm = BaseDBM::new(0);
        dbm.close().unwrap();

        assert_eq!(dbm.dimension() as usize, 1);
        assert_eq!(dbm[(0, 0)], POSITIVE);
        assert!(dbm.is_consistent());
    }

    #[test]
    fn unit_close_tightens_diagonal_to_at_most_zero() {
        let mut dbm = BaseDBM::new(2); // dimension = 3
        let n = dbm.dimension() as usize;

        // Force everything infinity (including diagonal and row 0).
        for i in 0..n {
            for j in 0..n {
                dbm[(i as Clock, j as Clock)] = INFINITY;
            }
        }

        dbm.close().unwrap();

        for i in 0..n {
            assert!(
                dbm[(i as Clock, i as Clock)] <= POSITIVE,
                "expected diagonal <= 0 after close"
            );
        }
    }

    #[test]
    fn unit_inconsistency_equivalence_simple_example() {
        // With one real clock x1:
        // DBM::new gives x1 >= 0 by (0,1) = 0≤.
        // Add x1 <= -1 by (1,0) = -1≤. This is inconsistent.
        let mut dbm = BaseDBM::new(1);
        dbm[(1, 0)] = Relation::weak(Bound::new(-1));

        dbm.close().unwrap();

        assert!(!dbm.is_consistent());
        assert!(dbm[(1, 1)] < POSITIVE);
    }

    #[test]
    fn prop_idempotency_random_consistent_dbms() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..1_500 {
            let mut dbm_a: BaseDBM = rng.random();
            let mut dbm_b: BaseDBM = dbm_a.clone();

            dbm_a.close().unwrap();
            dbm_b.close().unwrap();
            dbm_b.close().unwrap();

            let n = dbm_a.dimension() as usize;
            assert_eq!(n, dbm_b.dimension() as usize);

            for i in 0..n {
                for j in 0..n {
                    let i = i as Clock;
                    let j = j as Clock;
                    assert_eq!(dbm_a[(i, j)], dbm_b[(i, j)], "mismatch at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn prop_triangle_inequality_after_close_random_consistent_dbms() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..600 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap();

            let n = dbm.dimension() as usize;

            for i in 0..n {
                for j in 0..n {
                    let dij = dbm[(i as Clock, j as Clock)];

                    for k in 0..n {
                        let dik = dbm[(i as Clock, k as Clock)];
                        let dkj = dbm[(k as Clock, j as Clock)];

                        let rhs = if dik.is_infinity() || dkj.is_infinity() {
                            INFINITY
                        } else {
                            // For closed+consistent DBMs generated by StandardUniform,
                            // checked_add should never overflow.
                            dik.checked_add(dkj).unwrap()
                        };

                        assert!(
                            dij <= rhs,
                            "triangle violated: i={i}, j={j}, k={k}, dij={dij}, dik={dik}, dkj={dkj}, rhs={rhs}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prop_diagonal_non_positivity_after_close_random_consistent_dbms() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..1_500 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap();

            let n = dbm.dimension() as usize;
            for i in 0..n {
                let dii = dbm[(i as Clock, i as Clock)];
                assert!(
                    dii <= POSITIVE,
                    "diagonal not non-positive: i={i}, dii={dii}"
                );
            }
        }
    }

    #[test]
    fn prop_consistency_equivalence_after_close_random_consistent_dbms() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..2_000 {
            let mut dbm: BaseDBM = rng.random(); // uses StandardUniform<DBM>, now truly consistent
            dbm.close().unwrap();

            assert!(
                dbm.is_consistent(),
                "generator produced an inconsistent DBM"
            );

            let n = dbm.dimension() as usize;
            for i in 0..n {
                // close enforces dii <= 0, consistency implies dii >= 0, hence dii == 0<=
                assert_eq!(dbm[(i as Clock, i as Clock)], POSITIVE);
            }
        }
    }

    #[test]
    fn unit_consistent_random_dbms_are_consistent_after_close() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);

        for _ in 0..500 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap();
            assert!(dbm.is_consistent());
        }
    }

    #[test]
    fn prop_close_detects_inconsistency_from_contradictory_bounds() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(0xBAD0_CAFE);

        for _ in 0..5_000 {
            // At least 1 real clock so we can make it inconsistent.
            let clocks: Clock = rng.random_range(1..=8);
            let mut dbm = BaseDBM::new(clocks);

            // Pick u in [-50, 49], then set l = u + delta with delta in [1, 10],
            // ensuring l > u and still staying in a small range.
            let u: i16 = rng.random_range(-50..=49);
            let delta: i16 = rng.random_range(1..=10);
            let l: i16 = u + delta;

            // Encode contradiction on clock 1:
            //   x1 >= l    <=>   x0 - x1 <= -l   (cell 0,1)
            //   x1 <= u    <=>   x1 - x0 <= u    (cell 1,0)
            // Use WEAK on both sides so the interval [l,u] is empty iff l>u.
            dbm[(REFERENCE, 1)] = Relation::weak(Bound::new(-l));
            dbm[(1, REFERENCE)] = Relation::weak(Bound::new(u));

            // Closure should succeed (no overflow) but DBM should become inconsistent.
            dbm.close().unwrap();

            assert!(
                !dbm.is_consistent(),
                "expected inconsistent DBM after close, but is_consistent() returned true; \
                clocks={clocks}, l={l}, u={u}"
            );

            // Stronger: in a closed DBM, inconsistency implies some diagonal < 0≤.
            let n = dbm.dimension() as usize;
            let mut found_negative_diag = false;
            for i in 0..n {
                if dbm[(i as Clock, i as Clock)] < POSITIVE {
                    found_negative_diag = true;
                    break;
                }
            }
            assert!(
                found_negative_diag,
                "expected some diagonal < POSITIVE after closing an inconsistent DBM"
            );
        }
    }

    #[test]
    fn prop_close_detects_inconsistency_from_equal_bounds_but_strict() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(0xF00D_F00D);

        for _ in 0..5_000 {
            let clocks: Clock = rng.random_range(1..=8);
            let mut dbm = BaseDBM::new(clocks);

            let c: i16 = rng.random_range(-50..=50);

            // Make: x1 > c  and  x1 < c  (empty)
            // lower: x1 > c  <=>  x0 - x1 < -c
            // upper: x1 < c  <=>  x1 - x0 < c
            dbm[(REFERENCE, 1)] = Relation::strict(Bound::new(-c));
            dbm[(1, REFERENCE)] = Relation::strict(Bound::new(c));

            dbm.close().unwrap();
            assert!(!dbm.is_consistent());

            let mut found = false;
            let n = dbm.dimension() as usize;
            for i in 0..n {
                if dbm[(i as Clock, i as Clock)] < POSITIVE {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    #[test]
    fn unit_subseteq_reflexive_after_close() {
        let mut rng = StdRng::seed_from_u64(0xA11C_E001);
        for _ in 0..300 {
            let mut a: BaseDBM = rng.random();
            a.close().unwrap();
            assert!(a.subseteq(&a));
        }
    }

    #[test]
    fn prop_subseteq_monotone_under_relaxation() {
        let mut rng = StdRng::seed_from_u64(0xBADC_0FFE);

        for _ in 0..800 {
            let mut a: BaseDBM = rng.random();
            a.close().unwrap();

            // b is a relaxed version of a (elementwise >=).
            let mut b = a.clone();

            let n = a.dimension() as usize;

            // Relax some cells by replacing with INFINITY.
            // Relaxation preserves closure correctness (only loosening constraints).
            for _ in 0..(n * n / 2 + 1) {
                let i = rng.random_range(0..n) as Clock;
                let j = rng.random_range(0..n) as Clock;
                b[(i, j)] = INFINITY;
            }

            // Still safe to close; but relaxation doesn't require it. We'll close anyway.
            b.close().unwrap();

            assert!(a.subseteq(&b), "expected a ⊆ b after relaxing b");
        }
    }

    #[test]
    fn prop_subseteq_transitive_sanity() {
        let mut rng = StdRng::seed_from_u64(0xC0DE_CAFE);

        for _ in 0..500 {
            let mut a: BaseDBM = rng.random();
            a.close().unwrap();

            // b relaxes a
            let mut b = a.clone();
            let n = a.dimension() as usize;
            for _ in 0..(n + 1) {
                let i = rng.random_range(0..n) as Clock;
                let j = rng.random_range(0..n) as Clock;
                b[(i, j)] = INFINITY;
            }
            b.close().unwrap();

            // c relaxes b
            let mut c = b.clone();
            for _ in 0..(n + 1) {
                let i = rng.random_range(0..n) as Clock;
                let j = rng.random_range(0..n) as Clock;
                c[(i, j)] = INFINITY;
            }
            c.close().unwrap();

            assert!(a.subseteq(&b));
            assert!(b.subseteq(&c));
            assert!(a.subseteq(&c), "expected transitivity");
        }
    }

    #[test]
    fn unit_intersect_dimension_mismatch_returns_false_and_does_not_modify_self() {
        let mut a = BaseDBM::new(1); // dim = 2
        let b = BaseDBM::new(2); // dim = 3

        // Make a non-trivial so we can detect accidental mutation.
        a[(1, 0)] = Relation::weak(Bound::new(7));
        a[(0, 1)] = Relation::weak(Bound::new(-3));

        let before = a.clone();

        let ok = a.intersect(&b);
        assert!(!ok);

        let n = a.dimension() as usize;
        for i in 0..n {
            for j in 0..n {
                let i = i as Clock;
                let j = j as Clock;
                assert_eq!(a[(i, j)], before[(i, j)]);
            }
        }
    }

    #[test]
    fn unit_intersect_is_elementwise_min() {
        let mut a = BaseDBM::new(1); // dim = 2
        let mut b = BaseDBM::new(1);

        // Overwrite everything so we don't depend on DBM::new defaults.
        let n = a.dimension() as usize;
        for i in 0..n {
            for j in 0..n {
                a[(i as Clock, j as Clock)] = INFINITY;
                b[(i as Clock, j as Clock)] = INFINITY;
            }
        }

        // Pick a few cells where b is tighter, and a few where a is tighter.
        a[(0, 0)] = Relation::weak(Bound::new(0));
        b[(0, 0)] = Relation::weak(Bound::new(0));

        a[(0, 1)] = Relation::weak(Bound::new(10));
        b[(0, 1)] = Relation::strict(Bound::new(10)); // stricter => smaller

        a[(1, 0)] = Relation::weak(Bound::new(5));
        b[(1, 0)] = Relation::weak(Bound::new(7)); // a tighter

        a[(1, 1)] = Relation::weak(Bound::new(0));
        b[(1, 1)] = Relation::weak(Bound::new(0));

        let ok = a.intersect(&b);
        assert!(ok);

        assert_eq!(a[(0, 0)], Relation::weak(Bound::new(0)));
        assert_eq!(a[(0, 1)], Relation::strict(Bound::new(10)));
        assert_eq!(a[(1, 0)], Relation::weak(Bound::new(5)));
        assert_eq!(a[(1, 1)], Relation::weak(Bound::new(0)));
    }

    #[test]
    fn prop_intersect_matches_elementwise_min_random() {
        let mut rng = StdRng::seed_from_u64(0xA11C_E551);

        for _ in 0..2_000 {
            let clocks: Clock = rng.random_range(0..=8);
            let mut a = BaseDBM::new(clocks);
            let mut b = BaseDBM::new(clocks);

            let n = a.dimension() as usize;

            // Fill both matrices with arbitrary relations (including infinity).
            // Keep bounds moderate (but it doesn't matter; intersect doesn't add).
            for i in 0..n {
                for j in 0..n {
                    let ra = if rng.random_ratio(1, 8) {
                        INFINITY
                    } else {
                        let c: i16 = rng.random_range(-200..=200);
                        let s: Strictness = rng.random();
                        Relation::new(Bound::new(c), s)
                    };

                    let rb = if rng.random_ratio(1, 8) {
                        INFINITY
                    } else {
                        let c: i16 = rng.random_range(-200..=200);
                        let s: Strictness = rng.random();
                        Relation::new(Bound::new(c), s)
                    };

                    a[(i as Clock, j as Clock)] = ra;
                    b[(i as Clock, j as Clock)] = rb;
                }
            }

            // Compute expected = elementwise min of (a,b), without helpers.
            let before = a.clone();
            let ok = a.intersect(&b);
            assert!(ok);

            for i in 0..n {
                for j in 0..n {
                    let i = i as Clock;
                    let j = j as Clock;

                    let expected = {
                        let x = before[(i, j)];
                        let y = b[(i, j)];
                        if y < x { y } else { x }
                    };

                    assert_eq!(a[(i, j)], expected, "mismatch at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn prop_intersect_is_idempotent() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..1_000 {
            let mut a: BaseDBM = rng.random(); // your StandardUniform<DBM>
            let b = a.clone();

            let ok = a.intersect(&b);
            assert!(ok);

            let n = a.dimension() as usize;
            for i in 0..n {
                for j in 0..n {
                    let i = i as Clock;
                    let j = j as Clock;
                    assert_eq!(a[(i, j)], b[(i, j)]);
                }
            }
        }
    }

    #[test]
    fn prop_intersect_is_commutative_elementwise() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..1_500 {
            let clocks: Clock = rng.random_range(0..=8);
            let mut a = BaseDBM::new(clocks);
            let mut b = BaseDBM::new(clocks);

            let n = a.dimension() as usize;

            // Fill with arbitrary relations (including infinity).
            for i in 0..n {
                for j in 0..n {
                    let ra = if rng.random_ratio(1, 6) {
                        INFINITY
                    } else {
                        let c: i16 = rng.random_range(-100..=100);
                        let s: Strictness = rng.random();
                        Relation::new(Bound::new(c), s)
                    };

                    let rb = if rng.random_ratio(1, 6) {
                        INFINITY
                    } else {
                        let c: i16 = rng.random_range(-100..=100);
                        let s: Strictness = rng.random();
                        Relation::new(Bound::new(c), s)
                    };

                    a[(i as Clock, j as Clock)] = ra;
                    b[(i as Clock, j as Clock)] = rb;
                }
            }

            let mut a_meet_b = a.clone();
            let mut b_meet_a = b.clone();

            assert!(a_meet_b.intersect(&b));
            assert!(b_meet_a.intersect(&a));

            for i in 0..n {
                for j in 0..n {
                    let i = i as Clock;
                    let j = j as Clock;
                    assert_eq!(a_meet_b[(i, j)], b_meet_a[(i, j)], "mismatch at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn unit_tighten_changes_only_when_stricter() {
        let mut dbm = BaseDBM::new(2); // dim = 3
        let i: Clock = 1;
        let j: Clock = 2;

        // Start at infinity so tightening must change.
        dbm[(i, j)] = INFINITY;

        let changed = dbm.tighten(i, j, Relation::weak(Bound::new(7)));
        assert!(changed);
        assert_eq!(dbm[(i, j)], Relation::weak(Bound::new(7)));

        // Tighten with a weaker (larger) relation: should not change.
        let changed = dbm.tighten(i, j, Relation::weak(Bound::new(9)));
        assert!(!changed);
        assert_eq!(dbm[(i, j)], Relation::weak(Bound::new(7)));

        // Tighten with same: should not change.
        let changed = dbm.tighten(i, j, Relation::weak(Bound::new(7)));
        assert!(!changed);
        assert_eq!(dbm[(i, j)], Relation::weak(Bound::new(7)));

        // Tighten with strictly smaller: should change.
        let changed = dbm.tighten(i, j, Relation::strict(Bound::new(7))); // strict is smaller than weak for same bound
        assert!(changed);
        assert_eq!(dbm[(i, j)], Relation::strict(Bound::new(7)));
    }

    #[test]
    fn prop_tighten_is_min_and_idempotent_random() {
        let mut rng = StdRng::seed_from_u64(0x71A7_4E11);

        for _ in 0..5_000 {
            let clocks: Clock = rng.random_range(0..=8);
            let mut dbm = BaseDBM::new(clocks);
            let dim = dbm.dimension();

            // pick random cell
            let i: Clock = rng.random_range(0..dim);
            let j: Clock = rng.random_range(0..dim);

            // random current value
            let cur = if rng.random_ratio(1, 6) {
                INFINITY
            } else {
                let c: i16 = rng.random_range(-100..=100);
                let s: Strictness = rng.random();
                Relation::new(Bound::new(c), s)
            };
            dbm[(i, j)] = cur;

            // random candidate
            let rel = if rng.random_ratio(1, 6) {
                INFINITY
            } else {
                let c: i16 = rng.random_range(-100..=100);
                let s: Strictness = rng.random();
                Relation::new(Bound::new(c), s)
            };

            // expected min
            let expected = if rel < cur { rel } else { cur };

            let changed1 = dbm.tighten(i, j, rel);
            assert_eq!(dbm[(i, j)], expected);
            assert_eq!(changed1, rel < cur);

            // idempotent: tightening again with same rel changes nothing
            let changed2 = dbm.tighten(i, j, rel);
            assert!(!changed2);
            assert_eq!(dbm[(i, j)], expected);
        }
    }

    #[test]
    fn unit_intersect_constraint_reports_change_and_closes() {
        let mut dbm = BaseDBM::new(2); // dim=3, canonical non-negative zone
        // Make it closed/consistent to start.
        dbm.close().unwrap();
        assert!(dbm.is_consistent());

        // Apply a guard x1 - x0 <= 5  (upper bound)
        let changed = dbm.intersect_constraint(1, REFERENCE, Relation::weak(Bound::new(5)));
        dbm.close().unwrap();

        assert!(changed);

        // After guard+close, DBM must satisfy triangle inequality; check a cheap one:
        // d[1,1] <= d[1,0] + d[0,1]
        let d11 = dbm[(1, 1)];
        let d10 = dbm[(1, 0)];
        let d01 = dbm[(0, 1)];
        let rhs = if d10.is_infinity() || d01.is_infinity() {
            INFINITY
        } else {
            d10.checked_add(d01).unwrap()
        };
        assert!(d11 <= rhs);

        // Applying the same guard again should not report change
        let changed2 = dbm.intersect_constraint(1, REFERENCE, Relation::weak(Bound::new(5)));
        dbm.close().unwrap();
        assert!(!changed2);
    }

    #[test]
    fn prop_intersect_constraint_can_make_dbm_inconsistent() {
        let mut rng = StdRng::seed_from_u64(0xBAD0_CAFE);

        for _ in 0..3_000 {
            // at least 1 real clock so we can break it
            let clocks: Clock = rng.random_range(1..=8);
            let mut dbm = BaseDBM::new(clocks);
            dbm.close().unwrap();
            assert!(dbm.is_consistent());

            // Make contradictory bounds on clock 1:
            // x1 >= l  and x1 <= u with l > u.
            let u: i16 = rng.random_range(-50..=49);
            let delta: i16 = rng.random_range(1..=10);
            let l: i16 = u + delta;

            // lower: x0 - x1 <= -l
            let _ = dbm.intersect_constraint(REFERENCE, 1, Relation::weak(Bound::new(-l)));
            dbm.close().unwrap();
            // upper: x1 - x0 <= u
            let _ = dbm.intersect_constraint(1, REFERENCE, Relation::weak(Bound::new(u)));
            dbm.close().unwrap();

            // Must be inconsistent after closing
            assert!(!dbm.is_consistent());

            // In a closed DBM, inconsistency implies some diagonal < 0≤
            let n = dbm.dimension() as usize;
            let mut found = false;
            for i in 0..n {
                if dbm[(i as Clock, i as Clock)] < POSITIVE {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    #[test]
    fn unit_up_keeps_lower_bounds_drops_upper_bounds() {
        let mut dbm = BaseDBM::new(2);
        dbm.close().unwrap();

        // Put explicit bounds on x1
        dbm.set_lower(1, Relation::weak(Bound::new(-7))); // x0 - x1 <= -(-7)=? (stored form). Just ensure it persists.
        dbm.set_upper(1, Relation::weak(Bound::new(9))); // x1 - x0 <= 9

        // Also put an upper bound on x2
        dbm.set_upper(2, Relation::strict(Bound::new(3)));

        dbm.up();

        // Upper bounds for i>0 must be INFINITY
        assert!(dbm.upper(1).is_infinity());
        assert!(dbm.upper(2).is_infinity());

        // Lower bounds must be preserved
        assert_eq!(dbm.lower(1), Relation::weak(Bound::new(-7)));
        assert_eq!(dbm.lower(2), POSITIVE); // DBM::new default lower is (0,i)=0≤
    }

    #[test]
    fn prop_up_only_relaxes_and_preserves_closedness_if_already_closed() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..1_500 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap();
            assert!(dbm.is_consistent());

            // Snapshot all cells pre-up
            let dim = dbm.dimension();
            let n = dim as usize;
            let before = dbm.clone();

            dbm.up();

            // Check: for i>0, upper becomes infinity
            for i in 1..n {
                assert!(dbm.upper(i as Clock).is_infinity());
            }

            // Check: every cell is >= (i.e. not tighter) than before (up only relaxes).
            for i in 0..n {
                for j in 0..n {
                    let i = i as Clock;
                    let j = j as Clock;
                    // "relax" means value becomes larger or equal in DBM ordering
                    assert!(dbm[(i, j)] >= before[(i, j)], "tightened at ({i},{j})");
                }
            }

            // If it was closed, it should remain closed. Verify triangle inequality.
            for i in 0..n {
                for j in 0..n {
                    let dij = dbm[(i as Clock, j as Clock)];
                    for k in 0..n {
                        let dik = dbm[(i as Clock, k as Clock)];
                        let dkj = dbm[(k as Clock, j as Clock)];
                        let rhs = if dik.is_infinity() || dkj.is_infinity() {
                            INFINITY
                        } else {
                            dik.checked_add(dkj).unwrap()
                        };
                        assert!(dij <= rhs);
                    }
                }
            }
        }
    }

    #[test]
    fn unit_reset_zero_reference_is_noop() {
        let mut dbm = BaseDBM::new(3);
        dbm.close().unwrap();

        let before = dbm.clone();
        dbm.reset(REFERENCE);

        let n = dbm.dimension() as usize;
        for i in 0..n {
            for j in 0..n {
                let i = i as Clock;
                let j = j as Clock;
                assert_eq!(dbm[(i, j)], before[(i, j)]);
            }
        }
    }

    #[test]
    fn unit_reset_zero_enforces_zero_bounds() {
        let mut dbm = BaseDBM::new(2);
        dbm.close().unwrap();

        // Give x1 some bounds.
        dbm.set_lower(1, Relation::weak(Bound::new(-5)));
        dbm.set_upper(1, Relation::strict(Bound::new(7)));

        dbm.reset(1);

        // Must enforce x1 == 0: both directions 0≤
        assert_eq!(dbm.upper(1), POSITIVE);
        assert_eq!(dbm.lower(1), POSITIVE);
        assert_eq!(dbm[(1, 1)], POSITIVE);
    }

    #[test]
    fn prop_reset_zero_copies_row0_and_col0_random_closed_consistent() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..2_000 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap();
            assert!(dbm.is_consistent());

            let dim = dbm.dimension();
            if dim <= 1 {
                continue;
            }

            let clock: Clock = rng.random_range(1..dim);
            let before = dbm.clone();

            dbm.reset(clock);

            // 1) Row copy: d[clock, j] == before[0, j] for all j != clock
            // (j == clock is overwritten by the later diagonal enforcement)
            for j in 0..dim {
                if j == clock {
                    continue;
                }
                assert_eq!(
                    dbm[(clock, j)],
                    before[(REFERENCE, j)],
                    "row copy failed at (clock={clock}, j={j})"
                );
            }

            // 2) Column copy: d[i, clock] == before[i, 0] for all i != clock
            // (i == clock is overwritten by the later diagonal enforcement)
            for i in 0..dim {
                if i == clock {
                    continue;
                }
                assert_eq!(
                    dbm[(i, clock)],
                    before[(i, REFERENCE)],
                    "col copy failed at (i={i}, clock={clock})"
                );
            }

            // 3) Enforced exact reset constraints (x_clock == 0)
            assert_eq!(dbm.upper(clock), POSITIVE, "upper(clock) must be 0≤");
            assert_eq!(dbm.lower(clock), POSITIVE, "lower(clock) must be 0≤");
            assert_eq!(dbm[(clock, clock)], POSITIVE, "diagonal(clock) must be 0≤");
        }
    }

    #[test]
    fn unit_close_pair_noop_when_dij_is_infinity() {
        let mut rng = StdRng::seed_from_u64(0xA11C_E001);

        for _ in 0..500 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap(); // precondition: closed

            let n = dbm.dimension() as usize;
            if n <= 1 {
                continue;
            }

            let i: Clock = rng.random_range(0..(dbm.dimension()));
            let j: Clock = rng.random_range(0..(dbm.dimension()));

            // Force dij = ∞ (a relaxation). This should not require any updates.
            dbm[(i, j)] = INFINITY;

            let before = dbm.clone();
            dbm.close_pair(i, j).unwrap();

            // Must remain unchanged.
            let n2 = dbm.dimension() as usize;
            for r in 0..n2 {
                for c in 0..n2 {
                    let r = r as Clock;
                    let c = c as Clock;
                    assert_eq!(dbm[(r, c)], before[(r, c)], "changed at ({r},{c})");
                }
            }
        }
    }

    #[test]
    fn prop_close_pair_is_idempotent_after_application_when_result_is_consistent() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..2_000 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap(); // precondition: closed + consistent (by generator)

            let dim = dbm.dimension();
            let n = dim as usize;
            if n <= 1 {
                continue;
            }

            let i: Clock = rng.random_range(0..dim);
            let j: Clock = rng.random_range(0..dim);

            // Tighten (i,j) a bit (may create inconsistency, that's fine).
            let b: i16 = rng.random_range(-80..=80);
            let s: Strictness = rng.random();
            dbm.tighten(i, j, Relation::new(Bound::new(b), s));

            dbm.close_pair(i, j).unwrap();

            // Only assert idempotency when the outcome is consistent.
            if !dbm.is_consistent() {
                continue;
            }

            let once = dbm.clone();
            dbm.close_pair(i, j).unwrap();

            for r in 0..n {
                for c in 0..n {
                    let r = r as Clock;
                    let c = c as Clock;
                    assert_eq!(dbm[(r, c)], once[(r, c)], "changed at ({r},{c})");
                }
            }
        }
    }

    #[test]
    fn prop_close_pair_predictably_exposes_inconsistency_from_contradictory_bounds() {
        let mut rng = StdRng::seed_from_u64(0xBAD0_CAFE);

        for _ in 0..5_000 {
            let clocks: Clock = rng.random_range(1..=8);
            let mut dbm = BaseDBM::new(clocks);
            dbm.close().unwrap(); // closed

            let c: Clock = rng.random_range(1..=clocks);

            // Pick u, then l > u.
            let u: i16 = rng.random_range(-50..=49);
            let delta: i16 = rng.random_range(1..=10);
            let l: i16 = u + delta;

            // Set lower bound: x_c >= l  <=>  d[0,c] <= -l
            dbm[(REFERENCE, c)] = Relation::weak(Bound::new(-l));

            // Re-close so the precondition holds before using close_pair on (c,0).
            dbm.close().unwrap();

            // Tighten upper bound: x_c <= u  <=> d[c,0] <= u
            dbm.tighten(c, REFERENCE, Relation::weak(Bound::new(u)));
            dbm.close_pair(c, REFERENCE).unwrap();

            assert!(
                !dbm.is_consistent(),
                "expected inconsistency: c={c}, l={l}, u={u}"
            );

            let n = dbm.dimension() as usize;
            let mut found = false;
            for i in 0..n {
                if dbm[(i as Clock, i as Clock)] < POSITIVE {
                    found = true;
                    break;
                }
            }
            assert!(
                found,
                "expected negative diagonal after close_pair on empty zone"
            );
        }
    }

    #[test]
    fn prop_triangle_inequality_holds_after_close_pair_when_result_is_consistent() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..1_000 {
            let mut dbm: BaseDBM = rng.random();
            dbm.close().unwrap();

            let dim = dbm.dimension();
            let n = dim as usize;

            let i: Clock = rng.random_range(0..dim);
            let j: Clock = rng.random_range(0..dim);

            let b: i16 = rng.random_range(-80..=80);
            let s: Strictness = rng.random();
            dbm.tighten(i, j, Relation::new(Bound::new(b), s));
            dbm.close_pair(i, j).unwrap();

            if !dbm.is_consistent() {
                continue;
            }

            for a in 0..n {
                for b2 in 0..n {
                    let dab = dbm[(a as Clock, b2 as Clock)];
                    if dab.is_infinity() {
                        continue;
                    }
                    for c in 0..n {
                        let dbc = dbm[(b2 as Clock, c as Clock)];
                        let dac = dbm[(a as Clock, c as Clock)];

                        let rhs = if dbc.is_infinity() {
                            INFINITY
                        } else {
                            dab.checked_add(dbc).unwrap()
                        };

                        assert!(
                            dac <= rhs,
                            "triangle violated: a={a}, b={b2}, c={c}, dac={dac}, dab={dab}, dbc={dbc}, rhs={rhs}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unit_close_pair_can_create_negative_diagonal_when_guard_makes_zone_inconsistent() {
        // Classic inconsistency: x1 >= 0 and x1 <= -1.
        // In DBM terms: d[0,1] = 0≤ already from DBM::new, tighten d[1,0] = -1≤.
        // Then closing via pair (1,0) should produce d[0,0] <= -1 and/or d[1,1] <= -1.
        let mut dbm = BaseDBM::new(1);
        dbm.close().unwrap(); // ensure closed

        let changed = dbm.tighten(1, REFERENCE, Relation::weak(Bound::new(-1)));
        assert!(changed);

        dbm.close_pair(1, REFERENCE).unwrap();

        assert!(!dbm.is_consistent());
        assert!(dbm[(0, 0)] < POSITIVE || dbm[(1, 1)] < POSITIVE);
    }

    #[test]
    fn unit_close_pair_updates_paths_of_form_k_to_i_to_j_to_l() {
        // Construct a tiny case where only the new (i->j) edge makes a shorter cycle.
        //
        // We set:
        //   d[0,1] = 0≤
        //   d[2,0] = 0≤
        // and then tighten:
        //   d[1,2] = -3≤
        //
        // Then the path 0 -> 1 -> 2 -> 0 has weight -3, so closure must tighten d[0,0].
        let mut dbm = BaseDBM::new(2);
        dbm.close().unwrap(); // closed

        dbm[(0, 1)] = POSITIVE;
        dbm[(2, 0)] = POSITIVE;

        // Before tightening, diagonal is 0≤.
        assert_eq!(dbm[(0, 0)], POSITIVE);

        dbm.tighten(1, 2, Relation::weak(Bound::new(-3)));
        dbm.close_pair(1, 2).unwrap();

        assert!(
            dbm[(0, 0)] < POSITIVE,
            "expected diagonal to tighten via 0->1->2->0"
        );

        // Cross-check with full close from the same pre-state.
        let mut dbm2 = BaseDBM::new(2);
        dbm2.close().unwrap();
        dbm2[(0, 1)] = POSITIVE;
        dbm2[(2, 0)] = POSITIVE;
        dbm2.tighten(1, 2, Relation::weak(Bound::new(-3)));
        dbm2.close().unwrap();
        assert_eq!(dbm[(0, 0)], dbm2[(0, 0)]);
    }

    #[test]
    fn prop_close_pair_detects_inconsistency_from_contradictory_bounds_random() {
        let mut rng = StdRng::seed_from_u64(0xBAD0_CAFE);

        for _ in 0..5_000 {
            let clocks: Clock = rng.random_range(1..=8);
            let mut dbm = BaseDBM::new(clocks);
            dbm.close().unwrap(); // closed

            // Choose contradiction on some clock c>0: x_c >= l and x_c <= u with l > u.
            let c: Clock = rng.random_range(1..=clocks);

            let u: i16 = rng.random_range(-50..=49);
            let delta: i16 = rng.random_range(1..=10);
            let l: i16 = u + delta;

            // Lower: x_c >= l  <=>  d[0,c] <= -l
            // Upper: x_c <= u  <=>  d[c,0] <= u
            dbm[(REFERENCE, c)] = Relation::weak(Bound::new(-l));

            // Tighten just the upper bound and use close_pair(c,0).
            // (The lower bound was set directly, so re-close to satisfy precondition.)
            dbm.close().unwrap();

            dbm.tighten(c, REFERENCE, Relation::weak(Bound::new(u)));
            dbm.close_pair(c, REFERENCE).unwrap();

            assert!(
                !dbm.is_consistent(),
                "expected inconsistent after contradictory bounds: c={c}, l={l}, u={u}"
            );

            // In a closed DBM, inconsistency implies some diagonal < 0≤.
            let n = dbm.dimension() as usize;
            let mut found = false;
            for i in 0..n {
                if dbm[(i as Clock, i as Clock)] < POSITIVE {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    #[test]
    fn fmt_geogebra_conjunctions_empty_when_only_infinity_for_active_bounds() {
        // dimension = clocks + 1 = 3  => active clocks: 1,2
        let mut dbm = BaseDBM::new(2);

        // Ensure everything that could be printed is infinity:
        // - lower(i): (0,i)
        // - upper(i): (i,0)
        // - (i,j) for i!=j, i,j in ACTIVE..
        dbm[(REFERENCE, 1)] = INFINITY;
        dbm[(REFERENCE, 2)] = INFINITY;
        dbm[(1, REFERENCE)] = INFINITY;
        dbm[(2, REFERENCE)] = INFINITY;
        dbm[(1, 2)] = INFINITY;
        dbm[(2, 1)] = INFINITY;

        let s = dbm.fmt_geogebra_conjunctions(&["x", "y"]);
        assert_eq!(s, "");
    }

    #[test]
    fn fmt_geogebra_conjunctions_prints_lower_upper_and_diff_constraints() {
        // dimension = 3 => clocks: x1, x2
        let mut dbm = BaseDBM::new(2);

        // Lower bound printed as: "-label strictness bound" using lower(i) as stored (0,i).
        // Upper bound printed as: "label strictness bound" using upper(i) (i,0).
        // Diff printed as: "li - lj strictness bound" using cell (i,j).

        dbm[(REFERENCE, 1)] = Relation::new(Bound::new(-7), Strictness::Weak); // prints: "-x ≤ -7"
        dbm[(1, REFERENCE)] = Relation::new(Bound::new(5), Strictness::Strict); // prints: "x < 5"
        dbm[(1, 2)] = Relation::new(Bound::new(9), Strictness::Weak); // prints: "x - y ≤ 9"

        let out = dbm.fmt_geogebra_conjunctions(&["x", "y"]);

        // Order is deterministic given the loops:
        // for i=1: lower, upper, then j loop (j=1 skip diag, j=2 diff)
        // for i=2: lower/upper if present, then diffs (none set here)
        assert_eq!(out, "-x ≤ -7 && x < 5 && x - y ≤ 9 && -y ≤ 0");
    }

    #[test]
    fn fmt_geogebra_conjunctions_uses_labels_i_minus_1_mapping() {
        // dimension = 4 => active clocks: 1,2,3 => labels indices 0,1,2.
        let mut dbm = BaseDBM::new(3);

        dbm[(2, REFERENCE)] = Relation::new(Bound::new(11), Strictness::Weak); // clock 2 => labels[1]
        dbm[(REFERENCE, 3)] = Relation::new(Bound::new(-4), Strictness::Strict); // clock 3 => labels[2]

        let out = dbm.fmt_geogebra_conjunctions(&["a", "b", "c"]);
        // i=1 prints nothing, i=2 prints upper, i=3 prints lower
        assert_eq!(out, "-a ≤ 0 && -b ≤ 0 && b ≤ 11 && -c < -4");
    }

    #[test]
    fn fmt_geogebra_conjunctions_skips_diagonal_and_reference_cells_in_diff_loop() {
        // dimension = 3 => active clocks 1,2
        let mut dbm = BaseDBM::new(2);

        // These should NOT appear:
        dbm[(1, 1)] = Relation::new(Bound::new(-3), Strictness::Strict); // diagonal
        dbm[(2, 2)] = Relation::new(Bound::new(-3), Strictness::Strict); // diagonal
        dbm[(REFERENCE, 1)] = INFINITY; // lower is infinite
        dbm[(1, REFERENCE)] = INFINITY; // upper is infinite
        dbm[(REFERENCE, 2)] = INFINITY;
        dbm[(2, REFERENCE)] = INFINITY;

        // This SHOULD appear:
        dbm[(2, 1)] = Relation::new(Bound::new(6), Strictness::Weak); // "y - x ≤ 6"

        let out = dbm.fmt_geogebra_conjunctions(&["x", "y"]);
        assert_eq!(out, "y - x ≤ 6");
    }

    #[test]
    fn fmt_geogebra_conjunctions_separator_is_double_ampersand_with_spaces() {
        let mut dbm = BaseDBM::new(2);

        dbm[(REFERENCE, 1)] = Relation::new(Bound::new(-1), Strictness::Weak); // "-x ≤ -1"
        dbm[(1, REFERENCE)] = Relation::new(Bound::new(1), Strictness::Weak); // "x ≤ 1"

        let out = dbm.fmt_geogebra_conjunctions(&["x", "y"]);
        assert_eq!(out, "-x ≤ -1 && x ≤ 1 && -y ≤ 0");
        assert!(out.contains(" && "));
        assert!(!out.ends_with(" && "));
        assert!(!out.starts_with(" && "));
    }

    #[test]
    fn fmt_geogebra_conjunctions_can_print_zero_and_infinity_correctly() {
        // Just a small sanity: zero prints and infinity is skipped.
        let mut dbm = BaseDBM::new(1); // dimension=2 => active clock: 1 => labels len 1

        dbm[(1, REFERENCE)] = POSITIVE; // (0, ≤) prints "x ≤ 0"
        dbm[(REFERENCE, 1)] = INFINITY; // skipped

        let out = dbm.fmt_geogebra_conjunctions(&["x"]);
        assert_eq!(out, "x ≤ 0");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "labels must cover all active clocks excluding the reference clock")]
    fn fmt_geogebra_conjunctions_panics_on_wrong_label_len_in_debug() {
        let dbm = BaseDBM::new(2); // dimension=3 => active clocks 1,2 => labels.len must be 2
        let _ = dbm.fmt_geogebra_conjunctions(&["only_one"]);
    }
}
