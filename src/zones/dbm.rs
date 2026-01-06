use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::zones::{base_dbm::BaseDBM, clock::Clock, relation::Relation};

mod sealed {
    pub trait Sealed {}
}

pub trait DBMState: sealed::Sealed {}

#[derive(Clone, Copy, Debug, Default)]
pub struct Canonical;
impl sealed::Sealed for Canonical {}
impl DBMState for Canonical {}

#[derive(Clone, Copy, Debug, Default)]
pub struct Dirty;
impl sealed::Sealed for Dirty {}
impl DBMState for Dirty {}

#[derive(Clone, Copy, Debug, Default)]
pub struct Unchecked;
impl sealed::Sealed for Unchecked {}
impl DBMState for Unchecked {}

#[derive(Clone, Debug)]
pub struct DBM<S: DBMState> {
    base: BaseDBM,
    _state: PhantomData<S>,
}

pub type CanonicalZone = DBM<Canonical>;
pub type DirtyZone = DBM<Dirty>;
pub type UncheckedZone = DBM<Unchecked>;

impl CanonicalZone {
    #[inline(always)]
    pub fn new(clocks: Clock) -> Self {
        Self {
            base: BaseDBM::new(clocks),
            _state: PhantomData,
        }
    }

    #[inline(always)]
    pub fn tighten(
        mut self,
        i: Clock,
        j: Clock,
        relation: Relation,
    ) -> Result<Self, UncheckedZone> {
        if self.base.tighten(i, j, relation) {
            if self.base.close_pair(i, j).is_err() {
                return Err(UncheckedZone {
                    base: self.base,
                    _state: PhantomData,
                });
            }
        }
        Ok(self)
    }

    #[inline(always)]
    pub fn intersect_constraint(
        self,
        i: Clock,
        j: Clock,
        relation: Relation,
    ) -> Result<Self, UncheckedZone> {
        self.tighten(i, j, relation)
    }

    #[inline(always)]
    pub fn up(mut self) -> Self {
        self.base.up();
        self
    }

    #[inline(always)]
    pub fn subseteq(&self, other: &Self) -> bool {
        self.base.subseteq(&other.base)
    }

    #[inline(always)]
    pub fn intersect(mut self, other: &Self) -> DirtyZone {
        self.base.intersect(&other.base);
        self.dirty()
    }

    #[inline(always)]
    pub fn reset(mut self, clock: Clock) -> DirtyZone {
        self.base.reset(clock);
        self.dirty()
    }

    #[inline(always)]
    pub fn dirty(self) -> DirtyZone {
        DirtyZone {
            base: self.base,
            _state: PhantomData,
        }
    }

    #[inline(always)]
    pub fn fmt_geogebra_conjunctions(&self, labels: &[&str]) -> String {
        self.base.fmt_geogebra_conjunctions(labels)
    }
}

impl Index<(Clock, Clock)> for CanonicalZone {
    type Output = Relation;
    #[inline(always)]
    fn index(&self, idx: (Clock, Clock)) -> &Self::Output {
        &self.base[idx]
    }
}

impl DirtyZone {
    #[inline(always)]
    pub fn set(&mut self, i: Clock, j: Clock, relation: Relation) {
        self[(i, j)] = relation;
    }

    #[inline(always)]
    pub fn intersect(&mut self, other: &Self) {
        self.base.intersect(&other.base);
    }

    #[inline(always)]
    pub fn reset(&mut self, clock: Clock) {
        self.base.reset(clock);
    }

    #[inline(always)]
    pub fn clean(mut self) -> Result<CanonicalZone, UncheckedZone> {
        if self.base.close().is_err() {
            return Err(UncheckedZone {
                base: self.base,
                _state: PhantomData,
            });
        }
        Ok(CanonicalZone {
            base: self.base,
            _state: PhantomData,
        })
    }
}

impl Index<(Clock, Clock)> for DirtyZone {
    type Output = Relation;
    #[inline(always)]
    fn index(&self, idx: (Clock, Clock)) -> &Self::Output {
        &self.base[idx]
    }
}

impl IndexMut<(Clock, Clock)> for DirtyZone {
    #[inline(always)]
    fn index_mut(&mut self, idx: (Clock, Clock)) -> &mut Self::Output {
        &mut self.base[idx]
    }
}
