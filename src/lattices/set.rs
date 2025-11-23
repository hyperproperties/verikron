use std::{
    cmp::Ordering,
    collections::hash_set::{Drain, IntoIter, Iter},
    hash::Hash,
    iter::FromIterator,
};

use rustc_hash::{FxBuildHasher, FxHashSet};

use crate::lattices::lattice::{JoinSemiLattice, MeetSemiLattice};

#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct Set<T: Eq + Hash>(FxHashSet<T>);

impl<T: Eq + Hash> Set<T> {
    pub fn new() -> Self {
        Self(FxHashSet::with_hasher(FxBuildHasher::default()))
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(FxHashSet::with_capacity_and_hasher(
            capacity,
            FxBuildHasher::default(),
        ))
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> Iter<'_, T> {
        self.0.iter()
    }

    pub fn drain(&mut self) -> Drain<'_, T> {
        self.0.drain()
    }

    pub fn clear(&mut self) {
        self.0.clear()
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.0.retain(f)
    }

    pub fn inner(&self) -> &FxHashSet<T> {
        &self.0
    }

    pub fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }

    pub fn remove(&mut self, value: &T) -> bool {
        self.0.remove(value)
    }

    pub fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    pub fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.0.is_disjoint(&other.0)
    }

    pub fn union<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.union(&other.0)
    }

    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.intersection(&other.0)
    }

    pub fn difference<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.difference(&other.0)
    }

    pub fn symmetric_difference<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.0.symmetric_difference(&other.0)
    }

    pub fn union_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        let mut out = Set::with_capacity(self.len() + other.len());
        out.extend(self.iter().cloned());
        out.extend(other.iter().cloned());
        out
    }

    pub fn intersection_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        let mut out = Set::with_capacity(self.len().min(other.len()));
        for v in self.iter() {
            if other.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }

    pub fn difference_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        let mut out = Set::with_capacity(self.len());
        for v in self.iter() {
            if !other.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }

    pub fn symmetric_difference_owned(&self, other: &Self) -> Self
    where
        T: Clone,
    {
        let mut out = Set::with_capacity(self.len() + other.len());
        for v in self.iter() {
            if !other.contains(v) {
                out.insert(v.clone());
            }
        }
        for v in other.iter() {
            if !self.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }
}

impl<T: Eq + Hash> IntoIterator for Set<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Eq + Hash> Extend<T> for Set<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

impl<T: Eq + Hash> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(FxHashSet::from_iter(iter))
    }
}

impl<T: Eq + Hash> PartialOrd for Set<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let a = &self.0;
        let b = &other.0;

        let a_sub_b = a.is_subset(b);
        let b_sub_a = b.is_subset(a);

        match (a_sub_b, b_sub_a) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }

    fn le(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }
}

impl<T: Eq + Hash + Clone> JoinSemiLattice for Set<T> {
    fn join(&self, other: &Self) -> Self {
        // Quick trivial cases
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        // Optional: cheap superset checks that may avoid allocation if sets are nested
        if self.len() >= other.len() && self.is_superset(other) {
            return self.clone();
        }
        if other.len() >= self.len() && other.is_superset(self) {
            return other.clone();
        }

        // General case
        let mut set = Set::with_capacity(self.len() + other.len());
        set.extend(self.iter().cloned());
        set.extend(other.iter().cloned());
        set
    }
}

impl<T: Eq + Hash + Clone> MeetSemiLattice for Set<T> {
    fn meet(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Set::with_capacity(0); // or Set::new()
        }

        // Iterate smaller, look up in larger
        let (small, big) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };

        let mut out = Set::with_capacity(small.len());
        for v in small.iter() {
            if big.contains(v) {
                out.insert(v.clone());
            }
        }
        out
    }
}
