use bit_vec::BitVec;
use std::hash::Hash;

use crate::lattices::set::Set;

// TODO: Create a BitVec Lattice type. Maybe even a generic lattice driven visited set.

pub trait Visited<V>: Default {
    fn visit(&mut self, value: V) -> bool;

    fn is_visited(&self, value: &V) -> bool;
}

impl<V> Visited<V> for Set<V>
where
    V: Eq + Hash + Copy,
{
    #[inline]
    fn visit(&mut self, value: V) -> bool {
        self.insert(value)
    }

    #[inline]
    fn is_visited(&self, value: &V) -> bool {
        self.contains(&value)
    }
}

impl Visited<usize> for BitVec {
    #[inline]
    fn visit(&mut self, value: usize) -> bool {
        let len = self.len();
        if value >= len {
            let grow_by = value + 1 - len;

            // BitVec was initially sized from vertex_count().
            // Getting here means the graph is inconsistent
            // (successor >= vertex_count()).
            //
            // In debug builds we want this to scream,
            // but in release we prefer to recover instead of panicking.
            debug_assert!(
                false,
                "BfsFrontiersBitset: successor {} out of range (len = {}), \
                 growing BitVec by {} bits - inconsistent vertex_count()/successors",
                value, len, grow_by,
            );
            self.grow(grow_by, false);
        }

        if !self[value] {
            self.set(value, true);
            true
        } else {
            false
        }
    }

    #[inline]
    fn is_visited(&self, value: &usize) -> bool {
        self[*value]
    }
}
