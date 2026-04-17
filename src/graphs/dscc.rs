use std::{ops::Range, slice};

use bit_vec::BitVec;

use crate::graphs::{
    properties::{Properties, PropertyStoreType},
    quotient::{Quotient, QuotientType},
    scc::SCC,
    structure::VertexType,
};

/// Dense SCC decomposition on vertices `0..vertex_count()`.
///
/// `classes[v]` is the SCC of `v`.
/// `members` stores all vertices grouped by SCC, so vertices of the same SCC
/// occupy one contiguous range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DSCC {
    /// SCC id of each vertex.
    classes: Box<[usize]>,

    /// Vertices grouped by SCC.
    members: Box<[usize]>,

    /// Whether each SCC is recurrent.
    recurrent: BitVec,
}

impl DSCC {
    /// Creates a dense SCC decomposition.
    ///
    /// The caller must uphold the [`DSCC`] invariants.
    #[must_use]
    pub fn new(classes: Box<[usize]>, members: Box<[usize]>, recurrent: BitVec) -> Self {
        Self {
            classes,
            members,
            recurrent,
        }
    }

    /// Returns the number of vertices.
    #[must_use]
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.classes.len()
    }

    /// Returns the number of SCCs.
    #[must_use]
    #[inline]
    pub fn component_count(&self) -> usize {
        self.recurrent.len()
    }

    /// Returns the contiguous range of `members` belonging to `class`.
    #[must_use]
    #[inline]
    pub fn members_range(&self, class: usize) -> Range<usize> {
        let start = self
            .members
            .partition_point(|&vertex| self.classes[vertex] < class);

        let end = self
            .members
            .partition_point(|&vertex| self.classes[vertex] <= class);

        start..end
    }

    /// Returns the vertices in `class`.
    #[must_use]
    #[inline]
    pub fn members_slice(&self, class: usize) -> &[usize] {
        let range = self.members_range(class);
        &self.members[range]
    }
}

impl VertexType for DSCC {
    type Vertex = usize;
}

impl PropertyStoreType for DSCC {
    type Key = usize;
    type Property = usize;
}

impl Properties for DSCC {
    /// Returns the SCC id of `key`.
    #[inline]
    fn property(&self, key: Self::Key) -> Option<&Self::Property> {
        self.classes.get(key)
    }
}

impl QuotientType for DSCC {
    type Class = usize;
}

impl Quotient for DSCC {
    type Classes<'a>
        = Range<usize>
    where
        Self: 'a;

    type Members<'a>
        = std::iter::Copied<slice::Iter<'a, usize>>
    where
        Self: 'a;

    /// Returns all SCC ids.
    #[inline]
    fn classes(&self) -> Self::Classes<'_> {
        0..self.component_count()
    }

    /// Returns the SCC of `vertex`.
    #[inline]
    fn class(&self, vertex: Self::Vertex) -> Self::Class {
        self.classes[vertex]
    }

    /// Returns the vertices in `class`.
    #[inline]
    fn members(&self, class: Self::Class) -> Self::Members<'_> {
        self.members_slice(class).iter().copied()
    }
}

impl SCC for DSCC {
    /// Returns an arbitrary representative of `class`.
    #[inline]
    fn representative(&self, class: Self::Class) -> Self::Vertex {
        self.members_slice(class)[0]
    }

    /// Returns whether `class` is recurrent.
    #[inline]
    fn is_recurrent(&self, class: Self::Class) -> bool {
        self.recurrent.get(class).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graphs::{
        properties::Properties,
        quotient::{FiniteQuotient, Quotient},
        scc::SCC,
    };

    fn sample_dscc() -> DSCC {
        // Components:
        // 0 -> {1, 4}
        // 1 -> {0, 2}
        // 2 -> {3}
        DSCC::new(
            vec![1, 0, 1, 2, 0].into_boxed_slice(),
            vec![1, 4, 0, 2, 3].into_boxed_slice(),
            BitVec::from_iter([true, true, false]),
        )
    }

    #[test]
    fn accessors_return_expected_sizes() {
        let dscc = sample_dscc();

        assert_eq!(dscc.vertex_count(), 5);
        assert_eq!(dscc.component_count(), 3);
    }

    #[test]
    fn properties_expose_vertex_classes() {
        let dscc = sample_dscc();

        assert_eq!(dscc.property(0), Some(&1));
        assert_eq!(dscc.property(1), Some(&0));
        assert_eq!(dscc.property(2), Some(&1));
        assert_eq!(dscc.property(3), Some(&2));
        assert_eq!(dscc.property(4), Some(&0));
        assert_eq!(dscc.property(5), None);
    }

    #[test]
    fn quotient_classes_are_dense() {
        let dscc = sample_dscc();

        assert_eq!(dscc.classes().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(dscc.class_count(), 3);
        assert!(dscc.contains_class(&0));
        assert!(dscc.contains_class(&1));
        assert!(dscc.contains_class(&2));
        assert!(!dscc.contains_class(&3));
    }

    #[test]
    fn class_lookup_matches_expected_partition() {
        let dscc = sample_dscc();

        assert_eq!(dscc.class(0), 1);
        assert_eq!(dscc.class(1), 0);
        assert_eq!(dscc.class(2), 1);
        assert_eq!(dscc.class(3), 2);
        assert_eq!(dscc.class(4), 0);
    }

    #[test]
    fn members_ranges_are_correct() {
        let dscc = sample_dscc();

        assert_eq!(dscc.members_range(0), 0..2);
        assert_eq!(dscc.members_range(1), 2..4);
        assert_eq!(dscc.members_range(2), 4..5);
    }

    #[test]
    fn members_slices_are_correct() {
        let dscc = sample_dscc();

        assert_eq!(dscc.members_slice(0), &[1, 4]);
        assert_eq!(dscc.members_slice(1), &[0, 2]);
        assert_eq!(dscc.members_slice(2), &[3]);
    }

    #[test]
    fn quotient_members_iterator_matches_members_slice() {
        let dscc = sample_dscc();

        assert_eq!(dscc.members(0).collect::<Vec<_>>(), vec![1, 4]);
        assert_eq!(dscc.members(1).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(dscc.members(2).collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn representative_is_first_member_of_each_class() {
        let dscc = sample_dscc();

        assert_eq!(dscc.representative(0), 1);
        assert_eq!(dscc.representative(1), 0);
        assert_eq!(dscc.representative(2), 3);
    }

    #[test]
    fn recurrence_queries_match_bit_vector() {
        let dscc = sample_dscc();

        assert!(dscc.is_recurrent(0));
        assert!(dscc.is_recurrent(1));
        assert!(!dscc.is_recurrent(2));

        assert!(!dscc.is_trivial(0));
        assert!(!dscc.is_trivial(1));
        assert!(dscc.is_trivial(2));
    }

    #[test]
    fn strong_connectivity_matches_class_equality() {
        let dscc = sample_dscc();

        assert!(dscc.are_strongly_connected(0, 2));
        assert!(dscc.are_strongly_connected(1, 4));
        assert!(dscc.are_strongly_connected(3, 3));

        assert!(!dscc.are_strongly_connected(0, 1));
        assert!(!dscc.are_strongly_connected(2, 3));
        assert!(!dscc.are_strongly_connected(4, 3));
    }

    #[test]
    fn empty_dscc_is_valid() {
        let dscc = DSCC::new(
            Vec::<usize>::new().into_boxed_slice(),
            Vec::<usize>::new().into_boxed_slice(),
            BitVec::new(),
        );

        assert_eq!(dscc.vertex_count(), 0);
        assert_eq!(dscc.component_count(), 0);
        assert_eq!(dscc.classes().collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(dscc.property(0), None);
    }

    #[test]
    fn singleton_component_is_supported() {
        let dscc = DSCC::new(
            vec![0].into_boxed_slice(),
            vec![0].into_boxed_slice(),
            BitVec::from_iter([false]),
        );

        assert_eq!(dscc.vertex_count(), 1);
        assert_eq!(dscc.component_count(), 1);
        assert_eq!(dscc.class(0), 0);
        assert_eq!(dscc.members_slice(0), &[0]);
        assert_eq!(dscc.representative(0), 0);
        assert!(!dscc.is_recurrent(0));
        assert!(dscc.is_trivial(0));
    }

    #[test]
    fn constructor_rejects_mismatched_lengths() {
        let result = std::panic::catch_unwind(|| {
            DSCC::new(
                vec![0, 0].into_boxed_slice(),
                vec![0].into_boxed_slice(),
                BitVec::from_iter([true]),
            )
        });

        assert!(result.is_err());
    }

    #[test]
    fn constructor_rejects_invalid_class_id() {
        let result = std::panic::catch_unwind(|| {
            DSCC::new(
                vec![0, 1].into_boxed_slice(),
                vec![0, 1].into_boxed_slice(),
                BitVec::from_iter([true]),
            )
        });

        assert!(result.is_err());
    }

    #[test]
    fn constructor_rejects_missing_vertex_in_vertices_by_class() {
        let result = std::panic::catch_unwind(|| {
            DSCC::new(
                vec![0, 0].into_boxed_slice(),
                vec![0, 0].into_boxed_slice(),
                BitVec::from_iter([true]),
            )
        });

        assert!(result.is_err());
    }

    #[test]
    fn constructor_rejects_out_of_range_vertex_in_vertices_by_class() {
        let result = std::panic::catch_unwind(|| {
            DSCC::new(
                vec![0, 0].into_boxed_slice(),
                vec![0, 2].into_boxed_slice(),
                BitVec::from_iter([true]),
            )
        });

        assert!(result.is_err());
    }

    #[test]
    fn constructor_rejects_non_dense_classes() {
        let result = std::panic::catch_unwind(|| {
            DSCC::new(
                vec![0, 2].into_boxed_slice(),
                vec![0, 1].into_boxed_slice(),
                BitVec::from_iter([true, false, true]),
            )
        });

        assert!(result.is_err());
    }

    #[test]
    fn constructor_rejects_vertices_not_sorted_by_class() {
        let result = std::panic::catch_unwind(|| {
            // vertex 0 has class 1, vertex 1 has class 0, so [0, 1] is not sorted by class
            DSCC::new(
                vec![1, 0].into_boxed_slice(),
                vec![0, 1].into_boxed_slice(),
                BitVec::from_iter([true, true]),
            )
        });

        assert!(result.is_err());
    }
}
