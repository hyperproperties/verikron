use std::{marker::PhantomData, mem};

/// Generic property-store type.
pub trait PropertyStoreType {
    /// Key used to index the store.
    type Key;

    /// Property stored at each key.
    type Property;
}

/// Read access to properties.
pub trait Properties: PropertyStoreType {
    /// Returns the property of `key`.
    fn property(&self, key: Self::Key) -> Option<&Self::Property>;
}

/// Write access to existing properties.
pub trait WriteProperty: PropertyStoreType {
    /// Returns a mutable reference to the property of `key`.
    fn property_mut(&mut self, key: Self::Key) -> Option<&mut Self::Property>;

    /// Replaces the property of `key` and returns the old property.
    fn set_property(&mut self, key: Self::Key, property: Self::Property) -> Option<Self::Property>;
}

/// Structural insertion of a property at a specific key.
///
/// This is meant for stores that must stay aligned with an external graph.
/// For indexed stores, insertion behaves like `Vec::insert`.
pub trait InsertProperty: PropertyStoreType {
    /// Inserts `property` at `key`.
    ///
    /// Returns `true` on success.
    fn insert_property(&mut self, key: Self::Key, property: Self::Property) -> bool;
}

/// Structural removal of a property at a specific key.
///
/// This is meant for stores that must stay aligned with an external graph.
/// For indexed stores, removal behaves like `Vec::remove`.
pub trait RemoveProperty: PropertyStoreType {
    /// Removes and returns the property of `key`.
    fn remove_property(&mut self, key: Self::Key) -> Option<Self::Property>;
}

/// Property store indexed directly by `usize`.
///
/// The property for key `k` is stored in `properties[k]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexedProperties<P> {
    properties: Vec<P>,
}

impl<P> IndexedProperties<P> {
    /// Creates a store from an existing vector.
    ///
    /// `properties[k]` is the property of key `k`.
    #[must_use]
    #[inline]
    pub fn with_properties(properties: Vec<P>) -> Self {
        Self { properties }
    }
}

impl<P> FromIterator<P> for IndexedProperties<P> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        Self::with_properties(iter.into_iter().collect())
    }
}

impl<P> From<Vec<P>> for IndexedProperties<P> {
    #[inline]
    fn from(properties: Vec<P>) -> Self {
        Self::with_properties(properties)
    }
}

impl<P> PropertyStoreType for IndexedProperties<P> {
    type Key = usize;
    type Property = P;
}

impl<P> Properties for IndexedProperties<P> {
    #[inline]
    fn property(&self, key: Self::Key) -> Option<&Self::Property> {
        self.properties.get(key)
    }
}

impl<P> WriteProperty for IndexedProperties<P> {
    #[inline]
    fn property_mut(&mut self, key: Self::Key) -> Option<&mut Self::Property> {
        self.properties.get_mut(key)
    }

    #[inline]
    fn set_property(&mut self, key: Self::Key, property: Self::Property) -> Option<Self::Property> {
        self.properties
            .get_mut(key)
            .map(|slot| mem::replace(slot, property))
    }
}

impl<P> InsertProperty for IndexedProperties<P> {
    #[inline]
    fn insert_property(&mut self, key: Self::Key, property: Self::Property) -> bool {
        if key > self.properties.len() {
            return false;
        }

        self.properties.insert(key, property);
        true
    }
}

impl<P> RemoveProperty for IndexedProperties<P> {
    #[inline]
    fn remove_property(&mut self, key: Self::Key) -> Option<Self::Property> {
        (key < self.properties.len()).then(|| self.properties.remove(key))
    }
}

/// Empty property store.
///
/// This is useful when a graph has no vertex properties or no edge properties.
/// All lookups return `None`, all insertions succeed and are ignored, and all
/// removals return `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmptyProperties<K, P = ()> {
    marker: PhantomData<fn(K) -> P>,
}

impl<K, P> PropertyStoreType for EmptyProperties<K, P> {
    type Key = K;
    type Property = P;
}

impl<K, P> Properties for EmptyProperties<K, P> {
    #[inline]
    fn property(&self, _: Self::Key) -> Option<&Self::Property> {
        None
    }
}

impl<K, P> WriteProperty for EmptyProperties<K, P> {
    #[inline]
    fn property_mut(&mut self, _: Self::Key) -> Option<&mut Self::Property> {
        None
    }

    #[inline]
    fn set_property(&mut self, _: Self::Key, _: Self::Property) -> Option<Self::Property> {
        None
    }
}

impl<K, P> InsertProperty for EmptyProperties<K, P> {
    #[inline]
    fn insert_property(&mut self, _: Self::Key, _: Self::Property) -> bool {
        false
    }
}

impl<K, P> RemoveProperty for EmptyProperties<K, P> {
    #[inline]
    fn remove_property(&mut self, _: Self::Key) -> Option<Self::Property> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn indexed_properties_follow_vec_semantics() {
        let mut props = IndexedProperties::with_properties(vec!['a', 'b']);

        assert_eq!(props.property(0), Some(&'a'));
        assert_eq!(props.set_property(1, 'x'), Some('b'));
        assert_eq!(props.property(1), Some(&'x'));
    }

    #[test]
    fn empty_properties_are_always_empty() {
        let mut props = EmptyProperties::<usize, char>::default();

        assert_eq!(props.property(0), None);
        assert_eq!(props.property_mut(0), None);
        assert_eq!(props.set_property(0, 'x'), None);
        assert!(!props.insert_property(0, 'x'));
        assert_eq!(props.remove_property(0), None);
    }

    proptest! {
        #[test]
        fn prop_indexed_insert_matches_vec(
            values in prop::collection::vec(any::<u8>(), 0..16),
            index in 0usize..20,
            inserted in any::<u8>(),
        ) {
            let mut props = IndexedProperties::with_properties(values.clone());
            let mut expected = values;

            let ok = props.insert_property(index, inserted);
            let expected_ok = index <= expected.len();

            if expected_ok {
                expected.insert(index, inserted);
            }

            prop_assert_eq!(ok, expected_ok);
        }

        #[test]
        fn prop_indexed_remove_matches_vec(
            values in prop::collection::vec(any::<u8>(), 0..16),
            index in 0usize..20,
        ) {
            let mut props = IndexedProperties::with_properties(values.clone());
            let mut expected = values;

            let got = props.remove_property(index);
            let expected_got = (index < expected.len()).then(|| expected.remove(index));

            prop_assert_eq!(got, expected_got);
        }
    }
}
