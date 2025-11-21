use std::mem::MaybeUninit;

/// Reallocate `slice` to `new_len`, copying `prefix_len`
/// elements from the old slice into the beginning of the new allocation,
/// then calling `init_tail` on the remaining (possibly empty) tail region.
///
/// Requirements:
/// - `prefix_len <= slice.len()`
/// - `prefix_len <= new_len`
pub fn reallocate_with_prefix<T: Copy, F>(
    slice: &mut Box<[T]>,
    new_len: usize,
    prefix_len: usize,
    init_tail: F,
) where
    F: FnOnce(&mut [T]),
{
    debug_assert!(prefix_len <= slice.len());
    debug_assert!(prefix_len <= new_len);

    let mut new: Box<[MaybeUninit<T>]> = Box::<[T]>::new_uninit_slice(new_len);

    unsafe {
        // Copy the prefix that we keep.
        std::ptr::copy_nonoverlapping(slice.as_ptr(), new.as_mut_ptr() as *mut T, prefix_len);

        // Initialize the tail region (may be empty).
        let tail_uninit: &mut [MaybeUninit<T>] = &mut new[prefix_len..new_len];
        let tail: &mut [T] = &mut *(tail_uninit as *mut [MaybeUninit<T>] as *mut [T]);

        init_tail(tail);

        let new_init: Box<[T]> = new.assume_init();
        *slice = new_init;
    }
}

/// Grows a boxed slice by `extra` elements, cloning existing elements and
/// letting the caller initialize the newly added tail.
///
/// On success, `slice` is replaced with a new `Box<[T]>` of length
/// `old_len + extra`, and `init_tail` is called with a mutable slice
/// corresponding to the newly added region.
///
/// Returns `false` if the new length overflows `usize`.
pub fn grow_boxed_slice<T: Copy>(
    slice: &mut Box<[T]>,
    extra: usize,
    init_tail: impl FnOnce(&mut [T]),
) -> bool {
    if extra == 0 {
        init_tail(&mut []);
        return true;
    }

    let old_len = slice.len();
    let Some(new_len) = old_len.checked_add(extra) else {
        // Overflow: do nothing, signal failure.
        return false;
    };

    // Copy the old prefix and let the caller initialize the new tail
    // in the freshly allocated slice.
    reallocate_with_prefix(slice, new_len, old_len, init_tail);
    true
}

/// Shrinks a boxed slice by `remove` elements, letting the caller inspect
/// or modify the tail region being removed.
///
/// On success, `slice` is replaced with a new `Box<[T]>` of length
/// `old_len - remove`, and `with_tail` is called with a mutable slice
/// corresponding to the region that will be removed.
///
/// If `remove == 0`, `with_tail` is still called with an empty slice and
/// the function returns `true` without reallocating.
///
/// Returns `false` if `remove > slice.len()` (i.e. the new length would
/// underflow), in that case `slice` is left unchanged.
///
/// `T: Copy` is required so we can safely memcpy the prefix without
/// worrying about drop semantics.
pub fn shrink_boxed_slice<T: Copy>(
    slice: &mut Box<[T]>,
    remove: usize,
    with_tail: impl FnOnce(&mut [T]),
) -> bool {
    let old_len = slice.len();

    if remove == 0 {
        // Nothing to remove: still let the caller run any logic.
        with_tail(&mut []);
        return true;
    }

    let Some(new_len) = old_len.checked_sub(remove) else {
        // Underflow: cannot shrink by more than current length.
        return false;
    };

    // First, expose the old tail to the caller *on the original allocation*.
    {
        let full: &mut [T] = slice.as_mut();
        let (_head, tail) = full.split_at_mut(new_len);
        with_tail(tail);
    } // full & tail borrow end here, so we can re-borrow `slice`.

    // Now reallocate to the smaller size, copying the kept prefix.
    // We don't need to initialize anything beyond that, so `init_tail`
    // is just a no-op on an empty slice.
    reallocate_with_prefix(slice, new_len, new_len, |_tail: &mut [T]| {});

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::boxed::Box;

    #[test]
    fn grow_zero_extra_keeps_slice_unchanged() {
        let original = vec![1u32, 2, 3];
        let mut boxed: Box<[u32]> = original.clone().into_boxed_slice();

        let success = grow_boxed_slice(&mut boxed, 0, |tail| {
            // Tail should be empty when extra == 0
            assert_eq!(tail.len(), 0);
        });
        assert!(success);

        assert_eq!(&*boxed, &original[..]);
    }

    #[test]
    fn grow_by_one_appends_value() {
        let original = vec![10u32, 20, 30];
        let mut boxed: Box<[u32]> = original.clone().into_boxed_slice();

        let success = grow_boxed_slice(&mut boxed, 1, |tail| {
            assert_eq!(tail.len(), 1);
            tail[0] = 99;
        });
        assert!(success);

        assert_eq!(boxed.len(), original.len() + 1);
        assert_eq!(&boxed[..original.len()], &original[..]);
        assert_eq!(boxed[original.len()], 99);
    }

    #[test]
    fn grow_by_multiple_appends_block() {
        let original = vec![1u16, 2, 3];
        let mut boxed: Box<[u16]> = original.clone().into_boxed_slice();

        let tail_vals = [7u16, 8, 9];

        let success = grow_boxed_slice(&mut boxed, tail_vals.len(), |tail| {
            assert_eq!(tail.len(), tail_vals.len());
            tail.copy_from_slice(&tail_vals);
        });
        assert!(success);

        assert_eq!(boxed.len(), original.len() + tail_vals.len());
        assert_eq!(&boxed[..original.len()], &original[..]);
        assert_eq!(&boxed[original.len()..], &tail_vals[..]);
    }

    #[test]
    fn grow_multiple_times_preserves_prefix() {
        let mut boxed: Box<[u32]> = Box::new([]);

        // First grow
        let first = grow_boxed_slice(&mut boxed, 3, |tail| {
            tail.copy_from_slice(&[1, 2, 3]);
        });
        assert!(first);

        // Second grow
        let second = grow_boxed_slice(&mut boxed, 2, |tail| {
            tail.copy_from_slice(&[4, 5]);
        });
        assert!(second);

        assert_eq!(&boxed[..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn random_stress_grow_boxed_slice() {
        use rand::Rng;

        let mut rng = rand::rng();

        for _ in 0..500 {
            let len = rng.random_range(0..32);
            let head: Vec<u32> = (0..len).map(|_| rng.random()).collect();
            let extra = rng.random_range(0..8);
            let tail_vals: Vec<u32> = (0..extra).map(|_| rng.random()).collect();

            let old = head.clone();
            let mut boxed = head.into_boxed_slice();

            let success = grow_boxed_slice(&mut boxed, extra, |tail| {
                assert_eq!(tail.len(), extra);
                for (t, v) in tail.iter_mut().zip(tail_vals.iter()) {
                    *t = *v;
                }
            });
            assert!(success);

            // Length is old + extra.
            assert_eq!(boxed.len(), old.len() + extra);

            // Prefix preserved.
            assert_eq!(&boxed[..old.len()], &old[..]);

            // Tail matches the values we wrote.
            assert_eq!(&boxed[old.len()..], &tail_vals[..]);
        }
    }

    proptest! {
        #[test]
        fn grow_boxed_slice_appends_given_tail(
            head in proptest::collection::vec(any::<u16>(), 0..16),
            tail in proptest::collection::vec(any::<u16>(), 0..8),
        ) {
            let extra = tail.len();
            let old = head.clone();
            let mut boxed: Box<[u16]> = head.into_boxed_slice();

            let success = grow_boxed_slice(&mut boxed, extra, |tail_slice| {
                assert_eq!(tail_slice.len(), extra);
                tail_slice.copy_from_slice(&tail);
            });
            prop_assert!(success);

            prop_assert_eq!(boxed.len(), old.len() + extra);
            prop_assert_eq!(&boxed[..old.len()], &old[..]);
            prop_assert_eq!(&boxed[old.len()..], &tail[..]);
        }
    }

    #[test]
    fn shrink_zero_remove_keeps_slice_unchanged() {
        let original = vec![1u32, 2, 3];
        let mut boxed: Box<[u32]> = original.clone().into_boxed_slice();

        let mut called = false;
        let success = shrink_boxed_slice(&mut boxed, 0, |tail| {
            called = true;
            assert_eq!(tail.len(), 0, "tail must be empty when remove == 0");
        });

        assert!(success, "shrink with remove=0 must succeed");
        assert!(called, "with_tail must be called for remove=0");
        assert_eq!(&*boxed, &original[..], "slice must remain unchanged");
    }

    #[test]
    fn shrink_more_than_len_fails_and_keeps_slice() {
        let original = vec![1u32, 2, 3];
        let mut boxed: Box<[u32]> = original.clone().into_boxed_slice();

        let mut called = false;
        let success = shrink_boxed_slice(&mut boxed, 4, |_tail| {
            called = true;
        });

        assert!(!success, "shrink with remove > len must fail");
        assert!(!called, "with_tail must not be called on failure");
        assert_eq!(&*boxed, &original[..], "slice must remain unchanged");
    }

    #[test]
    fn shrink_by_one_removes_last_value() {
        let original = vec![10u32, 20, 30, 40];
        let mut boxed: Box<[u32]> = original.clone().into_boxed_slice();

        let mut tail_seen: Vec<u32> = Vec::new();
        let success = shrink_boxed_slice(&mut boxed, 1, |tail| {
            assert_eq!(tail.len(), 1);
            tail_seen.extend_from_slice(tail);
            // We could also mutate tail here, but it is going to be discarded anyway.
        });

        assert!(success);
        assert_eq!(tail_seen, vec![40], "tail must be the last removed element");

        assert_eq!(boxed.len(), original.len() - 1);
        assert_eq!(&boxed[..], &original[..original.len() - 1]);
    }

    #[test]
    fn shrink_entire_slice_results_in_empty() {
        let original = vec![1u8, 2, 3, 4];
        let mut boxed: Box<[u8]> = original.clone().into_boxed_slice();

        let mut tail_seen = Vec::new();
        let success = shrink_boxed_slice(&mut boxed, original.len(), |tail| {
            // Tail is the entire original slice.
            assert_eq!(tail.len(), original.len());
            tail_seen.extend_from_slice(tail);
        });

        assert!(success);
        assert_eq!(tail_seen, original, "tail must be the full original slice");
        assert_eq!(
            boxed.len(),
            0,
            "shrinking by full length must yield empty slice"
        );
    }

    #[test]
    fn shrink_multiple_times_preserves_head() {
        let original = vec![1u32, 2, 3, 4, 5];
        let mut boxed: Box<[u32]> = original.clone().into_boxed_slice();

        // First shrink: remove last 2 elements [4, 5]
        let mut tail1 = Vec::new();
        let ok1 = shrink_boxed_slice(&mut boxed, 2, |tail| {
            tail1.extend_from_slice(tail);
        });
        assert!(ok1);
        assert_eq!(tail1, vec![4, 5]);
        assert_eq!(&*boxed, &[1, 2, 3]);

        // Second shrink: remove last 1 element [3]
        let mut tail2 = Vec::new();
        let ok2 = shrink_boxed_slice(&mut boxed, 1, |tail| {
            tail2.extend_from_slice(tail);
        });
        assert!(ok2);
        assert_eq!(tail2, vec![3]);
        assert_eq!(&*boxed, &[1, 2]);
    }

    #[test]
    fn random_stress_shrink_boxed_slice() {
        use rand::Rng;

        let mut rng = rand::rng();

        for _ in 0..500 {
            let len = rng.random_range(0..32);
            let mut data: Vec<u32> = (0..len).map(|_| rng.random()).collect();
            let remove = rng.random_range(0..40); // allow > len cases

            let mut boxed: Box<[u32]> = data.clone().into_boxed_slice();

            let mut tail_seen = Vec::new();
            let success = shrink_boxed_slice(&mut boxed, remove, |tail| {
                tail_seen.extend_from_slice(tail);
            });

            if remove > data.len() {
                // Should fail, no changes.
                assert!(!success);
                assert_eq!(&*boxed, &data[..]);
                assert!(
                    tail_seen.is_empty(),
                    "closure must not be called on failure"
                );
            } else if remove == 0 {
                assert!(success);
                assert_eq!(&*boxed, &data[..]);
                assert!(tail_seen.is_empty(), "tail is empty when remove == 0");
            } else {
                // remove in 1..=len
                assert!(success);
                let new_len = data.len() - remove;
                let (head, tail) = data.split_at(new_len);

                assert_eq!(boxed.len(), new_len);
                assert_eq!(&boxed[..], head);
                assert_eq!(tail_seen, tail);
            }
        }
    }

    proptest! {
        #[test]
        fn shrink_boxed_slice_removes_tail_correctly(
            head in proptest::collection::vec(any::<u16>(), 0..32),
            remove in 0usize..40,  // allow > head.len()
        ) {
            let mut boxed: Box<[u16]> = head.clone().into_boxed_slice();
            let mut tail_seen: Vec<u16> = Vec::new();

            let success = shrink_boxed_slice(&mut boxed, remove, |tail| {
                tail_seen.extend_from_slice(tail);
            });

            if remove > head.len() {
                // Must fail, no change, closure not called.
                prop_assert!(!success);
                prop_assert_eq!(&*boxed, &head[..]);
                prop_assert!(tail_seen.is_empty());
            } else if remove == 0 {
                prop_assert!(success);
                prop_assert_eq!(&*boxed, &head[..]);
                prop_assert!(tail_seen.is_empty());
            } else {
                // 1..=head.len()
                prop_assert!(success);
                let new_len = head.len() - remove;
                let (expected_head, expected_tail) = head.split_at(new_len);

                prop_assert_eq!(boxed.len(), new_len);
                prop_assert_eq!(&boxed[..], expected_head);
                prop_assert_eq!(&tail_seen[..], expected_tail);
            }
        }

        // A combined property: grow then shrink by the same tail length
        // should be able to roundtrip to the original vector.
        #[test]
        fn grow_then_shrink_roundtrips(
            head in proptest::collection::vec(any::<u32>(), 0..16),
            tail in proptest::collection::vec(any::<u32>(), 0..8),
        ) {
            let old = head.clone();
            let tail_len = tail.len();
            let mut boxed: Box<[u32]> = head.into_boxed_slice();

            // First grow and append tail.
            let success_grow = grow_boxed_slice(&mut boxed, tail_len, |tail_slice| {
                tail_slice.copy_from_slice(&tail);
            });
            prop_assert!(success_grow);
            prop_assert_eq!(&boxed[..old.len()], &old[..]);
            prop_assert_eq!(&boxed[old.len()..], &tail[..]);

            // Now shrink by the same tail length, discard the tail.
            let mut shrink_tail_seen = Vec::new();
            let success_shrink = shrink_boxed_slice(&mut boxed, tail_len, |removed| {
                shrink_tail_seen.extend_from_slice(removed);
            });
            prop_assert!(success_shrink);

            prop_assert_eq!(&boxed[..], &old[..], "after grow+shrink, prefix must be original");
            prop_assert_eq!(&shrink_tail_seen[..], &tail[..], "removed tail must equal appended tail");
        }
    }
}
