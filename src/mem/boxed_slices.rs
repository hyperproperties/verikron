use std::mem::MaybeUninit;

/// Grows a boxed slice by `extra` elements, cloning existing elements and
/// letting the caller initialize the newly added tail.
///
/// On success, `slice` is replaced with a new `Box<[T]>` of length
/// `old_len + extra`, and `init_tail` is called with a mutable slice
/// corresponding to the newly added region.
///
/// Returns `None` if the new length overflows `usize`.
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

    let mut new: Box<[MaybeUninit<T>]> = Box::<[T]>::new_uninit_slice(new_len);

    unsafe {
        // Fast bulk copy of the initialized prefix.
        std::ptr::copy_nonoverlapping(slice.as_ptr(), new.as_mut_ptr() as *mut T, old_len);

        // Initialize the tail region.
        let new_tail_uninit: &mut [MaybeUninit<T>] = &mut new[old_len..new_len];
        let new_tail: &mut [T] = &mut *(new_tail_uninit as *mut [MaybeUninit<T>] as *mut [T]);

        init_tail(new_tail);

        let new_init: Box<[T]> = new.assume_init();
        *slice = new_init;
    }

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
}
