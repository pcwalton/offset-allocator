//! Extension functions not present in the original C++ `OffsetAllocator`.

use crate::small_float::SmallFloat;

/// Returns the minimum allocator size needed to hold an object of the given
/// size.
pub fn min_allocator_size(needed_object_size: u32) -> u32 {
    SmallFloat::from_u32_round_up(needed_object_size).to_u32()
}

#[cfg(test)]
mod tests {
    use crate::Allocator;

    use super::*;

    #[test]
    fn ext_min_allocator_size() {
        // Randomly generated integers on a log distribution, σ = 10.
        static TEST_OBJECT_SIZES: [u32; 42] = [
            0, 1, 2, 3, 4, 5, 8, 17, 23, 36, 51, 68, 87, 151, 165, 167, 201, 223, 306, 346, 394,
            411, 806, 969, 1404, 1798, 2236, 4281, 4745, 13989, 21095, 26594, 27146, 29679, 144685,
            153878, 495127, 727999, 1377073, 9440387, 41994490, 68520116,
        ];

        for needed_object_size in TEST_OBJECT_SIZES {
            let allocator_size = min_allocator_size(needed_object_size);
            let mut allocator: Allocator<u32> = Allocator::new(allocator_size);
            assert!(allocator.allocate(needed_object_size).is_some());
        }
    }
}
