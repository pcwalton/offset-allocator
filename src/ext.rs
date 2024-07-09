//! Extension functions not present in the original C++ `OffsetAllocator`.

use crate::small_float;

/// Returns the minimum allocator size needed to hold an object of the given
/// size.
pub fn min_allocator_size(needed_object_size: u32) -> u32 {
    small_float::float_to_uint(small_float::uint_to_float_round_up(needed_object_size))
}
