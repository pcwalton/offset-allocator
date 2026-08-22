// offset-allocator/src/node_index.rs

use std::fmt::{Debug, Display};

use nonmax::{NonMaxU16, NonMaxU32};

/// Determines the number of allocations that the allocator supports.
///
/// By default, [`Allocator`] and related functions use `u32`, which allows for
/// `u32::MAX - 1` allocations. You can, however, use `u16` instead, which
/// causes the allocator to use less memory but limits the number of allocations
/// within a single allocator to at most 65,534.
pub trait NodeIndex: Clone + Copy + Default {
    /// The `NonMax` version of this type.
    ///
    /// This is used extensively to optimize `enum` representations.
    type NonMax: NodeIndexNonMax + TryFrom<Self> + Into<Self>;

    /// The maximum value representable in this type.
    const MAX: u32;

    /// Converts from a unsigned 32-bit integer to an instance of this type.
    fn from_u32(val: u32) -> Self;

    /// Converts this type to an unsigned machine word.
    fn to_usize(self) -> usize;
}

/// The `NonMax` version of the [`NodeIndex`].
///
/// For example, for `u32`, the `NonMax` version is [`NonMaxU32`].
pub trait NodeIndexNonMax: Clone + Copy + PartialEq + Default + Debug + Display {
    /// Converts this type to an unsigned machine word.
    fn to_usize(self) -> usize;
}

impl NodeIndex for u32 {
    type NonMax = NonMaxU32;
    const MAX: u32 = u32::MAX;

    fn from_u32(val: u32) -> Self {
        val
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl NodeIndex for u16 {
    type NonMax = NonMaxU16;
    const MAX: u32 = u16::MAX as u32;

    fn from_u32(val: u32) -> Self {
        val as u16
    }

    fn to_usize(self) -> usize {
        self as usize
    }
}

impl NodeIndexNonMax for NonMaxU32 {
    fn to_usize(self) -> usize {
        u32::from(self) as usize
    }
}

impl NodeIndexNonMax for NonMaxU16 {
    fn to_usize(self) -> usize {
        u16::from(self) as usize
    }
}
