// offset-allocator/src/small_float.rs

//! This module handles operations around [`SmallFloat`], a custom struct that represents an
//! 8-bit unsigned floating point value that represents integer values using a 3-bit mantissa
//! and a 5-bit exponent. Each of these 256 values correspond to a specific bin, which determines
//! the size of the allocations supported by each bin.

/// An 8-bit unsigned floating point value representing an integer using a 3-bit mantissa and a 5-bit exponent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmallFloat(u32);

impl SmallFloat {
    /// The number of bits that represent the mantissa
    const MANTISSA_BITS: u32 = 3;

    /// The number of bits that represent the exponent
    const EXPONENT_BITS: u32 = 5;

    /// The number of possible values that can be stored in this `SmallFloat`
    const NUM_VALUES: usize = 1 << (Self::MANTISSA_BITS + Self::EXPONENT_BITS);

    /// The number of possible mantissa values. This number is a power of 2.
    const MANTISSA_VALUE: u32 = 1 << Self::MANTISSA_BITS;

    /// A mask that can be bitwise-anded with the float to get just the mantissa
    const MANTISSA_MASK: u32 = Self::MANTISSA_VALUE - 1;

    /// All possible values of a [`SmallFloat`] from smallest to largest
    pub fn values() -> impl ExactSizeIterator<Item = Self> {
        (0..Self::NUM_VALUES).map(|i| Self(i as u32))
    }

    /// The least [`SmallFloat`] greater than or equal to the given value
    pub fn from_u32_round_up(value: u32) -> Self {
        let mut exp = 0;
        let mut mantissa;

        if value < Self::MANTISSA_VALUE {
            // Denorm: 0..(MANTISSA_VALUE-1)
            mantissa = value
        } else {
            // Normalized: Hidden high bit always 1. Not stored. Just like float.
            let highest_set_bit = value.ilog2();
            let mantissa_start_bit = highest_set_bit - Self::MANTISSA_BITS;
            exp = mantissa_start_bit + 1;
            mantissa = (value >> mantissa_start_bit) & Self::MANTISSA_MASK;

            let low_bits_mask = (1 << mantissa_start_bit) - 1;

            // Round up!
            if (value & low_bits_mask) != 0 {
                mantissa += 1;
            }
        }

        // Using `+` instead of `|` allows mantissa->exp overflow for round up
        SmallFloat((exp << Self::MANTISSA_BITS) + mantissa)
    }

    /// The greatest [`SmallFloat`] less than or equal to the given value
    pub fn from_u32_round_down(value: u32) -> Self {
        let mut exp = 0;
        let mantissa;

        if value < Self::MANTISSA_VALUE {
            // Denorm: 0..(MANTISSA_VALUE-1)
            mantissa = value
        } else {
            // Normalized: Hidden high bit always 1. Not stored. Just like float.
            let highest_set_bit = value.ilog2();
            let mantissa_start_bit = highest_set_bit - Self::MANTISSA_BITS;
            exp = mantissa_start_bit + 1;
            mantissa = (value >> mantissa_start_bit) & Self::MANTISSA_MASK;
        }

        SmallFloat((exp << Self::MANTISSA_BITS) | mantissa)
    }

    /// The `u32` that holds the same value as the [`SmallFloat`]
    pub fn to_u32(self) -> u32 {
        let exponent = self.0 >> Self::MANTISSA_BITS;
        let mantissa = self.0 & Self::MANTISSA_MASK;
        if exponent == 0 {
            mantissa
        } else {
            (mantissa | Self::MANTISSA_VALUE) << (exponent - 1)
        }
    }

    /// Reinterprets the bits of the [`SmallFloat`] as a `u32` instead
    #[inline]
    pub fn reinterpret_as_u32(self) -> u32 {
        self.0
    }

    /// Reinterprets the bits of the `u32` as a [`SmallFloat`] instead
    #[inline]
    pub fn reinterpret_u32(data: u32) -> Self {
        Self(data)
    }
}

/// A map whose key is a [`SmallFloat`]. Internally represented as an array
#[derive(Debug)]
pub struct SmallFloatMap<T>([T; SmallFloat::NUM_VALUES]);

impl<T: Default + Copy> Default for SmallFloatMap<T> {
    fn default() -> Self {
        Self([T::default(); SmallFloat::NUM_VALUES])
    }
}

impl<T> std::ops::Index<SmallFloat> for SmallFloatMap<T> {
    type Output = T;

    fn index(&self, index: SmallFloat) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl<T> std::ops::IndexMut<SmallFloat> for SmallFloatMap<T> {
    fn index_mut(&mut self, index: SmallFloat) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_float_uint_to_float() {
        // Denorms, exp=1 and exp=2 + mantissa = 0 are all precise.
        // NOTE: Assuming 8 value (3 bit) mantissa.
        // If this test fails, please change this assumption!
        let precise_number_count = 17;
        for i in 0..precise_number_count {
            let round_up = SmallFloat::from_u32_round_up(i);
            let round_down = SmallFloat::from_u32_round_down(i);
            assert_eq!(SmallFloat::reinterpret_u32(i), round_up);
            assert_eq!(SmallFloat::reinterpret_u32(i), round_down);
        }

        // Test some random picked numbers
        struct NumberFloatUpDown {
            number: u32,
            up: u32,
            down: u32,
        }

        let test_data = [
            NumberFloatUpDown {
                number: 17,
                up: 17,
                down: 16,
            },
            NumberFloatUpDown {
                number: 118,
                up: 39,
                down: 38,
            },
            NumberFloatUpDown {
                number: 1024,
                up: 64,
                down: 64,
            },
            NumberFloatUpDown {
                number: 65536,
                up: 112,
                down: 112,
            },
            NumberFloatUpDown {
                number: 529445,
                up: 137,
                down: 136,
            },
            NumberFloatUpDown {
                number: 1048575,
                up: 144,
                down: 143,
            },
        ];

        for v in test_data {
            let round_up = SmallFloat::from_u32_round_up(v.number).reinterpret_as_u32();
            let round_down = SmallFloat::from_u32_round_down(v.number).reinterpret_as_u32();
            assert_eq!(round_up, v.up);
            assert_eq!(round_down, v.down);
        }
    }

    #[test]
    fn small_float_float_to_uint() {
        // Denorms, exp=1 and exp=2 + mantissa = 0 are all precise.
        // NOTE: Assuming 8 value (3 bit) mantissa.
        // If this test fails, please change this assumption!
        let precise_number_count = 17;
        for i in 0..precise_number_count {
            let v = SmallFloat::reinterpret_u32(i).to_u32();
            assert_eq!(i, v);
        }

        // Test that float->uint->float conversion is precise for all numbers
        // NOTE: Test values < 240. 240->4G = overflows 32 bit integer
        for i in (0..240).map(SmallFloat::reinterpret_u32) {
            let v = i.to_u32();
            let round_up = SmallFloat::from_u32_round_up(v);
            let round_down = SmallFloat::from_u32_round_down(v);
            assert_eq!(i, round_up);
            assert_eq!(i, round_down);
        }
    }
}
