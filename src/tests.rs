// offset-allocator/src/tests.rs

use crate::small_float;

#[test]
fn small_float_uint_to_float() {
    // Denorms, exp=1 and exp=2 + mantissa = 0 are all precise.
    // NOTE: Assuming 8 value (3 bit) mantissa.
    // If this test fails, please change this assumption!
    let precise_number_count = 17;
    for i in 0..precise_number_count {
        let round_up = small_float::uint_to_float_round_up(i);
        let round_down = small_float::uint_to_float_round_down(i);
        assert_eq!(i, round_up);
        assert_eq!(i, round_down);
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
        let round_up = small_float::uint_to_float_round_up(v.number);
        let round_down = small_float::uint_to_float_round_down(v.number);
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
        let v = small_float::float_to_uint(i);
        assert_eq!(i, v);
    }

    // Test that float->uint->float conversion is precise for all numbers
    // NOTE: Test values < 240. 240->4G = overflows 32 bit integer
    for i in 0..240 {
        let v = small_float::float_to_uint(i);
        let round_up = small_float::uint_to_float_round_up(v);
        let round_down = small_float::uint_to_float_round_down(v);
        assert_eq!(i, round_up);
        assert_eq!(i, round_down);
    }
}
