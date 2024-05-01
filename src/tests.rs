// offset-allocator/src/tests.rs

use std::array;

use crate::{small_float, Allocator};

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

#[test]
fn basic_offset_allocator() {
    let mut allocator = Allocator::new(1024 * 1024 * 256);
    let a = allocator.allocate(1337).unwrap();
    let offset: u32 = a.offset;
    assert_eq!(offset, 0);
    allocator.free(a);
}

#[test]
fn allocate_offset_allocator_simple() {
    let mut allocator: Allocator<u32> = Allocator::new(1024 * 1024 * 256);

    // Free merges neighbor empty nodes. Next allocation should also have offset = 0
    let a = allocator.allocate(0).unwrap();
    assert_eq!(a.offset, 0);

    let b = allocator.allocate(1).unwrap();
    assert_eq!(b.offset, 0);

    let c = allocator.allocate(123).unwrap();
    assert_eq!(c.offset, 1);

    let d = allocator.allocate(1234).unwrap();
    assert_eq!(d.offset, 124);

    allocator.free(a);
    allocator.free(b);
    allocator.free(c);
    allocator.free(d);

    // End: Validate that allocator has no fragmentation left. Should be 100% clean.
    let validate_all = allocator.allocate(1024 * 1024 * 256).unwrap();
    assert_eq!(validate_all.offset, 0);
    allocator.free(validate_all);
}

#[test]
fn allocate_offset_allocator_merge_trivial() {
    let mut allocator: Allocator<u32> = Allocator::new(1024 * 1024 * 256);

    // Free merges neighbor empty nodes. Next allocation should also have offset = 0
    let a = allocator.allocate(1337).unwrap();
    assert_eq!(a.offset, 0);
    allocator.free(a);

    let b = allocator.allocate(1337).unwrap();
    assert_eq!(b.offset, 0);
    allocator.free(b);

    // End: Validate that allocator has no fragmentation left. Should be 100% clean.
    let validate_all = allocator.allocate(1024 * 1024 * 256).unwrap();
    assert_eq!(validate_all.offset, 0);
    allocator.free(validate_all);
}

#[test]
fn allocate_offset_allocator_reuse_trivial() {
    let mut allocator: Allocator<u32> = Allocator::new(1024 * 1024 * 256);

    // Allocator should reuse node freed by A since the allocation C fits in the same bin (using pow2 size to be sure)
    let a = allocator.allocate(1024).unwrap();
    assert_eq!(a.offset, 0);

    let b = allocator.allocate(3456).unwrap();
    assert_eq!(b.offset, 1024);

    allocator.free(a);

    let c = allocator.allocate(1024).unwrap();
    assert_eq!(c.offset, 0);

    allocator.free(c);
    allocator.free(b);

    // End: Validate that allocator has no fragmentation left. Should be 100% clean.
    let validate_all = allocator.allocate(1024 * 1024 * 256).unwrap();
    assert_eq!(validate_all.offset, 0);
    allocator.free(validate_all);
}

#[test]
fn allocate_offset_allocator_reuse_complex() {
    let mut allocator: Allocator<u32> = Allocator::new(1024 * 1024 * 256);

    // Allocator should not reuse node freed by A since the allocation C doesn't fits in the same bin
    // However node D and E fit there and should reuse node from A
    let a = allocator.allocate(1024).unwrap();
    assert_eq!(a.offset, 0);

    let b = allocator.allocate(3456).unwrap();
    assert_eq!(b.offset, 1024);

    allocator.free(a);

    let c = allocator.allocate(2345).unwrap();
    assert_eq!(c.offset, 1024 + 3456);

    let d = allocator.allocate(456).unwrap();
    assert_eq!(d.offset, 0);

    let e = allocator.allocate(512).unwrap();
    assert_eq!(e.offset, 456);

    let report = allocator.storage_report();
    assert_eq!(report.total_free_space, 1024 * 1024 * 256 - 3456 - 2345 - 456 - 512);
    assert_ne!(report.largest_free_region, report.total_free_space);

    allocator.free(c);
    allocator.free(d);
    allocator.free(b);
    allocator.free(e);

    // End: Validate that allocator has no fragmentation left. Should be 100% clean.
    let validate_all = allocator.allocate(1024 * 1024 * 256).unwrap();
    assert_eq!(validate_all.offset, 0);
    allocator.free(validate_all);
}

#[test]
fn allocate_offset_allocator_zero_fragmentation() {
    let mut allocator: Allocator<u32> = Allocator::new(1024 * 1024 * 256);

            // Allocate 256x 1MB. Should fit. Then free four random slots and reallocate four slots.
            // Plus free four contiguous slots an allocate 4x larger slot. All must be zero fragmentation!
    let mut allocations: [_; 256] = array::from_fn(|i| {
        let allocation = allocator.allocate(1024 * 1024).unwrap();
        assert_eq!(allocation.offset, i as u32 * 1024 * 1024);
        allocation
    });

    let report = allocator.storage_report();
    assert_eq!(report.total_free_space, 0);
    assert_eq!(report.largest_free_region, 0);

    // Free four random slots
    allocator.free(allocations[243]);
    allocator.free(allocations[5]);
    allocator.free(allocations[123]);
    allocator.free(allocations[95]);

    // Free four contiguous slots (allocator must merge)
    allocator.free(allocations[151]);
    allocator.free(allocations[152]);
    allocator.free(allocations[153]);
    allocator.free(allocations[154]);

    allocations[243] = allocator.allocate(1024 * 1024).unwrap();
    allocations[5] = allocator.allocate(1024 * 1024).unwrap();
    allocations[123] = allocator.allocate(1024 * 1024).unwrap();
    allocations[95] = allocator.allocate(1024 * 1024).unwrap();
    allocations[151] = allocator.allocate(1024 * 1024 * 4).unwrap();    // 4x larger

    for (i, allocation) in allocations.iter().enumerate() {
        if !(152..155).contains(&i) {
            allocator.free(*allocation);
        }
    }

    let report2 = allocator.storage_report();
    assert_eq!(report2.total_free_space, 1024 * 1024 * 256);
    assert_eq!(report2.largest_free_region, 1024 * 1024 * 256);

    // End: Validate that allocator has no fragmentation left. Should be 100% clean.
    let validate_all = allocator.allocate(1024 * 1024 * 256).unwrap();
    assert_eq!(validate_all.offset, 0);
    allocator.free(validate_all);
}
