# `offset-allocator`

## Overview

This is a port of [Sebastian Aaltonen's `OffsetAllocator`] package for C++ to 100% safe Rust. It's a fast, simple, hard real time allocator. This is especially useful for managing GPU resources, and the goal is to use it in [Bevy].

The port has been made into more or less idiomatic Rust but is otherwise mostly line-for-line, preserving comments. That way, patches for the original `OffsetAllocator` should be readily transferable to this Rust port.

Please note that `offset-allocator` isn't a Rust allocator conforming to the `GlobalAlloc` trait. You can't use this crate as a drop-in replacement for the system allocator, `jemalloc`, `wee_alloc`, etc. The general algorithm that this crate uses could be adapted to construct a Rust allocator, but that's beyond the scope of this particular implementation. This is by design, so that this allocator can be used to manage resources that aren't just CPU memory: in particular, you can manage allocations inside GPU buffers with it. By contrast, Rust allocators are hard-wired to the CPU and can't be used to manage GPU resources.

## Description

This allocator is completely agnostic to what it's allocating: it only knows
about a contiguous block of memory of a specific size. That size need not be in
bytes: this is especially useful when allocating inside a buffer of fixed-size
structures. For example, if using this allocator to divide up a GPU index
buffer object, one might want to treat the units of allocation as 32-bit
floats.

From [the original README]:

> Fast hard realtime O(1) offset allocator with minimal fragmentation.

> Uses 256 bins with 8 bit floating point distribution (3 bit mantissa + 5 bit exponent) and a two level bitfield to find the next available bin using 2x LZCNT instructions to make all operations O(1). Bin sizes following the floating point distribution ensures hard bounds for memory overhead percentage regarless of size class. Pow2 bins would waste up to +100% memory (+50% on average). Our float bins waste up to +12.5% (+6.25% on average).

> The allocation metadata is stored in a separate data structure, making this allocator suitable for sub-allocating any resources, such as GPU heaps, buffers and arrays. Returns an offset to the first element of the allocated contiguous range.

## References

Again per [the original README]:

> This allocator is similar to the two-level segregated fit (TLSF) algorithm.

> Comparison paper shows that TLSF algorithm provides best in class performance and fragmentation: <https://www.researchgate.net/profile/Alfons-Crespo/publication/234785757_A_comparison_of_memory_allocators_for_real-time_applications/links/5421d8550cf2a39f4af765f4/A-comparison-of-memory-allocators-for-real-time-applications.pdf>

## Author

C++ version: Sebastian Aaltonen

Rust port: Patrick Walton, @pcwalton

## License

Licensed under the MIT license. See `LICENSE-MIT` for details.

## Code of conduct

`offset-allocator` follows the same code of conduct as Rust itself. Reports can be made to the project authors.

[Sebastian Aaltonen's `OffsetAllocator`]: https://github.com/sebbbi/OffsetAllocator
[the original README]: https://github.com/sebbbi/OffsetAllocator/blob/main/README.md
[Bevy]: https://github.com/bevyengine/bevy/
