# `offset-allocator`

This is a port of [Sebastian Aaltonen's `OffsetAllocator`] package for C++ to 100% safe Rust. It's a fast, simple, hard real time allocator. This is especially useful for managing GPU resources, and the goal is to use it in [Bevy].

The port has been made into more or less idiomatic Rust but is otherwise mostly line-for-line, preserving comments. That way, patches for the original `OffsetAllocator` should be readily transferable to this Rust port.

## Author

C++ version: Sebastian Aaltonen

Rust port: Patrick Walton, @pcwalton

## License

Licensed under the MIT license. See `LICENSE-MIT` for details.

## Code of conduct

`offset-allocator` follows the same code of conduct as Rust itself. Reports can be made to the project authors.

[Sebastian Aaltonen's `OffsetAllocator`]: https://github.com/sebbbi/OffsetAllocator
[Bevy]: https://github.com/bevyengine/bevy/
