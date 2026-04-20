# Crux

Crux is a With library for low-level compute: devices, memory, views,
programs, streams, and events.

This repository is initialized as a library scaffold. The pure
Session 1 surface from the Crux design is in place:

- shape and stride math
- dtype helpers
- view construction and view transforms
- memory placement helpers
- the initial public API split across `crux.core`, `crux.view`,
  `crux.device`, `crux.memory`, `crux.program`, `crux.stream`,
  and `crux.collective`

The pure surface is in place, and the first CPU runtime spike is now
implemented for the basic handle path: default device, heap-backed
memory, arena allocation, borrowed-memory interop, copy/fill
operations, CPU streams, completed events, and a first CPU interpreter
for the current IR subset. The runtime path now validates byte ranges
before dispatch/copy/fill, rejects invalid writable aliases, and uses
memmove-safe byte copies on CPU.
Compiled programs now carry real parameter signatures, named binding
validation, v2 `Param` and `SpecConstant` header handling, text-IR
compilation for the current scalar and loop subset, a first IR
validation pass for the current instruction subset, and programmatic
loop execution for the first CPU kernel slice: scalar add, 1D `i32`
parallel add, scalar reduction, nested 2D text-IR loops with multi-rank
index tuples, row-wise 2D reduction, 2D transpose, text-IR `fma`,
text-IR `neg`, text-IR `abs`, text-IR
`mod`, text-IR `add_sat/sub_sat`, text-IR bitwise
`and/or/xor/not/shl/shr`, text-IR bitcount `popcount/clz/ctz`, text-IR
`exp/log/log2`, text-IR `sin/cos/tanh`, text-IR `floor`, text-IR
`ceil`, text-IR `round`, text-IR `sqrt`, text-IR `rsqrt`, text-IR
`cast`, text-IR compare/select, text-IR `min/max/clamp`,
proven scalar `i64`, `f32`, and `f64` add plus pointwise `i64`, `f32`,
and `f64` add/`fma`, pointwise `u32` xor, pointwise `i32 -> f32` cast,
pointwise `i32` abs, pointwise `i32` mod, pointwise `i32` saturating
arithmetic, pointwise `i32` bitwise/shift/bitcount ops, pointwise `f32`
exp/log/log2/sin/cos/tanh/floor/ceil/round/sqrt/rsqrt, and a tiny 2D
matmul-shaped loop nest. The CPU kernel corpus now also has a public
`crux.kernels` module with canonical `ProgramSource` builders for map,
in-place map, reduction, transpose, and matmul kernels, plus builder-
backed execution over contiguous, non-contiguous, and broadcast-read
views. Unsupported scalar dtypes in the current
interpreter slice now reject at compile/validation time instead of
surprising the runtime, and text-IR indices are validated as `i32/u32`
only.
Remaining dtype breadth, collectives, and Metal remain ahead.

## Layout

- `lib/crux.w`: package root metadata
- `lib/crux/core.w`: handles, scalar/data types, shapes, strides, bindings
- `lib/crux/view.w`: pure view construction and transformation helpers
- `lib/crux/device.w`: device API and CPU device implementation
- `lib/crux/memory.w`: memory, arena, and placement APIs
- `lib/crux/program.w`: program compilation API
- `lib/crux/stream.w`: stream, event, and data movement APIs
- `lib/crux/collective.w`: collective API scaffold
- `lib/crux/ir.w`: IR instruction/program definitions and opcode table
- `lib/crux/ir_text.w`: minimal text IR parser for debug/test inputs
- `lib/crux/kernels.w`: canonical public CPU-kernel `ProgramSource` builders
- `lib/crux/cpu_interp.w`: CPU reference executor for the current IR subset
- `test/core_test.w`: shape, stride, dtype, and placement tests
- `test/view_test.w`: view semantics tests
- `test/cpu_interp_test.w`: scalar and 1D CPU interpreter execution tests
- `test/kernel_source_test.w`: compile/signature regressions for public kernel sources
- `test/kernel_cpu_test.w`: canonical CPU kernel corpus and execution-baseline tests
- `test/runtime_stub_test.w`: CPU runtime smoke and contract tests
- `test/ir_test.w`: IR data-model and source-plumbing tests
- `test/ir_text_test.w`: minimal text IR parser tests
- `test/view_smoke.w`: end-to-end pure view smoke test
- `scripts/test.sh`: runs the current file-based test suite
- `docs/`: copied design and implementation notes
- `docs/crux-milestone-board.md`: current milestone board and exit criteria

## Notes

- The scaffold follows the Crux implementation notes where it matters
  for current With support. Shape and stride metadata now use fixed
  arrays plus machine-sized integer types, and runtime handles now
  follow the opaque-pointer API model.
- `with.toml` is present for package identity, even though the current
  compiler remains primarily file-oriented.

## Quick Start

```sh
with check lib/crux/core.w
./scripts/test.sh
```

The current compiler build in this environment accepts `with test`
for individual source files but still expects an explicit `main`
wrapper, so the repo test runner shells out once per test file and
each test source keeps a tiny `main` that calls the local `test_*`
functions.
