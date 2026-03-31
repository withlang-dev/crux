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
memory, arena allocation, copy/fill operations, CPU streams, completed
events, and a first CPU interpreter for the current IR subset.
Compiled programs now carry real parameter signatures, named binding
validation, text-IR compilation for the straight-line scalar subset, a
first IR validation pass for the current instruction subset, and
programmatic loop execution for the first CPU kernel slice: scalar add,
1D `i32` parallel add, scalar reduction, text-IR `fma`, text-IR `neg`,
text-IR compare/select, text-IR `min/max/clamp`, proven scalar `f32`
add plus pointwise `f32` `fma`, and a tiny 2D matmul-shaped loop nest.
Richer IR text coverage,
collectives, and Metal remain ahead.

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
- `lib/crux/cpu_interp.w`: CPU reference executor for the current IR subset
- `test/core_test.w`: shape, stride, dtype, and placement tests
- `test/view_test.w`: view semantics tests
- `test/cpu_interp_test.w`: scalar and 1D CPU interpreter execution tests
- `test/kernel_cpu_test.w`: reduction and tiny matmul-shaped CPU kernel tests
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
