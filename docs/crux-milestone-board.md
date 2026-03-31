# Crux Milestone Board

This board tracks the implementation sequence for Crux based on the
design in `docs/crux-design.md`, the practical constraints in
`docs/crux-implementation-notes.md`, and the current repo state.

Status values:

- `done`: implemented and covered by the current test suite
- `in_progress`: partially implemented, but not yet a complete milestone
- `pending`: not started in a meaningful way

## Current Baseline

Status: `in_progress`

Implemented now:

- public substrate surface for device, memory, view, program, stream, and collective modules
- spec-aligned fixed-array shape/stride metadata and machine-sized offset types
- spec-aligned opaque handle model behind pointers in the public API
- shape/stride/dtype/view helpers
- CPU singleton device and heap-backed memory handles
- program records with parameter signatures
- text IR parsing for a minimal subset
- IR validation for the current supported subset
- CPU interpreter for `const`, `load`, `store`, `add/sub/mul/div/fma/neg`, `loop`, `parallel`, and block markers
- CPU interpreter for `eq/ne/lt/gt/le/ge/select` over the current scalar subset
- CPU interpreter for `min/max/clamp` over the current scalar subset
- dispatch-side validation for missing, extra, duplicate, dtype-mismatched, rank-mismatched, and broadcast-write bindings
- end-to-end tests for scalar add and 1D `i32` parallel add
- first kernel-corpus tests for scalar reduction and a tiny 2D matmul-shaped loop nest on CPU
- end-to-end text-IR `fma` coverage in the CPU kernel suite
- end-to-end text-IR `neg` coverage in the CPU kernel suite
- end-to-end text-IR compare/select coverage in the CPU kernel suite
- end-to-end text-IR `min/max/clamp` coverage across the CPU interpreter and kernel suites
- end-to-end `f32` coverage for scalar add and pointwise `fma`

Still missing from the baseline:

- broader dtype and opcode support
- backend split
- Metal runtime and codegen
- a broader kernel corpus beyond the first add/reduce/matmul slice
- collectives

## Milestone 0: Spec Alignment

Status: `done`

Goal:
Realign the public Crux substrate with the current With language surface before adding more backend and kernel complexity.

Scope:

- migrate `Shape` from named fields to `dims: [usize; 8]`
- migrate `Strides` from named fields to `elems: [isize; 8]`
- change `Size` to `usize` and `Stride` to `isize`
- update all view, IR, runtime, and test code that currently depends on the compatibility layout
- remove stale documentation that says Crux is using named fields because arrays are unavailable
- refactor runtime handles from by-value `i64` aliases to pointer-based opaque APIs
- make public handle-facing APIs use `*mut Device`, `*mut Memory`, `*mut Program`, `*mut Stream`, `*mut Event`, and `*mut Arena`

Primary files:

- `lib/crux/core.w`
- `lib/crux/view.w`
- `lib/crux/device.w`
- `lib/crux/memory.w`
- `lib/crux/program.w`
- `lib/crux/stream.w`
- `lib/crux/cpu_interp.w`
- `test/*`
- `README.md`

Exit criteria:

- `Shape` and `Strides` use fixed arrays and machine-sized integer types throughout the library
- the current test suite passes without the field-by-field compatibility layer
- the public handle model matches the current spec: opaque types are only used behind pointers
- no Crux docs claim that arrays or `usize`/`isize` are unavailable

Notes:

- completed in two passes: first arrays plus `usize`/`isize`, then pointer-to-opaque handles
- bare `null` still does not infer to `*mut Opaque`; use explicit casts when needed

## Milestone 1: CPU Substrate Completion

Status: `in_progress`

Goal:
Make the non-GPU substrate complete enough that CPU execution is a trustworthy reference path.

Completed so far:

- CPU copy and fill APIs
- CPU arena create/destroy/alloc/reset/used
- tests for copy/fill round-trips, broadcast-write rejection, float bit fills, and arena allocation/reset

Scope:

- make `DeviceInfo` and memory metadata less placeholder-like where practical
- keep validation rules at the API layer, especially grid and binding rules
- remove obvious stub-only behavior from stream/program/memory where the CPU backend can support it

Primary files:

- `lib/crux/device.w`
- `lib/crux/memory.w`
- `lib/crux/stream.w`
- `lib/crux/program.w`
- `test/runtime_stub_test.w`

Exit criteria:

- a user can allocate memory, fill it, copy between views, and free it through the public API
- arena allocation works on CPU with deterministic behavior
- stream dispatch validation catches contract violations before interpreter execution
- tests cover positive and negative cases for memory and stream semantics

## Milestone 2: CPU Reference Executor

Status: `in_progress`

Goal:
Turn the CPU interpreter into the correctness oracle for the backend-independent IR.

Scope:

- broaden interpreter dtype support beyond the current `i32` and `f32` slice
- add more scalar ops only when parser, validator, and interpreter all support them
- support richer indexing and loop patterns
- make interpreter errors precise and deterministic

Primary files:

- `lib/crux/ir.w`
- `lib/crux/ir_text.w`
- `lib/crux/cpu_interp.w`
- `test/ir_test.w`
- `test/ir_text_test.w`
- `test/cpu_interp_test.w`

Exit criteria:

- the interpreter can run a small but representative kernel set without special-case test construction
- every supported text-IR opcode is validated before dispatch
- unsupported ops fail at compile/validation time instead of surprising the runtime
- the CPU interpreter is clearly the reference execution path for future backend checks

## Milestone 3: Kernel Corpus on CPU

Status: `in_progress`

Goal:
Prove Crux can express and execute the first meaningful compute kernels on the CPU path.

Scope:

- add canonical kernel sources in repo form
- start with elementwise map, reduction, and a tiny matmul-shaped loop nest
- add correctness tests using real program compilation and dispatch
- keep kernels small and explicit; this is a substrate proof, not a tensor API yet

Completed so far:

- structured and text-IR elementwise add coverage
- text-IR scalar reduction coverage
- text-IR fused multiply-add coverage
- text-IR unary negate coverage
- text-IR compare/select coverage through a branchless ReLU-style pointwise kernel
- text-IR `min/max/clamp` coverage through scalar clamp and pointwise clip kernels
- `f32` coverage through scalar add and pointwise `fma` kernels
- structured-IR 2D `i32` matmul-shaped loop-nest coverage
- dedicated kernel-focused CPU test file in the suite

Primary files:

- `lib/crux/ir.w`
- `lib/crux/ir_text.w`
- `test/cpu_interp_test.w`
- new kernel-focused test files

Exit criteria:

- elementwise add works through both structured IR and text IR
- at least one reduction kernel runs correctly on CPU
- at least one small matrix-style kernel runs correctly on CPU
- the test suite can serve as a backend conformance baseline later

## Milestone 4: Backend Split and Metal Runtime Skeleton

Status: `pending`

Goal:
Separate the backend-independent surface from backend-specific execution and stand up a real Metal runtime path.

Scope:

- introduce backend-oriented module structure
- add Metal bridge/runtime files
- implement Metal devices, buffers, streams, events, and program handles
- preserve the CPU interpreter path as the reference oracle

Primary files:

- `lib/crux/backend/*`
- `lib/crux/device.w`
- `lib/crux/memory.w`
- `lib/crux/program.w`
- `lib/crux/stream.w`
- `runtime/crux_metal_bridge.m`

Exit criteria:

- CPU and Metal paths are structurally separate
- Metal device enumeration and basic resource management work
- stream/event/program lifetimes are real on Metal even before full kernel codegen

## Milestone 5: IR to Metal Codegen

Status: `pending`

Goal:
Run the first nontrivial Crux program on GPU and compare it against the CPU oracle.

Scope:

- lower the current structured IR subset to MSL
- compile and cache Metal pipelines
- bind views and metadata consistently with the byte-stride rules
- execute one kernel end to end and compare outputs to CPU

Primary files:

- `lib/crux/backend/metal_compiler.w`
- `lib/crux/program.w`
- `lib/crux/stream.w`
- `runtime/crux_metal_bridge.m`
- backend comparison tests

Exit criteria:

- one elementwise add kernel runs on Metal and matches CPU exactly
- compile failures surface as `CompileError`, not crashes or silent exits
- the first compile-cache path exists

## Milestone 6: Core Kernel Set

Status: `pending`

Goal:
Implement the small kernel basis needed for the future tensor layer and inference engine.

Scope:

- reductions
- matmul
- fused softmax
- attention-shaped kernels
- timing and performance plumbing where needed for iteration

Primary files:

- kernel sources/tests
- backend compiler/runtime files
- stream/event timing support

Exit criteria:

- the core validation programs from the design run on CPU and Metal
- backend results are checked against CPU for correctness
- compile cache and execution timing are usable enough for iteration

## Milestone 7: Handoff Surface for Weld

Status: `pending`

Goal:
Stop growing substrate concerns and expose a stable base for the future tensor and inference layers.

Scope:

- tighten API docs and examples
- confirm which parts of the substrate are stable
- keep tensor concerns out of Crux
- leave model and graph concerns for Weld

Exit criteria:

- Crux has a stable substrate contract
- the next layer can build tensors, autograd, and model execution without changing core runtime assumptions

## Immediate Next Sequence

The next three concrete implementation passes should be:

1. Finish the remaining `Milestone 1` substrate gaps around metadata, reclamation semantics, and runtime polish.
2. Push `Milestone 2` further by broadening the validated interpreter subset in lockstep across parser, validator, and executor.
3. Expand `Milestone 3` beyond the first add/reduce/matmul slice with more canonical kernels and stronger backend-conformance checks.

## Rules for Execution

- Do not work around compiler bugs in Crux implementation code.
- If a compiler defect blocks a clean implementation step, reduce it, file it against `QuixiAI/with`, and pause.
- Keep the CPU interpreter as the correctness oracle while new backends are added.
