# Crux Milestone Board

This board tracks the implementation sequence for Crux based on the
design in `docs/crux-design-v2.md`, the practical constraints in
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
- v2 `ProgramSource { ir, aux, strings, entry }` plus explicit-grid dispatch
- program records with parameter signatures derived from IR header instructions
- text IR parsing into self-contained v2 `ProgramSource` values
- `Param` and `SpecConstant` IR header support backed by the `ProgramSource.strings` pool
- IR validation for the current supported subset
- `memory_from_ptr`, `view_byte_range`, and `view_canonicalize`
- `Result`-returning copy, copy-bytes, fill, and dispatch APIs aligned with v2
- CPU interpreter for `const`, `load`, `store`, `add/sub/mul/div/mod/add_sat/sub_sat/and/or/xor/shl/shr/fma/neg/abs/not/popcount/clz/ctz/exp/log/log2/sin/cos/tanh/sqrt/rsqrt/floor/ceil/round/cast`, `loop`, `parallel`, and block markers
- CPU interpreter for `eq/ne/lt/gt/le/ge/select` over the current scalar subset
- CPU interpreter for `min/max/clamp` over the current scalar subset
- current scalar dtype support broadened beyond `i32/f32/f64` to include `i64` and `u32` for the current integer subset
- dispatch-side validation for missing, extra, duplicate, dtype-mismatched, rank-mismatched, and broadcast-write bindings
- end-to-end tests for scalar add and 1D `i32` parallel add
- first kernel-corpus tests for scalar reduction and a tiny 2D matmul-shaped loop nest on CPU
- end-to-end text-IR `fma` coverage in the CPU kernel suite
- end-to-end text-IR `neg` coverage in the CPU kernel suite
- end-to-end text-IR `abs` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `mod` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `add_sat/sub_sat` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `floor` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `ceil` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `round` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `sqrt` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `rsqrt` coverage through scalar and pointwise CPU tests
- end-to-end text-IR bitwise `and/or/xor/not/shl/shr` coverage through scalar and pointwise CPU tests
- end-to-end text-IR bitcount `popcount/clz/ctz` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `exp/log/log2` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `sin/cos/tanh` coverage through scalar and pointwise CPU tests
- end-to-end text-IR `cast` coverage through scalar and pointwise CPU tests
- end-to-end text-IR compare/select coverage in the CPU kernel suite
- end-to-end text-IR `min/max/clamp` coverage across the CPU interpreter and kernel suites
- end-to-end `f32` and `f64` coverage for scalar add and pointwise `fma`

Still missing from the baseline:

- optional `dispatch_auto`
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

## Milestone 0.5: V2 API Realignment

Status: `done`

Goal:
Realign the live Crux codebase with the new v2 design surface before pushing deeper into backend and kernel breadth.

Scope:

- change `compile` from device-handle based compilation to `compile(DeviceInfo, ProgramSource)`
- change `ProgramSource` to the v2 self-contained form: `ir`, `strings`, and `entry`
- move parameter declarations and spec constants into IR header instructions (`Param`, `SpecConstant`)
- derive `ProgramSig` from IR header instructions instead of the current side-channel source shape
- move text IR to a parse utility path instead of a direct `compile(..., ir_text)` input path
- change dispatch to the explicit-grid surface and add `dispatch_auto` only as a helper if needed
- change `copy`, `copy_bytes`, and `fill` to `Result[*mut Event, SubstrateError]`
- implement `memory_from_ptr`, `view_byte_range`, and `view_canonicalize`

Primary files:

- `lib/crux/core.w`
- `lib/crux/view.w`
- `lib/crux/memory.w`
- `lib/crux/program.w`
- `lib/crux/stream.w`
- `lib/crux/ir.w`
- `lib/crux/ir_text.w`
- `test/*`
- `README.md`

Exit criteria:

- the public API matches the v2 design for compile, dispatch, data movement, and memory interop
- the IR has a header model for `Param` and `SpecConstant` backed by the `ProgramSource.strings` pool
- `program_sig(prog)` is derived from compiled IR header instructions
- all existing CPU-path tests are updated to the new surface and pass
- no repo docs still describe the old `compile(device, source)` or implicit-grid dispatch model as the target design

## Milestone 1: CPU Substrate Completion

Status: `done`

Goal:
Make the non-GPU substrate complete enough that CPU execution is a trustworthy reference path.

Completed so far:

- CPU copy and fill APIs
- CPU arena create/destroy/alloc/reset/used
- tests for copy/fill round-trips, broadcast-write rejection, float bit fills, and arena allocation/reset
- borrowed-memory interop through `memory_from_ptr`
- `view_byte_range` / `view_canonicalize` wired into runtime validation and copy-path selection
- out-of-range view rejection for dispatch, copy, and fill
- memmove-safe overlapping `copy_bytes` behavior on CPU
- end-to-end borrowed-memory dispatch coverage and additional negative runtime tests

Scope:

- implement borrowed-memory interop through `memory_from_ptr`
- use `view_byte_range` / `view_canonicalize` for alias and copy-path validation
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
- borrowed memory works with the same copy/dispatch/view paths as owned memory
- arena allocation works on CPU with deterministic behavior
- stream dispatch validation catches contract violations before interpreter execution
- tests cover positive and negative cases for memory and stream semantics

## Milestone 2: CPU Reference Executor

Status: `done`

Goal:
Turn the CPU interpreter into the correctness oracle for the backend-independent IR.

Scope:

- broaden interpreter dtype support beyond the current `i32`, `f32`, and `f64` slice
- validate and execute the v2 IR header model cleanly before kernel execution begins
- add more scalar ops only when parser, validator, and interpreter all support them
- support richer indexing and loop patterns
- make interpreter errors precise and deterministic

Completed so far:

- validator now rejects unsupported current-interpreter scalar dtypes at compile time for load/store/arithmetic instead of deferring failure to runtime
- `abs` is implemented across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `mod` is implemented as an integer-only op across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `floor` is implemented as a float-only op across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `ceil` is implemented as a float-only op across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `round` is implemented as a float-only op across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `sqrt` is implemented as a float-only op across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `rsqrt` is implemented as a float-only op across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- integer-only bitwise `and/or/xor/not/shl/shr` are implemented across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- integer-only bitcount `popcount/clz/ctz` is implemented across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- float `exp/log/log2` are implemented across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- float `sin/cos/tanh` are implemented across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `cast` is implemented across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- `add_sat/sub_sat` are implemented as integer-only ops across parser, validator, interpreter, scalar tests, and a pointwise CPU kernel path
- v2 `SpecConstant` headers are parsed, validated, ignored cleanly by execution, and covered in parser/validator/interpreter tests
- nested text-IR loops with multi-rank `@loop` index tuples are covered through parser tests and a real 2D CPU dispatch
- dtype support is broadened beyond the original `i32/f32/f64` slice with validated and executed `i64` add and `u32` bitwise paths in scalar and pointwise tests
- text-IR loop bounds and index tuples now validate as `i32/u32` only, so unsupported index dtypes fail at compile time instead of surprising the runtime

Primary files:

- `lib/crux/ir.w`
- `lib/crux/ir_text.w`
- `lib/crux/cpu_interp.w`
- `test/ir_test.w`
- `test/ir_text_test.w`
- `test/cpu_interp_test.w`

Exit criteria:

- the interpreter can run a small but representative kernel set without special-case test construction
- the interpreter cleanly ignores or pre-processes IR header instructions after signature extraction
- every supported text-IR opcode is validated before dispatch
- unsupported ops fail at compile/validation time instead of surprising the runtime
- the CPU interpreter is clearly the reference execution path for future backend checks

## Milestone 3: Kernel Corpus on CPU

Status: `done`

Goal:
Prove Crux can express and execute a reusable baseline kernel corpus on the CPU path.

Scope:

- add canonical kernel sources in repo form through a public `crux.kernels` module
- cover map, in-place map, reduction, data-layout, and matrix-style kernels
- exercise contiguous, non-contiguous, and broadcast-read views
- add correctness tests using real program compilation and dispatch
- keep kernels small and explicit; this is a substrate proof, not a tensor API yet

Completed so far:

- public `crux.kernels` `ProgramSource` builders for canonical map, in-place map, reduction, transpose, and matmul kernels
- builder-backed `i32` 1D map coverage plus text-IR map parsing baseline
- builder-backed `f32` 2D map coverage over contiguous, transposed, and broadcast-read views
- builder-backed legal `InOut` in-place map coverage
- builder-backed scalar reduction, row-wise sum reduction, and row-wise max reduction coverage
- builder-backed integer and float matrix-style kernels
- builder-backed transpose coverage plus text-IR data-layout/parser baselines
- broad pointwise text-IR CPU coverage retained for the current supported opcode subset
- dedicated kernel-source signature regression and kernel-focused CPU execution tests in the suite

Primary files:

- `lib/crux/kernels.w`
- `test/kernel_source_test.w`
- `test/kernel_cpu_test.w`

Exit criteria:

- a public `crux.kernels` module exists with canonical `ProgramSource` builders
- the corpus includes map, in-place map, reduction, data-layout, and matrix-style kernels
- the corpus is exercised against contiguous, non-contiguous, and broadcast-read views
- at least one integer matrix-style kernel and one float matrix-style kernel run end to end
- the parser path still has map, reduction, and data-layout fixtures through `parse_ir_text`
- the CPU kernel suite can serve as a future backend conformance baseline

## Milestone 4: Backend Split and Metal Runtime Skeleton

Status: `pending`

Goal:
Separate the backend-independent surface from backend-specific execution and stand up a real Metal runtime path.

Scope:

- introduce backend-oriented module structure
- add Metal bridge/runtime files
- add the backend-side device registry needed for `compile(DeviceInfo, ...)`
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
- lower `Param` and `SpecConstant` headers into real kernel parameter / constant declarations
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

1. Push `Milestone 2` and `Milestone 3` further by broadening the validated interpreter subset and kernel corpus in lockstep.
2. Keep `Milestone 1` stable by extending runtime regressions whenever new CPU features stress copy, fill, aliasing, or borrowed-memory behavior.
3. Start `Milestone 4` only after the CPU oracle and CPU kernel corpus feel stable enough to serve as backend conformance tests.

## Rules for Execution

- Do not work around compiler bugs in Crux implementation code.
- If a compiler defect blocks a clean implementation step, reduce it, file it against `QuixiAI/with`, and pause.
- Keep the CPU interpreter as the correctness oracle while new backends are added.
