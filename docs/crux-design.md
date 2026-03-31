# Crux: A Compute Foundation for With (v2)

**Goal:** Replace JAX, PyTorch, tinygrad at the lowest layer.
Build upon this to replace vLLM with a hardware-agnostic inference engine.

**Design principle:** Follow the seams of the hardware, not the seams of ML.

---

## Part 1: The Six Primitives

```
Device   — a thing with memory and compute
Memory   — contiguous bytes on a device
View     — typed lens over memory (shape, strides, dtype)
Program  — compiled computation (immutable after compile)
Stream   — ordered command queue on a device
Event    — observable completion / dependency token
```

Everything above — tensors, gradients, attention, models — is built
from these six. The substrate does not know what a neural network is.

---

## Part 2: Type Definitions

### Core types (opaque handles)

```
type Device = opaque
type Memory = opaque
type Program = opaque
type Stream = opaque
type Event = opaque
type Arena = opaque
```

### View (value type, stack-allocated)

```
type View = {
    memory: *mut Memory,
    offset: usize,
    shape: Shape,
    strides: Strides,
    dtype: DType,
}

type Shape = {
    dims: [usize; 8],
    rank: i32,
}

type Strides = {
    elems: [isize; 8],
    rank: i32,
}
```

All layout math is defined in bytes. Strides are byte strides.
DType is purely interpretive — it tells the program how to read
the bytes, but the view and memory layers operate on raw bytes.

```
impl Shape:
    fn elem_count(self: &Self) -> usize:
        var n: usize = 1
        for i in 0..self.rank:
            n = n * self.dims[i]
        n

    fn is_scalar(self: &Self) -> bool:
        self.rank == 0

impl Strides:
    fn is_contiguous(self: &Self, shape: Shape) -> bool:
        var stride: isize = dtype_size(dtype) as isize
        var i = shape.rank - 1
        while i >= 0:
            if self.elems[i] != stride:
                return false
            stride = stride * shape.dims[i] as isize
            i = i - 1
        true

    fn is_broadcasted(self: &Self) -> bool:
        for i in 0..self.rank:
            if self.elems[i] == 0:
                return true
        false
```

### Data types

```
enum DType =
    | Int8 | Int16 | Int32 | Int64
    | UInt8 | UInt16 | UInt32 | UInt64
    | Float16 | Float32 | Float64
    | BFloat16

fn dtype_size(d: DType) -> usize:
    match d:
        Int8 | UInt8 -> 1
        Int16 | UInt16 | Float16 | BFloat16 -> 2
        Int32 | UInt32 | Float32 -> 4
        Int64 | UInt64 | Float64 -> 8
```

### Scalar (tagged value)

```
type Scalar = {
    dtype: DType,
    bits: u64,
}

fn scalar_i32(v: i32) -> Scalar:
    Scalar { dtype: DType.Int32, bits: transmute_i32_to_u64(v) }

fn scalar_f32(v: f32) -> Scalar:
    Scalar { dtype: DType.Float32, bits: transmute_f32_to_u64(v) }

fn scalar_f64(v: f64) -> Scalar:
    Scalar { dtype: DType.Float64, bits: transmute_f64_to_u64(v) }
```

v1 used an untagged union, relying on callers to keep dtype in
sync with the active union member. Tagged representation prevents
silent data corruption when scalars flow through spec constants,
fill operations, and constant descriptors.

### Device info

```
enum DeviceKind = CPU | GPU | Accelerator

type DeviceInfo = {
    name: str,
    kind: DeviceKind,
    memory_total: usize,
    memory_available: usize,
    max_workgroup_size: i32,
    max_grid_dims: [usize; 3],
    max_shared_memory: usize,
    memory_alignment: usize,
    memory_bandwidth_gbps: f64,
    preferred_vector_width: i32,
    subgroup_size: i32,
    unified_memory: bool,
}
```

### Program signature

```
enum ParamMode = In | Out | InOut | Scratch

type ParamDesc = {
    name: str,
    mode: ParamMode,
    rank: i32,
    dtype: DType,
}

type ProgramSig = {
    params: Vec[ParamDesc],
}
```

### Bindings (named dispatch)

```
type Bindings = {
    entries: Vec[BindEntry],
}

type BindEntry = {
    name: str,
    view: View,
}

fn bind(name: str, view: View) -> BindEntry:
    BindEntry { name, view }

fn bindings_from(entries: Vec[BindEntry]) -> Bindings:
    Bindings { entries }
```

### Errors

```
enum SubstrateError =
    | OutOfMemory
    | DeviceLost
    | CompileError(str)
    | ShapeMismatch(str)
    | DTypeMismatch(str)
    | InvalidView(str)
    | StreamError(str)
    | GridExceedsDevice(str)
    | BroadcastWriteViolation
    | Unsupported(str)
```

---

## Part 3: The API

### Device

```
fn devices() -> Vec[*mut Device]
fn device_info(d: *mut Device) -> DeviceInfo
fn default_device() -> *mut Device
```

### Memory

```
fn alloc(device: *mut Device, size: usize) -> Result[*mut Memory, SubstrateError]
fn free(mem: *mut Memory)
fn free_after(stream: *mut Stream, mem: *mut Memory)

fn memory_size(mem: *mut Memory) -> usize
fn memory_device(mem: *mut Memory) -> *mut Device
fn memory_ptr(mem: *mut Memory) -> *mut u8   // CPU-accessible only
```

**Lifetime rules:**
- `free` is immediate. Caller guarantees no pending operations
  reference this memory. Violation is UB.
- `free_after` enqueues deallocation on the stream. Memory is freed
  after all preceding operations on that stream complete.
- The substrate does not reference-count.

### Memory interop

```
fn memory_from_ptr(device: *mut Device, ptr: *mut u8, size: usize) -> *mut Memory
```

Returns a Memory handle that wraps an externally-owned pointer.
The internal Memory representation carries a `borrowed: bool` flag.
`free` is a no-op on borrowed memory — the caller owns the lifetime.

Required for zero-copy integration: mapped file I/O, another
library's allocation, pre-allocated CUDA/Metal buffers. Without
this, every integration requires a copy.

### Arena

```
fn arena_create(device: *mut Device, size: usize) -> Result[*mut Arena, SubstrateError]
fn arena_destroy(arena: *mut Arena)
fn arena_alloc(arena: *mut Arena, size: usize, align: usize) -> Result[*mut Memory, SubstrateError]
fn arena_alloc_view(arena: *mut Arena, shape: Shape, dtype: DType) -> Result[View, SubstrateError]
fn arena_reset(arena: *mut Arena)
fn arena_used(arena: *mut Arena) -> usize
```

Bump allocator on device memory. `arena_reset` reclaims all memory.
Common inference pattern: allocate arena per request, dispatch all
operations, reset when done. Zero per-object overhead.

`arena_alloc_view` is the common path — allocates aligned memory
and returns a contiguous View in one call, avoiding the separate
alloc-then-construct-view step.

### View

```
fn view(mem: *mut Memory, desc: ViewDesc) -> Result[View, SubstrateError]
fn view_contiguous(mem: *mut Memory, shape: Shape, dtype: DType) -> View
fn view_slice(v: View, dim: i32, start: usize, end: usize) -> View
fn view_transpose(v: View, dim0: i32, dim1: i32) -> View
fn view_reshape(v: View, shape: Shape) -> Result[View, SubstrateError]
fn view_broadcast(v: View, shape: Shape) -> Result[View, SubstrateError]
fn view_permute(v: View, order: [i32; 8]) -> View
fn view_expand(v: View, dim: i32, size: usize) -> View

fn view_is_contiguous(v: View) -> bool
fn view_is_broadcasted(v: View) -> bool
fn view_elem_count(v: View) -> usize
fn view_byte_size(v: View) -> usize
fn view_offset_of(v: View, indices: [usize; 8]) -> usize
fn view_byte_range(v: View) -> (usize, usize)
fn view_canonicalize(v: View) -> View
```

Views are value types. Creating a view never touches the device.
Views do not own memory. Multiple views may alias the same memory.

**`view_byte_range`** returns `(min_byte_offset, max_byte_offset)` —
the actual memory span touched by a strided view. This is NOT
`offset + elem_count * dtype_size`, which is wrong for non-contiguous
views (a transposed view's elements can span more bytes than that).
Computed by walking strides to find the minimum and maximum byte
offsets reachable from the view's index space. Required for:
- Alias validation at dispatch time
- Bounds-checking views against Memory size
- Safe cross-device copies of non-contiguous views

**`view_canonicalize`** collapses contiguous dimensions, normalizes
strides, and detects fast contiguous layout. Used internally by
backends to select optimized copy/dispatch paths.

### Data movement

```
fn copy(stream: *mut Stream, src: View, dst: View) -> *mut Event
fn copy_bytes(stream: *mut Stream,
              src: *mut Memory, src_offset: usize,
              dst: *mut Memory, dst_offset: usize,
              size: usize) -> *mut Event
fn fill(stream: *mut Stream, dst: View, value: Scalar) -> *mut Event
```

`copy` handles cross-device transfers transparently. Requires
matching dtype and element count. `copy_bytes` is raw.
`fill` is typed — the Scalar's tagged dtype must match the view's
dtype. Mismatch is an error, not silent reinterpretation.

### Program

```
fn compile(device_info: DeviceInfo, source: ProgramSource) -> Result[*mut Program, SubstrateError]
fn program_sig(prog: *mut Program) -> ProgramSig
fn program_destroy(prog: *mut Program)

type ProgramSource = {
    ir: Vec[IRInst],
    entry: str,
}
```

**`compile` takes `DeviceInfo`, not `*mut Device`.** The compilation
process only needs device capabilities (subgroup size, max shared
memory, max workgroup size), not a live device handle. This enables:
- Offline compilation (compile before device is acquired)
- Compilation on a different thread than the device owner
- Caching across devices with identical capabilities

Backends may internally require device handles for vendor compiler
APIs (e.g., MTLDevice for Metal library creation). This is handled
internally — the backend receives DeviceInfo and resolves the
device handle from its own state.

**Spec constants** are declared as IR instructions (see Part 5),
not as a separate sidecar. This makes the IR self-contained — a
single `Vec[IRInst]` fully describes the program.

**Compilation caching:** Per-process, keyed by
`hash(device_info.name, ir_bytes)`. O(1) lookup.

### Stream and execution

```
fn stream_create(device: *mut Device) -> *mut Stream
fn stream_destroy(stream: *mut Stream)
fn stream_sync(stream: *mut Stream)

fn dispatch(stream: *mut Stream, prog: *mut Program,
            grid: [usize; 3],
            bindings: Bindings) -> Result[*mut Event, SubstrateError]

fn event_wait(event: *mut Event)
fn event_is_done(event: *mut Event) -> bool
fn event_elapsed(start: *mut Event, end: *mut Event) -> f64
fn event_destroy(event: *mut Event)
```

Single `dispatch` function with explicit grid. v1 had both `dispatch`
and `dispatch_grid` — the only difference was whether grid was
explicit. Two functions that do almost the same thing is unnecessary
surface area at the substrate level.

If a convenience path is needed for programs with fixed grids:
```
fn dispatch_auto(stream: *mut Stream, prog: *mut Program,
                 bindings: Bindings) -> Result[*mut Event, SubstrateError]
```
Reads the grid from program metadata. This is a helper, not a
separate API concept.

`dispatch` returns Result — it fails on binding mismatch, dtype
mismatch, aliasing violation, or grid exceeding device limits. No
silent corruption.

---

## Part 4: Correctness Rules

### 4.1 Happens-Before

Three rules define the entire synchronization model:

**Rule 1: Stream ordering.**
Within a stream, operations execute in submission order. If A is
submitted before B on the same stream, A completes before B begins.
All writes from A are visible to B.

**Rule 2: Event ordering.**
`event_wait(e)` establishes happens-before from the operation that
produced `e` to all subsequent operations on the calling stream.

**Rule 3: Stream sync.**
`stream_sync(s)` establishes happens-before for all prior operations
on stream `s`. Equivalent to waiting on every event produced by `s`.

**Everything else is a data race.** No implicit synchronization.

### 4.2 Broadcast Views Are Read-Only

A broadcasted view (any dimension with stride = 0) is read-only.

- `copy(stream, src, dst)` where dst is broadcasted → `BroadcastWriteViolation` error
- `fill(stream, dst, value)` where dst is broadcasted → `BroadcastWriteViolation` error
- `dispatch` where an `Out` or `InOut` binding is broadcasted → `BroadcastWriteViolation` error
- `load(broadcasted_view, indices)` in IR → legal
- `store(broadcasted_view, indices, value)` in IR → UB (not validated at dispatch, UB at runtime)

The substrate validates at dispatch time for bindings. Within a
program, store-to-broadcast is the programmer's responsibility
(the compiler does not insert checks inside kernels).

### 4.3 Aliasing Rules Within a Dispatch

Default mode (strict):

- **Overlapping reads:** Legal. Multiple `In` bindings may alias.
- **Read + write overlap:** `In` and `Out`/`InOut` bindings that
  alias the same memory region are UB unless the program guarantees
  non-overlapping access at the index level.
- **Write + write overlap:** Two `Out` bindings aliasing the same
  region are UB unless they write to provably disjoint indices.
- **`InOut` self-alias:** A single `InOut` binding reading and
  writing the same view is legal — the program controls ordering.
- **`Scratch` aliasing:** `Scratch` bindings may alias anything.
  The substrate makes no assumptions about scratch lifetime.

Alias validation uses `view_byte_range` to detect overlapping
memory regions at dispatch time.

Future extension: explicit aliasing mode with `may_alias`/`noalias`
flags for cases where the programmer knows more than the substrate.

Across dispatches on the same stream: happens-before (Rule 1)
guarantees all writes are visible before the next dispatch reads.

Across streams: happens-before via events (Rule 2) or data race.

### 4.4 Grid Limits

`dispatch` fails with `GridExceedsDevice` if any grid dimension
exceeds `device_info.max_grid_dims[i]`. The substrate does not
silently clamp or split. Grid decomposition is the framework layer's
responsibility.

### 4.5 Subgroup Semantics

A subgroup is a fixed-width lockstep execution unit. Its size is
determined at compile time per device, available as
`device_info.subgroup_size`.

**Guarantees:**
- All threads in a subgroup execute the same instruction
  simultaneously (lockstep).
- No explicit synchronization is needed within a subgroup —
  `barrier()` is a workgroup-level operation, not subgroup.
- Divergent control flow within a subgroup causes masked execution
  (inactive lanes are masked, not stalled).

**Portability rule:**
Programs using `parallel[subgroup]` must not assume a specific
subgroup size. The loop range is `0..subgroup_size`, where
`subgroup_size` is a compile-time constant provided by the substrate.

```
// Correct: range from substrate constant
parallel[subgroup] lane in 0..SUBGROUP_SIZE:
    ...

// Incorrect: hardcoded warp size
parallel[subgroup] lane in 0..32:  // WRONG — breaks on AMD (64)
    ...
```

**Size by backend:**
- CUDA: 32 (warp)
- Metal: 32 (SIMD group, but query `device_info.subgroup_size`)
- AMD/HIP: 64 (wavefront)
- CPU: `preferred_vector_width` (4, 8, 16 depending on ISA)
- Vulkan: variable (query at runtime)

---

## Part 5: The Program IR

### Design goals

- Compilable in microseconds
- Expressive enough for dense linear algebra, reductions, scans,
  elementwise ops, attention, quantized matmul, KV cache updates
- Lowerable to Metal MSL, CUDA PTX, HIP, CPU SIMD, Vulkan SPIR-V
- No graph. Structured loops and memory access.

### Value model

Each instruction that produces a value writes to a destination
virtual register specified in `d0`. Values are referenced by
non-negative integers (virtual register numbers). Parameters are
referenced by negative integers.

```
Value referencing:
  >= 0  → virtual register (produced by an instruction)
  < 0   → parameter (e.g., -1 = first param, -2 = second)
```

A value is live from its definition (`d0` of the producing
instruction) until its last use. The IR is value-numbered and
SSA-like — each virtual register is defined exactly once.

This is not full SSA (no phi nodes, no explicit def-use chains),
but it is sufficient for:
- Liveness analysis
- Register allocation in the backend
- Backend lowering without reconstructing dataflow from
  sequential load/store patterns

The flash attention kernel generates ~15 live intermediates
(score, new_max, correction, inv_sum, acc[D], etc.). Without
explicit value numbering, the MSL/PTX emitter would have to
invent variable names and do its own liveness analysis. Virtual
registers make this the IR's responsibility, where it belongs.

### Execution space model

Three levels of parallelism mapping to hardware hierarchy:

```
grid        — independent work groups
workgroup   — cooperative threads with shared memory
subgroup    — lockstep lanes
```

Surface syntax:
```
parallel i in 0..N                    // compiler chooses level
parallel[grid] bi in 0..M/TILE       // one workgroup per iteration
parallel[workgroup] ti in 0..TILE    // one thread per iteration
parallel[subgroup] lane in 0..SG     // one lane per iteration
```

Bare `parallel` mapping heuristic:
- Outermost large range → grid
- Inner range within grid body → workgroup
- Innermost small range → sequential or subgroup

On CPU:
- `parallel[grid]` → thread pool tasks
- `parallel[workgroup]` → sequential within task
- `parallel[subgroup]` → SIMD lanes (auto-vectorization)

### Operations

```
// Spec constants (must appear before all other instructions)
spec_constant(name, dtype, default_value)

// Memory access
load(view, indices) -> scalar
store(view, indices, value)

// Scalar compute
add, sub, mul, div, mod, neg
fma(a, b, c)
add_sat, sub_sat

// Comparison
eq, ne, lt, gt, le, ge
min, max, clamp
select(cond, true_val, false_val)

// Math (float)
exp, log, log2, sqrt, rsqrt
sin, cos, tanh
abs, floor, ceil, round

// Bitwise
and, or, xor, not, shl, shr
popcount, clz, ctz

// Cast
cast(value, target_dtype)

// Control flow
loop(var, range, body)
parallel(var, range, body)
parallel[grid](var, range, body)
parallel[workgroup](var, range, body)
parallel[subgroup](var, range, body)
if(cond, then, else)

// Reduction (named op, parallel-safe)
reduce[sum](var, range, expr)
reduce[max](var, range, expr)
reduce[min](var, range, expr)
reduce[prod](var, range, expr)

// Memory declaration
local(name, shape, dtype)       // workgroup-shared
private(name, shape, dtype)     // per-thread scratch

// Synchronization
barrier()                       // workgroup barrier

// Hints (compiler may ignore)
@[tile_size(N)]
@[unroll(N)]
@[cache(workgroup)]
```

### Spec constant semantics

Spec constants are declared as IR instructions at the top of the
program, before any computation. They become compile-time constants
in the generated backend code.

```
spec_constant("TILE", DType.Int32, scalar_i32(16))
spec_constant("SUBGROUP_SIZE", DType.Int32, scalar_i32(32))
```

v1 stored spec constants as a separate `Vec[ConstantDesc]` sidecar
on `ProgramSource`. This meant the IR was not self-contained — you
needed two data structures to fully describe a program, and the
compiler had to cross-reference them during lowering. Making spec
constants IR instructions means:
- The IR is a single `Vec[IRInst]` that fully describes the program
- Hashing for compilation cache is `hash(ir_bytes)` — one blob
- Serialization/deserialization is one array, not two structures

### Reduction semantics

`reduce[op]` specifies an associative, commutative operation.
Backend chooses parallel strategy (tree, warp shuffle, atomic).

Identity elements:
- `sum` → 0
- `prod` → 1
- `max` → type minimum / -inf
- `min` → type maximum / +inf

Arbitrary reductions use sequential `loop` + accumulator.

### Memory spaces

```
global      — device memory (views point here)
local       — workgroup-shared (fast, small, explicit via `local`)
private     — per-thread registers/stack (explicit via `private`)
constant    — read-only, cached (In params with constant hint)
```

Shared memory is opt-in via `local` declaration. The compiler does
not infer shared memory placement. Explicit is predictable.

### IR internal representation

```
type IRInst = {
    op: IROp,
    dtype: DType,
    d0: i32,       // destination virtual register (or first operand)
    d1: i32,       // source operand 1
    d2: i32,       // source operand 2
    d3: i32,       // source operand 3 (fma, select, clamp)
}

enum IROp =
    // Spec constants
    | SpecConstant
    // Memory
    | Load | Store
    // Compute
    | Add | Sub | Mul | Div | Mod | Neg | FMA
    | AddSat | SubSat
    // Compare
    | Eq | Ne | Lt | Gt | Le | Ge | Min | Max | Clamp | Select
    // Math
    | Exp | Log | Log2 | Sqrt | Rsqrt
    | Sin | Cos | Tanh | Abs | Floor | Ceil | Round
    // Bitwise
    | And | Or | Xor | Not | Shl | Shr | Popcount | Clz | Ctz
    // Cast
    | Cast
    // Control
    | Loop | Parallel | ParallelGrid | ParallelWorkgroup | ParallelSubgroup
    | If | ReduceSum | ReduceMax | ReduceMin | ReduceProd
    // Memory decl
    | Local | Private
    // Sync
    | Barrier
    // Structure
    | BlockBegin | BlockEnd | Return
```

### IR text format

The IR text format exists for debugging, testing, and prototyping.
It is NOT an input to `compile`. To use text IR:

```
fn parse_ir_text(text: str) -> Result[Vec[IRInst], SubstrateError]
```

The caller converts text to structured IR, then passes structured
IR to `compile`. One input format into the compiler, one text
parser that lives outside the hot path.

---

## Part 6: Examples

### Elementwise add

```
program add(
    a: in [N] f32,
    b: in [N] f32,
    out: out [N] f32
):
    parallel i in 0..N:
        store(out, [i], load(a, [i]) + load(b, [i]))
```

### Matrix multiply (naive)

```
program matmul(
    a: in [M, K] f32,
    b: in [K, N] f32,
    out: out [M, N] f32
):
    parallel i in 0..M:
        parallel j in 0..N:
            var acc: f32 = 0.0
            for k in 0..K:
                acc = fma(load(a, [i, k]), load(b, [k, j]), acc)
            store(out, [i, j], acc)
```

### Tiled matrix multiply (explicit shared memory)

```
program matmul_tiled(
    a: in [M, K] f32,
    b: in [K, N] f32,
    out: out [M, N] f32
):
    local tile_a: [TILE, TILE] f32
    local tile_b: [TILE, TILE] f32

    parallel[grid] bi in 0..M/TILE:
        parallel[grid] bj in 0..N/TILE:
            var acc: [TILE, TILE] f32 = 0.0

            for bk in 0..K/TILE:
                parallel[workgroup] ti in 0..TILE:
                    parallel[workgroup] tj in 0..TILE:
                        store(tile_a, [ti, tj],
                              load(a, [bi*TILE+ti, bk*TILE+tj]))
                        store(tile_b, [ti, tj],
                              load(b, [bk*TILE+ti, bj*TILE+tj]))
                barrier()

                parallel[workgroup] ti in 0..TILE:
                    parallel[workgroup] tj in 0..TILE:
                        for tk in 0..TILE:
                            acc[ti][tj] = fma(
                                tile_a[ti][tk],
                                tile_b[tk][tj],
                                acc[ti][tj])
                barrier()

            parallel[workgroup] ti in 0..TILE:
                parallel[workgroup] tj in 0..TILE:
                    store(out, [bi*TILE+ti, bj*TILE+tj], acc[ti][tj])
```

### Fused softmax

```
program softmax(
    input: in [B, N] f32,
    output: out [B, N] f32
):
    parallel b in 0..B:
        let max_val = reduce[max](i, 0..N, load(input, [b, i]))

        var sum: f32 = 0.0
        for i in 0..N:
            let v = exp(load(input, [b, i]) - max_val)
            store(output, [b, i], v)
            sum = sum + v

        let inv_sum = 1.0 / sum
        for i in 0..N:
            store(output, [b, i], load(output, [b, i]) * inv_sum)
```

### Flash attention

```
program flash_attention(
    q: in [B, H, N, D] f32,
    k: in [B, H, S, D] f32,
    v: in [B, H, S, D] f32,
    out: out [B, H, N, D] f32,
    scale: constant f32
):
    local k_tile: [TILE_S, D] f32
    local v_tile: [TILE_S, D] f32

    parallel[grid] b in 0..B:
        parallel[grid] h in 0..H:
            parallel[workgroup] qi in 0..N:
                var max_val: f32 = -inf
                var sum: f32 = 0.0
                private acc: [D] f32 = 0.0

                for kv_block in 0..S/TILE_S:
                    parallel[workgroup] si in 0..TILE_S:
                        for d in 0..D:
                            store(k_tile, [si, d],
                                  load(k, [b, h, kv_block*TILE_S+si, d]))
                            store(v_tile, [si, d],
                                  load(v, [b, h, kv_block*TILE_S+si, d]))
                    barrier()

                    for si in 0..TILE_S:
                        var score: f32 = 0.0
                        for d in 0..D:
                            score = fma(load(q, [b, h, qi, d]),
                                       load(k_tile, [si, d]), score)
                        score = score * scale

                        let new_max = max(max_val, score)
                        let correction = exp(max_val - new_max)
                        sum = fma(sum, correction, exp(score - new_max))
                        for d in 0..D:
                            acc[d] = fma(acc[d], correction,
                                        exp(score - new_max) * load(v_tile, [si, d]))
                        max_val = new_max
                    barrier()

                let inv_sum = 1.0 / sum
                for d in 0..D:
                    store(out, [b, h, qi, d], acc[d] * inv_sum)
```

### Quantized matmul (INT8 × INT8 → INT32)

```
program quantized_matmul(
    a: in [M, K] i8,
    b: in [K, N] i8,
    out: out [M, N] i32,
    scale_a: in [M] f32,
    scale_b: in [N] f32
):
    parallel i in 0..M:
        parallel j in 0..N:
            var acc: i32 = 0
            for k in 0..K:
                acc = acc + cast(load(a, [i, k]), i32)
                         * cast(load(b, [k, j]), i32)
            store(out, [i, j], acc)
```

### KV cache update

```
program kv_cache_update(
    new_keys: in [B, H, 1, D] f32,
    new_vals: in [B, H, 1, D] f32,
    key_cache: inout [B, H, MAX_SEQ, D] f32,
    val_cache: inout [B, H, MAX_SEQ, D] f32,
    positions: in [B] i32
):
    parallel b in 0..B:
        let pos = cast(load(positions, [b]), usize)
        parallel h in 0..H:
            for d in 0..D:
                store(key_cache, [b, h, pos, d],
                      load(new_keys, [b, h, 0, d]))
                store(val_cache, [b, h, pos, d],
                      load(new_vals, [b, h, 0, d]))
```

All six use only the IR primitives. No new constructs needed.

---

## Part 7: Backend Architecture

### Backend trait

```
trait Backend =
    fn name(self: &Self) -> str
    fn create_device(self: &Self, index: i32) -> Result[*mut Device, SubstrateError]
    fn device_info(self: &Self, device: *mut Device) -> DeviceInfo
    fn alloc(self: &Self, device: *mut Device, size: usize) -> Result[*mut Memory, SubstrateError]
    fn free(self: &Self, mem: *mut Memory)
    fn compile(self: &Self, info: DeviceInfo, source: ProgramSource) -> Result[*mut Program, SubstrateError]
    fn dispatch(self: &Self, stream: *mut Stream, prog: *mut Program, grid: [usize; 3], bindings: Bindings) -> Result[*mut Event, SubstrateError]
    fn copy_bytes(self: &Self, stream: *mut Stream, src: *mut u8, dst: *mut u8, size: usize) -> *mut Event
    fn stream_create(self: &Self, device: *mut Device) -> *mut Stream
    fn stream_sync(self: &Self, stream: *mut Stream)
    fn event_wait(self: &Self, event: *mut Event)
    fn event_query(self: &Self, event: *mut Event) -> bool
    fn event_elapsed(self: &Self, start: *mut Event, end: *mut Event) -> f64
```

### Implementations (in order)

```
1. Metal    — IR → MSL → MTLLibrary → MTLComputePipelineState
2. CPU      — IR → With/LLVM. parallel → thread pool, subgroup → SIMD
3. CUDA     — IR → CUDA C → nvrtc → PTX (or direct PTX emission)
4. HIP      — Fork CUDA backend, swap API calls
5. Vulkan   — IR → SPIR-V
6. Gaudi    — IR → TPC-C
7. Hexagon  — IR → Hexagon SDK
```

### IR → MSL lowering (Metal)

```
parallel[grid] bi in 0..M/TILE
    → kernel void entry(
          uint2 gid [[threadgroup_position_in_grid]],
          uint2 tid [[thread_position_in_threadgroup]], ...)
      uint bi = gid.x;

parallel[workgroup] ti in 0..TILE
    → uint ti = tid.x;

parallel[subgroup] lane in 0..SUBGROUP_SIZE
    → uint lane = thread_index_in_simdgroup;

for k in 0..K
    → for (uint k = 0; k < K; k++) { ... }

load(a, [i, k])
    → a[i * a_stride_0 + k * a_stride_1]

store(out, [i, j], v)
    → out[i * out_stride_0 + j * out_stride_1] = v;

local tile: [N] f32
    → threadgroup float tile[N];

private acc: [D] f32
    → float acc[D];

barrier()
    → threadgroup_barrier(mem_flags::mem_threadgroup);

reduce[sum](i, 0..N, expr)
    → tree reduction in threadgroup memory + simd_shuffle_down

reduce[max](i, 0..N, expr)
    → same pattern with max instead of add
```

Strides passed as kernel arguments. Spec constants inlined as
`constant uint M = 4096;` etc.

### Future optimization path

v1: IR → MSL/CUDA C text → vendor compiler. Fast enough.
Long-term: direct AIR/PTX/SPIR-V emission for microsecond compile.

---

## Part 8: What Gets Built On Top

### Layer 1: Tensor library (Weld)

```
type Tensor = {
    storage: *mut Storage,
    view: View,
    grad_meta: *mut GradMeta,
}

fn tensor_from_data(device: *mut Device, data: *const u8, shape: Shape, dtype: DType) -> Tensor
fn tensor_zeros(device: *mut Device, shape: Shape, dtype: DType) -> Tensor
fn tensor_matmul(stream: *mut Stream, a: &Tensor, b: &Tensor) -> Tensor
fn tensor_softmax(stream: *mut Stream, input: &Tensor, dim: i32) -> Tensor
fn tensor_add(stream: *mut Stream, a: &Tensor, b: &Tensor) -> Tensor
fn tensor_to_cpu(stream: *mut Stream, t: &Tensor) -> Tensor
```

### Layer 2: Autograd (Weld)

```
type GradTensor = {
    tensor: Tensor,
    grad_fn: *mut GradFn,
    grad: Option[Tensor],
}

trait GradFn =
    fn backward(self: &Self, grad_output: &Tensor) -> Vec[Tensor]
```

### Layer 3: NN modules (Weld)

```
type Linear = { weight: Tensor, bias: Option[Tensor] }
type LayerNorm = { weight: Tensor, bias: Tensor, eps: f64 }
type Attention = { q_proj: Linear, k_proj: Linear, v_proj: Linear, o_proj: Linear }
type TransformerBlock = { attention: Attention, ffn: FFN, norm1: LayerNorm, norm2: LayerNorm }
```

### Layer 4: Inference engine (Weld)

```
type InferenceEngine = {
    model: Vec[TransformerBlock],
    kv_cache: KVCache,
    scheduler: BatchScheduler,
    tokenizer: Tokenizer,
}

fn generate(engine: *mut InferenceEngine, prompts: Vec[str],
            params: SamplingParams) -> Vec[str]
```

---

## Part 9: Implementation Plan

### Phase 1: Substrate core (8 sessions)

```
Session  Deliverable
──────── ────────────────────────────────────────────
  1      Type definitions: Shape, Strides, View, DType,
         Scalar (tagged), Bindings, SubstrateError.
         Shape/Strides methods. view_byte_range.
         view_canonicalize. Unit tests. Pure With.

  2      CPU backend: alloc/free via malloc,
         memory_from_ptr (borrowed flag), stream as
         sequential executor, event as done flag.
         dispatch validates bindings + alias checks.

  3      IR definition: IROp enum, IRInst struct with
         virtual register model. SpecConstant as IR
         instruction. parse_ir_text for testing. IR
         validation pass (value liveness, def-before-use).

  4      IR → CPU compiler: lower to With functions via
         LLVM. parallel → sequential. Virtual registers →
         local variables. First dispatch: elementwise add.

  5      Metal backend: MTLDevice, MTLBuffer, MTLCommand
         Queue. alloc/free/stream_sync working.

  6      IR → MSL compiler: emit shader strings, compile
         to MTLComputePipelineState. Virtual registers →
         MSL locals. First GPU dispatch: elementwise add
         on Metal.

  7      Cross-device copy: CPU↔GPU via MTLBuffer. fill
         (with Scalar dtype validation). Arena allocator
         with arena_alloc_view.

  8      Event timing, compilation cache (keyed by
         hash(device_info.name, ir_bytes)). Benchmark:
         elementwise add CPU vs Metal.
```

### Phase 2: Core programs (5 sessions)

```
Session  Deliverable
──────── ────────────────────────────────────────────
  9      Elementwise: add, mul, sub, div, neg, exp, log,
         tanh, relu, gelu, silu. Benchmark vs MPS.

 10      Reduction: sum, max, min, mean. Tree reduction
         in shared memory. Numerically stable.

 11      Matrix multiply: naive → tiled with shared memory.
         Benchmark vs MPSMatrixMultiplication.

 12      Fused: softmax, layer_norm, rms_norm. Single
         kernel per op.

 13      Data ops: transpose, concat, split, gather,
         scatter. Quantized matmul INT8.
```

### Phase 3–5

See Weld design doc for tensor library, transformer inference,
and scale-out phases.

---

## Part 10: Validation

The design is correct if and only if these six programs compile
and run on Metal using only the six primitives, with no additional
substrate concepts:

1. **Elementwise add** — memory, view, basic dispatch
2. **Matrix multiply** — parallel execution, shared memory, barriers
3. **Softmax** — reduction, online algorithms, multi-pass
4. **Flash attention** — tiling, barriers, online softmax, private memory
5. **Quantized matmul** — mixed dtype, casting
6. **KV cache update** — in-place mutation, dynamic indexing

If all six work cleanly, the abstraction is correct.

---

## Part 11: Design Decisions Log

| Decision | Rationale |
|---|---|
| No tensor primitive | Tensors mix ownership, view, and compute semantics. Separating them into Memory + View + Program is cleaner and more composable. |
| View is a value type | Zero-cost metadata. No allocation, no device touch. Multiple views alias safely. |
| Named bindings | Prevents positional fragility. Programs evolve without breaking dispatch sites. |
| Explicit shared memory | Compiler inference is fragile across backends. Explicit `local` is predictable. |
| Named reduction ops | Associativity and identity must be known for parallel correctness. Arbitrary reduction bodies are sequential fallback only. |
| Broadcast views read-only | stride=0 writes are physically aliased — every "element" maps to the same byte. Allowing writes would be silent corruption. |
| free_after on stream | Async reclamation without reference counting. The common safe path for device memory. |
| Arena as optional facility | Not every workload needs pooling, but inference workloads do. Arena is the right granularity — not per-object, not global GC. |
| arena_alloc_view | Common path is alloc + view construction. One call instead of two for the common case. |
| Grid exceeds limits → error | Silent clamping hides bugs. Grid decomposition belongs to the framework layer which has workload context. |
| Subgroup size from device | Hardcoding 32 breaks on AMD (64). Compile-time constant per device is the only portable approach. |
| Virtual registers in IR | Without value numbering, backend emitters reconstruct dataflow from sequential patterns. Flash attention has ~15 live intermediates — the IR should track them, not each backend independently. |
| Spec constants as IR instructions | Makes IR self-contained. One blob to hash, serialize, and compile. No sidecar cross-referencing. |
| Structured IR, no text input to compile | String parsing has no place in the hot path. Text parser is a separate utility function. |
| Dispatch returns Result | Fail loudly on mismatch. No silent corruption. |
| Single dispatch function with explicit grid | Two near-identical functions (dispatch vs dispatch_grid) is unnecessary surface area. Explicit grid is always known after compilation. |
| compile takes DeviceInfo, not Device | Enables offline compilation, cross-thread compilation, and caching across equivalent devices. |
| Tagged Scalar | Untagged union relies on callers keeping dtype in sync. Tagged representation prevents silent data corruption in fill, constants, and spec constants. |
| memory_from_ptr for interop | Zero-copy integration with external allocations. Without this, every integration requires a copy. |
| view_byte_range | Correct alias validation and bounds checking require knowing the actual byte span of a strided view, not just elem_count * dtype_size. |
| Byte-based strides | All layout math in bytes. DType is purely interpretive. Consistent, unambiguous. |
| Strict aliasing by default | Prevents silent data corruption. Explicit aliasing mode deferred to future extension. |