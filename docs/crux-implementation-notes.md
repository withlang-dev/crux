# Crux Implementation Notes (v2)

**Companion to:** `crux-design-v2.md`
**Audience:** The agent implementing this system in With.
**Purpose:** Practical details, API mappings, known constraints,
backend guidance, and implementation gotchas.

---

## Naming

The compute substrate is called **Crux**. The ML library built on
top is called **Weld**. All substrate types, files, and APIs use
the `crux_` prefix for C bridge functions and `Crux` prefix for
With types where disambiguation is needed.

```
lib/crux/              # substrate library root
lib/weld/              # ML library root (future)
crux_metal_bridge.c    # C bridge file naming
CruxDevice             # With type naming (when needed)
```

```
use crux          // substrate: memory, programs, streams, devices
use weld          // ML library: tensors, autograd, nn modules
use weld.serve    // inference engine: batching, KV cache, API
```

The core Crux types (`Device`, `Memory`, `View`, `Program`,
`Stream`, `Event`) do not get a `Crux` prefix in With code.

---

## Foundational Rules

These rules are implementation constraints, not suggestions.

### Rule 1: C bridge handles are pointer-backed

At the C bridge boundary, every opaque handle (`Device`, `Memory`,
`Program`, `Stream`, `Event`, `Arena`) is backed by a heap object.
The With-facing API uses opaque pointer types (`*mut Device`,
`*mut Memory`, etc.); any integer casts are bridge-internal only.

### Rule 2: Strides are bytes

All strides are byte strides throughout the substrate. Element
strides are derived only at backend/kernel argument boundaries.

### Rule 3: View offset arithmetic never multiplies by dtype size

Because strides are already in bytes, all offset computation is:

```
byte_offset = v.offset + sum(indices[i] * strides[i])
```

No additional `dtype_size` factor is ever applied in view math.

### Rule 4: Parameter refs are negative, virtual registers are non-negative

In instruction operands:

```
>= 0  -> virtual register
< 0   -> parameter index (-1 = first param, -2 = second, ...)
```

This rule is universal across validators, interpreters, and backends.

### Rule 5: Grid validation happens before backend dispatch

The API layer validates dispatch grids against device limits before
calling backend execution. Backends may revalidate defensively, but
the API layer is the authoritative contract.

### Rule 6: Shared memory is validated at compile time

Backends sum all `Local` declarations and reject programs that
exceed `max_shared_memory` with `CompileError`.

### Rule 7: Cache keys are capability-based

Compilation caches key on:

```
hash(backend_id, codegen_capabilities, ir_bytes, string_bytes)
```

not on device names or backend-instance identities.

### Rule 8: Collectives are primarily stream-level operations

The default collective API lives at the stream level. IR-level
collective ops are the backend-aware escape hatch, not the default
surface.

### Rule 9: Placement is about bytes, not tensor dimensions

Any memory placement or partitioning metadata describes byte regions.
Mapping tensor dimensions onto those regions is a higher-layer concern.

### Rule 10: Single-device programs stay portable

Programs that avoid explicit distributed constructs should run
unchanged on CPU, Metal, CUDA, HIP, or composite-device backends.

---

## File Structure

Expected target structure:

```
lib/crux/
├── core.w
├── errors.w
├── device.w
├── memory.w
├── view.w
├── program.w
├── stream.w
├── collective.w
├── ir.w
├── ir_text.w
├── backend/
│   ├── backend.w
│   ├── cpu.w
│   ├── cpu_compiler.w
│   ├── cpu_interp.w
│   ├── metal.w
│   └── metal_compiler.w
├── runtime/
│   └── crux_metal_bridge.m
└── test/
```

---

## 1. IR Value Model - Implementation

### Virtual register allocation

The IR uses a flat virtual register numbering scheme. Each instruction
that produces a value writes to `d0` as its destination register.
Backends lower virtual registers to local variables (MSL, CUDA C)
or SSA values (LLVM, SPIR-V).

```
Example: a[i] + b[i]

  IRInst { op: Load, d0: 0, d1: -1, d2: <index> }    // %0 = load(a, [i])
  IRInst { op: Load, d0: 1, d1: -2, d2: <index> }    // %1 = load(b, [i])
  IRInst { op: Add,  d0: 2, d1: 0, d2: 1 }           // %2 = add(%0, %1)
  IRInst { op: Store, d0: -3, d1: <index>, d2: 2 }   // store(out, [i], %2)
```

Parameter references:
- `-1` -> first parameter (a)
- `-2` -> second parameter (b)
- `-3` -> third parameter (out)

### Liveness analysis

A value is live from its definition (the instruction where it appears
as `d0`) until its last use (the last instruction that references it
in `d1`, `d2`, or `d3`). The backend performs a single reverse pass
to compute last-use points.

For the MSL/CUDA C backends (which emit text), liveness determines
when to declare variables and when temporaries can be reused. For
the LLVM backend, virtual registers map directly to SSA values and
LLVM handles allocation.

### Register pressure in complex kernels

Flash attention generates ~15 simultaneously live values:
`score`, `new_max`, `correction`, `inv_sum`, `max_val`, `sum`,
`acc[0..D]`, plus loop variables and tile indices.

The IR compiler must track this. Without virtual registers, the
MSL emitter would need to invent variable names and do its own
liveness analysis - duplicating work that belongs in the IR layer.

Backend strategy:
- **MSL/CUDA C:** Emit one local variable per virtual register.
  Rely on the vendor compiler (Metal shader compiler, nvcc) to
  do register allocation on the generated text. This is the v1
  approach and is sufficient.
- **LLVM (CPU):** Map virtual registers to LLVM SSA values directly.
  LLVM does full register allocation.
- **Future (direct PTX/AIR):** The IR compiler will need its own
  register allocator. Defer until direct emission is implemented.

### Instructions that don't produce values

`Store`, `Barrier`, `BlockBegin`, `BlockEnd` do not produce values.
Their `d0` field is unused (set to -1 by convention). The liveness
pass ignores them as producers.

---

## 2. IR Header Instructions - Implementation

### Param declarations

Param instructions declare kernel parameters. They must appear
at the top of the IR, after spec constants and before computation.

```
IRInst { op: Param, dtype: Float32, d0: 0, d1: 1, d2: 2 }
// d0=0 -> strings[0] = "a" (parameter name)
// d1=1 -> ParamMode.Out
// d2=2 -> rank 2
```

- `d0`: index into `ProgramSource.strings` (parameter name)
- `d1`: ParamMode enum value (0=In, 1=Out, 2=InOut, 3=Scratch)
- `d2`: tensor rank (0 for scalars)
- `dtype`: parameter dtype

During compilation, the backend collects all `Param` instructions
into a `ProgramSig`. `program_sig(prog)` returns this. At dispatch
time, each binding is validated against the signature:
- Binding name must match a declared parameter
- Binding view's dtype must match the parameter's dtype
- Binding view's rank must match the parameter's rank
- An `Out`/`InOut` binding on a parameter declared `In` is an error

### Spec constants

Spec constants are IR instructions with `op: SpecConstant`. They
must appear before `Param` instructions and before computation.

```
IRInst { op: SpecConstant, dtype: Int32, d0: 3, d1: 16, d2: 0 }
// d0=3 -> strings[3] = "TILE" (constant name)
// d1=16 -> default value low bits
// d2=0  -> default value high bits (zero for i32)
```

The `d0` field is an index into `ProgramSource.strings`. `d1`/`d2`
hold the default value bits (same encoding as Scalar.bits, split
across two i32 fields for the 64-bit case).

At compile time, the caller can override defaults by passing
actual values. The backend inlines spec constants as compile-time
constants in the generated code:

```
// MSL
constant uint TILE = 16;
constant uint SUBGROUP_SIZE = 32;

// CUDA
#define TILE 16
#define SUBGROUP_SIZE 32
```

### Cache key

The compilation cache key is
`hash(backend_id, codegen_capabilities, ir_bytes, string_bytes)`.

`codegen_capabilities` includes every DeviceInfo field that affects
code generation: `subgroup_size`, `max_shared_memory`,
`max_workgroup_size`, and `kind`. It excludes fields that don't
affect codegen (`memory_total`, `memory_available`,
`memory_bandwidth_gbps`). `device_info.name` is NOT part of the
key - two devices can share a name while differing in codegen-
relevant limits, and two devices with different names can produce
identical code if their capabilities match.

Since spec constants, param declarations, and the string pool are
all part of the hashed input, programs with different constant
values, different parameter signatures, or different names are
different cache entries.

---

## 3. View - Implementation Details

### Byte range computation

`view_byte_range(v: View) -> (usize, usize)` returns a half-open
byte range `[min_byte, end_byte)` - the actual memory span touched
by a strided view. `end_byte` is exclusive (one past the last byte
of the last element).

```
fn view_byte_range(v: View) -> (usize, usize):
    var min_off: isize = v.offset as isize
    var max_off: isize = v.offset as isize
    for i in 0..v.shape.rank:
        let stride = v.strides.elems[i]
        let extent = (v.shape.dims[i] - 1) as isize * stride
        if extent >= 0:
            max_off = max_off + extent
        else:
            min_off = min_off + extent
    // max_off is the offset of the last element's first byte;
    // add dtype_size to get the exclusive end
    (min_off as usize, (max_off + dtype_size(v.dtype) as isize) as usize)
```

Two views overlap if and only if their half-open byte ranges
intersect AND they reference the same Memory handle. This makes
alias checking a simple interval overlap test.

This handles negative strides (from transpose/flip), zero strides
(from broadcast), and non-contiguous layouts correctly.

### Canonicalization

`view_canonicalize(v: View) -> View` produces an equivalent view
with simplified metadata:

1. **Collapse contiguous dimensions.** If dimension `i` and `i+1`
   are contiguous (stride[i] == shape[i+1] * stride[i+1]), merge
   them into a single dimension with shape = shape[i] * shape[i+1].

2. **Normalize strides.** If shape[i] == 1, set stride[i] = 0
   (canonical form for degenerate dimensions).

3. **Detect fast contiguous.** If the result has rank 1 and
   stride[0] == dtype_size, set a "contiguous" flag that backends
   can use to select memcpy-style fast paths.

Canonicalization is used internally by `copy` and by the dispatch
alias checker. It is NOT applied automatically to user-created
views - the user's shape semantics are preserved.

### Contiguity

`view_is_contiguous(v: View) -> bool` checks whether the view
represents a dense, row-major layout with no gaps or reordering.
The check walks dimensions from innermost to outermost, verifying
that each stride equals the product of inner dimensions' sizes
times `dtype_size`.

This is more robust than `byte_range == elem_count * dtype_size`,
which would incorrectly return true for some non-contiguous views
where the byte range happens to match.

---

## 4. Memory Model - Implementation

### Borrowed memory

`memory_from_ptr` creates a Memory handle with `borrowed = true`.
Internal representation:

```
type MemoryInternal = {
    ptr: *mut u8,
    size: usize,
    device: *mut Device,
    borrowed: bool,
}
```

- `free` on borrowed memory is a no-op (no deallocation).
- `free_after` on borrowed memory is a no-op (enqueued but skips
  the actual deallocation step).
- `memory_ptr` works normally on borrowed memory.
- All other operations (copy, dispatch, view construction) work
  identically for borrowed and owned memory.

The caller is responsible for ensuring the external pointer remains
valid for the lifetime of the Memory handle and all operations that
reference it. Violation is UB, same as premature `free`.

### Stream ordering guarantees

Within a single stream:
- Operations execute in submission order.
- All writes from operation A are visible to operation B if A was
  submitted before B.
- This is the primary synchronization mechanism. Most programs
  use a single stream and rely entirely on this guarantee.

Cross-stream:
- No implicit ordering or visibility.
- `event_wait(e)` on stream S establishes happens-before from the
  operation that produced `e` to all subsequent operations on S.
- Without explicit synchronization, concurrent streams accessing
  the same memory are a data race. The substrate does not detect
  or prevent this.

### Aliasing validation at dispatch time

When `dispatch` is called, the substrate validates bindings:

1. Compute `view_byte_range` for every binding.
2. For every pair of bindings where at least one is `Out` or `InOut`:
   - If their byte ranges overlap and they reference the same Memory:
     - If strict mode (default): return `SubstrateError` describing
       the conflict.
     - Exception: a single `InOut` binding that reads and writes the
       same view is always legal.
3. All `In` bindings may freely overlap with each other.

This is O(n^2) in the number of bindings. For typical kernel
dispatches (2-6 bindings), this is negligible.

---

## 5. Compilation - Implementation

### DeviceInfo-based compilation

`compile` takes `DeviceInfo` rather than `*mut Device`. This means
the compiler only has access to device capabilities, not a live
device handle.

**Problem:** Some vendor APIs require a device handle to compile.
Metal needs an `MTLDevice` to create `MTLLibrary`. CUDA needs a
context for `nvrtcCompileProgram` (or a device for `cuModuleLoad`).

**Solution:** The Backend trait's `compile` method receives
`DeviceInfo` at the public API level. Internally, the backend
maintains a registry of device handles populated during
`create_device`. When compile is called:

1. Look up a compatible device handle from the backend's device
   registry. Match on codegen-relevant capabilities (subgroup_size,
   max_shared_memory, max_workgroup_size, kind), not on name.
2. Use the handle for vendor-specific compilation.
3. Return the compiled Program.

The public API contract is: "you only need DeviceInfo to compile."
The backend implementation detail is: "I secretly have the device
handle from when you created the device earlier."

**Compiled programs are capability-targeted.** A program compiled
for a given set of capabilities will work on any device with
compatible capabilities. If two Metal devices have the same
subgroup size and shared memory limits, the same compiled program
works on both.

If `compile` is called with a DeviceInfo that doesn't correspond
to any created device, the backend returns `CompileError`.

### Compilation cache

Per-process HashMap keyed by
`hash(backend_id, codegen_capabilities, ir_bytes, string_bytes)`.

- `codegen_capabilities` includes `subgroup_size`,
  `max_shared_memory`, `max_workgroup_size`, and `kind` - every
  field that affects generated code. It excludes fields like
  `memory_total` that don't affect codegen.
- `ir_bytes` + `string_bytes` cover instructions and the name
  pool. Different parameter signatures, constant values, or names
  produce different cache keys.
- `device_info.name` is NOT part of the key. Names are not a
  stable codegen identity.
- Cache stores compiled Program handles.
- Cache is NOT persistent across process restarts in v1.
- Thread safety: the cache is behind a mutex. Compile is not
  performance-critical (programs are compiled once).

### Future: persistent cache

Store `hash -> compiled_binary` on disk. Load compiled binaries
directly instead of recompiling. Requires stable binary formats
per backend (MTLBinaryArchive for Metal, cubin for CUDA).
Not v1.

---

## 6. Backend Implementation Guide

### What a backend must implement

Every function in the `Backend` trait, plus internal lowering from
IR to the backend's native representation.

### IR lowering (common pattern)

All backends follow the same structure:

1. **Walk IR instructions linearly.**
2. **Collect `Param` declarations** into the program signature.
   Generate kernel parameter declarations from them.
3. **Emit spec constants** as compile-time constants.
4. **Map virtual registers** to local variables (text backends)
   or SSA values (binary backends).
4. **Map parallel levels** to backend-specific constructs:
   - `ParallelGrid` -> Metal threadgroup position, CUDA blockIdx
   - `ParallelWorkgroup` -> Metal thread position, CUDA threadIdx
   - `ParallelSubgroup` -> Metal simdgroup lane, CUDA lane_id
5. **Map memory spaces:**
   - `Local` -> `threadgroup` (Metal), `__shared__` (CUDA)
   - `Private` -> thread-local variable
   - Global -> kernel parameter buffer
6. **Map reductions** to backend-specific parallel reductions:
   - Tree reduction in shared memory
   - Subgroup shuffles (simd_shuffle_down, __shfl_down_sync)
   - Final reduction across subgroups
7. **Emit strides as kernel arguments.** Views pass their
   strides as kernel parameters so the same compiled program
   works for any stride pattern.

### Metal-specific notes

- MTLComputePipelineState is the compiled program.
- MTLCommandBuffer + MTLComputeCommandEncoder per dispatch.
- Threadgroup memory declared in MSL with `threadgroup` qualifier.
- simd_shuffle_down for subgroup reductions.
- Strides passed via argument buffer or direct setBytes.
- MTLEvent for cross-encoder synchronization.
- Unified memory on Apple Silicon: CPU and GPU share the same
  physical memory. `memory_ptr` always works. Cross-device copy
  is a no-op (pointer is already accessible).

### CPU-specific notes

- Memory is malloc/free.
- Streams are sequential executors (single-threaded in v1).
- `parallel[grid]` maps to a thread pool dispatch.
- `parallel[workgroup]` within a grid task is sequential.
- `parallel[subgroup]` is a sequential loop (SIMD via LLVM
  auto-vectorization, not explicit in v1).
- Reductions are sequential loops.
- Events are simple done flags (atomic bool).
- All operations are synchronous in v1 (stream_sync is no-op).

### CUDA-specific notes (future)

- cuDeviceGet, cuCtxCreate for device.
- cuMemAlloc for memory.
- cuStream for streams.
- IR -> CUDA C text -> nvrtc -> PTX -> cuModuleLoadDataEx.
- `__shared__` for local memory.
- `__shfl_down_sync` for subgroup reductions.
- Warp size = 32 (from DeviceInfo.subgroup_size).

---

## 7. Error Handling

### Categories

**Validation errors** (caught at dispatch time):
- Binding name not found in program signature
- DType mismatch between binding and program parameter
- Shape rank mismatch
- Grid exceeds device limits
- Broadcast write violation (Out/InOut binding with stride=0)
- Aliasing violation (overlapping Out/InOut bindings)

**Compilation errors** (caught at compile time):
- Invalid IR (undefined virtual register reference)
- Unsupported operation for backend
- Shared memory exceeds device limit
- Workgroup size exceeds device limit

**Runtime errors** (caught during execution):
- Out of memory
- Device lost (GPU reset, thermal shutdown)
- Stream error (Metal command buffer failure)

### Error propagation

All fallible operations return `Result[T, SubstrateError]`.
The substrate never panics. The substrate never silently drops
errors. Every error is either returned to the caller or, for
truly unrecoverable conditions (device lost mid-dispatch),
reported via a callback mechanism.

---

## 8. Testing Strategy

### Correctness testing pattern

For every nontrivial kernel, the reference is the CPU path. A
backend implementation should be tested in this order:

1. Create deterministic or randomized inputs.
2. Run the CPU interpreter or CPU backend to produce the expected output.
3. Compile for the target backend.
4. Dispatch on the target backend.
5. Copy results back if needed.
6. Compare element-by-element with dtype-appropriate tolerance.

### Unit tests (pure With, no device)

- Shape arithmetic: elem_count, broadcasting
- Strides: is_contiguous, is_broadcasted
- View: byte_range, canonicalize, slice, transpose, reshape
- Scalar: construction, bit representation, dtype tag
- IR: parse_ir_text round-trip, validation pass
- Bindings: construction, lookup

### Integration tests (per backend)

Each backend runs the same test suite:

1. alloc -> free (basic lifecycle)
2. alloc -> view_contiguous -> fill -> read back (data round-trip)
3. alloc -> view -> copy -> read back (copy correctness)
4. Elementwise add (simplest dispatch)
5. Matrix multiply (parallel + shared memory)
6. Softmax (reduction)
7. Flash attention (all features)
8. Quantized matmul (mixed dtype)
9. KV cache update (in-place mutation)
10. Arena lifecycle (alloc, alloc_view, reset, reuse)
11. memory_from_ptr (borrowed memory lifecycle)
12. Alias validation (overlapping Out bindings -> error)
13. Broadcast write validation (Out with stride=0 -> error)
14. Grid exceeds limits -> error
15. Event timing accuracy

### Numerical verification

For operations with PyTorch equivalents, compare outputs within
tolerance (1e-5 for f32, 1e-3 for f16). Test inputs:
- Small deterministic inputs (manual verification)
- Random inputs at multiple scales (1e-6 to 1e6)
- Edge cases: zeros, infinities, NaN, denormals

---

## 9. Performance Targets (v1)

These are not hard requirements but directional goals for the
first implementation:

| Operation | Target | Comparison |
|---|---|---|
| Elementwise add (1M f32) | >80% of MPS bandwidth | MPS = Metal Performance Shaders |
| Matmul (2048x2048, f32) | >50% of MPSMatrixMultiplication | Tiled, shared memory |
| Softmax (batch=32, N=4096) | Within 2x of MPS | Fused, single kernel |
| Flash attention | Functional correctness | Performance optimization is v2 |
| Compile time (elementwise) | <10ms | IR -> MSL -> MTLLibrary |

v1 priority is correctness and completeness, not peak performance.
Performance optimization is iterative and benefits from profiling
data that only exists after the system works end-to-end.

---

## 10. Open Questions (Deferred)

These are known design questions that do not need to be resolved
for v1 but will need answers eventually:

1. **Graph capture.** Should Crux support recording a sequence of
   dispatches for replay? Or is this purely a Weld/higher-layer
   concern? Current answer: higher layer.

2. **Multi-device dispatch.** Should dispatch accept bindings from
   different devices? Current answer: no, single-device per dispatch,
   cross-device is orchestration.

3. **Persistent compilation cache.** Binary format stability across
   driver versions. Current answer: not v1.

4. **Explicit aliasing mode.** `may_alias`/`noalias` flags for
   expert users who know their access patterns. Current answer:
   future extension after strict mode is battle-tested.

5. **Dynamic shared memory.** Metal supports dynamic threadgroup
   memory size at dispatch time. Should the IR support this?
   Current answer: fixed sizes via spec constants are sufficient
   for v1.

---

## 11. Key Gotchas

### 1. With value semantics

Every struct assignment is a copy. Backend state should live behind
opaque handles, not in large structs passed around by value. Views
are intentionally safe to copy.

### 2. No implicit backend dynamic dispatch

If Crux grows a backend trait layer, dispatch will still be explicit:
manual vtables, tagged backend handles, or direct module calls.
Do not assume closure-based or OO-style dynamic dispatch.

### 3. Log full shader/compiler text on backend compile failure

Metal, CUDA, and future text-emitting backends should always include
the generated source when compilation fails. The emitted program is
the primary debugging artifact.

### 4. Metal `select` argument order is reversed relative to IR

IR uses `select(cond, true_val, false_val)`.
MSL uses `select(false_val, true_val, cond)`.
The Metal backend must swap operands when lowering.

### 5. Thread safety is per object kind

- Programs should be immutable and shareable.
- Streams should be treated as single-submitters unless explicitly synchronized.
- Events should be waitable from any host thread.
- Memory handle safety does not imply data-race safety for the underlying bytes.
