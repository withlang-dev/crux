# Crux — Implementation Notes

**Companion to:** crux-design.md
**Audience:** The agent implementing this system in With.
**Purpose:** Practical details, API mappings, gotchas, and
session-by-session guidance.

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

The core Crux types (Device, Memory, View, Program, Stream, Event)
do NOT get a Crux prefix in With code — they are the primary
vocabulary of the library and should be unadorned:

```
let device = default_device()
let mem = alloc(device, 1024)?
let view = view_contiguous(mem, shape(256), .Float32)
```

---

## Foundational Rules (decided, not negotiable)

These rules were established during design review. They prevent
drift and ambiguity. Violating any of them is a bug.

### Rule 1: C bridge handles are i64-cast pointers

At the C bridge boundary, every opaque handle (Device, Memory,
Program, Stream, Event, Arena) is a heap-allocated object whose
pointer is cast to `i64`. The With-facing API uses opaque pointer
types (`*mut Device`, `*mut Memory`, etc.) — the i64 conversion
is encapsulated inside the bridge layer.

```c
// C bridge: ObjC object → C struct pointer → i64
int64_t handle = (int64_t)(intptr_t)ctx;
```

```
// With side: c_import brings in bridge functions, we wrap them
// with proper opaque types:
fn alloc(device: *mut Device, size: usize) -> Result[*mut Memory, SubstrateError]:
    let handle = crux_cpu_alloc(device as i64, size as i64)
    if handle == 0:
        return Err(.OutOfMemory)
    Ok(handle as *mut Memory)
```

### Rule 2: Strides are in bytes

All strides throughout the entire system are byte strides.
No element strides anywhere in the substrate layer.

Element strides are computed at the backend boundary when setting
up shader arguments. This computation happens exactly once per
dispatch, inside the backend.

```
// Byte stride (stored in View):
byte_offset = v.offset + sum(indices[i] * strides.elems[i])

// Element stride (computed at dispatch for shader):
elem_stride[i] = byte_stride[i] / dtype.size()
```

### Rule 3: View.offset arithmetic never multiplies by dtype_size

Since strides are bytes, all offset computation is pure:

```
// view_slice:
new_offset = v.offset + start * v.strides.elems[dim] as usize
// NO dtype_size multiplication

// view_offset_of:
byte_offset = v.offset + sum(indices[i] * strides.elems[i])
// NO dtype_size multiplication
```

### Rule 4: Param references are negative, value references are non-negative

In the IR instruction encoding:

```
d0, d1, d2, d3 values:
  >= 0  →  instruction index (value produced by that instruction)
  < 0   →  parameter index (-1 = param 0, -2 = param 1, etc.)
```

This is unambiguous and requires no side-channel to distinguish
params from values. Every codegen path checks the sign.

### Rule 5: Grid validation happens at the API layer, before backend

`dispatch_grid` validates grid dimensions against `device_info.max_grid_dims`
BEFORE calling the backend. The backend may also validate (defense
in depth) but the API layer is the authoritative check. This ensures
consistent error behavior across all backends.

### Rule 6: Shared memory is validated at compile time

When compiling IR to a backend, the compiler sums all `local`
declarations and checks against `device_info.max_shared_memory`.
Exceeding the limit produces `CompileError`, not a runtime failure.

```
fn validate_shared_memory(prog: &IRProgram, device_info: &DeviceInfo) -> Result[Unit, SubstrateError]:
    var total_shared: usize = 0
    for inst in prog.insts:
        if inst.op == .Local:
            total_shared = total_shared + compute_local_size(inst, prog.aux)
    if total_shared > device_info.max_shared_memory:
        return Err(.CompileError("shared memory exceeds device limit: "
            ++ total_shared.to_string() ++ " > " ++ device_info.max_shared_memory.to_string()))
```

### Rule 7: Compilation cache key includes backend version

```
cache_key = hash(
    device_id,
    ir_bytes,
    spec_constant_values,
    backend_version_string,    // e.g., "metal-1.0" or "cuda-12.4"
)
```

This prevents stale pipelines after driver or backend updates.

### Rule 8: Collectives are stream-level operations

The primary collective API is at the stream level (`allreduce_sum`,
`allgather`, etc.), not inside IR programs. The backend decides the
mechanism:

- Multi-GPU: invokes NCCL/RCCL directly
- TPU: may compile an IR program containing IR-level collective ops
- Single device: no-op or local copy

IR-level collective ops (`CollectiveAllReduceSum`, etc.) exist for
backends that need intra-kernel collectives (TPU, Tenstorrent).
Programs using IR-level collectives are backend-aware — they're the
escape hatch, not the default.

### Rule 9: Placement is about bytes, not dimensions

`MemoryPlacement.Partitioned(N)` means "split this allocation across
N regions." Crux doesn't know which tensor dimension maps to which
partition. That's Weld's concern.

```
// Weld decides: shard dim 0 of a [4096, 4096] tensor across 8 regions
// Weld calls: alloc_placed(device, 4096 * 4096 * 4 / 8, .Partitioned(8))
// Crux sees: 8 regions, each with 8MB. That's all.
```

### Rule 10: Single-device programs work everywhere unchanged

A program that uses only `parallel`, `parallel[grid]`,
`parallel[workgroup]`, and `parallel[subgroup]` — without
`parallel[mesh]` or IR-level collectives — runs identically on
a single GPU, a TPU pod, or a multi-GPU node. The backend maps
execution to available resources.

Programs that use `parallel[mesh]` or IR-level collectives are
explicitly distributed and require a composite device.

---

## File Structure

```
lib/crux/
├── core.w                  # Shape, Strides, View, DType, Scalar, Bindings
├── error.w                 # SubstrateError enum
├── device.w                # Device API (dispatches to backend)
├── memory.w                # alloc, free, free_after, Arena, MemoryPlacement
├── view.w                  # View constructors and operations
├── program.w               # compile, ProgramSource, ProgramSig
├── stream.w                # Stream, Event, dispatch
├── collective.w            # allreduce, allgather, broadcast, reduce_scatter
├── ir.w                    # IROp, IRInst, IR builder, IR validation
├── ir_text.w               # Text parser (debug/testing only)
├── backend/
│   ├── backend.w           # Backend trait definition
│   ├── cpu.w               # CPU backend (impl Backend for CpuBackend)
│   ├── cpu_compiler.w      # IR → C emission + dlopen
│   ├── cpu_interp.w        # IR interpreter (correctness reference)
│   ├── metal.w             # Metal backend (impl Backend for MetalBackend)
│   └── metal_compiler.w    # IR → MSL emission
├── runtime/
│   └── crux_metal_bridge.m # Objective-C Metal bridge
├── test/
│   ├── test_view.w         # View arithmetic tests
│   ├── test_cpu.w          # CPU backend tests
│   ├── test_metal.w        # Metal backend tests
│   ├── test_ir.w           # IR parsing/validation tests
│   ├── test_elementwise.w  # Elementwise kernel tests
│   ├── test_matmul.w       # Matmul correctness tests
│   ├── test_softmax.w      # Softmax correctness tests
│   ├── test_collective.w   # Collective operation tests
│   └── bench/
│       ├── bench_elementwise.w
│       ├── bench_matmul.w
│       └── bench_softmax.w
└── kernels/
    ├── elementwise.ir      # Text IR for elementwise ops
    ├── matmul.ir           # Text IR for matrix multiply
    ├── softmax.ir          # Text IR for fused softmax
    └── attention.ir        # Text IR for flash attention
```

---

## Session 1: Type Definitions

### Shape and Strides

```
type Shape = {
    dims: [usize; 8],
    rank: i32,
}

type Strides = {
    elems: [isize; 8],
    rank: i32,
}
```

Direct indexing with `s.dims[i]` and `s.elems[i]`.

**Convenience constructors:**

```
fn shape1(d0: usize) -> Shape:
    Shape { dims: [d0, 0, 0, 0, 0, 0, 0, 0], rank: 1 }

fn shape2(d0: usize, d1: usize) -> Shape:
    Shape { dims: [d0, d1, 0, 0, 0, 0, 0, 0], rank: 2 }

fn shape3(d0: usize, d1: usize, d2: usize) -> Shape:
    Shape { dims: [d0, d1, d2, 0, 0, 0, 0, 0], rank: 3 }

fn shape4(d0: usize, d1: usize, d2: usize, d3: usize) -> Shape:
    Shape { dims: [d0, d1, d2, d3, 0, 0, 0, 0], rank: 4 }
```

**Shape methods:**

```
impl Shape:
    fn elem_count(self: &Self) -> usize:
        var n: usize = 1
        for i in 0..self.rank:
            n = n * self.dims[i]
        n

    fn is_scalar(self: &Self) -> bool:
        self.rank == 0
```

### Strides — the math

```
fn contiguous_strides(shape: &Shape, dtype: DType) -> Strides:
    let esize = dtype.size() as isize
    var st = Strides { elems: [0, 0, 0, 0, 0, 0, 0, 0], rank: shape.rank }
    if shape.rank == 0:
        return st
    // Last dimension stride = element size in bytes
    st.elems[shape.rank - 1] = esize
    var i = shape.rank - 2
    while i >= 0:
        st.elems[i] = st.elems[i + 1] * shape.dims[i + 1] as isize
        i = i - 1
    st
```

Example: `shape3(3, 4, 5)` with Float32 (4 bytes):
```
strides.elems = [80, 20, 4, 0, 0, 0, 0, 0]
// dim 2: 4 bytes (one f32)
// dim 1: 5 * 4 = 20 bytes (one row)
// dim 0: 4 * 20 = 80 bytes (one matrix)
```

**Strides methods:**

```
impl Strides:
    fn is_contiguous(self: &Self, shape: &Shape, dtype: DType) -> bool:
        var stride: isize = dtype.size() as isize
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

### DType

Uses the enum from the design doc directly. This enum is
**stable** — it enumerates hardware-native types only.
Quantization formats (Q4_K, AWQ, GPTQ, FP8, etc.) are not
dtypes. See "Quantization" section below.

```
enum DType =
    | Int8 | Int16 | Int32 | Int64
    | UInt8 | UInt16 | UInt32 | UInt64
    | Float16 | Float32 | Float64
    | BFloat16

impl DType:
    fn size(self) -> usize:
        match self
            .Int8 | .UInt8 => 1
            .Int16 | .UInt16 | .Float16 | .BFloat16 => 2
            .Int32 | .UInt32 | .Float32 => 4
            .Int64 | .UInt64 | .Float64 => 8

    fn is_float(self) -> bool:
        match self
            .Float16 | .Float32 | .Float64 | .BFloat16 => true
            _ => false

    fn is_signed(self) -> bool:
        match self
            .Int8 | .Int16 | .Int32 | .Int64 => true
            .Float16 | .Float32 | .Float64 | .BFloat16 => true
            _ => false
```

### Scalar

```
@[repr(C)]
type Scalar = union {
    i8_val: i8,
    i16_val: i16,
    i32_val: i32,
    i64_val: i64,
    u8_val: u8,
    u16_val: u16,
    u32_val: u32,
    u64_val: u64,
    f32_val: f32,
    f64_val: f64,
    bits: u64,
}

impl Scalar:
    fn from_i32(v: i32) -> Scalar: Scalar { i32_val: v }
    fn from_f32(v: f32) -> Scalar: Scalar { f32_val: v }
    fn from_f64(v: f64) -> Scalar: Scalar { f64_val: v }
    fn zero(dtype: DType) -> Scalar: Scalar { bits: 0 }
```

### MemoryPlacement

```
enum MemoryPlacement =
    | Local
    | Replicated
    | Partitioned(usize)

impl MemoryPlacement:
    fn regions(self: &Self) -> usize:
        match self
            .Local => 1
            .Replicated => 0
            .Partitioned(n) => n
```

On a single-region device, all placements behave identically.
`alloc` without placement defaults to `.Local`.

### DeviceInfo topology fields

```
// Single GPU / CPU:
//   region_count = 1, topology_rank = 0, topology_dims = [1, 1, 1]
//
// Multi-GPU (4x A100):
//   region_count = 4, topology_rank = 1, topology_dims = [4, 1, 1]
//
// TPU v4 pod (8x8 mesh):
//   region_count = 64, topology_rank = 2, topology_dims = [8, 8, 1]
//
// Tenstorrent Wormhole (8x8 grid):
//   region_count = 64, topology_rank = 2, topology_dims = [8, 8, 1]
```

The topology describes the physical mesh shape. The IR's
`parallel[mesh]` iterates over regions. The backend maps mesh
iterations to physical regions using the topology.

### View

```
type View = {
    memory: *mut Memory,
    offset: usize,           // byte offset from start of memory
    shape: Shape,
    strides: Strides,
    dtype: DType,
}
```

### View operations

All offset math follows Rule 2 (byte strides) and Rule 3 (no dtype
multiplication in offset arithmetic):

```
fn view_contiguous(mem: *mut Memory, shape: Shape, dtype: DType) -> View:
    View {
        memory: mem,
        offset: 0,
        shape,
        strides: contiguous_strides(shape, dtype),
        dtype,
    }

fn view_slice(v: View, dim: i32, start: usize, end: usize) -> View:
    { v with
        offset: v.offset + start * v.strides.elems[dim] as usize,
        shape: { v.shape with dims: {
            var d = v.shape.dims
            d[dim] = end - start
            d
        }},
    }

fn view_transpose(v: View, dim0: i32, dim1: i32) -> View:
    var out = v
    let s0 = v.strides.elems[dim0]
    let s1 = v.strides.elems[dim1]
    out.strides.elems[dim0] = s1
    out.strides.elems[dim1] = s0
    let d0 = v.shape.dims[dim0]
    let d1 = v.shape.dims[dim1]
    out.shape.dims[dim0] = d1
    out.shape.dims[dim1] = d0
    out

fn view_broadcast(v: View, target: Shape) -> Result[View, SubstrateError]:
    var out = v
    for i in 0..target.rank:
        if v.shape.dims[i] == 1 and target.dims[i] > 1:
            out.strides.elems[i] = 0
            out.shape.dims[i] = target.dims[i]
        else if v.shape.dims[i] != target.dims[i]:
            return Err(.ShapeMismatch("broadcast incompatible at dim " ++ i.to_string()))
    out

fn view_is_contiguous(v: &View) -> bool:
    v.strides.is_contiguous(v.shape, v.dtype)

fn view_is_broadcasted(v: &View) -> bool:
    for i in 0..v.shape.rank:
        if v.strides.elems[i] == 0 and v.shape.dims[i] > 1:
            return true
    false

fn view_elem_count(v: &View) -> usize:
    v.shape.elem_count()

fn view_byte_size(v: &View) -> usize:
    v.shape.elem_count() * v.dtype.size()

fn view_offset_of(v: &View, indices: [usize; 8]) -> usize:
    var off = v.offset
    for i in 0..v.shape.rank:
        off = off + indices[i] * v.strides.elems[i] as usize
    off
```

### Unit tests for session 1

```
test "shape elem_count":
    assert shape2(3, 4).elem_count() == 12
    assert shape3(2, 3, 4).elem_count() == 24
    assert shape1(0).elem_count() == 0

test "contiguous strides f32":
    let st = contiguous_strides(shape3(3, 4, 5), .Float32)
    assert st.elems[0] == 80    // 4*5*4
    assert st.elems[1] == 20    // 5*4
    assert st.elems[2] == 4     // 4

test "contiguous strides f64":
    let st = contiguous_strides(shape2(3, 4), .Float64)
    assert st.elems[0] == 32    // 4*8
    assert st.elems[1] == 8     // 8

test "view_slice offset":
    let mem: *mut Memory = null   // dummy for arithmetic tests
    let v = view_contiguous(mem, shape2(10, 20), .Float32)
    let sliced = view_slice(v, 0, 2, 5)
    assert sliced.offset == 2 * 80     // 2 * stride[0]
    assert sliced.shape.dims[0] == 3
    assert sliced.shape.dims[1] == 20

test "view_transpose swaps":
    let v = view_contiguous(null, shape2(3, 4), .Float32)
    let vt = view_transpose(v, 0, 1)
    assert vt.shape.dims[0] == 4
    assert vt.shape.dims[1] == 3
    assert vt.strides.elems[0] == 4     // was stride[1]
    assert vt.strides.elems[1] == 16    // was stride[0]

test "view_broadcast sets stride 0":
    let v = view_contiguous(null, shape2(1, 4), .Float32)
    let vb = view_broadcast(v, shape2(3, 4)).unwrap()
    assert vb.shape.dims[0] == 3
    assert vb.strides.elems[0] == 0
    assert view_is_broadcasted(vb) == true

test "view_offset_of":
    let v = view_contiguous(null, shape2(3, 4), .Float32)
    // element [1, 2] = offset + 1*16 + 2*4 = 24
    assert view_offset_of(v, [1, 2, 0, 0, 0, 0, 0, 0]) == 24
```

---

## Session 2: CPU Backend

### Backend trait and CPU implementation

The `Backend` trait uses standard dynamic dispatch:

```
trait Backend:
    fn name(self: &Self) -> str
    fn alloc(self: &Self, device: *mut Device, size: usize) -> Result[*mut Memory, SubstrateError]
    fn free(self: &Self, mem: *mut Memory)
    fn compile(self: &Self, device: *mut Device, source: &ProgramSource) -> Result[*mut Program, SubstrateError]
    fn dispatch(self: &Self, stream: *mut Stream, prog: *mut Program, bindings: &Bindings, grid: [usize; 3]) -> Result[*mut Event, SubstrateError]
    fn copy_bytes(self: &Self, stream: *mut Stream, src: *mut u8, dst: *mut u8, size: usize) -> *mut Event
    fn stream_create(self: &Self, device: *mut Device) -> *mut Stream
    fn stream_sync(self: &Self, stream: *mut Stream)
    fn event_wait(self: &Self, event: *mut Event)
    fn event_query(self: &Self, event: *mut Event) -> bool
    fn event_elapsed(self: &Self, start: *mut Event, end: *mut Event) -> f64
```

Backend routing uses `&dyn Backend`:

```
fn get_backend(device: *mut Device) -> &dyn Backend:
    let info = device_info(device)
    match info.kind
        .CPU => &CPU_BACKEND
        .GPU => &METAL_BACKEND
        .Accelerator => todo("accelerator backend")
```

### CPU memory (opaque types behind the API)

```
type CPUMemory = {
    ptr: *mut u8,
    size: usize,
}

type CpuBackend = {}

impl Backend for CpuBackend:
    fn name(self: &Self) -> str: "cpu"

    fn alloc(self: &Self, device: *mut Device, size: usize) -> Result[*mut Memory, SubstrateError]:
        let raw = unsafe: malloc(sizeof[CPUMemory]()) as *mut CPUMemory
        let data = unsafe: malloc(size) as *mut u8
        if data == null:
            return Err(.OutOfMemory)
        unsafe:
            (*raw).ptr = data
            (*raw).size = size
        Ok(raw as *mut Memory)

    fn free(self: &Self, mem: *mut Memory):
        let cpu_mem = mem as *mut CPUMemory
        unsafe:
            free((*cpu_mem).ptr as *mut c_void)
            free(cpu_mem as *mut c_void)

    // ... remaining trait methods
```

### Utility functions for CPU memory

```
fn cpu_memory_ptr(mem: *mut Memory) -> *mut u8:
    let cpu_mem = mem as *mut CPUMemory
    unsafe: (*cpu_mem).ptr

fn cpu_memory_size(mem: *mut Memory) -> usize:
    let cpu_mem = mem as *mut CPUMemory
    unsafe: (*cpu_mem).size
```

### Stream (CPU is synchronous)

```
type CPUStream = {
    device: *mut Device,
}

// Inside CpuBackend impl:
fn stream_create(self: &Self, device: *mut Device) -> *mut Stream:
    let s = unsafe: malloc(sizeof[CPUStream]()) as *mut CPUStream
    unsafe: (*s).device = device
    s as *mut Stream

fn stream_sync(self: &Self, stream: *mut Stream):
    // No-op: CPU is synchronous
    return
```

### Event (always done on CPU)

```
type CPUEvent = {
    done: bool,
}

fn cpu_event_create() -> *mut Event:
    let e = unsafe: malloc(sizeof[CPUEvent]()) as *mut CPUEvent
    unsafe: (*e).done = true
    e as *mut Event

// Inside CpuBackend impl:
fn event_query(self: &Self, event: *mut Event) -> bool:
    true  // CPU is synchronous

fn event_wait(self: &Self, event: *mut Event):
    return  // already done
```

### Binding validation

```
fn validate_bindings(sig: &ProgramSig, bindings: &Bindings) -> Result[Unit, SubstrateError]:
    for param in sig.params:
        var found = false
        for entry in bindings.entries:
            if entry.name == param.name:
                found = true
                if entry.view.dtype != param.dtype:
                    return Err(.DTypeMismatch("param " ++ param.name))
                if entry.view.shape.rank != param.rank:
                    return Err(.ShapeMismatch("rank mismatch: " ++ param.name))
                match param.mode
                    .Out | .InOut =>
                        if view_is_broadcasted(entry.view):
                            return Err(.BroadcastWriteViolation)
                    _ => ()
                break
        if not found:
            return Err(.ShapeMismatch("missing binding: " ++ param.name))
```

---

## Session 3: IR Definition

### IRInst layout

```
type IRInst = {
    op: IROp,
    dtype: DType,
    d0: i32,        // operand 0
    d1: i32,        // operand 1
    d2: i32,        // operand 2
    d3: i32,        // operand 3
}
// 24 bytes per instruction (enum discriminants + i32 fields)
```

### IRProgram

```
type IRProgram = {
    insts: Vec[IRInst],
    aux: Vec[i32],              // variable-length data (index tuples, shapes)
    param_names: Vec[str],
    param_modes: Vec[ParamMode],
    param_ranks: Vec[i32],
    param_dtypes: Vec[DType],
    num_params: i32,
}
```

### Encoding (Rule 4: negative = param, non-negative = value)

```
// Load: op=.Load, d0=param_ref (negative), d1=aux_base (index tuple start)
//   indices stored in aux[d1..d1+rank]
//   each index entry is a value ref or loop var ref

// Store: op=.Store, d0=param_ref, d1=aux_base, d2=value_ref

// BinOp: op=.Add/.Sub/etc, d0=lhs_ref, d1=rhs_ref

// UnOp: op=.Neg/.Abs/etc, d0=operand_ref

// FMA: op=.FMA, d0=a_ref, d1=b_ref, d2=c_ref

// Cast: op=.Cast, d0=operand_ref, dtype=target_type

// Loop: op=.Loop, d0=var_inst_idx, d1=start_ref, d2=end_ref, d3=body_block_id
// Parallel: same as Loop, op=.ParallelGrid/.ParallelWorkgroup/.ParallelSubgroup

// If: op=.If, d0=cond_ref, d1=then_block, d2=else_block

// ReduceSum/Max/Min/Prod: op=.ReduceSum, d0=var_inst, d1=start, d2=end, d3=body_expr_ref

// Local: op=.Local, d0=name_sym, d1=aux_base (shape), d2=rank, dtype=element type

// Barrier: op=.Barrier (no operands)

// BlockBegin: op=.BlockBegin, d0=block_id
// BlockEnd: op=.BlockEnd, d0=block_id

// Constant literal: op=.Const, dtype=type, d0=low_bits, d1=high_bits
```

### Text format

One instruction per line. `%N` references value N.
Parameters are declared at top. Indentation shows block nesting.

```
param a in [M, K] f32
param b in [K, N] f32
param out out [M, N] f32

parallel_grid i 0 M
  parallel_grid j 0 N
    %0 = const f32 0.0
    loop k 0 K
      %1 = load a [i, k]
      %2 = load b [k, j]
      %3 = fma %1 %2 %0
      %0 = %3
    store out [i, j] %0
```

The text parser is <300 lines. Tokenize by whitespace, map to IROp,
build IRInst/aux arrays.

### IR validation pass

Before compilation, check:

1. **Value refs in range:** All non-negative d0/d1/d2/d3 < instruction count
2. **Param refs in range:** All negative refs: abs(ref) - 1 < num_params
3. **Block nesting:** BlockBegin/BlockEnd matched
4. **Type consistency:** Binop operands same dtype, store value matches param dtype
5. **Reduction ops valid:** Only ReduceSum/Max/Min/Prod
6. **Local inside parallel:** Local declarations only inside grid/workgroup body
7. **Barrier inside workgroup:** Barrier only inside parallel[workgroup] body
8. **Shared memory total:** Sum of local declarations ≤ device max (Rule 6)
9. **Mesh inside program top level:** parallel[mesh] only at outermost parallel nesting
10. **Collective inside mesh:** IR-level collectives only inside parallel[mesh] body
11. **Collective consistency:** All mesh regions must execute identical collective sequence (statically verifiable for simple cases, UB for dynamic divergence)

Return list of `(instruction_index, error_message)`.

---

## Session 4: IR → CPU (Interpreter First)

### Interpreter architecture

Walk the IR linearly. Maintain a value table using parallel arrays:

```
type Interp = {
    values: Vec[u64],       // raw bits for each value
    dtypes: Vec[DType],     // dtype for each value
    loop_vars: Vec[i64],    // current loop variable values
    params: Vec[View],      // bound views
}
```

**Why parallel arrays instead of Vec[Scalar]:**
- No union overhead
- No struct copying
- Faster: no branching on dtype for storage, only for compute
- Raw bits (`u64`) hold any scalar value

### Interpreter dispatch

```
fn interp_exec(interp: &mut Interp, prog: &IRProgram):
    var ip = 0
    while ip < prog.insts.len():
        let inst = prog.insts[ip]
        match inst.op
            .Load =>
                let param_idx = (0 - inst.d0) - 1
                let view = interp.params[param_idx]
                let indices = read_indices(interp, prog, inst.d1, view.shape.rank)
                let byte_off = compute_offset(view, indices)
                let mem_ptr = cpu_memory_ptr(view.memory)
                let val = read_raw(mem_ptr, byte_off, view.dtype)
                interp.values.push(val)
                interp.dtypes.push(view.dtype)

            .Add =>
                let a = interp.values[inst.d0]
                let b = interp.values[inst.d1]
                let dt = interp.dtypes[inst.d0]
                let result = scalar_add_raw(a, b, dt)
                interp.values.push(result)
                interp.dtypes.push(dt)

            .Store =>
                let param_idx = (0 - inst.d0) - 1
                let view = interp.params[param_idx]
                let indices = read_indices(interp, prog, inst.d1, view.shape.rank)
                let byte_off = compute_offset(view, indices)
                let mem_ptr = cpu_memory_ptr(view.memory)
                let val = interp.values[inst.d2]
                write_raw(mem_ptr, byte_off, val, view.dtype)

            .Loop =>
                let var_idx = inst.d0
                let start = get_i64(interp, inst.d1)
                let end = get_i64(interp, inst.d2)
                let body_block = inst.d3
                for iter in start..end:
                    set_loop_var(interp, var_idx, iter)
                    exec_block(interp, prog, body_block)
            _ => todo("op: " ++ inst.op.to_string())
        ip = ip + 1
```

### scalar_add_raw (typed arithmetic on raw bits)

```
fn scalar_add_raw(a: u64, b: u64, dtype: DType) -> u64:
    match dtype
        .Float32 =>
            let fa = unsafe: transmute[f32](a as u32)
            let fb = unsafe: transmute[f32](b as u32)
            unsafe: transmute[u32](fa + fb) as u64
        .Float64 =>
            let fa = unsafe: transmute[f64](a)
            let fb = unsafe: transmute[f64](b)
            unsafe: transmute[u64](fa + fb)
        .Int32 =>
            ((a as i32) + (b as i32)) as u64
        .Int64 =>
            ((a as i64) + (b as i64)) as u64
        _ => todo("scalar_add for " ++ dtype.to_string())
```

Same pattern for sub, mul, div, etc. Each is ~20 lines of match dispatch.

### Test: elementwise add via interpreter

```
test "interp elementwise add":
    let cpu = CpuBackend {}
    let device = cpu.create_device(0)?
    let N = 16

    let a_mem = cpu.alloc(device, N * 4)?
    let b_mem = cpu.alloc(device, N * 4)?
    let out_mem = cpu.alloc(device, N * 4)?

    // Fill a with 1.0, b with 2.0
    let a_ptr = cpu_memory_ptr(a_mem)
    let b_ptr = cpu_memory_ptr(b_mem)
    for i in 0..N:
        write_f32(a_ptr, i * 4, 1.0)
        write_f32(b_ptr, i * 4, 2.0)

    let a_view = view_contiguous(a_mem, shape1(N), .Float32)
    let b_view = view_contiguous(b_mem, shape1(N), .Float32)
    let out_view = view_contiguous(out_mem, shape1(N), .Float32)

    let prog = parse_ir("
        param a in [N] f32
        param b in [N] f32
        param out out [N] f32
        parallel i 0 N
          %0 = load a [i]
          %1 = load b [i]
          %2 = add %0 %1
          store out [i] %2
    ")

    interp_dispatch(prog, [a_view, b_view, out_view])

    let out_ptr = cpu_memory_ptr(out_mem)
    for i in 0..N:
        assert read_f32(out_ptr, i * 4) == 3.0

    cpu.free(a_mem)
    cpu.free(b_mem)
    cpu.free(out_mem)
```

---

## Session 5: Metal Backend

### Objective-C bridge (crux_metal_bridge.m)

Compile with: `clang -c -x objective-c -fobjc-arc crux_metal_bridge.m -o crux_metal_bridge.o -framework Metal`

The C bridge uses i64 handles at the ABI boundary. The With layer
wraps these with `c_import` and casts to/from opaque pointer types:

```c
// crux_metal_bridge.m

#import <Metal/Metal.h>

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
} CruxMetalCtx;

// Device
int64_t crux_metal_create_device(void);
void crux_metal_destroy_device(int64_t ctx);
int64_t crux_metal_device_max_threadgroup_size(int64_t ctx);
int64_t crux_metal_device_max_shared_memory(int64_t ctx);

// Memory
int64_t crux_metal_alloc(int64_t ctx, int64_t size);
void crux_metal_free(int64_t buffer);
void* crux_metal_buffer_ptr(int64_t buffer);
int64_t crux_metal_buffer_size(int64_t buffer);

// Program (MSL compilation)
int64_t crux_metal_compile(int64_t ctx, const char* msl_source, const char* entry);
void crux_metal_destroy_program(int64_t pipeline);

// Stream (command buffer)
int64_t crux_metal_create_stream(int64_t ctx);
int64_t crux_metal_begin_command(int64_t stream);
void crux_metal_dispatch(int64_t cmd, int64_t pipeline,
                         int64_t* buffers, int64_t* offsets, int32_t buf_count,
                         int64_t* metadata_buf, int64_t metadata_offset,
                         uint32_t gx, uint32_t gy, uint32_t gz,
                         uint32_t tx, uint32_t ty, uint32_t tz);
int64_t crux_metal_commit(int64_t cmd);   // returns event handle
void crux_metal_wait(int64_t event);
int32_t crux_metal_event_done(int64_t event);
double crux_metal_event_elapsed(int64_t start_event, int64_t end_event);
void crux_metal_stream_sync(int64_t stream);
```

### With wrapper over C bridge

```
c_import "crux_metal_bridge.h"

type MetalBackend = {}

impl Backend for MetalBackend:
    fn name(self: &Self) -> str: "metal"

    fn alloc(self: &Self, device: *mut Device, size: usize) -> Result[*mut Memory, SubstrateError]:
        let handle = crux_metal_alloc(device as i64, size as i64)
        if handle == 0:
            return Err(.OutOfMemory)
        Ok(handle as *mut Memory)

    fn free(self: &Self, mem: *mut Memory):
        crux_metal_free(mem as i64)

    fn compile(self: &Self, device: *mut Device, source: &ProgramSource) -> Result[*mut Program, SubstrateError]:
        let msl = emit_msl(source, device_info(device))
        let handle = crux_metal_compile(device as i64, msl.as_cstr(), source.entry.as_cstr())
        if handle == 0:
            return Err(.CompileError("Metal compilation failed"))
        Ok(handle as *mut Program)

    // ... remaining trait methods
```

### Metal buffer binding layout (metadata struct)

Instead of one Metal buffer per stride array, pack all metadata
into a single buffer:

```c
// Metadata buffer layout:
// [param_count]
// For each param:
//   [rank, stride[0], stride[1], ..., stride[7]]
//   [shape[0], shape[1], ..., shape[7]]
// [spec_const_count]
// For each constant:
//   [value_bits_low, value_bits_high]
```

Metal buffer slots (max 31):
```
Buffer 0..N-1:  data buffers (one per param)
Buffer N:       metadata buffer (all strides, shapes, constants)
```

This scales to any number of parameters without hitting the 31-slot
limit, because metadata is packed into a single buffer.

### Memory model (Apple Silicon)

Use `MTLResourceStorageModeShared` for session 5. The CPU can
read/write buffer contents directly via `[buffer contents]`. No
staging buffers needed on unified memory.

Optimization (later): `MTLResourceStorageModePrivate` for GPU-only
intermediates (KV cache, activation buffers). Requires explicit
GPU-side copy for initialization.

### Stream model (session 5: simple)

One `MTLCommandBuffer` per dispatch. Commit immediately.
`event_wait` calls `[commandBuffer waitUntilCompleted]`.

Optimization (later): batch multiple dispatches into one command
buffer, commit on stream_sync or when buffer count reaches threshold.

---

## Session 6: IR → MSL Compiler

### MSL kernel template

```metal
#include <metal_stdlib>
using namespace metal;

struct Metadata {
    int param_count;
    // ... packed strides, shapes, constants
};

kernel void ENTRY_NAME(
    device const float* param_0 [[buffer(0)]],
    device float* param_1 [[buffer(1)]],
    // ... one per parameter
    constant Metadata& meta [[buffer(LAST)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // ... generated code
}
```

### IR → MSL translation rules

```
parallel[grid] i in 0..N
    → uint i = gid.x;
      // (or gid.y, gid.z for nested grid parallels)

parallel[workgroup] j in 0..TILE
    → uint j = tid.x;

parallel[subgroup] lane in 0..SG
    → uint lane = simd_lane;

for k in 0..K
    → for (uint k = 0; k < K; k++) { ... }

load(a, [i, k])
    → param_0[i * meta.strides_0[0] + k * meta.strides_0[1]]
    // element strides computed from byte strides at dispatch

store(out, [i, j], v)
    → param_1[i * meta.strides_1[0] + j * meta.strides_1[1]] = v;

local tile: [TILE, TILE] f32
    → threadgroup float tile[TILE * TILE];

private acc: [D] f32
    → float acc[D];

barrier()
    → threadgroup_barrier(mem_flags::mem_threadgroup);

reduce[sum](i, 0..N, expr)
    → threadgroup float shared[THREADGROUP_SIZE];
      shared[tid.x] = expr;
      threadgroup_barrier(...);
      for (uint s = THREADGROUP_SIZE/2; s > 0; s >>= 1) {
          if (tid.x < s) shared[tid.x] += shared[tid.x + s];
          threadgroup_barrier(...);
      }
      float result = shared[0];

reduce[max](i, 0..N, expr)
    → same pattern with max() instead of +

fma(a, b, c)
    → fma(a, b, c)   // Metal has native fma

select(c, t, f)
    → select(f, t, c)   // Metal select has reversed arg order!
```

### Grid/threadgroup size computation

For 1D kernels (elementwise):
```
threadgroup_size = min(256, pipeline.maxTotalThreadsPerThreadgroup)
grid_size = ceil_div(N, threadgroup_size)
dispatch: grid=[grid_size, 1, 1], threadgroup=[threadgroup_size, 1, 1]
```

For 2D kernels (matmul tiled):
```
grid_x = ceil_div(M, TILE)
grid_y = ceil_div(N, TILE)
dispatch: grid=[grid_x, grid_y, 1], threadgroup=[TILE, TILE, 1]
```

### MSL string building

Build MSL as a With string via concatenation:

```
fn emit_msl(prog: &IRProgram, device_info: &DeviceInfo) -> str:
    var msl = "#include <metal_stdlib>\nusing namespace metal;\n\n"
    msl = msl ++ emit_metadata_struct(prog)
    msl = msl ++ emit_kernel_signature(prog)
    msl = msl ++ emit_kernel_body(prog, device_info)
    msl
```

Each `emit_*` function walks the IR and appends MSL text.
The whole compiler is ~500-800 lines.

### Multi-backend dispatch with comptime

When compiling for different backends, use `comptime if` to
eliminate dead code paths:

```
fn compile_for_device[B: Backend](backend: &B, device: *mut Device, source: &ProgramSource) -> Result[*mut Program, SubstrateError]:
    comptime if B == MetalBackend:
        let msl = emit_msl(source, device_info(device))
        backend.compile_msl(device, msl)
    else if B == CpuBackend:
        let c_code = emit_c(source)
        backend.compile_c(device, c_code)
    else:
        comptime_error("unsupported backend")
```

---

## Sessions 7-8: Copy, Arena, Timing

### Cross-device copy (Apple Silicon)

On unified memory, CPU↔GPU copy is memcpy through `[buffer contents]`:

```c
void crux_metal_copy_to_device(int64_t dst_buf, int64_t dst_offset,
                                const void* src, int64_t size) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)(intptr_t)dst_buf;
    memcpy((uint8_t*)[buffer contents] + dst_offset, src, (size_t)size);
    // On macOS with managed mode, would need didModifyRange here
}

void crux_metal_copy_from_device(void* dst, int64_t src_buf,
                                  int64_t src_offset, int64_t size) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(void*)(intptr_t)src_buf;
    memcpy(dst, (uint8_t*)[buffer contents] + src_offset, (size_t)size);
}
```

### Arena (Metal)

```c
typedef struct {
    id<MTLBuffer> buffer;
    int64_t size;
    int64_t used;
} CruxMetalArena;

int64_t crux_metal_arena_alloc(int64_t arena_handle, int64_t size, int64_t align) {
    CruxMetalArena* arena = (CruxMetalArena*)(intptr_t)arena_handle;
    int64_t aligned = (arena->used + align - 1) & ~(align - 1);
    if (aligned + size > arena->size) return 0;  // OOM

    // Create sub-alloc descriptor
    typedef struct { int64_t buffer; int64_t offset; int64_t size; } SubAlloc;
    SubAlloc* sub = malloc(sizeof(SubAlloc));
    sub->buffer = (int64_t)(intptr_t)arena->buffer;
    sub->offset = aligned;
    sub->size = size;
    arena->used = aligned + size;
    return (int64_t)(intptr_t)sub;
}
```

**Metal alignment:** Buffer offsets for `setBuffer:offset:atIndex:`
must be 256-byte aligned. Arena alloc uses `align=256` by default.

### Event timing

```c
double crux_metal_event_elapsed(int64_t start_handle, int64_t end_handle) {
    id<MTLCommandBuffer> start = (__bridge id<MTLCommandBuffer>)(void*)(intptr_t)start_handle;
    id<MTLCommandBuffer> end_buf = (__bridge id<MTLCommandBuffer>)(void*)(intptr_t)end_handle;
    return [end_buf GPUEndTime] - [start GPUStartTime];
}
```

### Compilation cache

```
type CompileCache = HashMap[u64, *mut Program]

fn compile_cache_key(prog: &IRProgram, device: *mut Device, backend_version: str) -> u64:
    var h: u64 = 14695981039346656037   // FNV offset basis
    h = fnv_hash_i64(h, device as i64)
    for inst in prog.insts:
        h = fnv_hash_bytes(h, inst_as_bytes(inst), 24)
    for c in prog.spec_constants:
        h = fnv_hash_u64(h, c.value.bits)
    h = fnv_hash_str(h, backend_version)
    h

fn compile_cached(cache: &mut CompileCache, prog: &IRProgram, device: *mut Device, backend: &dyn Backend) -> Result[*mut Program, SubstrateError]:
    let key = compile_cache_key(prog, device, backend.name())
    match cache.get(key)
        Some(program) => Ok(program)
        None =>
            let program = backend.compile(device, prog)?
            cache.insert(key, program)
            Ok(program)
```

---

## Correctness Testing Pattern

For every kernel, the reference is the CPU interpreter. Test flow:

```
1. Create input data (known values or random)
2. Run interpreter on CPU → expected output
3. Compile for Metal
4. Dispatch on Metal → actual output
5. Copy Metal output to CPU
6. Compare element-by-element with tolerance
```

Tolerances:
```
Float32:  atol=1e-6,  rtol=1e-5
Float16:  atol=1e-3,  rtol=1e-2
BFloat16: atol=1e-2,  rtol=1e-1
Int types: exact match
```

---

## Performance Targets

| Session | Kernel | Target | Reference |
|---|---|---|---|
| 8 | Elementwise add (N=1M) | >80% mem bandwidth | ~0.12ms on M2 |
| 11 | Matmul 4096×4096 | >50% of MPS | ~38ms on M2 |
| 12 | Softmax (B=64, N=4096) | <1ms | MPS reference |
| 23 | Flash attention | within 3x of MLX | measure MLX |
| 25 | GPT-2 124M tok/s | within 2x llama.cpp | ~100 tok/s on M2 |

---

## Key Gotchas

### 1. With value semantics

Every struct assignment is a copy (for Copy types) or move (for
non-Copy types). Backend state MUST be opaque handles (pointer types),
not With structs passed by value. View is safe to copy — it's a value
type by design, and `impl Copy for View` is legal since all its
fields are Copy (the raw pointer `*mut Memory` is Copy).

### 2. Metal shader compilation errors

Always log the full MSL source on compilation failure.
`[error localizedDescription]` gives the Metal compiler error.
This is the #1 debugging scenario.

### 3. BFloat16 in Metal

Not native. Store as `uint16_t`. Convert in shader:
```metal
float bf16_to_f32(ushort bf) {
    return as_type<float>(uint(bf) << 16);
}
ushort f32_to_bf16(float f) {
    return ushort(as_type<uint>(f) >> 16);
}
```

### 4. Metal select() argument order

Metal's `select(a, b, cond)` returns `a` when cond is false,
`b` when true. This is reversed from C's `cond ? t : f`.
The IR → MSL compiler must swap arguments:
```
IR: select(cond, true_val, false_val)
MSL: select(false_val, true_val, cond)
```

### 5. Thread safety

- Programs: immutable, safe across streams and CPU threads
- Streams: NOT safe across CPU threads
- Events: safe to wait on from any thread
- Device: safe to use from any thread (Metal handles locking)
- Memory: the handle is safe to pass around; concurrent access
  to the underlying data follows the aliasing/happens-before rules

---

## Quantization Implementation (Weld layer, sessions 26+)

Quantization lives in Weld. Crux never sees it. This section is
guidance for the agent implementing Weld's quant support.

### QuantScheme struct

```
type QuantScheme = {
    name: str,                    // "q4_k", "awq_4bit", "fp8_e4m3"
    storage_dtype: DType,         // what's in memory (UInt8, UInt32)
    compute_dtype: DType,         // what compute sees (Float16, Float32)
    block_size: i32,              // elements per quant block
    bits_per_weight: i32,         // 2, 3, 4, 8
    has_scales: bool,
    has_zeros: bool,
    scale_dtype: DType,
    decode_program: *mut Program, // Crux IR that decodes one block
}
```

### Decode program contract

A decode program receives packed weight bytes, scale/zero-point
data, and an output buffer. It writes `block_size` elements of
`compute_dtype` to the output. The program is a normal Crux IR
program compiled per-device.

Memory layout per block:
```
[packed_weights: block_size * bits_per_weight / 8 bytes]
[scale: scale_dtype.size() bytes]                       // if has_scales
[zero_point: scale_dtype.size() bytes]                  // if has_zeros
```

### Registry

```
// Global registry — populated at startup
var quant_registry: Vec[QuantScheme] = Vec.new()

fn register_quant_scheme(scheme: QuantScheme):
    quant_registry.push(scheme)

fn get_quant_scheme(name: str) -> Option[*mut QuantScheme]:
    for i in 0..quant_registry.len():
        if quant_registry.get(i).name == name:
            return Some(&mut quant_registry[i])
    None
```

### Fused quantized matmul pattern

The quantized matmul does NOT decode all weights first and then
multiply. It decodes per-tile inside the tiled matmul loop:

```
// Pseudocode for quant-aware tiled matmul
parallel[grid] bi in 0..M/TILE:
    parallel[grid] bj in 0..N/TILE:
        var acc: [TILE, TILE] f16 = 0.0
        for bk in 0..K/block_size:
            // Decode one block of B weights into shared memory
            local b_decoded: [block_size, TILE] f16
            parallel[workgroup] ti in 0..TILE:
                decode_block(b_packed, bk, bj*TILE+ti, b_decoded)
            barrier()
            // Standard tiled multiply with decoded tile
            ...
```

The decode step is the `decode_program` from the `QuantScheme`,
inlined or called as a subroutine.

### Adding a new format

When a new quant format drops (happens quarterly):

1. Define the `QuantScheme` struct with storage layout
2. Write a decode IR program (~10-30 instructions)
3. Call `register_quant_scheme`
4. Write a weight loader that reads the format's file layout
   (GGUF, safetensors, etc.) into Memory with correct blocking

No Crux changes. No DType changes. No recompilation of existing
kernels. The fused matmul template picks up the new decode program
automatically.

### Known formats to support (as of 2026)

```
Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0   — GGUF family
GPTQ 4-bit                             — per-group, asymmetric
AWQ 4-bit                              — activation-aware
FP8 E4M3, FP8 E5M2                     — hardware FP8
INT8 symmetric, INT8 asymmetric        — standard PTQ
NF4                                     — QLoRA format
MXFP4, MXFP6                           — microscaling
```

Each is a QuantScheme with a decode program. Priority order:
Q4_K (most GGUF models), GPTQ 4-bit (most HuggingFace models),
FP8 E4M3 (fastest on H100/MI300), INT8 symmetric (widest support).