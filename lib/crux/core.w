pub let MAX_RANK: i32 = 8

// Runtime resources are exposed as opaque handle types behind pointers.
pub type Device = opaque
pub type Memory = opaque
pub type Program = opaque
pub type Stream = opaque
pub type Event = opaque
pub type Arena = opaque

pub type Size = usize
pub type Stride = isize

pub fn null_device -> *mut Device: null
pub fn null_memory -> *mut Memory: null
pub fn null_program -> *mut Program: null
pub fn null_stream -> *mut Stream: null
pub fn null_event -> *mut Event: null
pub fn null_arena -> *mut Arena: null

pub type Dim3 {
    x: Size,
    y: Size,
    z: Size,
}
impl Copy for Dim3

pub fn dim3(x: Size, y: Size, z: Size) -> Dim3:
    Dim3 { x, y, z }

pub type Shape {
    dims: [Size; 8],
    rank: i32,
}
impl Copy for Shape

pub fn shape_scalar -> Shape:
    Shape { dims: [0usize; 8], rank: 0 }

pub fn shape1(d0: Size) -> Shape:
    Shape { dims: [d0, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize], rank: 1 }

pub fn shape2(d0: Size, d1: Size) -> Shape:
    Shape { dims: [d0, d1, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize], rank: 2 }

pub fn shape3(d0: Size, d1: Size, d2: Size) -> Shape:
    Shape { dims: [d0, d1, d2, 0usize, 0usize, 0usize, 0usize, 0usize], rank: 3 }

pub fn shape4(d0: Size, d1: Size, d2: Size, d3: Size) -> Shape:
    Shape { dims: [d0, d1, d2, d3, 0usize, 0usize, 0usize, 0usize], rank: 4 }

pub fn shape5(d0: Size, d1: Size, d2: Size, d3: Size, d4: Size) -> Shape:
    Shape { dims: [d0, d1, d2, d3, d4, 0usize, 0usize, 0usize], rank: 5 }

pub fn shape6(d0: Size, d1: Size, d2: Size, d3: Size, d4: Size, d5: Size) -> Shape:
    Shape { dims: [d0, d1, d2, d3, d4, d5, 0usize, 0usize], rank: 6 }

pub fn shape7(d0: Size, d1: Size, d2: Size, d3: Size, d4: Size, d5: Size, d6: Size) -> Shape:
    Shape { dims: [d0, d1, d2, d3, d4, d5, d6, 0usize], rank: 7 }

pub fn shape8(d0: Size, d1: Size, d2: Size, d3: Size, d4: Size, d5: Size, d6: Size, d7: Size) -> Shape:
    Shape { dims: [d0, d1, d2, d3, d4, d5, d6, d7], rank: 8 }

pub fn shape_get(shape: Shape, i: i32) -> Size:
    if i < 0 or i >= MAX_RANK:
        return 0usize
    shape.dims[i]

pub fn shape_set(shape: Shape, i: i32, value: Size) -> Shape:
    if i < 0 or i >= MAX_RANK:
        return shape
    var out = shape
    out.dims[i] = value
    out

pub fn shape_is_valid(shape: Shape) -> bool:
    shape.rank >= 0 and shape.rank <= MAX_RANK

pub fn shape_equal(a: Shape, b: Shape) -> bool:
    if a.rank != b.rank:
        return false
    for i in 0..a.rank:
        if shape_get(a, i) != shape_get(b, i):
            return false
    true

pub fn shape_elem_count(shape: Shape) -> Size:
    if shape.rank == 0:
        return 1usize
    var n: Size = 1usize
    for i in 0..shape.rank:
        n = n * shape_get(shape, i)
    n

pub fn shape_is_scalar(shape: Shape) -> bool:
    shape.rank == 0

pub type Strides {
    elems: [Stride; 8],
    rank: i32,
}
impl Copy for Strides

pub fn strides_zero -> Strides:
    Strides { elems: [0isize; 8], rank: 0 }

pub fn strides_get(strides: Strides, i: i32) -> Stride:
    if i < 0 or i >= MAX_RANK:
        return 0isize
    strides.elems[i]

pub fn strides_set(strides: Strides, i: i32, value: Stride) -> Strides:
    if i < 0 or i >= MAX_RANK:
        return strides
    var out = strides
    out.elems[i] = value
    out

pub fn strides_is_valid(strides: Strides) -> bool:
    strides.rank >= 0 and strides.rank <= MAX_RANK

pub fn strides_equal(a: Strides, b: Strides) -> bool:
    if a.rank != b.rank:
        return false
    for i in 0..a.rank:
        if strides_get(a, i) != strides_get(b, i):
            return false
    true

pub enum DType {
    Int8
    | Int16
    | Int32
    | Int64
    | UInt8
    | UInt16
    | UInt32
    | UInt64
    | Float16
    | Float32
    | Float64
    | BFloat16
}
impl Copy for DType

pub fn dtype_size(dtype: DType) -> Size:
    match dtype:
        .Int8 | .UInt8 => 1usize
        .Int16 | .UInt16 | .Float16 | .BFloat16 => 2usize
        .Int32 | .UInt32 | .Float32 => 4usize
        .Int64 | .UInt64 | .Float64 => 8usize

pub fn contiguous_strides(shape: Shape, dtype: DType) -> Strides:
    let elem_size = dtype_size(dtype) as Stride
    var strides = strides_zero()
    if shape.rank == 0:
        return strides
    strides = strides_set(strides, shape.rank - 1, elem_size)
    strides.rank = shape.rank
    var i = shape.rank - 2
    while i >= 0:
        let next = strides_get(strides, i + 1) * shape_get(shape, i + 1) as Stride
        strides = strides_set(strides, i, next)
        strides.rank = shape.rank
        i = i - 1
    strides

pub fn strides_is_contiguous(strides: Strides, shape: Shape, dtype: DType) -> bool:
    let expected = contiguous_strides(shape, dtype)
    strides_equal(strides, expected)

pub fn strides_is_broadcasted(strides: Strides) -> bool:
    for i in 0..strides.rank:
        if strides_get(strides, i) == 0:
            return true
    false

pub type Scalar {
    bits: u64,
    dtype: DType,
}
impl Copy for Scalar

pub type IRInst {
    op: i32,
    dtype: DType,
    d0: i32,
    d1: i32,
    d2: i32,
    d3: i32,
}
impl Copy for IRInst

pub fn scalar_i32(v: i32) -> Scalar:
    Scalar { bits: v as u64, dtype: .Int32 }

pub fn scalar_i8(v: i8) -> Scalar:
    let raw: u8 = unsafe { transmute[u8](v) }
    Scalar { bits: raw as u64, dtype: .Int8 }

pub fn scalar_i16(v: i16) -> Scalar:
    let raw: u16 = unsafe { transmute[u16](v) }
    Scalar { bits: raw as u64, dtype: .Int16 }

pub fn scalar_i64(v: i64) -> Scalar:
    let raw: u64 = unsafe { transmute[u64](v) }
    Scalar { bits: raw, dtype: .Int64 }

pub fn scalar_u8(v: u8) -> Scalar:
    Scalar { bits: v as u64, dtype: .UInt8 }

pub fn scalar_u16(v: u16) -> Scalar:
    Scalar { bits: v as u64, dtype: .UInt16 }

pub fn scalar_u32(v: u32) -> Scalar:
    Scalar { bits: v as u64, dtype: .UInt32 }

pub fn scalar_u64(v: u64) -> Scalar:
    Scalar { bits: v, dtype: .UInt64 }

pub fn scalar_f32(v: f32) -> Scalar:
    let raw: u32 = unsafe { transmute[u32](v) }
    Scalar { bits: raw as u64, dtype: .Float32 }

pub fn scalar_f64(v: f64) -> Scalar:
    let raw: u64 = unsafe { transmute[u64](v) }
    Scalar { bits: raw, dtype: .Float64 }

pub fn scalar_f16_bits(bits: u16) -> Scalar:
    Scalar { bits: bits as u64, dtype: .Float16 }

pub fn scalar_bf16_bits(bits: u16) -> Scalar:
    Scalar { bits: bits as u64, dtype: .BFloat16 }

pub fn scalar_low_i32(bits: u64) -> i32:
    let raw: u32 = bits as u32
    unsafe { transmute[i32](raw) }

pub fn scalar_high_i32(bits: u64) -> i32:
    let raw: u32 = (bits >> 32) as u32
    unsafe { transmute[i32](raw) }

pub fn scalar_bits_from_words(low: i32, high: i32) -> u64:
    let low_u: u32 = unsafe { transmute[u32](low) }
    let high_u: u32 = unsafe { transmute[u32](high) }
    (low_u as u64) | ((high_u as u64) << 32)

pub enum DeviceKind { CPU | GPU | Accelerator }
impl Copy for DeviceKind

pub type DeviceInfo {
    name: str,
    kind: DeviceKind,
    memory_total: Size,
    memory_available: Size,
    max_workgroup_size: i32,
    max_grid_dims: Dim3,
    max_shared_memory: Size,
    memory_alignment: Size,
    memory_bandwidth_gbps: f64,
    preferred_vector_width: i32,
    subgroup_size: i32,
    unified_memory: bool,
}
impl Copy for DeviceInfo

pub enum ParamMode { In | Out | InOut | Scratch }
impl Copy for ParamMode

pub type View {
    memory: *mut Memory,
    offset: Size,
    shape: Shape,
    strides: Strides,
    dtype: DType,
}
impl Copy for View

pub type ViewDesc {
    shape: Shape,
    strides: Strides,
    dtype: DType,
    offset: Size = 0,
}
impl Copy for ViewDesc

pub type ParamDesc {
    name: str,
    mode: ParamMode,
    rank: i32,
    dtype: DType,
}
impl Copy for ParamDesc

pub type ProgramSig {
    params: Vec[ParamDesc],
}

pub type ProgramSource {
    ir: Vec[IRInst],
    aux: Vec[i32],
    strings: Vec[str],
    entry: str,
}

pub type BindEntry {
    name: str,
    view: View,
}
impl Copy for BindEntry

pub type Bindings {
    entries: Vec[BindEntry],
}

pub fn bind(name: str, view: View) -> BindEntry:
    BindEntry { name, view }

pub fn bindings_from(entries: Vec[BindEntry]) -> Bindings:
    Bindings { entries }
