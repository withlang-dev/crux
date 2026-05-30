use c_import("stdlib.h")

use crux.cpu_interp
use crux.core
use crux.device
use crux.errors
use crux.program
use crux.view

let CPU_STREAM_BYTES: i32 = 32
let CPU_EVENT_BYTES: i32 = 32

type CPUStream {
    base: *mut c_void,
    device: *mut Device,
}

type CPUEvent {
    base: *mut c_void,
    done: bool,
}

fn cpu_event_create() -> Result[*mut Event, SubstrateError]:
    let raw_opt = malloc(CPU_EVENT_BYTES)
    if raw_opt == None:
        return Err(.OutOfMemory)
    let raw = raw_opt.unwrap()
    let event = raw as *mut CPUEvent
    unsafe:
        (*event).base = raw
        (*event).done = true
    Ok(event as *mut Event)

fn stream_device(stream: *mut Stream) -> *mut Device:
    if stream == null:
        return default_device()
    unsafe { (*(stream as *mut CPUStream)).device }

fn null_view -> View:
    View {
        memory: null_memory(),
        offset: 0usize,
        shape: shape_scalar(),
        strides: strides_zero(),
        dtype: .Int32,
    }

fn range_fits(total: Size, offset: Size, count: Size) -> bool:
    if offset > total:
        return false
    count <= total - offset

fn validate_runtime_view(stream_dev: *mut Device, view: View) -> Result[i32, SubstrateError]:
    if view.memory == null:
        return Err(.InvalidView("view memory is null"))
    if memory_device(view.memory) != stream_dev:
        return Err(.StreamError("view memory device mismatch"))
    let span = view_byte_range(view)
    if span.0 > span.1:
        return Err(.InvalidView("view byte range is invalid"))
    if span.1 > memory_size(view.memory):
        return Err(.InvalidView("view byte range exceeds memory"))
    Ok(0)

fn resolve_bindings(stream_dev: *mut Device, sig: ProgramSig, bindings: Bindings) -> Result[Vec[View], SubstrateError]:
    var bi: i32 = 0
    while bi < bindings.entries.len() as i32:
        let name = bindings.entries[bi].name
        var next = bi + 1
        while next < bindings.entries.len() as i32:
            if bindings.entries[next].name == name:
                return Err(.ShapeMismatch("duplicate binding"))
            next = next + 1
        var known = false
        for pi in 0..sig.params.len():
            if sig.params[pi].name == name:
                known = true
                break
        if not known:
            return Err(.ShapeMismatch("unexpected binding"))
        bi = bi + 1
    let ordered: Vec[View] = Vec.new()
    for pi in 0..sig.params.len():
        let param = sig.params[pi]
        var found = false
        var matched = null_view()
        for bind_index in 0..bindings.entries.len():
            let entry = bindings.entries[bind_index]
            if entry.name == param.name:
                matched = entry.view
                found = true
                break
        if not found:
            return Err(.ShapeMismatch("missing binding"))
        if matched.dtype != param.dtype:
            return Err(.DTypeMismatch("binding dtype mismatch"))
        if matched.shape.rank != param.rank:
            return Err(.ShapeMismatch("binding rank mismatch"))
        let _ = validate_runtime_view(stream_dev, matched)?
        if (param.mode == .Out or param.mode == .InOut) and view_is_broadcasted(matched):
            return Err(.BroadcastWriteViolation)
        ordered.push(matched)
    Ok(ordered)

fn views_overlap(left: View, right: View) -> bool:
    if left.memory != right.memory:
        return false
    let left_range = view_byte_range(left)
    let right_range = view_byte_range(right)
    left_range.0 < right_range.1 and right_range.0 < left_range.1

fn validate_aliasing(sig: ProgramSig, ordered: Vec[View]) -> Result[i32, SubstrateError]:
    var left: i32 = 0
    while left < ordered.len() as i32:
        var right = left + 1
        while right < ordered.len() as i32:
            let left_mode = sig.params[left].mode
            let right_mode = sig.params[right].mode
            if (left_mode != .In or right_mode != .In) and views_overlap(ordered[left], ordered[right]):
                return Err(.ShapeMismatch("overlapping writable bindings"))
            right = right + 1
        left = left + 1
    Ok(0)

fn copy_n_bytes(src: *mut u8, dst: *mut u8, size: Size):
    if size == 0usize:
        return
    let src_addr = src as Size
    let dst_addr = dst as Size
    if src_addr < dst_addr and dst_addr < src_addr + size:
        var i = size
        while i > 0usize:
            i = i - 1usize
            unsafe:
                *(dst + i as i64) = *(src + i as i64)
        return
    var i: Size = 0usize
    while i < size:
        unsafe:
            *(dst + i as i64) = *(src + i as i64)
        i = i + 1usize

fn fill_n_bytes(dst: *mut u8, size: Size, value: Scalar):
    var i: Size = 0usize
    while i < size:
        let shift: u32 = (i * 8usize) as u32
        let byte = (value.bits >> shift) as u8
        unsafe:
            *(dst + i as i64) = byte
        i = i + 1usize

fn copy_indexed(src: View, dst: View, indices: Shape, dim: i32, elem_size: Size) -> Result[i32, SubstrateError]:
    if dim == src.shape.rank:
        let src_base = memory_ptr(src.memory)
        let dst_base = memory_ptr(dst.memory)
        if src_base == null or dst_base == null:
            return Err(.InvalidView("copy view memory is null"))
        let src_ptr = unsafe { src_base + view_offset_of(src, indices) as i64 }
        let dst_ptr = unsafe { dst_base + view_offset_of(dst, indices) as i64 }
        copy_n_bytes(src_ptr, dst_ptr, elem_size)
        return Ok(0)
    let extent = shape_get(src.shape, dim)
    var i: Size = 0usize
    while i < extent:
        let next = shape_set(indices, dim, i)
        let _ = copy_indexed(src, dst, next, dim + 1, elem_size)?
        i = i + 1usize
    Ok(0)

fn fill_indexed(dst: View, value: Scalar, indices: Shape, dim: i32, elem_size: Size) -> Result[i32, SubstrateError]:
    if dim == dst.shape.rank:
        let dst_base = memory_ptr(dst.memory)
        if dst_base == null:
            return Err(.InvalidView("fill view memory is null"))
        let dst_ptr = unsafe { dst_base + view_offset_of(dst, indices) as i64 }
        fill_n_bytes(dst_ptr, elem_size, value)
        return Ok(0)
    let extent = shape_get(dst.shape, dim)
    var i: Size = 0usize
    while i < extent:
        let next = shape_set(indices, dim, i)
        let _ = fill_indexed(dst, value, next, dim + 1, elem_size)?
        i = i + 1usize
    Ok(0)

pub fn stream_create(device: *mut Device) -> *mut Stream:
    let raw_opt = malloc(CPU_STREAM_BYTES)
    if raw_opt == None:
        return null
    let raw = raw_opt.unwrap()
    let stream = raw as *mut CPUStream
    unsafe:
        (*stream).base = raw
        (*stream).device = if device == null then default_device() else device
    stream as *mut Stream

pub fn stream_destroy(stream: *mut Stream):
    if stream == null:
        return
    let raw = unsafe { (*(stream as *mut CPUStream)).base }
    if raw != null:
        let _ = realloc(raw, 0)

pub fn stream_sync(stream: *mut Stream):
    let _ = stream

pub fn dispatch(stream: *mut Stream, prog: *mut Program, grid: [Size; 3], bindings: Bindings) -> Result[*mut Event, SubstrateError]:
    if stream == null:
        return Err(.StreamError("stream handle is null"))
    if prog == null:
        return Err(.CompileError("program handle is null"))
    let stream_dev = stream_device(stream)
    let limits = device_info(stream_dev).max_grid_dims
    if grid[0] > limits.x or grid[1] > limits.y or grid[2] > limits.z:
        return Err(.GridExceedsDevice("grid exceeds CPU device limits"))
    let prog_dev = program_device(prog)?
    if prog_dev != stream_dev:
        return Err(.StreamError("program device does not match stream"))
    let sig = program_sig(prog)
    let ordered = resolve_bindings(stream_dev, sig, bindings)?
    let _ = validate_aliasing(sig, ordered)?
    let ir = program_ir(prog)?
    let _ = interp_dispatch(ir, sig, ordered)?
    cpu_event_create()

pub fn copy_bytes(stream: *mut Stream, src: *mut Memory, src_offset: Size, dst: *mut Memory, dst_offset: Size, size: Size) -> Result[*mut Event, SubstrateError]:
    if stream == null:
        return Err(.StreamError("stream handle is null"))
    if src == null or dst == null:
        return Err(.InvalidView("copy memory is null"))
    let stream_dev = stream_device(stream)
    if memory_device(src) != stream_dev or memory_device(dst) != stream_dev:
        return Err(.StreamError("copy memory device mismatch"))
    let src_base = memory_ptr(src)
    let dst_base = memory_ptr(dst)
    if src_base == null or dst_base == null:
        return Err(.InvalidView("copy memory is null"))
    if not range_fits(memory_size(src), src_offset, size) or not range_fits(memory_size(dst), dst_offset, size):
        return Err(.ShapeMismatch("copy bytes range exceeds memory"))
    let src_ptr = unsafe { src_base + src_offset as i64 }
    let dst_ptr = unsafe { dst_base + dst_offset as i64 }
    copy_n_bytes(src_ptr, dst_ptr, size)
    cpu_event_create()

pub fn copy_view(stream: *mut Stream, src: View, dst: View) -> Result[*mut Event, SubstrateError]:
    if stream == null:
        return Err(.StreamError("stream handle is null"))
    let src_norm = view_canonicalize(src)
    let dst_norm = view_canonicalize(dst)
    let stream_dev = stream_device(stream)
    let _ = validate_runtime_view(stream_dev, src_norm)?
    let _ = validate_runtime_view(stream_dev, dst_norm)?
    if src_norm.dtype != dst_norm.dtype:
        return Err(.DTypeMismatch("copy dtype mismatch"))
    if view_elem_count(src_norm) != view_elem_count(dst_norm):
        return Err(.ShapeMismatch("copy requires matching element count"))
    if view_is_broadcasted(dst_norm):
        return Err(.BroadcastWriteViolation)
    if view_is_contiguous(src_norm) and view_is_contiguous(dst_norm):
        return copy_bytes(stream, src_norm.memory, src_norm.offset, dst_norm.memory, dst_norm.offset, view_byte_size(src_norm))
    if src_norm.shape.rank != dst_norm.shape.rank or not shape_equal(src_norm.shape, dst_norm.shape):
        return Err(.ShapeMismatch("copy requires matching canonical shapes"))
    let indices = shape_scalar()
    let _ = copy_indexed(src_norm, dst_norm, indices, 0, dtype_size(src_norm.dtype))?
    cpu_event_create()

pub fn fill(stream: *mut Stream, dst: View, value: Scalar) -> Result[*mut Event, SubstrateError]:
    if stream == null:
        return Err(.StreamError("stream handle is null"))
    let dst_norm = view_canonicalize(dst)
    let _ = validate_runtime_view(stream_device(stream), dst_norm)?
    if value.dtype != dst_norm.dtype:
        return Err(.DTypeMismatch("fill scalar dtype mismatch"))
    if view_is_broadcasted(dst_norm):
        return Err(.BroadcastWriteViolation)
    if view_is_contiguous(dst_norm):
        let base = memory_ptr(dst_norm.memory)
        if base == null:
            return Err(.InvalidView("fill view memory is null"))
        let elem_size = dtype_size(dst_norm.dtype)
        let count = view_elem_count(dst_norm)
        var i: Size = 0usize
        while i < count:
            let dst_ptr = unsafe { base + (dst_norm.offset + i * elem_size) as i64 }
            fill_n_bytes(dst_ptr, elem_size, value)
            i = i + 1usize
        return cpu_event_create()
    let indices = shape_scalar()
    let _ = fill_indexed(dst_norm, value, indices, 0, dtype_size(dst_norm.dtype))?
    cpu_event_create()

pub fn event_wait(event: *mut Event):
    let _ = event

pub fn event_is_done(event: *mut Event) -> bool:
    if event == null:
        return true
    unsafe { (*(event as *mut CPUEvent)).done }

pub fn event_elapsed(start: *mut Event, end: *mut Event) -> f64:
    let _ = start
    let _ = end
    0.0

pub fn event_destroy(event: *mut Event):
    if event == null:
        return
    let raw = unsafe { (*(event as *mut CPUEvent)).base }
    if raw != null:
        let _ = realloc(raw, 0)
