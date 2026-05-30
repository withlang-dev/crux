use crux.collective
use crux.core
use crux.device
use crux.ir_text
use crux.memory
use crux.program
use crux.stream
use crux.view

fn unit_grid -> [Size; 3]:
    [1usize, 1usize, 1usize]

fn compile_text_source(text: str) -> ProgramSource:
    match parse_ir_text(text):
        Ok(v) => v
        Err(_) =>
            assert(false)
            program_source("main")

fn test_default_device_info_host_backed:
    let device = default_device()
    let info = device_info(device)
    assert(device != null_device())
    assert(info.name == "cpu")
    assert(info.kind == .CPU)
    assert(info.memory_total > 0usize)
    assert(info.memory_available > 0usize)
    assert(info.max_workgroup_size == 1024)
    assert(info.memory_alignment > 0usize)
    assert(info.memory_bandwidth_gbps > 0.0)
    assert(info.unified_memory)

fn test_alloc_roundtrip:
    let mem = match alloc(default_device(), 16usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    assert(mem != null_memory())
    assert(memory_size(mem) == 16usize)
    assert(memory_device(mem) == default_device())
    let ptr = memory_ptr(mem) as *mut i32
    assert(ptr != null)
    unsafe:
        *ptr = 123
    assert(unsafe *ptr == 123)
    free(mem)

fn test_memory_from_ptr_is_borrowed:
    let owner = match alloc(default_device(), 8usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let ptr = memory_ptr(owner)
    unsafe:
        *(ptr + 0i64) = 7u8
        *(ptr + 1i64) = 9u8
    let borrowed = memory_from_ptr(default_device(), ptr, 8usize)
    assert(borrowed != null_memory())
    assert(memory_ptr(borrowed) == ptr)
    free(borrowed)
    assert(unsafe *(ptr + 0i64) == 7u8)
    assert(unsafe *(ptr + 1i64) == 9u8)
    free(owner)

fn test_arena_allocation_lifecycle:
    let arena = match arena_create(default_device(), 4096usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_arena()
    assert(arena != null_arena())
    assert(arena_used(arena) == 0usize)

    let left = match arena_alloc(arena, 16usize, 8usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let right = match arena_alloc(arena, 8usize, 8usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    assert(memory_size(left) == 16usize)
    assert(memory_size(right) == 8usize)
    assert(arena_used(arena) == 24usize)

    let left_ptr = memory_ptr(left) as *mut i32
    let right_ptr = memory_ptr(right) as *mut i32
    assert(left_ptr != null)
    assert(right_ptr != null)
    unsafe:
        *left_ptr = 7
        *right_ptr = 9
    assert(unsafe *left_ptr == 7)
    assert(unsafe *right_ptr == 9)

    free(left)
    free(right)
    arena_reset(arena)
    assert(arena_used(arena) == 0usize)
    arena_destroy(arena)

fn test_arena_alloc_rejects_overflow:
    let arena = match arena_create(default_device(), 16usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_arena()
    let first = match arena_alloc(arena, 8usize, 8usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let got_oom = match arena_alloc(arena, 8usize, 32usize):
        Err(.OutOfMemory) => true
        _ => false
    assert(got_oom)
    free(first)
    arena_destroy(arena)

fn test_program_source_defaults:
    let src = program_source("main")
    assert(src.entry == "main")
    assert(src.ir.len() == 0)
    assert(src.aux.len() == 0)
    assert(src.strings.len() == 0)

fn test_compile_and_dispatch:
    let program = match compile(device_info(default_device()), program_source("main")):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    assert(program != null_program())
    assert(stream != null_stream())
    let bindings = Bindings { entries: Vec.new() }
    let event = match dispatch(stream, program, unit_grid(), bindings):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event != null_event())
    assert(event_is_done(event))
    event_wait(event)
    event_destroy(event)
    stream_destroy(stream)
    program_destroy(program)

fn test_dispatch_writes_through_borrowed_memory:
    let owner = match alloc(default_device(), 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let borrowed = memory_from_ptr(default_device(), memory_ptr(owner), 4usize)
    let program = match compile(device_info(default_device()), compile_text_source("param out out [] i32\n%0 = const i32 7\nstore out [] %0\nreturn\n")):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("out", view_contiguous(borrowed, shape_scalar(), .Int32)))
    let event = match dispatch(stream, program, unit_grid(), bindings_from(entries)):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(event))
    assert(unsafe *(memory_ptr(owner) as *mut i32) == 7)
    event_destroy(event)
    stream_destroy(stream)
    program_destroy(program)
    free(borrowed)
    free(owner)

fn test_dispatch_rejects_oversized_grid:
    let program = match compile(device_info(default_device()), program_source("main")):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let bindings = Bindings { entries: Vec.new() }
    let got_limit_error = match dispatch(stream, program, [70000usize, 1usize, 1usize], bindings):
        Err(.GridExceedsDevice(_)) => true
        _ => false
    assert(got_limit_error)
    stream_destroy(stream)
    program_destroy(program)

fn test_dispatch_rejects_unexpected_binding:
    let program = match compile(device_info(default_device()), program_source("main")):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("extra", view_contiguous(null_memory(), shape_scalar(), .Int32)))
    let got_shape_error = match dispatch(stream, program, unit_grid(), bindings_from(entries)):
        Err(.ShapeMismatch(_)) => true
        _ => false
    assert(got_shape_error)
    stream_destroy(stream)
    program_destroy(program)

fn test_dispatch_rejects_duplicate_binding_names:
    let program = match compile(device_info(default_device()), compile_text_source("param a in [] i32\nreturn\n")):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    let value = view_contiguous(null_memory(), shape_scalar(), .Int32)
    entries.push(bind("a", value))
    entries.push(bind("a", value))
    let got_shape_error = match dispatch(stream, program, unit_grid(), bindings_from(entries)):
        Err(.ShapeMismatch(_)) => true
        _ => false
    assert(got_shape_error)
    stream_destroy(stream)
    program_destroy(program)

fn test_dispatch_rejects_overlapping_writable_bindings:
    let program = match compile(device_info(default_device()), compile_text_source("param a out [] i32\nparam b out [] i32\nreturn\n")):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let mem = match alloc(default_device(), 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let same = view_contiguous(mem, shape_scalar(), .Int32)
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", same))
    entries.push(bind("b", same))
    let got_shape_error = match dispatch(stream, program, unit_grid(), bindings_from(entries)):
        Err(.ShapeMismatch(_)) => true
        _ => false
    assert(got_shape_error)
    free(mem)
    stream_destroy(stream)
    program_destroy(program)

fn test_dispatch_rejects_out_of_range_view:
    let program = match compile(device_info(default_device()), compile_text_source("param a in [] i32\nreturn\n")):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let mem = match alloc(default_device(), 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let bad = View {
        memory: mem,
        offset: 4usize,
        shape: shape_scalar(),
        strides: strides_zero(),
        dtype: .Int32,
    }
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", bad))
    let got_invalid = match dispatch(stream, program, unit_grid(), bindings_from(entries)):
        Err(.InvalidView(_)) => true
        _ => false
    assert(got_invalid)
    free(mem)
    stream_destroy(stream)
    program_destroy(program)

fn test_collective_reports_unsupported:
    let src = view_contiguous(null_memory(), shape1(4usize), .Float32)
    let dst = view_contiguous(null_memory(), shape1(4usize), .Float32)
    let got_unsupported = match allreduce_sum(stream_create(default_device()), src, dst):
        Err(.Unsupported(_)) => true
        _ => false
    assert(got_unsupported)

fn test_copy_and_fill_roundtrip:
    let stream = stream_create(default_device())
    let src_mem = match alloc(default_device(), 16usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let dst_mem = match alloc(default_device(), 16usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    let src_view = view_contiguous(src_mem, shape1(4usize), .Int32)
    let dst_view = view_contiguous(dst_mem, shape1(4usize), .Int32)

    let fill_event = match fill(stream, src_view, scalar_i32(11)):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(fill_event))
    event_destroy(fill_event)

    let copy_event = match copy_view(stream, src_view, dst_view):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(copy_event))
    event_destroy(copy_event)

    let ptr = memory_ptr(dst_mem) as *mut i32
    assert(unsafe *(ptr + 0i64) == 11)
    assert(unsafe *(ptr + 1i64) == 11)
    assert(unsafe *(ptr + 2i64) == 11)
    assert(unsafe *(ptr + 3i64) == 11)

    stream_destroy(stream)
    free(src_mem)
    free(dst_mem)

fn test_fill_float32_writes_bits:
    let stream = stream_create(default_device())
    let mem = match alloc(default_device(), 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let dst = view_contiguous(mem, shape_scalar(), .Float32)
    let event = match fill(stream, dst, scalar_f32(1.5)):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(event))
    event_destroy(event)
    let ptr = memory_ptr(mem) as *mut f32
    assert(unsafe *ptr == 1.5)
    stream_destroy(stream)
    free(mem)

fn test_copy_and_fill_reject_broadcast_writes:
    let stream = stream_create(default_device())
    let src_mem = match alloc(default_device(), 48usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let dst_mem = match alloc(default_device(), 16usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let src = view_contiguous(src_mem, shape2(3usize, 4usize), .Float32)
    let row = view_contiguous(dst_mem, shape2(1usize, 4usize), .Float32)
    let dst = match view_broadcast(row, shape2(3usize, 4usize)):
        Ok(v) => v
        Err(_) =>
            assert(false)
            row

    let copy_error = match copy_view(stream, src, dst):
        Err(.BroadcastWriteViolation) => true
        _ => false
    assert(copy_error)

    let fill_error = match fill(stream, dst, scalar_f32(2.0)):
        Err(.BroadcastWriteViolation) => true
        _ => false
    assert(fill_error)

    stream_destroy(stream)
    free(src_mem)
    free(dst_mem)

fn test_copy_bytes_roundtrip:
    let stream = stream_create(default_device())
    let src_mem = match alloc(default_device(), 8usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let dst_mem = match alloc(default_device(), 8usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    let src_ptr = memory_ptr(src_mem)
    unsafe:
        *(src_ptr + 0i64) = 1u8
        *(src_ptr + 1i64) = 2u8
        *(src_ptr + 2i64) = 3u8
        *(src_ptr + 3i64) = 4u8

    let event = match copy_bytes(stream, src_mem, 0usize, dst_mem, 2usize, 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(event))
    event_destroy(event)

    let dst_ptr = memory_ptr(dst_mem)
    assert(unsafe *(dst_ptr + 2i64) == 1u8)
    assert(unsafe *(dst_ptr + 3i64) == 2u8)
    assert(unsafe *(dst_ptr + 4i64) == 3u8)
    assert(unsafe *(dst_ptr + 5i64) == 4u8)

    stream_destroy(stream)
    free(src_mem)
    free(dst_mem)

fn test_copy_bytes_handles_overlap_like_memmove:
    let stream = stream_create(default_device())
    let mem = match alloc(default_device(), 8usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let ptr = memory_ptr(mem)
    unsafe:
        *(ptr + 0i64) = 1u8
        *(ptr + 1i64) = 2u8
        *(ptr + 2i64) = 3u8
        *(ptr + 3i64) = 4u8
        *(ptr + 4i64) = 5u8
        *(ptr + 5i64) = 6u8

    let event = match copy_bytes(stream, mem, 0usize, mem, 2usize, 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(event))
    event_destroy(event)

    assert(unsafe *(ptr + 0i64) == 1u8)
    assert(unsafe *(ptr + 1i64) == 2u8)
    assert(unsafe *(ptr + 2i64) == 1u8)
    assert(unsafe *(ptr + 3i64) == 2u8)
    assert(unsafe *(ptr + 4i64) == 3u8)
    assert(unsafe *(ptr + 5i64) == 4u8)

    stream_destroy(stream)
    free(mem)

fn test_copy_rejects_out_of_range_view:
    let stream = stream_create(default_device())
    let src_mem = match alloc(default_device(), 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let dst_mem = match alloc(default_device(), 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let src = view_contiguous(src_mem, shape_scalar(), .Int32)
    let bad_dst = View {
        memory: dst_mem,
        offset: 4usize,
        shape: shape_scalar(),
        strides: strides_zero(),
        dtype: .Int32,
    }
    let got_invalid = match copy_view(stream, src, bad_dst):
        Err(.InvalidView(_)) => true
        _ => false
    assert(got_invalid)
    stream_destroy(stream)
    free(src_mem)
    free(dst_mem)

fn test_fill_rejects_out_of_range_view:
    let stream = stream_create(default_device())
    let mem = match alloc(default_device(), 4usize):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let bad = View {
        memory: mem,
        offset: 4usize,
        shape: shape_scalar(),
        strides: strides_zero(),
        dtype: .Int32,
    }
    let got_invalid = match fill(stream, bad, scalar_i32(1)):
        Err(.InvalidView(_)) => true
        _ => false
    assert(got_invalid)
    stream_destroy(stream)
    free(mem)

fn test_event_stub_defaults:
    assert(event_is_done(null_event()))
    assert(event_elapsed(null_event(), null_event()) == 0.0)

fn main:
    test_default_device_info_host_backed()
    test_alloc_roundtrip()
    test_memory_from_ptr_is_borrowed()
    test_arena_allocation_lifecycle()
    test_arena_alloc_rejects_overflow()
    test_program_source_defaults()
    test_compile_and_dispatch()
    test_dispatch_writes_through_borrowed_memory()
    test_dispatch_rejects_oversized_grid()
    test_dispatch_rejects_unexpected_binding()
    test_dispatch_rejects_duplicate_binding_names()
    test_dispatch_rejects_overlapping_writable_bindings()
    test_dispatch_rejects_out_of_range_view()
    test_collective_reports_unsupported()
    test_copy_and_fill_roundtrip()
    test_fill_float32_writes_bits()
    test_copy_and_fill_reject_broadcast_writes()
    test_copy_bytes_roundtrip()
    test_copy_bytes_handles_overlap_like_memmove()
    test_copy_rejects_out_of_range_view()
    test_fill_rejects_out_of_range_view()
    test_event_stub_defaults()
