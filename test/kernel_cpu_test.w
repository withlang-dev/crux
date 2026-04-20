use crux.core
use crux.device
use crux.ir_text
use crux.kernels
use crux.memory
use crux.program
use crux.stream
use crux.view

fn unit_grid -> [Size; 3]:
    [1usize, 1usize, 1usize]

fn write_i32(mem: *mut Memory, index: i32, value: i32):
    let ptr = memory_ptr(mem) as *mut i32
    unsafe:
        *(ptr + index as i64) = value

fn read_i32(mem: *mut Memory, index: i32) -> i32:
    let ptr = memory_ptr(mem) as *mut i32
    unsafe: *(ptr + index as i64)

fn write_i64(mem: *mut Memory, index: i32, value: i64):
    let ptr = memory_ptr(mem) as *mut i64
    unsafe:
        *(ptr + index as i64) = value

fn read_i64(mem: *mut Memory, index: i32) -> i64:
    let ptr = memory_ptr(mem) as *mut i64
    unsafe: *(ptr + index as i64)

fn write_u32(mem: *mut Memory, index: i32, value: u32):
    let ptr = memory_ptr(mem) as *mut u32
    unsafe:
        *(ptr + index as i64) = value

fn read_u32(mem: *mut Memory, index: i32) -> u32:
    let ptr = memory_ptr(mem) as *mut u32
    unsafe: *(ptr + index as i64)

fn write_f32(mem: *mut Memory, index: i32, value: f32):
    let ptr = memory_ptr(mem) as *mut f32
    unsafe:
        *(ptr + index as i64) = value

fn read_f32_bits(mem: *mut Memory, index: i32) -> u32:
    let ptr = memory_ptr(mem) as *mut f32
    let value = unsafe: *(ptr + index as i64)
    unsafe: transmute[u32](value)

fn read_f32(mem: *mut Memory, index: i32) -> f32:
    let ptr = memory_ptr(mem) as *mut f32
    unsafe: *(ptr + index as i64)

fn write_f64(mem: *mut Memory, index: i32, value: f64):
    let ptr = memory_ptr(mem) as *mut f64
    unsafe:
        *(ptr + index as i64) = value

fn read_f64_bits(mem: *mut Memory, index: i32) -> u64:
    let ptr = memory_ptr(mem) as *mut f64
    let value = unsafe: *(ptr + index as i64)
    unsafe: transmute[u64](value)

fn abs_f32(v: f32) -> f32:
    if v < 0.0:
        return -v
    v

fn assert_close_f32(actual: f32, expected: f32, tolerance: f32):
    assert(abs_f32(actual - expected) <= tolerance)

fn compile_text_source(text: str) -> ProgramSource:
    match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            program_source("main")

fn compile_program(source: ProgramSource) -> *mut Program:
    match compile(device_info(default_device()), source)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

fn test_dispatch_map_add_i32_from_builder:
    let bytes = 16usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)

    write_i32(b_mem, 0, 10)
    write_i32(b_mem, 1, 20)
    write_i32(b_mem, 2, 30)
    write_i32(b_mem, 3, 40)

    let prog = compile_program(kernel_map_add_1d_i32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(4usize), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(4usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(4usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 11)
    assert(read_i32(out_mem, 1) == 22)
    assert(read_i32(out_mem, 2) == 33)
    assert(read_i32(out_mem, 3) == 44)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_map_add_i32_from_text_ir:
    let bytes = 16usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 5)
    write_i32(a_mem, 1, 6)
    write_i32(a_mem, 2, 7)
    write_i32(a_mem, 3, 8)

    write_i32(b_mem, 0, 1)
    write_i32(b_mem, 1, 2)
    write_i32(b_mem, 2, 3)
    write_i32(b_mem, 3, 4)

    let prog = compile_program(compile_text_source("param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = add %2 %3\nstore out [@0] %4\nblock_end 1\nreturn\n"))

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(4usize), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(4usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(4usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 6)
    assert(read_i32(out_mem, 1) == 8)
    assert(read_i32(out_mem, 2) == 10)
    assert(read_i32(out_mem, 3) == 12)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_map_add_f32_transposed_input_from_builder:
    let bytes = 24usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, 4.0)
    write_f32(a_mem, 2, 2.0)
    write_f32(a_mem, 3, 5.0)
    write_f32(a_mem, 4, 3.0)
    write_f32(a_mem, 5, 6.0)

    write_f32(b_mem, 0, 10.0)
    write_f32(b_mem, 1, 20.0)
    write_f32(b_mem, 2, 30.0)
    write_f32(b_mem, 3, 40.0)
    write_f32(b_mem, 4, 50.0)
    write_f32(b_mem, 5, 60.0)

    let a_view = view_transpose(view_contiguous(a_mem, shape2(3usize, 2usize), .Float32), 0, 1)
    let b_view = view_contiguous(b_mem, shape2(2usize, 3usize), .Float32)
    let out_view = view_contiguous(out_mem, shape2(2usize, 3usize), .Float32)

    let prog = compile_program(kernel_map_add_2d_f32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", a_view))
    entries.push(bind("b", b_view))
    entries.push(bind("out", out_view))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert_close_f32(read_f32(out_mem, 0), 11.0, 0.0001)
    assert_close_f32(read_f32(out_mem, 1), 22.0, 0.0001)
    assert_close_f32(read_f32(out_mem, 2), 33.0, 0.0001)
    assert_close_f32(read_f32(out_mem, 3), 44.0, 0.0001)
    assert_close_f32(read_f32(out_mem, 4), 55.0, 0.0001)
    assert_close_f32(read_f32(out_mem, 5), 66.0, 0.0001)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_map_add_f32_broadcast_read_from_builder:
    let a_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 12usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, 2.0)
    write_f32(a_mem, 2, 3.0)
    write_f32(a_mem, 3, 4.0)
    write_f32(a_mem, 4, 5.0)
    write_f32(a_mem, 5, 6.0)

    write_f32(b_mem, 0, 0.5)
    write_f32(b_mem, 1, 1.5)
    write_f32(b_mem, 2, 2.5)

    let a_view = view_contiguous(a_mem, shape2(2usize, 3usize), .Float32)
    let base_b = view_contiguous(b_mem, shape2(1usize, 3usize), .Float32)
    let b_view = match view_broadcast(base_b, shape2(2usize, 3usize))
        Ok(v) => v
        Err(_) =>
            assert(false)
            base_b
    let out_view = view_contiguous(out_mem, shape2(2usize, 3usize), .Float32)

    let prog = compile_program(kernel_map_add_2d_f32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", a_view))
    entries.push(bind("b", b_view))
    entries.push(bind("out", out_view))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert_close_f32(read_f32(out_mem, 0), 1.5, 0.0001)
    assert_close_f32(read_f32(out_mem, 1), 3.5, 0.0001)
    assert_close_f32(read_f32(out_mem, 2), 5.5, 0.0001)
    assert_close_f32(read_f32(out_mem, 3), 4.5, 0.0001)
    assert_close_f32(read_f32(out_mem, 4), 6.5, 0.0001)
    assert_close_f32(read_f32(out_mem, 5), 8.5, 0.0001)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_inplace_add_i32_from_builder:
    let bytes = 16usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)

    write_i32(b_mem, 0, 10)
    write_i32(b_mem, 1, 20)
    write_i32(b_mem, 2, 30)
    write_i32(b_mem, 3, 40)

    let prog = compile_program(kernel_map_inplace_add_1d_i32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(4usize), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(4usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(a_mem, 0) == 11)
    assert(read_i32(a_mem, 1) == 22)
    assert(read_i32(a_mem, 2) == 33)
    assert(read_i32(a_mem, 3) == 44)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)

fn test_dispatch_reduce_sum_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let a_mem = match alloc(default_device(), n * 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)
    write_i32(out_mem, 0, 0)

    let src = compile_text_source("param a in [N] i32\nparam out inout [] i32\n%0 = const i32 0\n%1 = const i32 4\nloop 0 %0 %1 1\nblock_begin 1\n%2 = load out []\n%3 = load a [@0]\n%4 = add %2 %3\nstore out [] %4\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 10)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_reduce_sum_i32_from_builder:
    let a_mem = match alloc(default_device(), 16usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)
    write_i32(out_mem, 0, 99)

    let prog = compile_program(kernel_reduce_sum_1d_i32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(4usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 10)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_matmul_i32_from_builder:
    let a_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 16usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)
    write_i32(a_mem, 4, 5)
    write_i32(a_mem, 5, 6)

    write_i32(b_mem, 0, 7)
    write_i32(b_mem, 1, 8)
    write_i32(b_mem, 2, 9)
    write_i32(b_mem, 3, 10)
    write_i32(b_mem, 4, 11)
    write_i32(b_mem, 5, 12)

    write_i32(out_mem, 0, 99)
    write_i32(out_mem, 1, 99)
    write_i32(out_mem, 2, 99)
    write_i32(out_mem, 3, 99)

    let prog = compile_program(kernel_matmul_2d_i32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape2(3usize, 2usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape2(2usize, 2usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 58)
    assert(read_i32(out_mem, 1) == 64)
    assert(read_i32(out_mem, 2) == 139)
    assert(read_i32(out_mem, 3) == 154)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_matmul_f32_from_builder:
    let a_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 16usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, 2.0)
    write_f32(a_mem, 2, 3.0)
    write_f32(a_mem, 3, 4.0)
    write_f32(a_mem, 4, 5.0)
    write_f32(a_mem, 5, 6.0)

    write_f32(b_mem, 0, 0.5)
    write_f32(b_mem, 1, 1.5)
    write_f32(b_mem, 2, 2.0)
    write_f32(b_mem, 3, -1.0)
    write_f32(b_mem, 4, 1.0)
    write_f32(b_mem, 5, 0.25)

    let prog = compile_program(kernel_matmul_2d_f32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Float32)))
    entries.push(bind("b", view_contiguous(b_mem, shape2(3usize, 2usize), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape2(2usize, 2usize), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert_close_f32(read_f32(out_mem, 0), 7.5, 0.0001)
    assert_close_f32(read_f32(out_mem, 1), 0.25, 0.0001)
    assert_close_f32(read_f32(out_mem, 2), 18.0, 0.0001)
    assert_close_f32(read_f32(out_mem, 3), 2.5, 0.0001)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_nested_loop_copy_2d_from_text_ir:
    let rows: i32 = 2
    let cols: i32 = 3
    let count = rows * cols
    let bytes = count as Size * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 11)
    write_i32(a_mem, 1, 12)
    write_i32(a_mem, 2, 13)
    write_i32(a_mem, 3, 21)
    write_i32(a_mem, 4, 22)
    write_i32(a_mem, 5, 23)

    let src = compile_text_source("param a in [M,N] i32\nparam out out [M,N] i32\n%0 = const i32 0\n%1 = const i32 2\n%2 = const i32 3\nloop 0 %0 %1 1\nblock_begin 1\nloop 1 %0 %2 2\nblock_begin 2\n%3 = load a [@0, @1]\nstore out [@0, @1] %3\nblock_end 2\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape2(2usize, 3usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 11)
    assert(read_i32(out_mem, 1) == 12)
    assert(read_i32(out_mem, 2) == 13)
    assert(read_i32(out_mem, 3) == 21)
    assert(read_i32(out_mem, 4) == 22)
    assert(read_i32(out_mem, 5) == 23)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_row_reduce_sum_2d_from_text_ir:
    let rows: i32 = 2
    let cols: i32 = 3
    let count = rows * cols
    let bytes = count as Size * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), rows as Size * 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)
    write_i32(a_mem, 4, 5)
    write_i32(a_mem, 5, 6)

    write_i32(out_mem, 0, 99)
    write_i32(out_mem, 1, 99)

    let src = compile_text_source("param a in [M,N] i32\nparam out out [M] i32\n%0 = const i32 0\n%1 = const i32 2\n%2 = const i32 3\nloop 0 %0 %1 1\nblock_begin 1\nstore out [@0] %0\nloop 1 %0 %2 2\nblock_begin 2\n%3 = load out [@0]\n%4 = load a [@0, @1]\n%5 = add %3 %4\nstore out [@0] %5\nblock_end 2\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(2usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 6)
    assert(read_i32(out_mem, 1) == 15)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_row_reduce_sum_2d_i32_from_builder:
    let a_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)
    write_i32(a_mem, 4, 5)
    write_i32(a_mem, 5, 6)
    write_i32(out_mem, 0, 99)
    write_i32(out_mem, 1, 99)

    let prog = compile_program(kernel_reduce_sum_rows_2d_i32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(2usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 6)
    assert(read_i32(out_mem, 1) == 15)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_reduce_max_rows_2d_f32_from_builder:
    let a_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, -3.0)
    write_f32(a_mem, 2, 2.0)
    write_f32(a_mem, 3, 7.0)
    write_f32(a_mem, 4, 0.5)
    write_f32(a_mem, 5, 6.0)

    let prog = compile_program(kernel_reduce_max_rows_2d_f32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(2usize), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert_close_f32(read_f32(out_mem, 0), 2.0, 0.0001)
    assert_close_f32(read_f32(out_mem, 1), 7.0, 0.0001)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_transpose_2d_i32_from_builder:
    let a_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 24usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)
    write_i32(a_mem, 4, 5)
    write_i32(a_mem, 5, 6)

    let prog = compile_program(kernel_transpose_2d_i32_source())

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape2(3usize, 2usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 1)
    assert(read_i32(out_mem, 1) == 4)
    assert(read_i32(out_mem, 2) == 2)
    assert(read_i32(out_mem, 3) == 5)
    assert(read_i32(out_mem, 4) == 3)
    assert(read_i32(out_mem, 5) == 6)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_transpose_2d_from_text_ir:
    let rows: i32 = 2
    let cols: i32 = 3
    let count = rows * cols
    let bytes = count as Size * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)
    write_i32(a_mem, 4, 5)
    write_i32(a_mem, 5, 6)

    let src = compile_text_source("param a in [M,N] i32\nparam out out [N,M] i32\n%0 = const i32 0\n%1 = const i32 2\n%2 = const i32 3\nloop 0 %0 %1 1\nblock_begin 1\nloop 1 %0 %2 2\nblock_begin 2\n%3 = load a [@0, @1]\nstore out [@1, @0] %3\nblock_end 2\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape2(3usize, 2usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 1)
    assert(read_i32(out_mem, 1) == 4)
    assert(read_i32(out_mem, 2) == 2)
    assert(read_i32(out_mem, 3) == 5)
    assert(read_i32(out_mem, 4) == 3)
    assert(read_i32(out_mem, 5) == 6)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_fma_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let c_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)

    write_i32(b_mem, 0, 10)
    write_i32(b_mem, 1, 20)
    write_i32(b_mem, 2, 30)
    write_i32(b_mem, 3, 40)

    write_i32(c_mem, 0, 100)
    write_i32(c_mem, 1, 200)
    write_i32(c_mem, 2, 300)
    write_i32(c_mem, 3, 400)

    let src = compile_text_source("param a in [N] i32\nparam b in [N] i32\nparam c in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = load c [@0]\n%5 = fma %2 %3 %4\nstore out [@0] %5\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Int32)))
    entries.push(bind("c", view_contiguous(c_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 110)
    assert(read_i32(out_mem, 1) == 240)
    assert(read_i32(out_mem, 2) == 390)
    assert(read_i32(out_mem, 3) == 560)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(c_mem)
    free(out_mem)

fn test_dispatch_neg_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, -2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, -4)

    let src = compile_text_source("param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = neg %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == -1)
    assert(read_i32(out_mem, 1) == 2)
    assert(read_i32(out_mem, 2) == -3)
    assert(read_i32(out_mem, 3) == 4)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_xor_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 3)
    write_i32(a_mem, 1, 5)
    write_i32(a_mem, 2, 12)
    write_i32(a_mem, 3, 7)

    write_i32(b_mem, 0, 1)
    write_i32(b_mem, 1, 6)
    write_i32(b_mem, 2, 10)
    write_i32(b_mem, 3, 3)

    let src = compile_text_source("param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = xor %2 %3\nstore out [@0] %4\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 2)
    assert(read_i32(out_mem, 1) == 3)
    assert(read_i32(out_mem, 2) == 6)
    assert(read_i32(out_mem, 3) == 4)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_shift_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 3)
    write_i32(a_mem, 2, 7)
    write_i32(a_mem, 3, 16)

    write_i32(b_mem, 0, 1)
    write_i32(b_mem, 1, 2)
    write_i32(b_mem, 2, 1)
    write_i32(b_mem, 3, 3)

    let src = compile_text_source("param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = shl %2 %3\n%5 = shr %4 %3\nstore out [@0] %5\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 1)
    assert(read_i32(out_mem, 1) == 3)
    assert(read_i32(out_mem, 2) == 7)
    assert(read_i32(out_mem, 3) == 16)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_bitcount_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, 3)
    write_i32(a_mem, 2, 8)
    write_i32(a_mem, 3, 40)

    let src = compile_text_source("param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = popcount %2\n%4 = clz %2\n%5 = ctz %2\n%6 = add %3 %4\n%7 = add %6 %5\nstore out [@0] %7\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 32)
    assert(read_i32(out_mem, 1) == 32)
    assert(read_i32(out_mem, 2) == 32)
    assert(read_i32(out_mem, 3) == 31)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_add_sat_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 2147483647)
    write_i32(a_mem, 1, 2147483640)
    write_i32(a_mem, 2, -2147483648)
    write_i32(a_mem, 3, -10)

    write_i32(b_mem, 0, 1)
    write_i32(b_mem, 1, 10)
    write_i32(b_mem, 2, -1)
    write_i32(b_mem, 3, -20)

    let src = compile_text_source("param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = add_sat %2 %3\nstore out [@0] %4\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 2147483647)
    assert(read_i32(out_mem, 1) == 2147483647)
    assert(read_i32(out_mem, 2) == -2147483648)
    assert(read_i32(out_mem, 3) == -30)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_exp_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)
    write_f32(a_mem, 1, 0.0)
    write_f32(a_mem, 2, 0.0)
    write_f32(a_mem, 3, 0.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = exp %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(1.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_log_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, 1.0)
    write_f32(a_mem, 2, 1.0)
    write_f32(a_mem, 3, 1.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = log %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(0.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_log2_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, 2.0)
    write_f32(a_mem, 2, 4.0)
    write_f32(a_mem, 3, 8.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = log2 %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(2.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(3.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_sin_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)
    write_f32(a_mem, 1, 0.0)
    write_f32(a_mem, 2, 0.0)
    write_f32(a_mem, 3, 0.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = sin %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(0.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_cos_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)
    write_f32(a_mem, 1, 0.0)
    write_f32(a_mem, 2, 0.0)
    write_f32(a_mem, 3, 0.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = cos %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(1.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_tanh_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)
    write_f32(a_mem, 1, 0.0)
    write_f32(a_mem, 2, 0.0)
    write_f32(a_mem, 3, 0.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = tanh %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(0.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_float_math_nontrivial_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)
    write_f32(a_mem, 1, 1.0)
    write_f32(a_mem, 2, -1.0)
    write_f32(a_mem, 3, 0.5)

    let exp_src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = exp %2\nstore out [@0] %3\nblock_end 1\nreturn\n")
    let exp_prog = match compile(device_info(default_device()), exp_src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let exp_entries: Vec[BindEntry] = Vec.new()
    exp_entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    exp_entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))
    let stream = stream_create(default_device())
    let exp_event = match dispatch(stream, exp_prog, unit_grid(), bindings_from(exp_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(exp_event))
    assert_close_f32(read_f32(out_mem, 0), 1.0, 0.001)
    assert_close_f32(read_f32(out_mem, 1), 2.7182817, 0.02)
    assert_close_f32(read_f32(out_mem, 2), 0.36787945, 0.02)
    assert_close_f32(read_f32(out_mem, 3), 1.6487212, 0.02)
    event_destroy(exp_event)
    program_destroy(exp_prog)

    write_f32(a_mem, 0, 0.0)
    write_f32(a_mem, 1, 1.0)
    write_f32(a_mem, 2, -1.0)
    write_f32(a_mem, 3, 0.5)
    let tanh_src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = tanh %2\nstore out [@0] %3\nblock_end 1\nreturn\n")
    let tanh_prog = match compile(device_info(default_device()), tanh_src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let tanh_entries: Vec[BindEntry] = Vec.new()
    tanh_entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    tanh_entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))
    let tanh_event = match dispatch(stream, tanh_prog, unit_grid(), bindings_from(tanh_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(tanh_event))
    assert_close_f32(read_f32(out_mem, 0), 0.0, 0.001)
    assert_close_f32(read_f32(out_mem, 1), 0.7615942, 0.02)
    assert_close_f32(read_f32(out_mem, 2), -0.7615942, 0.02)
    assert_close_f32(read_f32(out_mem, 3), 0.46211717, 0.02)
    event_destroy(tanh_event)
    program_destroy(tanh_prog)

    stream_destroy(stream)
    free(a_mem)
    free(out_mem)

fn test_dispatch_relu_from_text_ir:
    let count: i32 = 5
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, -3)
    write_i32(a_mem, 1, -1)
    write_i32(a_mem, 2, 0)
    write_i32(a_mem, 3, 2)
    write_i32(a_mem, 4, 5)

    let src = compile_text_source("param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 5\n%2 = const i32 0\nparallel 0 %0 %1 1\nblock_begin 1\n%3 = load a [@0]\n%4 = lt %3 %2\n%5 = select %4 %2 %3\nstore out [@0] %5\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 0)
    assert(read_i32(out_mem, 1) == 0)
    assert(read_i32(out_mem, 2) == 0)
    assert(read_i32(out_mem, 3) == 2)
    assert(read_i32(out_mem, 4) == 5)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_clip_from_text_ir:
    let count: i32 = 6
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, -5)
    write_i32(a_mem, 1, -1)
    write_i32(a_mem, 2, 0)
    write_i32(a_mem, 3, 3)
    write_i32(a_mem, 4, 6)
    write_i32(a_mem, 5, 9)

    let src = compile_text_source("param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 6\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = max %2 %0\n%4 = min %3 %1\nstore out [@0] %4\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 0)
    assert(read_i32(out_mem, 1) == 0)
    assert(read_i32(out_mem, 2) == 0)
    assert(read_i32(out_mem, 3) == 3)
    assert(read_i32(out_mem, 4) == 6)
    assert(read_i32(out_mem, 5) == 6)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_fma_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let c_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, 2.0)
    write_f32(a_mem, 2, 3.0)
    write_f32(a_mem, 3, 4.0)

    write_f32(b_mem, 0, 0.5)
    write_f32(b_mem, 1, 1.5)
    write_f32(b_mem, 2, 2.0)
    write_f32(b_mem, 3, 2.5)

    write_f32(c_mem, 0, 1.0)
    write_f32(c_mem, 1, 1.0)
    write_f32(c_mem, 2, 1.0)
    write_f32(c_mem, 3, 1.0)

    let src = compile_text_source("param a in [N] f32\nparam b in [N] f32\nparam c in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = load c [@0]\n%5 = fma %2 %3 %4\nstore out [@0] %5\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Float32)))
    entries.push(bind("c", view_contiguous(c_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(1.5).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(4.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(7.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(11.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(c_mem)
    free(out_mem)

fn test_dispatch_fma_f64_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 8usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let c_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f64(a_mem, 0, 1.0)
    write_f64(a_mem, 1, 2.0)
    write_f64(a_mem, 2, 3.0)
    write_f64(a_mem, 3, 4.0)

    write_f64(b_mem, 0, 0.5)
    write_f64(b_mem, 1, 1.5)
    write_f64(b_mem, 2, 2.0)
    write_f64(b_mem, 3, 2.5)

    write_f64(c_mem, 0, 1.0)
    write_f64(c_mem, 1, 1.0)
    write_f64(c_mem, 2, 1.0)
    write_f64(c_mem, 3, 1.0)

    let src = compile_text_source("param a in [N] f64\nparam b in [N] f64\nparam c in [N] f64\nparam out out [N] f64\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = load c [@0]\n%5 = fma %2 %3 %4\nstore out [@0] %5\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float64)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Float64)))
    entries.push(bind("c", view_contiguous(c_mem, shape1(n), .Float64)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float64)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f64_bits(out_mem, 0) == scalar_f64(1.5).bits)
    assert(read_f64_bits(out_mem, 1) == scalar_f64(4.0).bits)
    assert(read_f64_bits(out_mem, 2) == scalar_f64(7.0).bits)
    assert(read_f64_bits(out_mem, 3) == scalar_f64(11.0).bits)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(c_mem)
    free(out_mem)

fn test_dispatch_cast_i32_to_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let in_bytes = n * 4usize
    let out_bytes = n * 4usize
    let a_mem = match alloc(default_device(), in_bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), out_bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 1)
    write_i32(a_mem, 1, -2)
    write_i32(a_mem, 2, 3)
    write_i32(a_mem, 3, 4)

    let src = compile_text_source("param a in [N] i32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = cast f32 %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(-2.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(3.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(4.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_abs_i32_from_text_ir:
    let count: i32 = 5
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, -3)
    write_i32(a_mem, 1, -1)
    write_i32(a_mem, 2, 0)
    write_i32(a_mem, 3, 2)
    write_i32(a_mem, 4, -5)

    let src = compile_text_source("param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 5\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = abs %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 3)
    assert(read_i32(out_mem, 1) == 1)
    assert(read_i32(out_mem, 2) == 0)
    assert(read_i32(out_mem, 3) == 2)
    assert(read_i32(out_mem, 4) == 5)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_mod_i32_from_text_ir:
    let count: i32 = 5
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i32(a_mem, 0, 10)
    write_i32(a_mem, 1, 11)
    write_i32(a_mem, 2, 12)
    write_i32(a_mem, 3, 13)
    write_i32(a_mem, 4, 14)

    write_i32(b_mem, 0, 3)
    write_i32(b_mem, 1, 3)
    write_i32(b_mem, 2, 5)
    write_i32(b_mem, 3, 5)
    write_i32(b_mem, 4, 4)

    let src = compile_text_source("param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 5\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = mod %2 %3\nstore out [@0] %4\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 1)
    assert(read_i32(out_mem, 1) == 2)
    assert(read_i32(out_mem, 2) == 2)
    assert(read_i32(out_mem, 3) == 3)
    assert(read_i32(out_mem, 4) == 2)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_floor_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, -1.25)
    write_f32(a_mem, 1, -0.1)
    write_f32(a_mem, 2, 0.0)
    write_f32(a_mem, 3, 2.75)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = floor %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(-2.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(-1.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(2.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_ceil_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, -1.25)
    write_f32(a_mem, 1, -0.1)
    write_f32(a_mem, 2, 0.0)
    write_f32(a_mem, 3, 2.75)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = ceil %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(-1.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(3.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_round_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, -1.25)
    write_f32(a_mem, 1, -0.1)
    write_f32(a_mem, 2, 0.0)
    write_f32(a_mem, 3, 2.75)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = round %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(-1.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(3.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_sqrt_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)
    write_f32(a_mem, 1, 1.0)
    write_f32(a_mem, 2, 4.0)
    write_f32(a_mem, 3, 9.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = sqrt %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(2.0).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(3.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_rsqrt_f32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)
    write_f32(a_mem, 1, 4.0)
    write_f32(a_mem, 2, 16.0)
    write_f32(a_mem, 3, 64.0)

    let src = compile_text_source("param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = rsqrt %2\nstore out [@0] %3\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Float32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(1.0).bits as u32)
    assert(read_f32_bits(out_mem, 1) == scalar_f32(0.5).bits as u32)
    assert(read_f32_bits(out_mem, 2) == scalar_f32(0.25).bits as u32)
    assert(read_f32_bits(out_mem, 3) == scalar_f32(0.125).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_dispatch_add_i64_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 8usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i64(a_mem, 0, 10 as i64)
    write_i64(a_mem, 1, 20 as i64)
    write_i64(a_mem, 2, 30 as i64)
    write_i64(a_mem, 3, 40 as i64)

    write_i64(b_mem, 0, 1 as i64)
    write_i64(b_mem, 1, 2 as i64)
    write_i64(b_mem, 2, 3 as i64)
    write_i64(b_mem, 3, 4 as i64)

    let src = compile_text_source("param a in [N] i64\nparam b in [N] i64\nparam out out [N] i64\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = add %2 %3\nstore out [@0] %4\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int64)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Int64)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int64)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i64(out_mem, 0) == 11 as i64)
    assert(read_i64(out_mem, 1) == 22 as i64)
    assert(read_i64(out_mem, 2) == 33 as i64)
    assert(read_i64(out_mem, 3) == 44 as i64)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_xor_u32_from_text_ir:
    let count: i32 = 4
    let n = count as Size
    let bytes = n * 4usize
    let a_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_u32(a_mem, 0, 12 as u32)
    write_u32(a_mem, 1, 7 as u32)
    write_u32(a_mem, 2, 255 as u32)
    write_u32(a_mem, 3, 1024 as u32)

    write_u32(b_mem, 0, 10 as u32)
    write_u32(b_mem, 1, 3 as u32)
    write_u32(b_mem, 2, 15 as u32)
    write_u32(b_mem, 3, 1 as u32)

    let src = compile_text_source("param a in [N] u32\nparam b in [N] u32\nparam out out [N] u32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = xor %2 %3\nstore out [@0] %4\nblock_end 1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .UInt32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .UInt32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .UInt32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_u32(out_mem, 0) == 6 as u32)
    assert(read_u32(out_mem, 1) == 4 as u32)
    assert(read_u32(out_mem, 2) == 240 as u32)
    assert(read_u32(out_mem, 3) == 1025 as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn main:
    test_dispatch_map_add_i32_from_builder()
    test_dispatch_map_add_i32_from_text_ir()
    test_dispatch_map_add_f32_transposed_input_from_builder()
    test_dispatch_map_add_f32_broadcast_read_from_builder()
    test_dispatch_inplace_add_i32_from_builder()
    test_dispatch_reduce_sum_from_text_ir()
    test_dispatch_reduce_sum_i32_from_builder()
    test_dispatch_matmul_i32_from_builder()
    test_dispatch_matmul_f32_from_builder()
    test_dispatch_nested_loop_copy_2d_from_text_ir()
    test_dispatch_row_reduce_sum_2d_i32_from_builder()
    test_dispatch_row_reduce_sum_2d_from_text_ir()
    test_dispatch_reduce_max_rows_2d_f32_from_builder()
    test_dispatch_transpose_2d_i32_from_builder()
    test_dispatch_transpose_2d_from_text_ir()
    test_dispatch_fma_from_text_ir()
    test_dispatch_neg_from_text_ir()
    test_dispatch_xor_from_text_ir()
    test_dispatch_shift_from_text_ir()
    test_dispatch_bitcount_from_text_ir()
    test_dispatch_add_sat_from_text_ir()
    test_dispatch_exp_f32_from_text_ir()
    test_dispatch_log_f32_from_text_ir()
    test_dispatch_log2_f32_from_text_ir()
    test_dispatch_sin_f32_from_text_ir()
    test_dispatch_cos_f32_from_text_ir()
    test_dispatch_tanh_f32_from_text_ir()
    test_dispatch_float_math_nontrivial_f32_from_text_ir()
    test_dispatch_relu_from_text_ir()
    test_dispatch_clip_from_text_ir()
    test_dispatch_fma_f32_from_text_ir()
    test_dispatch_fma_f64_from_text_ir()
    test_dispatch_cast_i32_to_f32_from_text_ir()
    test_dispatch_abs_i32_from_text_ir()
    test_dispatch_mod_i32_from_text_ir()
    test_dispatch_floor_f32_from_text_ir()
    test_dispatch_ceil_f32_from_text_ir()
    test_dispatch_round_f32_from_text_ir()
    test_dispatch_sqrt_f32_from_text_ir()
    test_dispatch_rsqrt_f32_from_text_ir()
    test_dispatch_add_i64_from_text_ir()
    test_dispatch_xor_u32_from_text_ir()
