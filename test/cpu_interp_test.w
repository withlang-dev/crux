use crux.core
use crux.device
use crux.ir
use crux.ir_text
use crux.memory
use crux.program
use crux.stream
use crux.view

fn unit_grid -> [Size; 3]:
    [1usize, 1usize, 1usize]

fn fill_i32(mem: *mut Memory, n: i32, base_value: i32):
    let ptr = memory_ptr(mem) as *mut i32
    var i: i32 = 0
    while i < n:
        unsafe:
            *(ptr + i as i64) = base_value + i
        i = i + 1

fn write_i32(mem: *mut Memory, index: i32, value: i32):
    let ptr = memory_ptr(mem) as *mut i32
    unsafe:
        *(ptr + index as i64) = value

fn read_i32(mem: *mut Memory, index: i32) -> i32:
    let ptr = memory_ptr(mem) as *mut i32
    unsafe: *(ptr + index as i64)

fn write_i8(mem: *mut Memory, index: i32, value: i8):
    let ptr = memory_ptr(mem) as *mut i8
    unsafe:
        *(ptr + index as i64) = value

fn read_i8(mem: *mut Memory, index: i32) -> i8:
    let ptr = memory_ptr(mem) as *mut i8
    unsafe: *(ptr + index as i64)

fn write_i16(mem: *mut Memory, index: i32, value: i16):
    let ptr = memory_ptr(mem) as *mut i16
    unsafe:
        *(ptr + index as i64) = value

fn read_i16(mem: *mut Memory, index: i32) -> i16:
    let ptr = memory_ptr(mem) as *mut i16
    unsafe: *(ptr + index as i64)

fn write_i64(mem: *mut Memory, index: i32, value: i64):
    let ptr = memory_ptr(mem) as *mut i64
    unsafe:
        *(ptr + index as i64) = value

fn read_i64(mem: *mut Memory, index: i32) -> i64:
    let ptr = memory_ptr(mem) as *mut i64
    unsafe: *(ptr + index as i64)

fn write_u8(mem: *mut Memory, index: i32, value: u8):
    let ptr = memory_ptr(mem) as *mut u8
    unsafe:
        *(ptr + index as i64) = value

fn read_u8(mem: *mut Memory, index: i32) -> u8:
    let ptr = memory_ptr(mem) as *mut u8
    unsafe: *(ptr + index as i64)

fn write_u16(mem: *mut Memory, index: i32, value: u16):
    let ptr = memory_ptr(mem) as *mut u16
    unsafe:
        *(ptr + index as i64) = value

fn read_u16(mem: *mut Memory, index: i32) -> u16:
    let ptr = memory_ptr(mem) as *mut u16
    unsafe: *(ptr + index as i64)

fn write_u32(mem: *mut Memory, index: i32, value: u32):
    let ptr = memory_ptr(mem) as *mut u32
    unsafe:
        *(ptr + index as i64) = value

fn read_u32(mem: *mut Memory, index: i32) -> u32:
    let ptr = memory_ptr(mem) as *mut u32
    unsafe: *(ptr + index as i64)

fn write_u64(mem: *mut Memory, index: i32, value: u64):
    let ptr = memory_ptr(mem) as *mut u64
    unsafe:
        *(ptr + index as i64) = value

fn read_u64(mem: *mut Memory, index: i32) -> u64:
    let ptr = memory_ptr(mem) as *mut u64
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

fn alloc_memory(bytes: usize) -> *mut Memory:
    match alloc(default_device(), bytes)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

fn abs_f32(v: f32) -> f32:
    if v < 0.0:
        return -v
    v

fn assert_close_f32(actual: f32, expected: f32, tolerance: f32):
    assert(abs_f32(actual - expected) <= tolerance)

fn compile_program(source: ProgramSource) -> *mut Program:
    match compile(device_info(default_device()), source)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

fn dispatch_entries(stream: *mut Stream, prog: *mut Program, entries: Vec[BindEntry]) -> *mut Event:
    match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

fn build_reduce_1d_source(op: i32, dtype: DType, count: i32) -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 1, dtype))
    insts.push(ir_param_inst(1, .Out, 0, dtype))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, count))

    let reduce_meta = aux.len() as i32
    aux.push(0)
    aux.push(1)
    aux.push(3)
    insts.push(ir_inst(op, dtype, 2, 0, 1, reduce_meta))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    let load_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, dtype, 3, ir_param_ref(0), load_index, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_STORE, dtype, ir_param_ref(1), 0, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    ProgramSource {
        ir: insts,
        aux,
        strings,
        entry: "main",
    }

fn build_parallel_add_source(n: i32) -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 1, .Int32))
    insts.push(ir_param_inst(1, .In, 1, .Int32))
    insts.push(ir_param_inst(2, .Out, 1, .Int32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, n))
    insts.push(ir_inst(IROP_PARALLEL, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 2, ir_param_ref(0), 0, 0))

    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 3, ir_param_ref(1), 1, 0))

    insts.push(ir_inst(IROP_ADD, .Int32, 4, 2, 3, 0))

    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(2), 2, 4, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    ProgramSource {
        ir: insts,
        aux,
        strings,
        entry: "main",
    }

fn compile_text_source(text: str) -> ProgramSource:
    match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            program_source("main")

fn test_compile_ir_text_scalar_add:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 4
        *(memory_ptr(b_mem) as *mut i32) = 5

    let src = compile_text_source("param a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 9)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_parallel_add_i32:
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

    fill_i32(a_mem, count, 1)
    fill_i32(b_mem, count, 10)

    let prog = match compile(device_info(default_device()), build_parallel_add_source(count))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sig = program_sig(prog)
    assert(sig.params.len() == 3)
    assert(sig.params[0].mode == .In)
    assert(sig.params[2].mode == .Out)
    assert(sig.params[2].rank == 1)
    assert(sig.params[2].dtype == .Int32)

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
    assert(read_i32(out_mem, 0) == 11)
    assert(read_i32(out_mem, 1) == 13)
    assert(read_i32(out_mem, 2) == 15)
    assert(read_i32(out_mem, 3) == 17)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_dispatch_parallel_add_from_text_ir:
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

    fill_i32(a_mem, count, 1)
    fill_i32(b_mem, count, 10)

    let src = compile_text_source("param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\n%3 = load b [@0]\n%4 = add %2 %3\nstore out [@0] %4\nblock_end 1\nreturn\n")

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
    assert(read_i32(out_mem, 0) == 11)
    assert(read_i32(out_mem, 1) == 13)
    assert(read_i32(out_mem, 2) == 15)
    assert(read_i32(out_mem, 3) == 17)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_clamp:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 9

    let src = compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = const i32 0\n%1 = const i32 7\n%2 = load a []\n%3 = clamp %2 %0 %1\nstore out [] %3\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 7)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_add_with_spec_header:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 4
        *(memory_ptr(b_mem) as *mut i32) = 5

    let src = compile_text_source("spec_constant TILE i32 16\nparam a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sig = program_sig(prog)
    assert(sig.params.len() == 3)
    assert(sig.params[0].name == "a")
    assert(sig.params[1].name == "b")
    assert(sig.params[2].name == "out")
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 9)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_bitwise_i32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 6
        *(memory_ptr(b_mem) as *mut i32) = 10

    let src = compile_text_source("param a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = xor %0 %1\nstore out [] %2\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 12)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_not_i32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 5

    let src = compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = not %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == ~5)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_shift_i32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 3
        *(memory_ptr(b_mem) as *mut i32) = 2

    let src = compile_text_source("param a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = shl %0 %1\n%3 = shr %2 %1\nstore out [] %3\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 3)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_bitcount_i32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 40

    let src = compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = popcount %0\n%2 = clz %0\n%3 = ctz %0\n%4 = add %1 %2\n%5 = add %4 %3\nstore out [] %5\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 31)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_sat_i32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 2147483647
        *(memory_ptr(b_mem) as *mut i32) = 1

    let add_src = compile_text_source("param a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = add_sat %0 %1\nstore out [] %2\nreturn\n")
    let add_prog = match compile(device_info(default_device()), add_src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let add_entries: Vec[BindEntry] = Vec.new()
    add_entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    add_entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    add_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let add_event = match dispatch(stream, add_prog, unit_grid(), bindings_from(add_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(add_event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 2147483647)
    event_destroy(add_event)
    program_destroy(add_prog)

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = -2147483648
        *(memory_ptr(b_mem) as *mut i32) = 1

    let sub_src = compile_text_source("param a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = sub_sat %0 %1\nstore out [] %2\nreturn\n")
    let sub_prog = match compile(device_info(default_device()), sub_src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sub_entries: Vec[BindEntry] = Vec.new()
    sub_entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    sub_entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    sub_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let sub_event = match dispatch(stream, sub_prog, unit_grid(), bindings_from(sub_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(sub_event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == -2147483648)
    event_destroy(sub_event)
    program_destroy(sub_prog)

    stream_destroy(stream)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_exp_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = exp %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(1.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_log_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = log %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_log2_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 8.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = log2 %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(3.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_sin_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = sin %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_cos_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = cos %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(1.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_tanh_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 0.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = tanh %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_float_math_nontrivial_f32:
    let input_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let stream = stream_create(default_device())

    write_f32(input_mem, 0, 1.0)
    let exp_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = exp %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let exp_entries: Vec[BindEntry] = Vec.new()
    exp_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    exp_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let exp_event = match dispatch(stream, exp_prog, unit_grid(), bindings_from(exp_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(exp_event))
    assert_close_f32(read_f32(out_mem, 0), 2.7182817, 0.02)
    event_destroy(exp_event)
    program_destroy(exp_prog)

    write_f32(input_mem, 0, 2.7182817)
    let log_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = log %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let log_entries: Vec[BindEntry] = Vec.new()
    log_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    log_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let log_event = match dispatch(stream, log_prog, unit_grid(), bindings_from(log_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(log_event))
    assert_close_f32(read_f32(out_mem, 0), 1.0, 0.02)
    event_destroy(log_event)
    program_destroy(log_prog)

    write_f32(input_mem, 0, 3.0)
    let log2_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = log2 %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let log2_entries: Vec[BindEntry] = Vec.new()
    log2_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    log2_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let log2_event = match dispatch(stream, log2_prog, unit_grid(), bindings_from(log2_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(log2_event))
    assert_close_f32(read_f32(out_mem, 0), 1.5849625, 0.02)
    event_destroy(log2_event)
    program_destroy(log2_prog)

    write_f32(input_mem, 0, 0.5)
    let sin_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = sin %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sin_entries: Vec[BindEntry] = Vec.new()
    sin_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    sin_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let sin_event = match dispatch(stream, sin_prog, unit_grid(), bindings_from(sin_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(sin_event))
    assert_close_f32(read_f32(out_mem, 0), 0.47942555, 0.01)
    event_destroy(sin_event)
    program_destroy(sin_prog)

    let cos_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = cos %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let cos_entries: Vec[BindEntry] = Vec.new()
    cos_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    cos_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let cos_event = match dispatch(stream, cos_prog, unit_grid(), bindings_from(cos_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(cos_event))
    assert_close_f32(read_f32(out_mem, 0), 0.87758255, 0.01)
    event_destroy(cos_event)
    program_destroy(cos_prog)

    write_f32(input_mem, 0, 1.0)
    let tanh_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = tanh %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let tanh_entries: Vec[BindEntry] = Vec.new()
    tanh_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    tanh_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let tanh_event = match dispatch(stream, tanh_prog, unit_grid(), bindings_from(tanh_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(tanh_event))
    assert_close_f32(read_f32(out_mem, 0), 0.7615942, 0.02)
    event_destroy(tanh_event)
    program_destroy(tanh_prog)

    write_f32(input_mem, 0, 2.0)
    let sqrt_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = sqrt %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sqrt_entries: Vec[BindEntry] = Vec.new()
    sqrt_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    sqrt_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let sqrt_event = match dispatch(stream, sqrt_prog, unit_grid(), bindings_from(sqrt_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(sqrt_event))
    assert_close_f32(read_f32(out_mem, 0), 1.4142135, 0.002)
    event_destroy(sqrt_event)
    program_destroy(sqrt_prog)

    let rsqrt_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = rsqrt %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let rsqrt_entries: Vec[BindEntry] = Vec.new()
    rsqrt_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    rsqrt_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let rsqrt_event = match dispatch(stream, rsqrt_prog, unit_grid(), bindings_from(rsqrt_entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()
    assert(event_is_done(rsqrt_event))
    assert_close_f32(read_f32(out_mem, 0), 0.70710677, 0.002)
    event_destroy(rsqrt_event)
    program_destroy(rsqrt_prog)

    stream_destroy(stream)
    free(input_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_float_math_domain_errors:
    let input_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let stream = stream_create(default_device())

    write_f32(input_mem, 0, 0.0)
    let log_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = log %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let log_entries: Vec[BindEntry] = Vec.new()
    log_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    log_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let got_log_error = match dispatch(stream, log_prog, unit_grid(), bindings_from(log_entries))
        Err(.Unsupported(_)) => true
        _ => false
    assert(got_log_error)
    program_destroy(log_prog)

    write_f32(input_mem, 0, -1.0)
    let sqrt_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = sqrt %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sqrt_entries: Vec[BindEntry] = Vec.new()
    sqrt_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    sqrt_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let got_sqrt_error = match dispatch(stream, sqrt_prog, unit_grid(), bindings_from(sqrt_entries))
        Err(.Unsupported(_)) => true
        _ => false
    assert(got_sqrt_error)
    program_destroy(sqrt_prog)

    write_f32(input_mem, 0, 0.0)
    let rsqrt_prog = match compile(device_info(default_device()), compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = rsqrt %0\nstore out [] %1\nreturn\n"))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let rsqrt_entries: Vec[BindEntry] = Vec.new()
    rsqrt_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Float32)))
    rsqrt_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))
    let got_rsqrt_error = match dispatch(stream, rsqrt_prog, unit_grid(), bindings_from(rsqrt_entries))
        Err(.Unsupported(_)) => true
        _ => false
    assert(got_rsqrt_error)
    program_destroy(rsqrt_prog)

    stream_destroy(stream)
    free(input_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_add_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.25)
    write_f32(b_mem, 0, 2.5)

    let src = compile_text_source("param a in [] f32\nparam b in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(3.75).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_add_f64:
    let a_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f64(a_mem, 0, 1.25)
    write_f64(b_mem, 0, 2.5)

    let src = compile_text_source("param a in [] f64\nparam b in [] f64\nparam out out [] f64\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float64)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Float64)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float64)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f64_bits(out_mem, 0) == scalar_f64(3.75).bits)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_cast_i32_to_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 7

    let src = compile_text_source("param a in [] i32\nparam out out [] f32\n%0 = load a []\n%1 = cast f32 %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(7.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_abs_i32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = -9

    let src = compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = abs %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 9)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_mod_i32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 17
        *(memory_ptr(b_mem) as *mut i32) = 5

    let src = compile_text_source("param a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = mod %0 %1\nstore out [] %2\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(unsafe: *(memory_ptr(out_mem) as *mut i32) == 2)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_floor_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, -1.25)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = floor %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(-2.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_ceil_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, -1.25)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = ceil %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(-1.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_round_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, -1.25)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = round %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(-1.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_sqrt_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 9.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = sqrt %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(3.0).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_rsqrt_f32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 4.0)

    let src = compile_text_source("param a in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = rsqrt %0\nstore out [] %1\nreturn\n")

    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_f32_bits(out_mem, 0) == scalar_f32(0.5).bits as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_add_i64:
    let a_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 8usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_i64(a_mem, 0, 10000000000 as i64)
    write_i64(b_mem, 0, 2345678901 as i64)

    let src = compile_text_source("param a in [] i64\nparam b in [] i64\nparam out out [] i64\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n")
    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int64)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int64)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int64)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_i64(out_mem, 0) == 12345678901 as i64)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_scalar_xor_u32:
    let a_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4usize)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_u32(a_mem, 0, 12 as u32)
    write_u32(b_mem, 0, 10 as u32)

    let src = compile_text_source("param a in [] u32\nparam b in [] u32\nparam out out [] u32\n%0 = load a []\n%1 = load b []\n%2 = xor %0 %1\nstore out [] %2\nreturn\n")
    let prog = match compile(device_info(default_device()), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .UInt32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .UInt32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .UInt32)))

    let event = match dispatch(stream, prog, unit_grid(), bindings_from(entries))
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_event()

    assert(event_is_done(event))
    assert(read_u32(out_mem, 0) == 6 as u32)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(b_mem)
    free(out_mem)

fn test_compile_ir_text_parallel_grid_copy_i32:
    let a_mem = alloc_memory(12usize)
    let out_mem = alloc_memory(12usize)
    write_i32(a_mem, 0, 5)
    write_i32(a_mem, 1, 6)
    write_i32(a_mem, 2, 7)

    let prog = compile_program(compile_text_source("param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 3\nparallel_grid 0 %0 %1 1\nblock_begin 1\n%2 = load a [@0]\nstore out [@0] %2\nblock_end 1\nreturn\n"))
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(3usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(3usize), .Int32)))

    let event = dispatch_entries(stream, prog, entries)
    assert(event_is_done(event))
    assert(read_i32(out_mem, 0) == 5)
    assert(read_i32(out_mem, 1) == 6)
    assert(read_i32(out_mem, 2) == 7)

    event_destroy(event)
    stream_destroy(stream)
    program_destroy(prog)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_if_local_private_barrier_parallel_workgroup:
    let cond_mem = alloc_memory(4usize)
    let a_mem = alloc_memory(4usize)
    let out_mem = alloc_memory(4usize)
    let prog = compile_program(compile_text_source("param cond in [] i32\nparam a in [] i32\nparam out out [] i32\n%0 = const i32 1\nlocal tmp [%0] i32\nprivate scratch [%0] i32\n%1 = const i32 0\n%2 = load cond []\nparallel_workgroup 0 %1 %0 1\nblock_begin 1\nif %2 2 3\nblock_begin 2\n%3 = load a []\nstore tmp [@0] %3\nbarrier\n%4 = load tmp [@0]\nstore scratch [@0] %4\n%5 = load scratch [@0]\nstore out [] %5\nblock_end 2\nblock_begin 3\n%6 = const i32 9\nstore out [] %6\nblock_end 3\nblock_end 1\nreturn\n"))
    let stream = stream_create(default_device())

    write_i32(cond_mem, 0, 1)
    write_i32(a_mem, 0, 7)
    write_i32(out_mem, 0, 0)
    let true_entries: Vec[BindEntry] = Vec.new()
    true_entries.push(bind("cond", view_contiguous(cond_mem, shape_scalar(), .Int32)))
    true_entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    true_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let true_event = dispatch_entries(stream, prog, true_entries)
    assert(event_is_done(true_event))
    assert(read_i32(out_mem, 0) == 7)
    event_destroy(true_event)

    write_i32(cond_mem, 0, 0)
    write_i32(out_mem, 0, 0)
    let false_entries: Vec[BindEntry] = Vec.new()
    false_entries.push(bind("cond", view_contiguous(cond_mem, shape_scalar(), .Int32)))
    false_entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    false_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let false_event = dispatch_entries(stream, prog, false_entries)
    assert(event_is_done(false_event))
    assert(read_i32(out_mem, 0) == 9)
    event_destroy(false_event)

    stream_destroy(stream)
    program_destroy(prog)
    free(cond_mem)
    free(a_mem)
    free(out_mem)

fn test_compile_ir_text_if_without_else_parallel_subgroup:
    let cond_mem = alloc_memory(4usize)
    let out_mem = alloc_memory(4usize)
    let prog = compile_program(compile_text_source("param cond in [] i32\nparam out out [] i32\n%0 = const i32 0\n%1 = const i32 1\n%2 = load cond []\nparallel_subgroup 0 %0 %1 1\nblock_begin 1\nif %2 2\nblock_begin 2\n%3 = const i32 5\nstore out [] %3\nblock_end 2\nblock_end 1\nreturn\n"))
    let stream = stream_create(default_device())

    write_i32(cond_mem, 0, 0)
    write_i32(out_mem, 0, 11)
    let false_entries: Vec[BindEntry] = Vec.new()
    false_entries.push(bind("cond", view_contiguous(cond_mem, shape_scalar(), .Int32)))
    false_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let false_event = dispatch_entries(stream, prog, false_entries)
    assert(event_is_done(false_event))
    assert(read_i32(out_mem, 0) == 11)
    event_destroy(false_event)

    write_i32(cond_mem, 0, 1)
    write_i32(out_mem, 0, 11)
    let true_entries: Vec[BindEntry] = Vec.new()
    true_entries.push(bind("cond", view_contiguous(cond_mem, shape_scalar(), .Int32)))
    true_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let true_event = dispatch_entries(stream, prog, true_entries)
    assert(event_is_done(true_event))
    assert(read_i32(out_mem, 0) == 5)
    event_destroy(true_event)

    stream_destroy(stream)
    program_destroy(prog)
    free(cond_mem)
    free(out_mem)

fn test_compile_structured_reductions_i32_and_f32:
    let i32_mem = alloc_memory(16usize)
    let i32_out = alloc_memory(4usize)
    write_i32(i32_mem, 0, 1)
    write_i32(i32_mem, 1, 2)
    write_i32(i32_mem, 2, 3)
    write_i32(i32_mem, 3, 4)

    let f32_mem = alloc_memory(12usize)
    let f32_out = alloc_memory(4usize)
    write_f32(f32_mem, 0, 1.5)
    write_f32(f32_mem, 1, -2.0)
    write_f32(f32_mem, 2, 0.75)

    let stream = stream_create(default_device())

    let sum_prog = compile_program(build_reduce_1d_source(IROP_REDUCE_SUM, .Int32, 4))
    let sum_entries: Vec[BindEntry] = Vec.new()
    sum_entries.push(bind("a", view_contiguous(i32_mem, shape1(4usize), .Int32)))
    sum_entries.push(bind("out", view_contiguous(i32_out, shape_scalar(), .Int32)))
    let sum_event = dispatch_entries(stream, sum_prog, sum_entries)
    assert(event_is_done(sum_event))
    assert(read_i32(i32_out, 0) == 10)
    event_destroy(sum_event)
    program_destroy(sum_prog)

    let prod_prog = compile_program(build_reduce_1d_source(IROP_REDUCE_PROD, .Int32, 4))
    let prod_entries: Vec[BindEntry] = Vec.new()
    prod_entries.push(bind("a", view_contiguous(i32_mem, shape1(4usize), .Int32)))
    prod_entries.push(bind("out", view_contiguous(i32_out, shape_scalar(), .Int32)))
    let prod_event = dispatch_entries(stream, prod_prog, prod_entries)
    assert(event_is_done(prod_event))
    assert(read_i32(i32_out, 0) == 24)
    event_destroy(prod_event)
    program_destroy(prod_prog)

    let max_prog = compile_program(build_reduce_1d_source(IROP_REDUCE_MAX, .Float32, 3))
    let max_entries: Vec[BindEntry] = Vec.new()
    max_entries.push(bind("a", view_contiguous(f32_mem, shape1(3usize), .Float32)))
    max_entries.push(bind("out", view_contiguous(f32_out, shape_scalar(), .Float32)))
    let max_event = dispatch_entries(stream, max_prog, max_entries)
    assert(event_is_done(max_event))
    assert_close_f32(read_f32(f32_out, 0), 1.5, 0.001)
    event_destroy(max_event)
    program_destroy(max_prog)

    let min_prog = compile_program(build_reduce_1d_source(IROP_REDUCE_MIN, .Float32, 3))
    let min_entries: Vec[BindEntry] = Vec.new()
    min_entries.push(bind("a", view_contiguous(f32_mem, shape1(3usize), .Float32)))
    min_entries.push(bind("out", view_contiguous(f32_out, shape_scalar(), .Float32)))
    let min_event = dispatch_entries(stream, min_prog, min_entries)
    assert(event_is_done(min_event))
    assert_close_f32(read_f32(f32_out, 0), -2.0, 0.001)
    event_destroy(min_event)
    program_destroy(min_prog)

    stream_destroy(stream)
    free(i32_mem)
    free(i32_out)
    free(f32_mem)
    free(f32_out)

fn test_compile_ir_text_collectives_identity:
    let input_mem = alloc_memory(4usize)
    let out_mem = alloc_memory(4usize)
    let stream = stream_create(default_device())
    write_i32(input_mem, 0, 23)

    let sum_prog = compile_program(compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = collective_allreduce_sum %0\nstore out [] %1\nreturn\n"))
    let sum_entries: Vec[BindEntry] = Vec.new()
    sum_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Int32)))
    sum_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let sum_event = dispatch_entries(stream, sum_prog, sum_entries)
    assert(event_is_done(sum_event))
    assert(read_i32(out_mem, 0) == 23)
    event_destroy(sum_event)
    program_destroy(sum_prog)

    let max_prog = compile_program(compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = collective_allreduce_max %0\nstore out [] %1\nreturn\n"))
    let max_entries: Vec[BindEntry] = Vec.new()
    max_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Int32)))
    max_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let max_event = dispatch_entries(stream, max_prog, max_entries)
    assert(event_is_done(max_event))
    assert(read_i32(out_mem, 0) == 23)
    event_destroy(max_event)
    program_destroy(max_prog)

    let gather_prog = compile_program(compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = collective_allgather %0\nstore out [] %1\nreturn\n"))
    let gather_entries: Vec[BindEntry] = Vec.new()
    gather_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Int32)))
    gather_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let gather_event = dispatch_entries(stream, gather_prog, gather_entries)
    assert(event_is_done(gather_event))
    assert(read_i32(out_mem, 0) == 23)
    event_destroy(gather_event)
    program_destroy(gather_prog)

    let broadcast_prog = compile_program(compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = collective_broadcast %0\nstore out [] %1\nreturn\n"))
    let broadcast_entries: Vec[BindEntry] = Vec.new()
    broadcast_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Int32)))
    broadcast_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let broadcast_event = dispatch_entries(stream, broadcast_prog, broadcast_entries)
    assert(event_is_done(broadcast_event))
    assert(read_i32(out_mem, 0) == 23)
    event_destroy(broadcast_event)
    program_destroy(broadcast_prog)

    let scatter_prog = compile_program(compile_text_source("param a in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = collective_reduce_scatter %0\nstore out [] %1\nreturn\n"))
    let scatter_entries: Vec[BindEntry] = Vec.new()
    scatter_entries.push(bind("a", view_contiguous(input_mem, shape_scalar(), .Int32)))
    scatter_entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))
    let scatter_event = dispatch_entries(stream, scatter_prog, scatter_entries)
    assert(event_is_done(scatter_event))
    assert(read_i32(out_mem, 0) == 23)
    event_destroy(scatter_event)
    program_destroy(scatter_prog)

    stream_destroy(stream)
    free(input_mem)
    free(out_mem)

fn test_compile_ir_text_exact_bit_constants:
    let stream = stream_create(default_device())

    let u64_mem = alloc_memory(8usize)
    let u64_prog = compile_program(compile_text_source("param out out [] u64\n%0 = const u64 bits:0x0102030405060708\nstore out [] %0\nreturn\n"))
    let u64_entries: Vec[BindEntry] = Vec.new()
    u64_entries.push(bind("out", view_contiguous(u64_mem, shape_scalar(), .UInt64)))
    let u64_event = dispatch_entries(stream, u64_prog, u64_entries)
    assert(event_is_done(u64_event))
    assert(read_u64(u64_mem, 0) == 72623859790382856u64)
    event_destroy(u64_event)
    program_destroy(u64_prog)

    let f64_mem = alloc_memory(8usize)
    let f64_prog = compile_program(compile_text_source("param out out [] f64\n%0 = const f64 bits:0x4008000000000000\nstore out [] %0\nreturn\n"))
    let f64_entries: Vec[BindEntry] = Vec.new()
    f64_entries.push(bind("out", view_contiguous(f64_mem, shape_scalar(), .Float64)))
    let f64_event = dispatch_entries(stream, f64_prog, f64_entries)
    assert(event_is_done(f64_event))
    assert(read_f64_bits(f64_mem, 0) == 4613937818241073152u64)
    event_destroy(f64_event)
    program_destroy(f64_prog)

    let f16_mem = alloc_memory(2usize)
    let f16_prog = compile_program(compile_text_source("param out out [] f16\n%0 = const f16 bits:0x3c00\nstore out [] %0\nreturn\n"))
    let f16_entries: Vec[BindEntry] = Vec.new()
    f16_entries.push(bind("out", view_contiguous(f16_mem, shape_scalar(), .Float16)))
    let f16_event = dispatch_entries(stream, f16_prog, f16_entries)
    assert(event_is_done(f16_event))
    assert(read_u16(f16_mem, 0) == 15360 as u16)
    event_destroy(f16_event)
    program_destroy(f16_prog)

    let bf16_mem = alloc_memory(2usize)
    let bf16_prog = compile_program(compile_text_source("param out out [] bf16\n%0 = const bf16 bits:0x3f80\nstore out [] %0\nreturn\n"))
    let bf16_entries: Vec[BindEntry] = Vec.new()
    bf16_entries.push(bind("out", view_contiguous(bf16_mem, shape_scalar(), .BFloat16)))
    let bf16_event = dispatch_entries(stream, bf16_prog, bf16_entries)
    assert(event_is_done(bf16_event))
    assert(read_u16(bf16_mem, 0) == 16256 as u16)
    event_destroy(bf16_event)
    program_destroy(bf16_prog)

    stream_destroy(stream)
    free(u64_mem)
    free(f64_mem)
    free(f16_mem)
    free(bf16_mem)

fn test_compile_ir_text_widened_integer_dtypes:
    let stream = stream_create(default_device())

    let i8_a = alloc_memory(1usize)
    let i8_b = alloc_memory(1usize)
    let i8_out = alloc_memory(1usize)
    write_i8(i8_a, 0, 10 as i8)
    write_i8(i8_b, 0, 20 as i8)
    let i8_prog = compile_program(compile_text_source("param a in [] i8\nparam b in [] i8\nparam out out [] i8\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n"))
    let i8_entries: Vec[BindEntry] = Vec.new()
    i8_entries.push(bind("a", view_contiguous(i8_a, shape_scalar(), .Int8)))
    i8_entries.push(bind("b", view_contiguous(i8_b, shape_scalar(), .Int8)))
    i8_entries.push(bind("out", view_contiguous(i8_out, shape_scalar(), .Int8)))
    let i8_event = dispatch_entries(stream, i8_prog, i8_entries)
    assert(event_is_done(i8_event))
    assert(read_i8(i8_out, 0) == 30 as i8)
    event_destroy(i8_event)
    program_destroy(i8_prog)

    let i16_a = alloc_memory(2usize)
    let i16_b = alloc_memory(2usize)
    let i16_out = alloc_memory(2usize)
    write_i16(i16_a, 0, 32760 as i16)
    write_i16(i16_b, 0, 10 as i16)
    let i16_prog = compile_program(compile_text_source("param a in [] i16\nparam b in [] i16\nparam out out [] i16\n%0 = load a []\n%1 = load b []\n%2 = add_sat %0 %1\nstore out [] %2\nreturn\n"))
    let i16_entries: Vec[BindEntry] = Vec.new()
    i16_entries.push(bind("a", view_contiguous(i16_a, shape_scalar(), .Int16)))
    i16_entries.push(bind("b", view_contiguous(i16_b, shape_scalar(), .Int16)))
    i16_entries.push(bind("out", view_contiguous(i16_out, shape_scalar(), .Int16)))
    let i16_event = dispatch_entries(stream, i16_prog, i16_entries)
    assert(event_is_done(i16_event))
    assert(read_i16(i16_out, 0) == 32767 as i16)
    event_destroy(i16_event)
    program_destroy(i16_prog)

    let u8_a = alloc_memory(1usize)
    let u8_b = alloc_memory(1usize)
    let u8_out = alloc_memory(1usize)
    write_u8(u8_a, 0, 3 as u8)
    write_u8(u8_b, 0, 10 as u8)
    let u8_prog = compile_program(compile_text_source("param a in [] u8\nparam b in [] u8\nparam out out [] u8\n%0 = load a []\n%1 = load b []\n%2 = sub_sat %0 %1\nstore out [] %2\nreturn\n"))
    let u8_entries: Vec[BindEntry] = Vec.new()
    u8_entries.push(bind("a", view_contiguous(u8_a, shape_scalar(), .UInt8)))
    u8_entries.push(bind("b", view_contiguous(u8_b, shape_scalar(), .UInt8)))
    u8_entries.push(bind("out", view_contiguous(u8_out, shape_scalar(), .UInt8)))
    let u8_event = dispatch_entries(stream, u8_prog, u8_entries)
    assert(event_is_done(u8_event))
    assert(read_u8(u8_out, 0) == 0 as u8)
    event_destroy(u8_event)
    program_destroy(u8_prog)

    let u16_a = alloc_memory(2usize)
    let u16_b = alloc_memory(2usize)
    let u16_out = alloc_memory(2usize)
    write_u16(u16_a, 0, 3 as u16)
    write_u16(u16_b, 0, 2 as u16)
    let u16_prog = compile_program(compile_text_source("param a in [] u16\nparam b in [] u16\nparam out out [] u16\n%0 = load a []\n%1 = load b []\n%2 = shl %0 %1\n%3 = shr %2 %1\nstore out [] %3\nreturn\n"))
    let u16_entries: Vec[BindEntry] = Vec.new()
    u16_entries.push(bind("a", view_contiguous(u16_a, shape_scalar(), .UInt16)))
    u16_entries.push(bind("b", view_contiguous(u16_b, shape_scalar(), .UInt16)))
    u16_entries.push(bind("out", view_contiguous(u16_out, shape_scalar(), .UInt16)))
    let u16_event = dispatch_entries(stream, u16_prog, u16_entries)
    assert(event_is_done(u16_event))
    assert(read_u16(u16_out, 0) == 3 as u16)
    event_destroy(u16_event)
    program_destroy(u16_prog)

    let u64_out = alloc_memory(8usize)
    let u64_prog = compile_program(compile_text_source("param out out [] u64\n%0 = const u64 bits:0xf0f0f0f0f0f0f0f0\n%1 = popcount %0\nstore out [] %1\nreturn\n"))
    let u64_entries: Vec[BindEntry] = Vec.new()
    u64_entries.push(bind("out", view_contiguous(u64_out, shape_scalar(), .UInt64)))
    let u64_event = dispatch_entries(stream, u64_prog, u64_entries)
    assert(event_is_done(u64_event))
    assert(read_u64(u64_out, 0) == 32u64)
    event_destroy(u64_event)
    program_destroy(u64_prog)

    let cast_i16_in = alloc_memory(2usize)
    let cast_i16_out = alloc_memory(4usize)
    write_i16(cast_i16_in, 0, 7 as i16)
    let cast_i16_prog = compile_program(compile_text_source("param a in [] i16\nparam out out [] f32\n%0 = load a []\n%1 = cast f32 %0\nstore out [] %1\nreturn\n"))
    let cast_i16_entries: Vec[BindEntry] = Vec.new()
    cast_i16_entries.push(bind("a", view_contiguous(cast_i16_in, shape_scalar(), .Int16)))
    cast_i16_entries.push(bind("out", view_contiguous(cast_i16_out, shape_scalar(), .Float32)))
    let cast_i16_event = dispatch_entries(stream, cast_i16_prog, cast_i16_entries)
    assert(event_is_done(cast_i16_event))
    assert(read_f32_bits(cast_i16_out, 0) == scalar_f32(7.0).bits as u32)
    event_destroy(cast_i16_event)
    program_destroy(cast_i16_prog)

    let cast_u16_in = alloc_memory(4usize)
    let cast_u16_out = alloc_memory(2usize)
    write_f32(cast_u16_in, 0, 7.0)
    let cast_u16_prog = compile_program(compile_text_source("param a in [] f32\nparam out out [] u16\n%0 = load a []\n%1 = cast u16 %0\nstore out [] %1\nreturn\n"))
    let cast_u16_entries: Vec[BindEntry] = Vec.new()
    cast_u16_entries.push(bind("a", view_contiguous(cast_u16_in, shape_scalar(), .Float32)))
    cast_u16_entries.push(bind("out", view_contiguous(cast_u16_out, shape_scalar(), .UInt16)))
    let cast_u16_event = dispatch_entries(stream, cast_u16_prog, cast_u16_entries)
    assert(event_is_done(cast_u16_event))
    assert(read_u16(cast_u16_out, 0) == 7 as u16)
    event_destroy(cast_u16_event)
    program_destroy(cast_u16_prog)

    stream_destroy(stream)
    free(i8_a)
    free(i8_b)
    free(i8_out)
    free(i16_a)
    free(i16_b)
    free(i16_out)
    free(u8_a)
    free(u8_b)
    free(u8_out)
    free(u16_a)
    free(u16_b)
    free(u16_out)
    free(u64_out)
    free(cast_i16_in)
    free(cast_i16_out)
    free(cast_u16_in)
    free(cast_u16_out)

fn test_compile_ir_text_float16_and_bfloat16_runtime:
    let stream = stream_create(default_device())

    let f16_a = alloc_memory(2usize)
    let f16_b = alloc_memory(2usize)
    let f16_out = alloc_memory(2usize)
    write_u16(f16_a, 0, 15360 as u16)
    write_u16(f16_b, 0, 16384 as u16)
    let f16_add_prog = compile_program(compile_text_source("param a in [] f16\nparam b in [] f16\nparam out out [] f16\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n"))
    let f16_add_entries: Vec[BindEntry] = Vec.new()
    f16_add_entries.push(bind("a", view_contiguous(f16_a, shape_scalar(), .Float16)))
    f16_add_entries.push(bind("b", view_contiguous(f16_b, shape_scalar(), .Float16)))
    f16_add_entries.push(bind("out", view_contiguous(f16_out, shape_scalar(), .Float16)))
    let f16_add_event = dispatch_entries(stream, f16_add_prog, f16_add_entries)
    assert(event_is_done(f16_add_event))
    assert(read_u16(f16_out, 0) == 16896 as u16)
    event_destroy(f16_add_event)
    program_destroy(f16_add_prog)

    let f16_cast_prog = compile_program(compile_text_source("param a in [] f16\nparam out out [] f32\n%0 = load a []\n%1 = cast f32 %0\nstore out [] %1\nreturn\n"))
    let f16_cast_out = alloc_memory(4usize)
    write_u16(f16_a, 0, 16896 as u16)
    let f16_cast_entries: Vec[BindEntry] = Vec.new()
    f16_cast_entries.push(bind("a", view_contiguous(f16_a, shape_scalar(), .Float16)))
    f16_cast_entries.push(bind("out", view_contiguous(f16_cast_out, shape_scalar(), .Float32)))
    let f16_cast_event = dispatch_entries(stream, f16_cast_prog, f16_cast_entries)
    assert(event_is_done(f16_cast_event))
    assert(read_f32_bits(f16_cast_out, 0) == scalar_f32(3.0).bits as u32)
    event_destroy(f16_cast_event)
    program_destroy(f16_cast_prog)

    let f32_to_f16_in = alloc_memory(4usize)
    let f32_to_f16_out = alloc_memory(2usize)
    write_f32(f32_to_f16_in, 0, 1.5)
    let f32_to_f16_prog = compile_program(compile_text_source("param a in [] f32\nparam out out [] f16\n%0 = load a []\n%1 = cast f16 %0\nstore out [] %1\nreturn\n"))
    let f32_to_f16_entries: Vec[BindEntry] = Vec.new()
    f32_to_f16_entries.push(bind("a", view_contiguous(f32_to_f16_in, shape_scalar(), .Float32)))
    f32_to_f16_entries.push(bind("out", view_contiguous(f32_to_f16_out, shape_scalar(), .Float16)))
    let f32_to_f16_event = dispatch_entries(stream, f32_to_f16_prog, f32_to_f16_entries)
    assert(event_is_done(f32_to_f16_event))
    assert(read_u16(f32_to_f16_out, 0) == 15872 as u16)
    event_destroy(f32_to_f16_event)
    program_destroy(f32_to_f16_prog)

    let bf16_a = alloc_memory(2usize)
    let bf16_b = alloc_memory(2usize)
    let bf16_out = alloc_memory(2usize)
    write_u16(bf16_a, 0, 16256 as u16)
    write_u16(bf16_b, 0, 16384 as u16)
    let bf16_add_prog = compile_program(compile_text_source("param a in [] bf16\nparam b in [] bf16\nparam out out [] bf16\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n"))
    let bf16_add_entries: Vec[BindEntry] = Vec.new()
    bf16_add_entries.push(bind("a", view_contiguous(bf16_a, shape_scalar(), .BFloat16)))
    bf16_add_entries.push(bind("b", view_contiguous(bf16_b, shape_scalar(), .BFloat16)))
    bf16_add_entries.push(bind("out", view_contiguous(bf16_out, shape_scalar(), .BFloat16)))
    let bf16_add_event = dispatch_entries(stream, bf16_add_prog, bf16_add_entries)
    assert(event_is_done(bf16_add_event))
    assert(read_u16(bf16_out, 0) == 16448 as u16)
    event_destroy(bf16_add_event)
    program_destroy(bf16_add_prog)

    let bf16_cast_prog = compile_program(compile_text_source("param a in [] bf16\nparam out out [] f32\n%0 = load a []\n%1 = cast f32 %0\nstore out [] %1\nreturn\n"))
    let bf16_cast_out = alloc_memory(4usize)
    write_u16(bf16_a, 0, 16448 as u16)
    let bf16_cast_entries: Vec[BindEntry] = Vec.new()
    bf16_cast_entries.push(bind("a", view_contiguous(bf16_a, shape_scalar(), .BFloat16)))
    bf16_cast_entries.push(bind("out", view_contiguous(bf16_cast_out, shape_scalar(), .Float32)))
    let bf16_cast_event = dispatch_entries(stream, bf16_cast_prog, bf16_cast_entries)
    assert(event_is_done(bf16_cast_event))
    assert(read_f32_bits(bf16_cast_out, 0) == scalar_f32(3.0).bits as u32)
    event_destroy(bf16_cast_event)
    program_destroy(bf16_cast_prog)

    let f32_to_bf16_in = alloc_memory(4usize)
    let f32_to_bf16_out = alloc_memory(2usize)
    write_f32(f32_to_bf16_in, 0, 1.5)
    let f32_to_bf16_prog = compile_program(compile_text_source("param a in [] f32\nparam out out [] bf16\n%0 = load a []\n%1 = cast bf16 %0\nstore out [] %1\nreturn\n"))
    let f32_to_bf16_entries: Vec[BindEntry] = Vec.new()
    f32_to_bf16_entries.push(bind("a", view_contiguous(f32_to_bf16_in, shape_scalar(), .Float32)))
    f32_to_bf16_entries.push(bind("out", view_contiguous(f32_to_bf16_out, shape_scalar(), .BFloat16)))
    let f32_to_bf16_event = dispatch_entries(stream, f32_to_bf16_prog, f32_to_bf16_entries)
    assert(event_is_done(f32_to_bf16_event))
    assert(read_u16(f32_to_bf16_out, 0) == 16320 as u16)
    event_destroy(f32_to_bf16_event)
    program_destroy(f32_to_bf16_prog)

    stream_destroy(stream)
    free(f16_a)
    free(f16_b)
    free(f16_out)
    free(f16_cast_out)
    free(f32_to_f16_in)
    free(f32_to_f16_out)
    free(bf16_a)
    free(bf16_b)
    free(bf16_out)
    free(bf16_cast_out)
    free(f32_to_bf16_in)
    free(f32_to_bf16_out)

fn main:
    test_compile_ir_text_scalar_add()
    test_dispatch_parallel_add_i32()
    test_dispatch_parallel_add_from_text_ir()
    test_compile_ir_text_scalar_clamp()
    test_compile_ir_text_scalar_add_with_spec_header()
    test_compile_ir_text_scalar_bitwise_i32()
    test_compile_ir_text_scalar_not_i32()
    test_compile_ir_text_scalar_shift_i32()
    test_compile_ir_text_scalar_bitcount_i32()
    test_compile_ir_text_scalar_sat_i32()
    test_compile_ir_text_scalar_exp_f32()
    test_compile_ir_text_scalar_log_f32()
    test_compile_ir_text_scalar_log2_f32()
    test_compile_ir_text_scalar_sin_f32()
    test_compile_ir_text_scalar_cos_f32()
    test_compile_ir_text_scalar_tanh_f32()
    test_compile_ir_text_scalar_float_math_nontrivial_f32()
    test_compile_ir_text_scalar_float_math_domain_errors()
    test_compile_ir_text_scalar_add_f32()
    test_compile_ir_text_scalar_add_f64()
    test_compile_ir_text_scalar_cast_i32_to_f32()
    test_compile_ir_text_scalar_abs_i32()
    test_compile_ir_text_scalar_mod_i32()
    test_compile_ir_text_scalar_floor_f32()
    test_compile_ir_text_scalar_ceil_f32()
    test_compile_ir_text_scalar_round_f32()
    test_compile_ir_text_scalar_sqrt_f32()
    test_compile_ir_text_scalar_rsqrt_f32()
    test_compile_ir_text_scalar_add_i64()
    test_compile_ir_text_scalar_xor_u32()
    test_compile_ir_text_parallel_grid_copy_i32()
    test_compile_ir_text_if_local_private_barrier_parallel_workgroup()
    test_compile_ir_text_if_without_else_parallel_subgroup()
    test_compile_structured_reductions_i32_and_f32()
    test_compile_ir_text_collectives_identity()
    test_compile_ir_text_exact_bit_constants()
    test_compile_ir_text_widened_integer_dtypes()
    test_compile_ir_text_float16_and_bfloat16_runtime()
