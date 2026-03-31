use crux.core
use crux.device
use crux.ir
use crux.memory
use crux.program
use crux.stream
use crux.view

fn fill_i32(mem: *mut Memory, n: i32, base_value: i32):
    let ptr = memory_ptr(mem) as *mut i32
    var i: i32 = 0
    while i < n:
        unsafe:
            *(ptr + i as i64) = base_value + i
        i = i + 1

fn read_i32(mem: *mut Memory, index: i32) -> i32:
    let ptr = memory_ptr(mem) as *mut i32
    unsafe: *(ptr + index as i64)

fn write_f32(mem: *mut Memory, index: i32, value: f32):
    let ptr = memory_ptr(mem) as *mut f32
    unsafe:
        *(ptr + index as i64) = value

fn read_f32_bits(mem: *mut Memory, index: i32) -> u32:
    let ptr = memory_ptr(mem) as *mut f32
    let value = unsafe: *(ptr + index as i64)
    unsafe: transmute[u32](value)

fn build_parallel_add_ir(n: i32) -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    param_names.push("a")
    param_modes.push(.In)
    param_ranks.push(1)
    param_dtypes.push(.Int32)

    param_names.push("b")
    param_modes.push(.In)
    param_ranks.push(1)
    param_dtypes.push(.Int32)

    param_names.push("out")
    param_modes.push(.Out)
    param_ranks.push(1)
    param_dtypes.push(.Int32)

    insts.push(ir_inst(IROP_CONST, .Int32, 0, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, n, 0, 0, 0))
    insts.push(ir_inst(IROP_PARALLEL, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, ir_param_ref(0), 0, 0, 0))

    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, ir_param_ref(1), 1, 0, 0))

    insts.push(ir_inst(IROP_ADD, .Int32, 4, 5, 0, 0))

    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(2), 2, 6, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 3,
    }

fn test_compile_ir_text_scalar_add:
    let a_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 4
        *(memory_ptr(b_mem) as *mut i32) = 5

    var src = program_source("main")
    src.ir_text = "param a in [] i32\nparam b in [] i32\nparam out out [] i32\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, bindings_from(entries))
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

    var src = program_source("main")
    src.ir = build_parallel_add_ir(count)
    let prog = match compile(default_device(), src)
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
    let event = match dispatch(stream, prog, bindings_from(entries))
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

    var src = program_source("main")
    src.ir_text = "param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%4 = load a [@0]\n%5 = load b [@0]\n%6 = add %4 %5\nstore out [@0] %6\nblock_end 1\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, bindings_from(entries))
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
    let a_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    unsafe:
        *(memory_ptr(a_mem) as *mut i32) = 9

    var src = program_source("main")
    src.ir_text = "param a in [] i32\nparam out out [] i32\n%0 = const i32 0\n%1 = const i32 7\n%2 = load a []\n%3 = clamp %2 %0 %1\nstore out [] %3\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let event = match dispatch(stream, prog, bindings_from(entries))
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

fn test_compile_ir_text_scalar_add_f32:
    let a_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let b_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()
    let out_mem = match alloc(default_device(), 4)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_memory()

    write_f32(a_mem, 0, 1.25)
    write_f32(b_mem, 0, 2.5)

    var src = program_source("main")
    src.ir_text = "param a in [] f32\nparam b in [] f32\nparam out out [] f32\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let stream = stream_create(default_device())
    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape_scalar(), .Float32)))
    entries.push(bind("b", view_contiguous(b_mem, shape_scalar(), .Float32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Float32)))

    let event = match dispatch(stream, prog, bindings_from(entries))
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

fn main:
    test_compile_ir_text_scalar_add()
    test_dispatch_parallel_add_i32()
    test_dispatch_parallel_add_from_text_ir()
    test_compile_ir_text_scalar_clamp()
    test_compile_ir_text_scalar_add_f32()
