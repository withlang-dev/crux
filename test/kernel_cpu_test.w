use crux.core
use crux.device
use crux.ir
use crux.memory
use crux.program
use crux.stream
use crux.view

fn write_i32(mem: *mut Memory, index: i32, value: i32):
    let ptr = memory_ptr(mem) as *mut i32
    unsafe:
        *(ptr + index as i64) = value

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

fn build_matmul_ir(m: i32, n: i32, k: i32) -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    param_names.push("a")
    param_modes.push(.In)
    param_ranks.push(2)
    param_dtypes.push(.Int32)

    param_names.push("b")
    param_modes.push(.In)
    param_ranks.push(2)
    param_dtypes.push(.Int32)

    param_names.push("out")
    param_modes.push(.Out)
    param_ranks.push(2)
    param_dtypes.push(.Int32)

    insts.push(ir_inst(IROP_CONST, .Int32, 0, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, m, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, n, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, k, 0, 0, 0))

    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    insts.push(ir_inst(IROP_LOOP, .Int32, 1, 0, 2, 2))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))

    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(2), 0, 0, 0))

    insts.push(ir_inst(IROP_LOOP, .Int32, 2, 0, 3, 3))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 3, 0, 0, 0))

    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Int32, ir_param_ref(2), 2, 0, 0))

    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(2))
    insts.push(ir_inst(IROP_LOAD, .Int32, ir_param_ref(0), 4, 0, 0))

    aux.push(ir_loop_ref(2))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Int32, ir_param_ref(1), 6, 0, 0))

    insts.push(ir_inst(IROP_MUL, .Int32, 12, 13, 0, 0))
    insts.push(ir_inst(IROP_ADD, .Int32, 11, 14, 0, 0))

    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(2), 8, 15, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 3, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
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

    var src = program_source("main")
    src.ir_text = "param a in [N] i32\nparam out inout [] i32\n%0 = const i32 0\n%1 = const i32 4\nloop 0 %0 %1 1\nblock_begin 1\n%4 = load out []\n%5 = load a [@0]\n%6 = add %4 %5\nstore out [] %6\nblock_end 1\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape_scalar(), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, bindings_from(entries))
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

fn test_dispatch_matmul_i32:
    let m: i32 = 2
    let n: i32 = 2
    let k: i32 = 3
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

    var src = program_source("main")
    src.ir = build_matmul_ir(m, n, k)

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape2(2usize, 3usize), .Int32)))
    entries.push(bind("b", view_contiguous(b_mem, shape2(3usize, 2usize), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape2(2usize, 2usize), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, bindings_from(entries))
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

    var src = program_source("main")
    src.ir_text = "param a in [N] i32\nparam b in [N] i32\nparam c in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%4 = load a [@0]\n%5 = load b [@0]\n%6 = load c [@0]\n%7 = fma %4 %5 %6\nstore out [@0] %7\nblock_end 1\nreturn\n"

    let prog = match compile(default_device(), src)
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
    let event = match dispatch(stream, prog, bindings_from(entries))
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

    var src = program_source("main")
    src.ir_text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%4 = load a [@0]\n%5 = neg %4\nstore out [@0] %5\nblock_end 1\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, bindings_from(entries))
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

    var src = program_source("main")
    src.ir_text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 5\n%2 = const i32 0\nparallel 0 %0 %1 1\nblock_begin 1\n%5 = load a [@0]\n%6 = lt %5 %2\n%7 = select %6 %2 %5\nstore out [@0] %7\nblock_end 1\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, bindings_from(entries))
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

    var src = program_source("main")
    src.ir_text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 6\nparallel 0 %0 %1 1\nblock_begin 1\n%4 = load a [@0]\n%5 = max %4 %0\n%6 = min %5 %1\nstore out [@0] %6\nblock_end 1\nreturn\n"

    let prog = match compile(default_device(), src)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()

    let entries: Vec[BindEntry] = Vec.new()
    entries.push(bind("a", view_contiguous(a_mem, shape1(n), .Int32)))
    entries.push(bind("out", view_contiguous(out_mem, shape1(n), .Int32)))

    let stream = stream_create(default_device())
    let event = match dispatch(stream, prog, bindings_from(entries))
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

    var src = program_source("main")
    src.ir_text = "param a in [N] f32\nparam b in [N] f32\nparam c in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = const i32 4\nparallel 0 %0 %1 1\nblock_begin 1\n%4 = load a [@0]\n%5 = load b [@0]\n%6 = load c [@0]\n%7 = fma %4 %5 %6\nstore out [@0] %7\nblock_end 1\nreturn\n"

    let prog = match compile(default_device(), src)
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
    let event = match dispatch(stream, prog, bindings_from(entries))
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

fn main:
    test_dispatch_reduce_sum_from_text_ir()
    test_dispatch_matmul_i32()
    test_dispatch_fma_from_text_ir()
    test_dispatch_neg_from_text_ir()
    test_dispatch_relu_from_text_ir()
    test_dispatch_clip_from_text_ir()
    test_dispatch_fma_f32_from_text_ir()
