use crux.core
use crux.ir

fn make_source(insts: Vec[IRInst], aux: Vec[i32], strings: Vec[str]) -> ProgramSource:
    ProgramSource {
        ir: insts,
        aux,
        strings,
        entry: "main",
    }

fn ir_const_f32_zero(dest: i32) -> IRInst:
    ir_inst(IROP_CONST, .Float32, dest, 0, 0, 0)

pub fn kernel_map_add_1d_i32_source() -> ProgramSource:
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
    insts.push(ir_const_i32(1, 4))
    insts.push(ir_inst(IROP_PARALLEL, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 2, ir_param_ref(0), a_index, 0))

    let b_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 3, ir_param_ref(1), b_index, 0))

    insts.push(ir_inst(IROP_ADD, .Int32, 4, 2, 3, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(2), out_index, 4, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_map_add_2d_f32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 2, .Float32))
    insts.push(ir_param_inst(1, .In, 2, .Float32))
    insts.push(ir_param_inst(2, .Out, 2, .Float32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_const_i32(2, 3))
    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_LOOP, .Int32, 1, 0, 2, 2))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Float32, 3, ir_param_ref(0), a_index, 0))

    let b_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Float32, 4, ir_param_ref(1), b_index, 0))

    insts.push(ir_inst(IROP_ADD, .Float32, 5, 3, 4, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_STORE, .Float32, ir_param_ref(2), out_index, 5, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_map_inplace_add_1d_i32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")

    insts.push(ir_param_inst(0, .InOut, 1, .Int32))
    insts.push(ir_param_inst(1, .In, 1, .Int32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 4))
    insts.push(ir_inst(IROP_PARALLEL, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 2, ir_param_ref(0), a_index, 0))

    let b_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 3, ir_param_ref(1), b_index, 0))

    insts.push(ir_inst(IROP_ADD, .Int32, 4, 2, 3, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(0), out_index, 4, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_reduce_sum_1d_i32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 1, .Int32))
    insts.push(ir_param_inst(1, .Out, 0, .Int32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 4))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(1), 0, 0, 0))
    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    insts.push(ir_inst(IROP_LOAD, .Int32, 2, ir_param_ref(1), 0, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 3, ir_param_ref(0), a_index, 0))

    insts.push(ir_inst(IROP_ADD, .Int32, 4, 2, 3, 0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(1), 0, 4, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_reduce_sum_rows_2d_i32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 2, .Int32))
    insts.push(ir_param_inst(1, .Out, 1, .Int32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_const_i32(2, 3))
    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    let init_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(1), init_index, 0, 0))

    insts.push(ir_inst(IROP_LOOP, .Int32, 1, 0, 2, 2))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 3, ir_param_ref(1), out_index, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Int32, 4, ir_param_ref(0), a_index, 0))

    insts.push(ir_inst(IROP_ADD, .Int32, 5, 3, 4, 0))

    let store_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(1), store_index, 5, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_reduce_max_rows_2d_f32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 2, .Float32))
    insts.push(ir_param_inst(1, .Out, 1, .Float32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_const_i32(2, 3))
    insts.push(ir_const_i32(3, 1))
    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))

    let first_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(0)
    insts.push(ir_inst(IROP_LOAD, .Float32, 4, ir_param_ref(0), first_index, 0))

    let init_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Float32, ir_param_ref(1), init_index, 4, 0))

    insts.push(ir_inst(IROP_LOOP, .Int32, 1, 3, 2, 2))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_LOAD, .Float32, 5, ir_param_ref(1), out_index, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Float32, 6, ir_param_ref(0), a_index, 0))

    insts.push(ir_inst(IROP_MAX, .Float32, 7, 5, 6, 0))

    let store_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Float32, ir_param_ref(1), store_index, 7, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_transpose_2d_i32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 2, .Int32))
    insts.push(ir_param_inst(1, .Out, 2, .Int32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_const_i32(2, 3))
    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_LOOP, .Int32, 1, 0, 2, 2))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Int32, 3, ir_param_ref(0), a_index, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(1))
    aux.push(ir_loop_ref(0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(1), out_index, 3, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_matmul_2d_i32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 2, .Int32))
    insts.push(ir_param_inst(1, .In, 2, .Int32))
    insts.push(ir_param_inst(2, .Out, 2, .Int32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_const_i32(2, 2))
    insts.push(ir_const_i32(3, 3))
    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_LOOP, .Int32, 1, 0, 2, 2))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))

    let init_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(2), init_index, 0, 0))

    insts.push(ir_inst(IROP_LOOP, .Int32, 2, 0, 3, 3))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 3, 0, 0, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Int32, 4, ir_param_ref(2), out_index, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(2))
    insts.push(ir_inst(IROP_LOAD, .Int32, 5, ir_param_ref(0), a_index, 0))

    let b_index = aux.len() as i32
    aux.push(ir_loop_ref(2))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Int32, 6, ir_param_ref(1), b_index, 0))

    insts.push(ir_inst(IROP_MUL, .Int32, 7, 5, 6, 0))
    insts.push(ir_inst(IROP_ADD, .Int32, 8, 4, 7, 0))

    let store_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(2), store_index, 8, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 3, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

pub fn kernel_matmul_2d_f32_source() -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    strings.push("out")

    insts.push(ir_param_inst(0, .In, 2, .Float32))
    insts.push(ir_param_inst(1, .In, 2, .Float32))
    insts.push(ir_param_inst(2, .Out, 2, .Float32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_const_i32(2, 2))
    insts.push(ir_const_i32(3, 3))
    insts.push(ir_const_f32_zero(4))
    insts.push(ir_inst(IROP_LOOP, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_LOOP, .Int32, 1, 0, 2, 2))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))

    let init_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_STORE, .Float32, ir_param_ref(2), init_index, 4, 0))

    insts.push(ir_inst(IROP_LOOP, .Int32, 2, 0, 3, 3))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 3, 0, 0, 0))

    let out_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Float32, 5, ir_param_ref(2), out_index, 0))

    let a_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(2))
    insts.push(ir_inst(IROP_LOAD, .Float32, 6, ir_param_ref(0), a_index, 0))

    let b_index = aux.len() as i32
    aux.push(ir_loop_ref(2))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_LOAD, .Float32, 7, ir_param_ref(1), b_index, 0))

    insts.push(ir_inst(IROP_FMA, .Float32, 8, 6, 7, 5))

    let store_index = aux.len() as i32
    aux.push(ir_loop_ref(0))
    aux.push(ir_loop_ref(1))
    insts.push(ir_inst(IROP_STORE, .Float32, ir_param_ref(2), store_index, 8, 0))

    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 3, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)
