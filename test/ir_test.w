use crux.core
use crux.device
use crux.ir
use crux.program

fn make_source(insts: Vec[IRInst], aux: Vec[i32], strings: Vec[str]) -> ProgramSource:
    ProgramSource {
        ir: insts,
        aux,
        strings,
        entry: "main",
    }

fn make_demo_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_inst(IROP_LOAD, .Float32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_i64_demo_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    strings.push("out")
    insts.push(ir_param_inst(0, .In, 0, .Int64))
    insts.push(ir_param_inst(1, .In, 0, .Int64))
    insts.push(ir_param_inst(2, .Out, 0, .Int64))
    insts.push(ir_inst(IROP_LOAD, .Int64, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_LOAD, .Int64, 1, ir_param_ref(1), 0, 0))
    insts.push(ir_inst(IROP_ADD, .Int64, 2, 0, 1, 0))
    insts.push(ir_inst(IROP_STORE, .Int64, ir_param_ref(2), 0, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_u32_demo_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    strings.push("out")
    insts.push(ir_param_inst(0, .In, 0, .UInt32))
    insts.push(ir_param_inst(1, .In, 0, .UInt32))
    insts.push(ir_param_inst(2, .Out, 0, .UInt32))
    insts.push(ir_inst(IROP_LOAD, .UInt32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_LOAD, .UInt32, 1, ir_param_ref(1), 0, 0))
    insts.push(ir_inst(IROP_XOR, .UInt32, 2, 0, 1, 0))
    insts.push(ir_inst(IROP_STORE, .UInt32, ir_param_ref(2), 0, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_param_ref_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_inst(IROP_LOAD, .Float32, 0, ir_param_ref(1), 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_store_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("out")
    insts.push(ir_param_inst(0, .Out, 0, .Float32))
    insts.push(ir_const_i32(0, 7))
    insts.push(ir_inst(IROP_STORE, .Float32, ir_param_ref(0), 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_loop_block_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 4))
    insts.push(ir_inst(IROP_PARALLEL, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_index_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("out")
    insts.push(ir_param_inst(0, .In, 1, .Int32))
    insts.push(ir_param_inst(1, .Out, 1, .Int32))
    insts.push(ir_const_i32(0, 0))
    insts.push(ir_inst(IROP_CAST, .Int64, 1, 0, 0, 0))
    aux.push(1)
    insts.push(ir_inst(IROP_LOAD, .Int32, 2, ir_param_ref(0), 0, 0))
    aux.push(1)
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(1), 1, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_fma_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("c")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_const_i32(0, 2))
    insts.push(ir_const_i32(1, 3))
    insts.push(ir_inst(IROP_LOAD, .Float32, 2, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_FMA, .Int32, 3, 0, 1, 2))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_neg_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_NEG, .Float32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_abs_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_ABS, .Float32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_mod_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_param_inst(1, .In, 0, .Float32))
    insts.push(ir_inst(IROP_LOAD, .Float32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_LOAD, .Float32, 1, ir_param_ref(1), 0, 0))
    insts.push(ir_inst(IROP_MOD, .Float32, 2, 0, 1, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_sat_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_param_inst(1, .In, 0, .Float32))
    insts.push(ir_inst(IROP_LOAD, .Float32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_LOAD, .Float32, 1, ir_param_ref(1), 0, 0))
    insts.push(ir_inst(IROP_ADD_SAT, .Float32, 2, 0, 1, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_bitwise_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_param_inst(1, .In, 0, .Float32))
    insts.push(ir_inst(IROP_LOAD, .Float32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_LOAD, .Float32, 1, ir_param_ref(1), 0, 0))
    insts.push(ir_inst(IROP_AND, .Float32, 2, 0, 1, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_not_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_NOT, .Float32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_shift_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    strings.push("b")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_param_inst(1, .In, 0, .Float32))
    insts.push(ir_inst(IROP_LOAD, .Float32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_LOAD, .Float32, 1, ir_param_ref(1), 0, 0))
    insts.push(ir_inst(IROP_SHL, .Float32, 2, 0, 1, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_bitcount_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_POPCOUNT, .Float32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_exp_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_EXP, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_log_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_LOG, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_log2_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_LOG2, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_sin_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_SIN, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_cos_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_COS, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_tanh_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_TANH, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_floor_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_FLOOR, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_ceil_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_CEIL, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_round_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_inst(IROP_ROUND, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_sqrt_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 4))
    insts.push(ir_inst(IROP_SQRT, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_rsqrt_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 4))
    insts.push(ir_inst(IROP_RSQRT, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_compare_result_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 2))
    insts.push(ir_const_i32(1, 3))
    insts.push(ir_inst(IROP_LT, .Float32, 2, 0, 1, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_select_condition_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("cond")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_inst(IROP_LOAD, .Float32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_const_i32(2, 3))
    insts.push(ir_inst(IROP_SELECT, .Int32, 3, 0, 1, 2))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_select_arm_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("other")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_const_i32(0, 1))
    insts.push(ir_const_i32(1, 2))
    insts.push(ir_inst(IROP_LOAD, .Float32, 2, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_SELECT, .Int32, 3, 0, 1, 2))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_clamp_dtype_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("hi")
    insts.push(ir_param_inst(0, .In, 0, .Float32))
    insts.push(ir_const_i32(0, 5))
    insts.push(ir_const_i32(1, 0))
    insts.push(ir_inst(IROP_LOAD, .Float32, 2, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_CLAMP, .Int32, 3, 0, 1, 2))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_bad_header_order_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    insts.push(ir_const_i32(0, 1))
    insts.push(ir_param_inst(0, .In, 0, .Int32))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_spec_constant_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("TILE")
    strings.push("a")
    strings.push("out")
    insts.push(ir_spec_constant_inst(0, .Int32, 16, 0))
    insts.push(ir_param_inst(1, .In, 0, .Int32))
    insts.push(ir_param_inst(2, .Out, 0, .Int32))
    insts.push(ir_inst(IROP_LOAD, .Int32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(1), 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_duplicate_spec_constant_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("TILE")
    insts.push(ir_spec_constant_inst(0, .Int32, 16, 0))
    insts.push(ir_spec_constant_inst(0, .Int32, 32, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_param_spec_name_collision_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("shared")
    insts.push(ir_spec_constant_inst(0, .Int32, 16, 0))
    insts.push(ir_param_inst(0, .In, 0, .Int32))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_cast_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    insts.push(ir_param_inst(0, .In, 0, .Int32))
    insts.push(ir_inst(IROP_LOAD, .Int32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_CAST, .Float32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_u16_cast_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    insts.push(ir_param_inst(0, .In, 0, .Int32))
    insts.push(ir_inst(IROP_LOAD, .Int32, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_CAST, .UInt16, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_int16_load_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("a")
    insts.push(ir_param_inst(0, .In, 0, .Int16))
    insts.push(ir_inst(IROP_LOAD, .Int16, 0, ir_param_ref(0), 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_if_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 1))
    insts.push(ir_inst(IROP_IF, .Int32, 0, 0, 1, -1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))
    insts.push(ir_const_i32(1, 7))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_reduce_collective_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    insts.push(ir_const_i32(0, 0))
    insts.push(ir_const_i32(1, 2))
    aux.push(0)
    aux.push(1)
    aux.push(2)
    insts.push(ir_inst(IROP_REDUCE_SUM, .Int32, 3, 0, 1, 0))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 1, 0, 0, 0))
    insts.push(ir_const_i32(2, 5))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_COLLECTIVE_ALLREDUCE_SUM, .Int32, 4, 3, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn make_local_private_source -> ProgramSource:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let strings: Vec[str] = Vec.new()

    strings.push("tmp")
    strings.push("scratch")
    insts.push(ir_const_i32(0, 1))
    aux.push(0)
    insts.push(ir_inst(IROP_LOCAL, .Int32, 0, 1, 0, 0))
    aux.push(0)
    insts.push(ir_inst(IROP_PRIVATE, .Int32, 1, 1, 1, 0))
    insts.push(ir_inst(IROP_LOAD, .Int32, 1, ir_local_ref(0), 0, 0))
    insts.push(ir_inst(IROP_STORE, .Int32, ir_private_ref(0), 1, 1, 0))
    insts.push(ir_inst(IROP_BARRIER, .Int32, 0, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    make_source(insts, aux, strings)

fn compile_is_compile_error(source: ProgramSource) -> bool:
    let prog = match compile(device_info(default_device()), source):
        Ok(v) => v
        Err(_) => return true
    program_destroy(prog)
    false

fn test_param_ref_helpers:
    assert(ir_param_ref(0) == -1)
    assert(ir_param_ref(2) == -3)
    assert(ir_is_param_ref(-2))
    assert(not ir_is_param_ref(4))
    assert(ir_param_index(-4) == 3)

fn test_ir_inst_builder:
    let inst = ir_inst(IROP_ADD, .Float32, 1, 2, 3, 0)
    assert(inst.op == IROP_ADD)
    assert(inst.d0 == 1)
    assert(inst.d1 == 2)
    assert(inst.d2 == 3)

fn test_ir_program_defaults:
    let prog = ir_program()
    assert(prog.insts.len() == 0)
    assert(prog.aux.len() == 0)
    assert(ir_value_count(prog) == 0)

fn test_ir_program_literal_layout:
    let source = make_demo_source()
    assert(source.ir.len() == 3)
    assert(source.aux.len() == 0)
    assert(source.strings.len() == 1)
    assert(source.ir[0].op == IROP_PARAM)
    assert(source.ir[1].op == IROP_LOAD)

fn test_program_source_carries_ir:
    let source = make_demo_source()
    assert(source.entry == "main")
    assert(source.ir.len() == 3)
    assert(source.strings.len() == 1)
    assert(source.strings[0] == "a")

fn test_ir_validation_accepts_demo_ir:
    assert(not compile_is_compile_error(make_demo_source()))

fn test_ir_validation_accepts_i64_and_u32:
    assert(not compile_is_compile_error(make_i64_demo_source()))
    assert(not compile_is_compile_error(make_u32_demo_source()))

fn test_ir_validation_accepts_cast:
    assert(not compile_is_compile_error(make_cast_source()))

fn test_ir_validation_accepts_spec_constant_headers:
    let prog = match compile(device_info(default_device()), make_spec_constant_source()):
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sig = program_sig(prog)
    assert(sig.params.len() == 2)
    assert(sig.params[0].name == "a")
    assert(sig.params[1].name == "out")
    program_destroy(prog)

fn test_ir_validation_accepts_u16_cast_and_int16_load:
    assert(not compile_is_compile_error(make_u16_cast_source()))
    assert(not compile_is_compile_error(make_int16_load_source()))

fn test_ir_validation_accepts_if_reduce_collective_and_scratch:
    assert(not compile_is_compile_error(make_if_source()))
    assert(not compile_is_compile_error(make_reduce_collective_source()))
    assert(not compile_is_compile_error(make_local_private_source()))

fn test_ir_validation_rejects_bad_param_ref:
    assert(compile_is_compile_error(make_bad_param_ref_source()))

fn test_ir_validation_rejects_store_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_store_dtype_source()))

fn test_ir_validation_rejects_bad_loop_block:
    assert(compile_is_compile_error(make_bad_loop_block_source()))

fn test_ir_validation_rejects_bad_index_dtype:
    assert(compile_is_compile_error(make_bad_index_dtype_source()))

fn test_ir_validation_rejects_fma_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_fma_dtype_source()))

fn test_ir_validation_rejects_neg_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_neg_dtype_source()))

fn test_ir_validation_rejects_abs_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_abs_dtype_source()))

fn test_ir_validation_rejects_mod_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_mod_dtype_source()))

fn test_ir_validation_rejects_sat_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_sat_dtype_source()))

fn test_ir_validation_rejects_bitwise_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_bitwise_dtype_source()))

fn test_ir_validation_rejects_not_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_not_dtype_source()))

fn test_ir_validation_rejects_shift_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_shift_dtype_source()))

fn test_ir_validation_rejects_bitcount_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_bitcount_dtype_source()))

fn test_ir_validation_rejects_exp_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_exp_dtype_source()))

fn test_ir_validation_rejects_log_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_log_dtype_source()))

fn test_ir_validation_rejects_log2_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_log2_dtype_source()))

fn test_ir_validation_rejects_sin_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_sin_dtype_source()))

fn test_ir_validation_rejects_cos_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_cos_dtype_source()))

fn test_ir_validation_rejects_tanh_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_tanh_dtype_source()))

fn test_ir_validation_rejects_floor_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_floor_dtype_source()))

fn test_ir_validation_rejects_ceil_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_ceil_dtype_source()))

fn test_ir_validation_rejects_round_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_round_dtype_source()))

fn test_ir_validation_rejects_sqrt_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_sqrt_dtype_source()))

fn test_ir_validation_rejects_rsqrt_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_rsqrt_dtype_source()))

fn test_ir_validation_rejects_bad_compare_result:
    assert(compile_is_compile_error(make_bad_compare_result_source()))

fn test_ir_validation_rejects_bad_select_condition:
    assert(compile_is_compile_error(make_bad_select_condition_source()))

fn test_ir_validation_rejects_bad_select_arms:
    assert(compile_is_compile_error(make_bad_select_arm_source()))

fn test_ir_validation_rejects_bad_clamp_dtype:
    assert(compile_is_compile_error(make_bad_clamp_dtype_source()))

fn test_ir_validation_rejects_bad_header_order:
    assert(compile_is_compile_error(make_bad_header_order_source()))

fn test_ir_validation_rejects_duplicate_spec_constants:
    assert(compile_is_compile_error(make_duplicate_spec_constant_source()))

fn test_ir_validation_rejects_param_spec_name_collision:
    assert(compile_is_compile_error(make_param_spec_name_collision_source()))

fn main:
    test_param_ref_helpers()
    test_ir_inst_builder()
    test_ir_program_defaults()
    test_ir_program_literal_layout()
    test_program_source_carries_ir()
    test_ir_validation_accepts_demo_ir()
    test_ir_validation_accepts_i64_and_u32()
    test_ir_validation_accepts_cast()
    test_ir_validation_accepts_spec_constant_headers()
    test_ir_validation_accepts_u16_cast_and_int16_load()
    test_ir_validation_accepts_if_reduce_collective_and_scratch()
    test_ir_validation_rejects_bad_param_ref()
    test_ir_validation_rejects_store_dtype_mismatch()
    test_ir_validation_rejects_bad_loop_block()
    test_ir_validation_rejects_bad_index_dtype()
    test_ir_validation_rejects_fma_dtype_mismatch()
    test_ir_validation_rejects_neg_dtype_mismatch()
    test_ir_validation_rejects_abs_dtype_mismatch()
    test_ir_validation_rejects_mod_dtype_mismatch()
    test_ir_validation_rejects_sat_dtype_mismatch()
    test_ir_validation_rejects_bitwise_dtype_mismatch()
    test_ir_validation_rejects_not_dtype_mismatch()
    test_ir_validation_rejects_shift_dtype_mismatch()
    test_ir_validation_rejects_bitcount_dtype_mismatch()
    test_ir_validation_rejects_exp_dtype_mismatch()
    test_ir_validation_rejects_log_dtype_mismatch()
    test_ir_validation_rejects_log2_dtype_mismatch()
    test_ir_validation_rejects_sin_dtype_mismatch()
    test_ir_validation_rejects_cos_dtype_mismatch()
    test_ir_validation_rejects_tanh_dtype_mismatch()
    test_ir_validation_rejects_floor_dtype_mismatch()
    test_ir_validation_rejects_ceil_dtype_mismatch()
    test_ir_validation_rejects_round_dtype_mismatch()
    test_ir_validation_rejects_sqrt_dtype_mismatch()
    test_ir_validation_rejects_rsqrt_dtype_mismatch()
    test_ir_validation_rejects_bad_compare_result()
    test_ir_validation_rejects_bad_select_condition()
    test_ir_validation_rejects_bad_select_arms()
    test_ir_validation_rejects_bad_clamp_dtype()
    test_ir_validation_rejects_bad_header_order()
    test_ir_validation_rejects_duplicate_spec_constants()
    test_ir_validation_rejects_param_spec_name_collision()
