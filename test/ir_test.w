use crux.core
use crux.device
use crux.ir
use crux.program

fn make_demo_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    param_names.push("a")
    param_modes.push(.In)
    param_ranks.push(1)
    param_dtypes.push(.Float32)

    aux.push(0)
    insts.push(ir_inst(IROP_LOAD, .Float32, ir_param_ref(0), 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Float32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 1,
    }

fn make_bad_param_ref_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    param_names.push("a")
    param_modes.push(.In)
    param_ranks.push(1)
    param_dtypes.push(.Float32)

    aux.push(0)
    insts.push(ir_inst(IROP_LOAD, .Float32, ir_param_ref(1), 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Float32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 1,
    }

fn make_bad_store_dtype_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    param_names.push("out")
    param_modes.push(.Out)
    param_ranks.push(0)
    param_dtypes.push(.Float32)

    insts.push(ir_inst(IROP_CONST, .Int32, 7, 0, 0, 0))
    insts.push(ir_inst(IROP_STORE, .Float32, ir_param_ref(0), 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Float32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 1,
    }

fn make_bad_loop_block_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    insts.push(ir_inst(IROP_CONST, .Int32, 0, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, 4, 0, 0, 0))
    insts.push(ir_inst(IROP_PARALLEL, .Int32, 0, 0, 1, 1))
    insts.push(ir_inst(IROP_BLOCK_BEGIN, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_BLOCK_END, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 0,
    }

fn make_bad_fma_dtype_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    insts.push(ir_inst(IROP_CONST, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, 3, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Float32, 4, 0, 0, 0))
    insts.push(ir_inst(IROP_FMA, .Int32, 0, 1, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 0,
    }

fn make_bad_neg_dtype_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    insts.push(ir_inst(IROP_CONST, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_NEG, .Float32, 0, 0, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Float32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 0,
    }

fn make_bad_compare_result_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    insts.push(ir_inst(IROP_CONST, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, 3, 0, 0, 0))
    insts.push(ir_inst(IROP_LT, .Float32, 0, 1, 0, 0))
    insts.push(ir_inst(IROP_RETURN, .Float32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 0,
    }

fn make_bad_select_condition_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    insts.push(ir_inst(IROP_CONST, .Float32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, 3, 0, 0, 0))
    insts.push(ir_inst(IROP_SELECT, .Int32, 0, 1, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 0,
    }

fn make_bad_select_arm_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    insts.push(ir_inst(IROP_CONST, .Int32, 1, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, 2, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Float32, 3, 0, 0, 0))
    insts.push(ir_inst(IROP_SELECT, .Int32, 0, 1, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 0,
    }

fn make_bad_clamp_dtype_ir -> IRProgram:
    let insts: Vec[IRInst] = Vec.new()
    let aux: Vec[i32] = Vec.new()
    let param_names: Vec[str] = Vec.new()
    let param_modes: Vec[ParamMode] = Vec.new()
    let param_ranks: Vec[i32] = Vec.new()
    let param_dtypes: Vec[DType] = Vec.new()

    insts.push(ir_inst(IROP_CONST, .Int32, 5, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Int32, 0, 0, 0, 0))
    insts.push(ir_inst(IROP_CONST, .Float32, 9, 0, 0, 0))
    insts.push(ir_inst(IROP_CLAMP, .Int32, 0, 1, 2, 0))
    insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))

    IRProgram {
        insts,
        aux,
        param_names,
        param_modes,
        param_ranks,
        param_dtypes,
        num_params: 0,
    }


fn compile_is_compile_error(ir: IRProgram) -> bool:
    var src = program_source("main")
    src.ir = ir
    let prog = match compile(default_device(), src)
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
    let inst = ir_inst(IROP_ADD, .Float32, 1, 2, 0, 0)
    assert(inst.op == IROP_ADD)
    assert(inst.d0 == 1)
    assert(inst.d1 == 2)

fn test_ir_program_defaults:
    let prog = ir_program()
    let insts = prog.insts
    let aux = prog.aux
    let names = prog.param_names
    assert(prog.num_params == 0)
    assert(insts.len() == 0)
    assert(aux.len() == 0)
    assert(names.len() == 0)

fn test_ir_program_literal_layout:
    let prog = make_demo_ir()
    let insts = prog.insts
    let aux = prog.aux
    let names = prog.param_names
    let modes = prog.param_modes
    let ranks = prog.param_ranks
    let dtypes = prog.param_dtypes
    assert(prog.num_params == 1)
    assert(insts.len() == 2)
    assert(aux.len() == 1)
    assert(names.len() == 1)
    assert(modes.len() == 1)
    assert(ranks.len() == 1)
    assert(dtypes.len() == 1)
    let load = insts[0]
    assert(load.op == IROP_LOAD)

fn test_program_source_carries_ir:
    var src = program_source("main")
    src.ir = make_demo_ir()
    let ir = src.ir
    let insts = ir.insts
    let names = ir.param_names
    assert(src.entry == "main")
    assert(insts.len() == 2)
    assert(names.len() == 1)

fn test_ir_validation_accepts_demo_ir:
    assert(not compile_is_compile_error(make_demo_ir()))

fn test_ir_validation_rejects_bad_param_ref:
    assert(compile_is_compile_error(make_bad_param_ref_ir()))

fn test_ir_validation_rejects_store_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_store_dtype_ir()))

fn test_ir_validation_rejects_bad_loop_block:
    assert(compile_is_compile_error(make_bad_loop_block_ir()))

fn test_ir_validation_rejects_fma_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_fma_dtype_ir()))

fn test_ir_validation_rejects_neg_dtype_mismatch:
    assert(compile_is_compile_error(make_bad_neg_dtype_ir()))

fn test_ir_validation_rejects_bad_compare_result:
    assert(compile_is_compile_error(make_bad_compare_result_ir()))

fn test_ir_validation_rejects_bad_select_condition:
    assert(compile_is_compile_error(make_bad_select_condition_ir()))

fn test_ir_validation_rejects_bad_select_arms:
    assert(compile_is_compile_error(make_bad_select_arm_ir()))

fn test_ir_validation_rejects_bad_clamp_dtype:
    assert(compile_is_compile_error(make_bad_clamp_dtype_ir()))

fn main:
    test_param_ref_helpers()
    test_ir_inst_builder()
    test_ir_program_defaults()
    test_ir_program_literal_layout()
    test_program_source_carries_ir()
    test_ir_validation_accepts_demo_ir()
    test_ir_validation_rejects_bad_param_ref()
    test_ir_validation_rejects_store_dtype_mismatch()
    test_ir_validation_rejects_bad_loop_block()
    test_ir_validation_rejects_fma_dtype_mismatch()
    test_ir_validation_rejects_neg_dtype_mismatch()
    test_ir_validation_rejects_bad_compare_result()
    test_ir_validation_rejects_bad_select_condition()
    test_ir_validation_rejects_bad_select_arms()
    test_ir_validation_rejects_bad_clamp_dtype()
