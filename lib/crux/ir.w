use crux.core
use crux.errors

pub let IROP_CONST: i32 = 0
pub let IROP_LOAD: i32 = 1
pub let IROP_STORE: i32 = 2
pub let IROP_ADD: i32 = 3
pub let IROP_SUB: i32 = 4
pub let IROP_MUL: i32 = 5
pub let IROP_DIV: i32 = 6
pub let IROP_MOD: i32 = 7
pub let IROP_NEG: i32 = 8
pub let IROP_FMA: i32 = 9
pub let IROP_ADD_SAT: i32 = 10
pub let IROP_SUB_SAT: i32 = 11
pub let IROP_EQ: i32 = 12
pub let IROP_NE: i32 = 13
pub let IROP_LT: i32 = 14
pub let IROP_GT: i32 = 15
pub let IROP_LE: i32 = 16
pub let IROP_GE: i32 = 17
pub let IROP_MIN: i32 = 18
pub let IROP_MAX: i32 = 19
pub let IROP_CLAMP: i32 = 20
pub let IROP_SELECT: i32 = 21
pub let IROP_EXP: i32 = 22
pub let IROP_LOG: i32 = 23
pub let IROP_LOG2: i32 = 24
pub let IROP_SQRT: i32 = 25
pub let IROP_RSQRT: i32 = 26
pub let IROP_SIN: i32 = 27
pub let IROP_COS: i32 = 28
pub let IROP_TANH: i32 = 29
pub let IROP_ABS: i32 = 30
pub let IROP_FLOOR: i32 = 31
pub let IROP_CEIL: i32 = 32
pub let IROP_ROUND: i32 = 33
pub let IROP_AND: i32 = 34
pub let IROP_OR: i32 = 35
pub let IROP_XOR: i32 = 36
pub let IROP_NOT: i32 = 37
pub let IROP_SHL: i32 = 38
pub let IROP_SHR: i32 = 39
pub let IROP_POPCOUNT: i32 = 40
pub let IROP_CLZ: i32 = 41
pub let IROP_CTZ: i32 = 42
pub let IROP_CAST: i32 = 43
pub let IROP_LOOP: i32 = 44
pub let IROP_PARALLEL: i32 = 45
pub let IROP_PARALLEL_GRID: i32 = 46
pub let IROP_PARALLEL_WORKGROUP: i32 = 47
pub let IROP_PARALLEL_SUBGROUP: i32 = 48
pub let IROP_IF: i32 = 49
pub let IROP_REDUCE_SUM: i32 = 50
pub let IROP_REDUCE_MAX: i32 = 51
pub let IROP_REDUCE_MIN: i32 = 52
pub let IROP_REDUCE_PROD: i32 = 53
pub let IROP_LOCAL: i32 = 54
pub let IROP_PRIVATE: i32 = 55
pub let IROP_BARRIER: i32 = 56
pub let IROP_BLOCK_BEGIN: i32 = 57
pub let IROP_BLOCK_END: i32 = 58
pub let IROP_RETURN: i32 = 59
pub let IROP_COLLECTIVE_ALLREDUCE_SUM: i32 = 60
pub let IROP_COLLECTIVE_ALLREDUCE_MAX: i32 = 61
pub let IROP_COLLECTIVE_ALLGATHER: i32 = 62
pub let IROP_COLLECTIVE_BROADCAST: i32 = 63
pub let IROP_COLLECTIVE_REDUCE_SCATTER: i32 = 64

pub type IRInst {
    op: i32,
    dtype: DType,
    d0: i32,
    d1: i32,
    d2: i32,
    d3: i32,
}

pub type IRProgram {
    insts: Vec[IRInst],
    aux: Vec[i32],
    param_names: Vec[str],
    param_modes: Vec[ParamMode],
    param_ranks: Vec[i32],
    param_dtypes: Vec[DType],
    num_params: i32,
}

pub fn ir_inst(op: i32, dtype: DType, d0: i32, d1: i32, d2: i32, d3: i32) -> IRInst:
    IRInst { op, dtype, d0, d1, d2, d3 }

pub fn ir_program -> IRProgram:
    IRProgram {
        insts: Vec.new(),
        aux: Vec.new(),
        param_names: Vec.new(),
        param_modes: Vec.new(),
        param_ranks: Vec.new(),
        param_dtypes: Vec.new(),
        num_params: 0,
    }

pub fn ir_param_ref(param_index: i32) -> i32:
    -param_index - 1

pub fn ir_is_param_ref(ref: i32) -> bool:
    ref < 0

pub fn ir_param_index(ref: i32) -> i32:
    if ref >= 0:
        return -1
    -ref - 1

let IR_LOOP_REF_BASE: i32 = 1024

pub fn ir_loop_ref(loop_index: i32) -> i32:
    -(IR_LOOP_REF_BASE + loop_index)

pub fn ir_is_loop_ref(ref: i32) -> bool:
    ref <= -IR_LOOP_REF_BASE

pub fn ir_loop_index(ref: i32) -> i32:
    if ref > -IR_LOOP_REF_BASE:
        return -1
    -ref - IR_LOOP_REF_BASE

let IR_MAX_LOOP_SLOTS: i32 = 8

fn is_param_binding_ref(ref: i32) -> bool:
    ref < 0 and not ir_is_loop_ref(ref)

fn validate_value_ref(prog: IRProgram, ref: i32) -> Result[i32, SubstrateError]:
    let inst_count = prog.insts.len() as i32
    if ref < 0 or ref >= inst_count:
        return Err(.CompileError("ir value ref out of range"))
    Ok(ref)

fn validate_scalar_ref(prog: IRProgram, ref: i32) -> Result[i32, SubstrateError]:
    if ir_is_loop_ref(ref):
        let slot = ir_loop_index(ref)
        if slot < 0 or slot >= IR_MAX_LOOP_SLOTS:
            return Err(.CompileError("ir loop ref out of range"))
        return Ok(ref)
    if is_param_binding_ref(ref):
        return Err(.CompileError("ir index refs cannot point to params"))
    validate_value_ref(prog, ref)

fn validate_param_ref(prog: IRProgram, ref: i32) -> Result[i32, SubstrateError]:
    if not is_param_binding_ref(ref):
        return Err(.CompileError("ir param ref is invalid"))
    let index = ir_param_index(ref)
    if index < 0 or index >= prog.num_params:
        return Err(.CompileError("ir param ref out of range"))
    Ok(index)

fn validate_aux_window(prog: IRProgram, aux_base: i32, width: i32) -> Result[i32, SubstrateError]:
    let aux_len = prog.aux.len() as i32
    if aux_base < 0:
        return Err(.CompileError("ir aux base is invalid"))
    if width < 0:
        return Err(.CompileError("ir aux width is invalid"))
    if aux_base + width > aux_len:
        return Err(.CompileError("ir aux range is invalid"))
    Ok(aux_base)

fn validate_index_tuple(prog: IRProgram, aux_base: i32, rank: i32) -> Result[i32, SubstrateError]:
    let _ = validate_aux_window(prog, aux_base, rank)?
    var i: i32 = 0
    while i < rank:
        let ref = prog.aux[aux_base + i]
        let _ = validate_scalar_ref(prog, ref)?
        i = i + 1
    Ok(aux_base)

fn validate_block_nesting(prog: IRProgram) -> Result[i32, SubstrateError]:
    let stack: Vec[i32] = Vec.new()
    var depth: i32 = 0
    for ip in 0..prog.insts.len():
        let inst = prog.insts[ip]
        if inst.op == IROP_BLOCK_BEGIN:
            stack.push(inst.d0)
            depth = depth + 1
            continue
        if inst.op == IROP_BLOCK_END:
            if depth == 0:
                return Err(.CompileError("ir block end without begin"))
            let expected = stack[depth - 1]
            if expected != inst.d0:
                return Err(.CompileError("ir block id mismatch"))
            depth = depth - 1
    if depth != 0:
        return Err(.CompileError("ir block begin without end"))
    Ok(0)

pub fn validate_ir(prog: IRProgram, sig: ProgramSig) -> Result[i32, SubstrateError]:
    if sig.params.len() as i32 != prog.num_params:
        return Err(.CompileError("ir param signature mismatch"))
    let _ = validate_block_nesting(prog)?
    for ip in 0..prog.insts.len():
        let inst = prog.insts[ip]

        if inst.op == IROP_CONST or inst.op == IROP_BLOCK_BEGIN or inst.op == IROP_BLOCK_END or inst.op == IROP_RETURN:
            continue

        if inst.op == IROP_LOAD:
            let param_index = validate_param_ref(prog, inst.d0)?
            let param = sig.params[param_index]
            if inst.dtype != param.dtype:
                return Err(.CompileError("ir load dtype mismatch"))
            let _ = validate_index_tuple(prog, inst.d1, param.rank)?
            continue

        if inst.op == IROP_STORE:
            let param_index = validate_param_ref(prog, inst.d0)?
            let param = sig.params[param_index]
            let _ = validate_index_tuple(prog, inst.d1, param.rank)?
            let value_ref = validate_value_ref(prog, inst.d2)?
            let value_inst = prog.insts[value_ref]
            if value_inst.dtype != param.dtype:
                return Err(.CompileError("ir store dtype mismatch"))
            continue

        if inst.op == IROP_ADD or inst.op == IROP_SUB or inst.op == IROP_MUL or inst.op == IROP_DIV or inst.op == IROP_MIN or inst.op == IROP_MAX:
            let lhs_ref = validate_value_ref(prog, inst.d0)?
            let rhs_ref = validate_value_ref(prog, inst.d1)?
            let lhs = prog.insts[lhs_ref]
            let rhs = prog.insts[rhs_ref]
            if lhs.dtype != rhs.dtype or lhs.dtype != inst.dtype:
                return Err(.CompileError("ir binary op dtype mismatch"))
            continue

        if inst.op == IROP_EQ or inst.op == IROP_NE or inst.op == IROP_LT or inst.op == IROP_GT or inst.op == IROP_LE or inst.op == IROP_GE:
            let lhs_ref = validate_value_ref(prog, inst.d0)?
            let rhs_ref = validate_value_ref(prog, inst.d1)?
            let lhs = prog.insts[lhs_ref]
            let rhs = prog.insts[rhs_ref]
            if lhs.dtype != rhs.dtype:
                return Err(.CompileError("ir compare operand dtype mismatch"))
            if inst.dtype != .Int32:
                return Err(.CompileError("ir compare result dtype must be i32"))
            continue

        if inst.op == IROP_FMA:
            let a_ref = validate_value_ref(prog, inst.d0)?
            let b_ref = validate_value_ref(prog, inst.d1)?
            let c_ref = validate_value_ref(prog, inst.d2)?
            let a = prog.insts[a_ref]
            let b = prog.insts[b_ref]
            let c = prog.insts[c_ref]
            if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != inst.dtype:
                return Err(.CompileError("ir fma dtype mismatch"))
            continue

        if inst.op == IROP_NEG:
            let value_ref = validate_value_ref(prog, inst.d0)?
            let value = prog.insts[value_ref]
            if value.dtype != inst.dtype:
                return Err(.CompileError("ir neg dtype mismatch"))
            continue

        if inst.op == IROP_SELECT:
            let cond_ref = validate_value_ref(prog, inst.d0)?
            let true_ref = validate_value_ref(prog, inst.d1)?
            let false_ref = validate_value_ref(prog, inst.d2)?
            let cond = prog.insts[cond_ref]
            let on_true = prog.insts[true_ref]
            let on_false = prog.insts[false_ref]
            if cond.dtype != .Int32:
                return Err(.CompileError("ir select condition dtype must be i32"))
            if on_true.dtype != on_false.dtype or on_true.dtype != inst.dtype:
                return Err(.CompileError("ir select arm dtype mismatch"))
            continue

        if inst.op == IROP_CLAMP:
            let value_ref = validate_value_ref(prog, inst.d0)?
            let lo_ref = validate_value_ref(prog, inst.d1)?
            let hi_ref = validate_value_ref(prog, inst.d2)?
            let value = prog.insts[value_ref]
            let lo = prog.insts[lo_ref]
            let hi = prog.insts[hi_ref]
            if value.dtype != lo.dtype or value.dtype != hi.dtype or value.dtype != inst.dtype:
                return Err(.CompileError("ir clamp dtype mismatch"))
            continue

        if inst.op == IROP_LOOP or inst.op == IROP_PARALLEL:
            if inst.d0 < 0 or inst.d0 >= IR_MAX_LOOP_SLOTS:
                return Err(.CompileError("ir loop slot out of range"))
            let _ = validate_scalar_ref(prog, inst.d1)?
            let _ = validate_scalar_ref(prog, inst.d2)?
            if inst.d3 < 0:
                return Err(.CompileError("ir block id is invalid"))
            let next_ip = ip + 1
            if next_ip >= prog.insts.len():
                return Err(.CompileError("ir loop body is missing"))
            let begin = prog.insts[next_ip]
            if begin.op != IROP_BLOCK_BEGIN or begin.d0 != inst.d3:
                return Err(.CompileError("ir loop body must start with matching block"))
            continue

        return Err(.CompileError("ir op not supported by current validator"))
    Ok(0)
