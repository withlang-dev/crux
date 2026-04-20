use crux.core
use crux.errors

pub let IROP_PARAM: i32 = 0
pub let IROP_SPEC_CONSTANT: i32 = 1
pub let IROP_CONST: i32 = 2
pub let IROP_LOAD: i32 = 3
pub let IROP_STORE: i32 = 4
pub let IROP_ADD: i32 = 5
pub let IROP_SUB: i32 = 6
pub let IROP_MUL: i32 = 7
pub let IROP_DIV: i32 = 8
pub let IROP_MOD: i32 = 9
pub let IROP_NEG: i32 = 10
pub let IROP_FMA: i32 = 11
pub let IROP_ADD_SAT: i32 = 12
pub let IROP_SUB_SAT: i32 = 13
pub let IROP_EQ: i32 = 14
pub let IROP_NE: i32 = 15
pub let IROP_LT: i32 = 16
pub let IROP_GT: i32 = 17
pub let IROP_LE: i32 = 18
pub let IROP_GE: i32 = 19
pub let IROP_MIN: i32 = 20
pub let IROP_MAX: i32 = 21
pub let IROP_CLAMP: i32 = 22
pub let IROP_SELECT: i32 = 23
pub let IROP_EXP: i32 = 24
pub let IROP_LOG: i32 = 25
pub let IROP_LOG2: i32 = 26
pub let IROP_SQRT: i32 = 27
pub let IROP_RSQRT: i32 = 28
pub let IROP_SIN: i32 = 29
pub let IROP_COS: i32 = 30
pub let IROP_TANH: i32 = 31
pub let IROP_ABS: i32 = 32
pub let IROP_FLOOR: i32 = 33
pub let IROP_CEIL: i32 = 34
pub let IROP_ROUND: i32 = 35
pub let IROP_AND: i32 = 36
pub let IROP_OR: i32 = 37
pub let IROP_XOR: i32 = 38
pub let IROP_NOT: i32 = 39
pub let IROP_SHL: i32 = 40
pub let IROP_SHR: i32 = 41
pub let IROP_POPCOUNT: i32 = 42
pub let IROP_CLZ: i32 = 43
pub let IROP_CTZ: i32 = 44
pub let IROP_CAST: i32 = 45
pub let IROP_LOOP: i32 = 46
pub let IROP_PARALLEL: i32 = 47
pub let IROP_PARALLEL_GRID: i32 = 48
pub let IROP_PARALLEL_WORKGROUP: i32 = 49
pub let IROP_PARALLEL_SUBGROUP: i32 = 50
pub let IROP_IF: i32 = 51
pub let IROP_REDUCE_SUM: i32 = 52
pub let IROP_REDUCE_MAX: i32 = 53
pub let IROP_REDUCE_MIN: i32 = 54
pub let IROP_REDUCE_PROD: i32 = 55
pub let IROP_LOCAL: i32 = 56
pub let IROP_PRIVATE: i32 = 57
pub let IROP_BARRIER: i32 = 58
pub let IROP_BLOCK_BEGIN: i32 = 59
pub let IROP_BLOCK_END: i32 = 60
pub let IROP_RETURN: i32 = 61
pub let IROP_COLLECTIVE_ALLREDUCE_SUM: i32 = 62
pub let IROP_COLLECTIVE_ALLREDUCE_MAX: i32 = 63
pub let IROP_COLLECTIVE_ALLGATHER: i32 = 64
pub let IROP_COLLECTIVE_BROADCAST: i32 = 65
pub let IROP_COLLECTIVE_REDUCE_SCATTER: i32 = 66

pub fn ir_op_name(op: i32) -> str:
    if op == IROP_PARAM: return "param"
    if op == IROP_SPEC_CONSTANT: return "spec_constant"
    if op == IROP_CONST: return "const"
    if op == IROP_LOAD: return "load"
    if op == IROP_STORE: return "store"
    if op == IROP_ADD: return "add"
    if op == IROP_SUB: return "sub"
    if op == IROP_MUL: return "mul"
    if op == IROP_DIV: return "div"
    if op == IROP_MOD: return "mod"
    if op == IROP_NEG: return "neg"
    if op == IROP_FMA: return "fma"
    if op == IROP_ADD_SAT: return "add_sat"
    if op == IROP_SUB_SAT: return "sub_sat"
    if op == IROP_EQ: return "eq"
    if op == IROP_NE: return "ne"
    if op == IROP_LT: return "lt"
    if op == IROP_GT: return "gt"
    if op == IROP_LE: return "le"
    if op == IROP_GE: return "ge"
    if op == IROP_MIN: return "min"
    if op == IROP_MAX: return "max"
    if op == IROP_CLAMP: return "clamp"
    if op == IROP_SELECT: return "select"
    if op == IROP_EXP: return "exp"
    if op == IROP_LOG: return "log"
    if op == IROP_LOG2: return "log2"
    if op == IROP_SQRT: return "sqrt"
    if op == IROP_RSQRT: return "rsqrt"
    if op == IROP_SIN: return "sin"
    if op == IROP_COS: return "cos"
    if op == IROP_TANH: return "tanh"
    if op == IROP_ABS: return "abs"
    if op == IROP_FLOOR: return "floor"
    if op == IROP_CEIL: return "ceil"
    if op == IROP_ROUND: return "round"
    if op == IROP_AND: return "and"
    if op == IROP_OR: return "or"
    if op == IROP_XOR: return "xor"
    if op == IROP_NOT: return "not"
    if op == IROP_SHL: return "shl"
    if op == IROP_SHR: return "shr"
    if op == IROP_POPCOUNT: return "popcount"
    if op == IROP_CLZ: return "clz"
    if op == IROP_CTZ: return "ctz"
    if op == IROP_CAST: return "cast"
    if op == IROP_LOOP: return "loop"
    if op == IROP_PARALLEL: return "parallel"
    if op == IROP_PARALLEL_GRID: return "parallel_grid"
    if op == IROP_PARALLEL_WORKGROUP: return "parallel_workgroup"
    if op == IROP_PARALLEL_SUBGROUP: return "parallel_subgroup"
    if op == IROP_IF: return "if"
    if op == IROP_REDUCE_SUM: return "reduce_sum"
    if op == IROP_REDUCE_MAX: return "reduce_max"
    if op == IROP_REDUCE_MIN: return "reduce_min"
    if op == IROP_REDUCE_PROD: return "reduce_prod"
    if op == IROP_LOCAL: return "local"
    if op == IROP_PRIVATE: return "private"
    if op == IROP_BARRIER: return "barrier"
    if op == IROP_BLOCK_BEGIN: return "block_begin"
    if op == IROP_BLOCK_END: return "block_end"
    if op == IROP_RETURN: return "return"
    if op == IROP_COLLECTIVE_ALLREDUCE_SUM: return "collective_allreduce_sum"
    if op == IROP_COLLECTIVE_ALLREDUCE_MAX: return "collective_allreduce_max"
    if op == IROP_COLLECTIVE_ALLGATHER: return "collective_allgather"
    if op == IROP_COLLECTIVE_BROADCAST: return "collective_broadcast"
    if op == IROP_COLLECTIVE_REDUCE_SCATTER: return "collective_reduce_scatter"
    "unknown"

pub type IRProgram {
    insts: Vec[IRInst],
    aux: Vec[i32],
}

type StorageDesc {
    rank: i32,
    dtype: DType,
}

pub fn ir_inst(op: i32, dtype: DType, d0: i32, d1: i32, d2: i32, d3: i32) -> IRInst:
    IRInst { op, dtype, d0, d1, d2, d3 }

pub fn ir_program -> IRProgram:
    IRProgram {
        insts: Vec.new(),
        aux: Vec.new(),
    }

pub fn ir_param_inst(name_index: i32, mode: ParamMode, rank: i32, dtype: DType) -> IRInst:
    ir_inst(IROP_PARAM, dtype, name_index, ir_param_mode_code(mode), rank, 0)

pub fn ir_spec_constant_inst(name_index: i32, dtype: DType, low_bits: i32, high_bits: i32) -> IRInst:
    ir_inst(IROP_SPEC_CONSTANT, dtype, name_index, low_bits, high_bits, 0)

pub fn ir_const_bits(dest: i32, dtype: DType, low_bits: i32, high_bits: i32) -> IRInst:
    ir_inst(IROP_CONST, dtype, dest, low_bits, high_bits, 0)

pub fn ir_const_scalar(dest: i32, value: Scalar) -> IRInst:
    ir_const_bits(dest, value.dtype, scalar_low_i32(value.bits), scalar_high_i32(value.bits))

pub fn ir_const_i8(dest: i32, value: i8) -> IRInst:
    ir_const_scalar(dest, scalar_i8(value))

pub fn ir_const_i16(dest: i32, value: i16) -> IRInst:
    ir_const_scalar(dest, scalar_i16(value))

pub fn ir_const_i32(dest: i32, value: i32) -> IRInst:
    ir_const_scalar(dest, scalar_i32(value))

pub fn ir_const_i64(dest: i32, value: i64) -> IRInst:
    ir_const_scalar(dest, scalar_i64(value))

pub fn ir_const_u8(dest: i32, value: u8) -> IRInst:
    ir_const_scalar(dest, scalar_u8(value))

pub fn ir_const_u16(dest: i32, value: u16) -> IRInst:
    ir_const_scalar(dest, scalar_u16(value))

pub fn ir_const_u32(dest: i32, value: u32) -> IRInst:
    ir_const_scalar(dest, scalar_u32(value))

pub fn ir_const_u64(dest: i32, value: u64) -> IRInst:
    ir_const_scalar(dest, scalar_u64(value))

pub fn ir_const_f32(dest: i32, value: f32) -> IRInst:
    ir_const_scalar(dest, scalar_f32(value))

pub fn ir_const_f64(dest: i32, value: f64) -> IRInst:
    ir_const_scalar(dest, scalar_f64(value))

pub fn ir_const_f16_bits(dest: i32, bits: u16) -> IRInst:
    ir_const_scalar(dest, scalar_f16_bits(bits))

pub fn ir_const_bf16_bits(dest: i32, bits: u16) -> IRInst:
    ir_const_scalar(dest, scalar_bf16_bits(bits))

pub fn ir_param_mode_code(mode: ParamMode) -> i32:
    match mode
        .In => 0
        .Out => 1
        .InOut => 2
        .Scratch => 3

pub fn ir_param_mode_from_code(code: i32) -> Result[ParamMode, SubstrateError]:
    if code == 0:
        return Ok(.In)
    if code == 1:
        return Ok(.Out)
    if code == 2:
        return Ok(.InOut)
    if code == 3:
        return Ok(.Scratch)
    Err(.CompileError("invalid param mode code"))

pub fn ir_param_ref(param_index: i32) -> i32:
    -param_index - 1

pub fn ir_is_param_ref(ref: i32) -> bool:
    ref < 0 and ref > -IR_LOOP_REF_BASE

pub fn ir_param_index(ref: i32) -> i32:
    if not ir_is_param_ref(ref):
        return -1
    -ref - 1

let IR_LOOP_REF_BASE: i32 = 1024
let IR_LOCAL_REF_BASE: i32 = 2048
let IR_PRIVATE_REF_BASE: i32 = 3072

pub fn ir_loop_ref(loop_index: i32) -> i32:
    -(IR_LOOP_REF_BASE + loop_index)

pub fn ir_is_loop_ref(ref: i32) -> bool:
    ref <= -IR_LOOP_REF_BASE and ref > -IR_LOCAL_REF_BASE

pub fn ir_loop_index(ref: i32) -> i32:
    if not ir_is_loop_ref(ref):
        return -1
    -ref - IR_LOOP_REF_BASE

pub fn ir_local_ref(local_index: i32) -> i32:
    -(IR_LOCAL_REF_BASE + local_index)

pub fn ir_is_local_ref(ref: i32) -> bool:
    ref <= -IR_LOCAL_REF_BASE and ref > -IR_PRIVATE_REF_BASE

pub fn ir_local_index(ref: i32) -> i32:
    if not ir_is_local_ref(ref):
        return -1
    -ref - IR_LOCAL_REF_BASE

pub fn ir_private_ref(private_index: i32) -> i32:
    -(IR_PRIVATE_REF_BASE + private_index)

pub fn ir_is_private_ref(ref: i32) -> bool:
    ref <= -IR_PRIVATE_REF_BASE

pub fn ir_private_index(ref: i32) -> i32:
    if not ir_is_private_ref(ref):
        return -1
    -ref - IR_PRIVATE_REF_BASE

let IR_MAX_LOOP_SLOTS: i32 = 8

pub fn ir_is_header_op(op: i32) -> bool:
    op == IROP_PARAM or op == IROP_SPEC_CONSTANT

pub fn ir_is_value_producer(op: i32) -> bool:
    op == IROP_CONST or op == IROP_LOAD or op == IROP_ADD or op == IROP_SUB or op == IROP_MUL or op == IROP_DIV or op == IROP_MOD or op == IROP_ADD_SAT or op == IROP_SUB_SAT or op == IROP_AND or op == IROP_OR or op == IROP_XOR or op == IROP_SHL or op == IROP_SHR or op == IROP_MIN or op == IROP_MAX or op == IROP_EQ or op == IROP_NE or op == IROP_LT or op == IROP_GT or op == IROP_LE or op == IROP_GE or op == IROP_FMA or op == IROP_NEG or op == IROP_ABS or op == IROP_NOT or op == IROP_POPCOUNT or op == IROP_CLZ or op == IROP_CTZ or op == IROP_EXP or op == IROP_LOG or op == IROP_LOG2 or op == IROP_SQRT or op == IROP_RSQRT or op == IROP_SIN or op == IROP_COS or op == IROP_TANH or op == IROP_FLOOR or op == IROP_CEIL or op == IROP_ROUND or op == IROP_SELECT or op == IROP_CLAMP or op == IROP_CAST or op == IROP_REDUCE_SUM or op == IROP_REDUCE_MAX or op == IROP_REDUCE_MIN or op == IROP_REDUCE_PROD or op == IROP_COLLECTIVE_ALLREDUCE_SUM or op == IROP_COLLECTIVE_ALLREDUCE_MAX or op == IROP_COLLECTIVE_ALLGATHER or op == IROP_COLLECTIVE_BROADCAST or op == IROP_COLLECTIVE_REDUCE_SCATTER

pub fn ir_value_count(prog: IRProgram) -> i32:
    var count: i32 = 0
    for ip in 0..prog.insts.len():
        let inst = prog.insts[ip]
        if ir_is_value_producer(inst.op):
            count = count + 1
    count

fn is_param_binding_ref(ref: i32) -> bool:
    ir_is_param_ref(ref)

fn validate_value_layout(prog: IRProgram) -> Result[i32, SubstrateError]:
    var count: i32 = 0
    for ip in 0..prog.insts.len():
        let inst = prog.insts[ip]
        if not ir_is_value_producer(inst.op):
            continue
        if inst.d0 < 0:
            return Err(.CompileError("ir value id is invalid"))
        var other_ip: i32 = 0
        while other_ip < ip:
            let other = prog.insts[other_ip]
            if ir_is_value_producer(other.op) and other.d0 == inst.d0:
                return Err(.CompileError("ir value ids must be unique"))
            other_ip = other_ip + 1
        count = count + 1
    var expected: i32 = 0
    while expected < count:
        var found = false
        for ip in 0..prog.insts.len():
            let inst = prog.insts[ip]
            if ir_is_value_producer(inst.op) and inst.d0 == expected:
                found = true
                break
        if not found:
            return Err(.CompileError("ir value ids must form a dense 0-based range"))
        expected = expected + 1
    Ok(count)

fn value_inst_by_ref(prog: IRProgram, ref: i32) -> Result[IRInst, SubstrateError]:
    for ip in 0..prog.insts.len():
        let inst = prog.insts[ip]
        if ir_is_value_producer(inst.op) and inst.d0 == ref:
            return Ok(inst)
    Err(.CompileError("ir value ref out of range"))

fn validate_value_ref(value_count: i32, ref: i32) -> Result[i32, SubstrateError]:
    if ref < 0 or ref >= value_count:
        return Err(.CompileError("ir value ref out of range"))
    Ok(ref)

fn validate_scalar_ref(prog: IRProgram, value_count: i32, ref: i32) -> Result[i32, SubstrateError]:
    if ir_is_loop_ref(ref):
        let slot = ir_loop_index(ref)
        if slot < 0 or slot >= IR_MAX_LOOP_SLOTS:
            return Err(.CompileError("ir loop ref out of range"))
        return Ok(ref)
    if is_param_binding_ref(ref):
        return Err(.CompileError("ir index refs cannot point to params"))
    validate_value_ref(value_count, ref)

fn validate_param_ref(sig: ProgramSig, ref: i32) -> Result[i32, SubstrateError]:
    if not is_param_binding_ref(ref):
        return Err(.CompileError("ir param ref is invalid"))
    let index = ir_param_index(ref)
    if index < 0 or index >= sig.params.len() as i32:
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

fn validate_index_tuple(prog: IRProgram, value_count: i32, aux_base: i32, rank: i32) -> Result[i32, SubstrateError]:
    let _ = validate_aux_window(prog, aux_base, rank)?
    var i: i32 = 0
    while i < rank:
        let ref = prog.aux[aux_base + i]
        let checked_ref = validate_scalar_ref(prog, value_count, ref)?
        if not ir_is_loop_ref(checked_ref):
            let value = value_inst_by_ref(prog, checked_ref)?
            if value.dtype != .Int32 and value.dtype != .UInt32:
                return Err(.CompileError("ir index dtype must be i32 or u32"))
        i = i + 1
    Ok(aux_base)

fn validate_block_nesting(prog: IRProgram) -> Result[i32, SubstrateError]:
    let stack: Vec[i32] = Vec.new()
    var depth: i32 = 0
    for ip in 0..prog.insts.len():
        let inst = prog.insts[ip]
        if inst.op == IROP_BLOCK_BEGIN:
            if depth < stack.len() as i32:
                stack[depth] = inst.d0
            else:
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

fn find_block_end_ip(prog: IRProgram, begin_ip: i32, block_id: i32) -> Result[i32, SubstrateError]:
    if begin_ip < 0 or begin_ip >= prog.insts.len() as i32:
        return Err(.CompileError("ir block begin is missing"))
    let begin = prog.insts[begin_ip]
    if begin.op != IROP_BLOCK_BEGIN or begin.d0 != block_id:
        return Err(.CompileError("ir block must start with matching block_begin"))
    var depth: i32 = 1
    var ip = begin_ip + 1
    while ip < prog.insts.len() as i32:
        let inst = prog.insts[ip]
        if inst.op == IROP_BLOCK_BEGIN:
            depth = depth + 1
        if inst.op == IROP_BLOCK_END:
            depth = depth - 1
            if depth == 0:
                if inst.d0 != block_id:
                    return Err(.CompileError("ir block id mismatch"))
                return Ok(ip)
        ip = ip + 1
    Err(.CompileError("ir block begin without end"))

fn validate_if_layout(prog: IRProgram, ip: i32, then_block: i32, else_block: i32) -> Result[i32, SubstrateError]:
    let then_begin = ip + 1
    let then_end = find_block_end_ip(prog, then_begin, then_block)?
    if else_block < 0:
        return Ok(then_end)
    let else_begin = then_end + 1
    let _ = find_block_end_ip(prog, else_begin, else_block)?
    Ok(else_begin)

fn current_scalar_dtype_supported(dtype: DType) -> bool:
    dtype == .Int8 or dtype == .Int16 or dtype == .Int32 or dtype == .Int64 or dtype == .UInt8 or dtype == .UInt16 or dtype == .UInt32 or dtype == .UInt64 or dtype == .Float16 or dtype == .Float32 or dtype == .Float64 or dtype == .BFloat16

fn current_int_dtype_supported(dtype: DType) -> bool:
    dtype == .Int8 or dtype == .Int16 or dtype == .Int32 or dtype == .Int64 or dtype == .UInt8 or dtype == .UInt16 or dtype == .UInt32 or dtype == .UInt64

fn current_float_dtype_supported(dtype: DType) -> bool:
    dtype == .Float16 or dtype == .Float32 or dtype == .Float64 or dtype == .BFloat16

fn current_signed_int_dtype_supported(dtype: DType) -> bool:
    dtype == .Int8 or dtype == .Int16 or dtype == .Int32 or dtype == .Int64

fn current_sat_dtype_supported(dtype: DType) -> bool:
    current_int_dtype_supported(dtype)

fn cast_supported(from: DType, to: DType) -> bool:
    if from == to:
        return current_scalar_dtype_supported(from)
    if not current_scalar_dtype_supported(from) or not current_scalar_dtype_supported(to):
        return false
    true

pub fn validate_ir(prog: IRProgram, sig: ProgramSig) -> Result[i32, SubstrateError]:
    let value_count = validate_value_layout(prog)?
    let _ = validate_block_nesting(prog)?
    let locals: Vec[StorageDesc] = Vec.new()
    let privates: Vec[StorageDesc] = Vec.new()
    for ip in 0..prog.insts.len():
        let inst = prog.insts[ip]

        if inst.op == IROP_PARAM or inst.op == IROP_SPEC_CONSTANT or inst.op == IROP_BLOCK_BEGIN or inst.op == IROP_BLOCK_END or inst.op == IROP_RETURN or inst.op == IROP_BARRIER:
            continue

        if inst.op == IROP_CONST:
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir const dtype is not supported by current interpreter"))
            continue

        if inst.op == IROP_LOAD:
            var actual_load_desc = StorageDesc { rank: 0, dtype: .Int32 }
            if ir_is_param_ref(inst.d1):
                let param_index = validate_param_ref(sig, inst.d1)?
                actual_load_desc = StorageDesc { rank: sig.params[param_index].rank, dtype: sig.params[param_index].dtype }
            else:
                let _ = 0
            if ir_is_local_ref(inst.d1):
                let local_index = ir_local_index(inst.d1)
                if local_index < 0 or local_index >= locals.len() as i32:
                    return Err(.CompileError("ir local load ref out of range"))
                actual_load_desc = locals[local_index]
            else if ir_is_private_ref(inst.d1):
                let private_index = ir_private_index(inst.d1)
                if private_index < 0 or private_index >= privates.len() as i32:
                    return Err(.CompileError("ir private load ref out of range"))
                actual_load_desc = privates[private_index]
            else if not ir_is_param_ref(inst.d1):
                return Err(.CompileError("ir load ref is invalid"))
            if inst.dtype != actual_load_desc.dtype:
                return Err(.CompileError("ir load dtype mismatch"))
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir load dtype not supported by current interpreter"))
            let _ = validate_index_tuple(prog, value_count, inst.d2, actual_load_desc.rank)?
            continue

        if inst.op == IROP_STORE:
            var actual_store_desc = StorageDesc { rank: 0, dtype: .Int32 }
            if ir_is_param_ref(inst.d0):
                let param_index = validate_param_ref(sig, inst.d0)?
                actual_store_desc = StorageDesc { rank: sig.params[param_index].rank, dtype: sig.params[param_index].dtype }
            else:
                let _ = 0
            if ir_is_local_ref(inst.d0):
                let local_index = ir_local_index(inst.d0)
                if local_index < 0 or local_index >= locals.len() as i32:
                    return Err(.CompileError("ir local store ref out of range"))
                actual_store_desc = locals[local_index]
            else if ir_is_private_ref(inst.d0):
                let private_index = ir_private_index(inst.d0)
                if private_index < 0 or private_index >= privates.len() as i32:
                    return Err(.CompileError("ir private store ref out of range"))
                actual_store_desc = privates[private_index]
            else if not ir_is_param_ref(inst.d0):
                return Err(.CompileError("ir store ref is invalid"))
            if not current_scalar_dtype_supported(actual_store_desc.dtype):
                return Err(.CompileError("ir store dtype not supported by current interpreter"))
            let _ = validate_index_tuple(prog, value_count, inst.d1, actual_store_desc.rank)?
            let value_ref = validate_value_ref(value_count, inst.d2)?
            let value_inst = value_inst_by_ref(prog, value_ref)?
            if value_inst.dtype != actual_store_desc.dtype:
                return Err(.CompileError("ir store dtype mismatch"))
            continue

        if inst.op == IROP_LOCAL or inst.op == IROP_PRIVATE:
            if inst.d1 < 0 or inst.d1 > MAX_RANK:
                return Err(.CompileError("ir scratch rank is invalid"))
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir scratch dtype not supported by current interpreter"))
            let _ = validate_index_tuple(prog, value_count, inst.d2, inst.d1)?
            let desc = StorageDesc { rank: inst.d1, dtype: inst.dtype }
            if inst.op == IROP_LOCAL:
                locals.push(desc)
            else:
                privates.push(desc)
            continue

        if inst.op == IROP_ADD or inst.op == IROP_SUB or inst.op == IROP_MUL or inst.op == IROP_DIV or inst.op == IROP_MOD or inst.op == IROP_MIN or inst.op == IROP_MAX:
            let lhs_ref = validate_value_ref(value_count, inst.d1)?
            let rhs_ref = validate_value_ref(value_count, inst.d2)?
            let lhs = value_inst_by_ref(prog, lhs_ref)?
            let rhs = value_inst_by_ref(prog, rhs_ref)?
            if lhs.dtype != rhs.dtype or lhs.dtype != inst.dtype:
                return Err(.CompileError("ir binary op dtype mismatch"))
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir binary op dtype not supported by current interpreter"))
            if inst.op == IROP_MOD and not current_int_dtype_supported(inst.dtype):
                return Err(.CompileError("ir mod dtype must be integer"))
            continue

        if inst.op == IROP_ADD_SAT or inst.op == IROP_SUB_SAT:
            let lhs_ref = validate_value_ref(value_count, inst.d1)?
            let rhs_ref = validate_value_ref(value_count, inst.d2)?
            let lhs = value_inst_by_ref(prog, lhs_ref)?
            let rhs = value_inst_by_ref(prog, rhs_ref)?
            if lhs.dtype != rhs.dtype or lhs.dtype != inst.dtype:
                return Err(.CompileError("ir saturating binary op dtype mismatch"))
            if not current_sat_dtype_supported(inst.dtype):
                return Err(.CompileError("ir saturating binary op dtype must be supported integer"))
            continue

        if inst.op == IROP_AND or inst.op == IROP_OR or inst.op == IROP_XOR or inst.op == IROP_SHL or inst.op == IROP_SHR:
            let lhs_ref = validate_value_ref(value_count, inst.d1)?
            let rhs_ref = validate_value_ref(value_count, inst.d2)?
            let lhs = value_inst_by_ref(prog, lhs_ref)?
            let rhs = value_inst_by_ref(prog, rhs_ref)?
            if lhs.dtype != rhs.dtype or lhs.dtype != inst.dtype:
                return Err(.CompileError("ir bitwise operand dtype mismatch"))
            if not current_int_dtype_supported(inst.dtype):
                return Err(.CompileError("ir bitwise dtype must be integer"))
            continue

        if inst.op == IROP_EQ or inst.op == IROP_NE or inst.op == IROP_LT or inst.op == IROP_GT or inst.op == IROP_LE or inst.op == IROP_GE:
            let lhs_ref = validate_value_ref(value_count, inst.d1)?
            let rhs_ref = validate_value_ref(value_count, inst.d2)?
            let lhs = value_inst_by_ref(prog, lhs_ref)?
            let rhs = value_inst_by_ref(prog, rhs_ref)?
            if lhs.dtype != rhs.dtype:
                return Err(.CompileError("ir compare operand dtype mismatch"))
            if inst.dtype != .Int32:
                return Err(.CompileError("ir compare result dtype must be i32"))
            if not current_scalar_dtype_supported(lhs.dtype):
                return Err(.CompileError("ir compare dtype not supported by current interpreter"))
            continue

        if inst.op == IROP_FMA:
            let a_ref = validate_value_ref(value_count, inst.d1)?
            let b_ref = validate_value_ref(value_count, inst.d2)?
            let c_ref = validate_value_ref(value_count, inst.d3)?
            let a = value_inst_by_ref(prog, a_ref)?
            let b = value_inst_by_ref(prog, b_ref)?
            let c = value_inst_by_ref(prog, c_ref)?
            if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != inst.dtype:
                return Err(.CompileError("ir fma dtype mismatch"))
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir fma dtype not supported by current interpreter"))
            continue

        if inst.op == IROP_NEG or inst.op == IROP_ABS:
            let value_ref = validate_value_ref(value_count, inst.d1)?
            let value = value_inst_by_ref(prog, value_ref)?
            if value.dtype != inst.dtype:
                return Err(.CompileError("ir unary op dtype mismatch"))
            if inst.op == IROP_NEG:
                if not current_signed_int_dtype_supported(inst.dtype) and not current_float_dtype_supported(inst.dtype):
                    return Err(.CompileError("ir neg dtype not supported by current interpreter"))
            else:
                if not current_signed_int_dtype_supported(inst.dtype) and not current_float_dtype_supported(inst.dtype):
                    return Err(.CompileError("ir abs dtype not supported by current interpreter"))
            continue

        if inst.op == IROP_NOT:
            let value_ref = validate_value_ref(value_count, inst.d1)?
            let value = value_inst_by_ref(prog, value_ref)?
            if value.dtype != inst.dtype:
                return Err(.CompileError("ir bitwise unary dtype mismatch"))
            if not current_int_dtype_supported(inst.dtype):
                return Err(.CompileError("ir bitwise unary dtype must be integer"))
            continue

        if inst.op == IROP_POPCOUNT or inst.op == IROP_CLZ or inst.op == IROP_CTZ:
            let value_ref = validate_value_ref(value_count, inst.d1)?
            let value = value_inst_by_ref(prog, value_ref)?
            if value.dtype != inst.dtype:
                return Err(.CompileError("ir bitcount unary dtype mismatch"))
            if not current_int_dtype_supported(inst.dtype):
                return Err(.CompileError("ir bitcount unary dtype must be integer"))
            continue

        if inst.op == IROP_EXP or inst.op == IROP_LOG or inst.op == IROP_LOG2 or inst.op == IROP_SQRT or inst.op == IROP_RSQRT or inst.op == IROP_SIN or inst.op == IROP_COS or inst.op == IROP_TANH or inst.op == IROP_FLOOR or inst.op == IROP_CEIL or inst.op == IROP_ROUND:
            let value_ref = validate_value_ref(value_count, inst.d1)?
            let value = value_inst_by_ref(prog, value_ref)?
            if value.dtype != inst.dtype:
                return Err(.CompileError("ir float unary dtype mismatch"))
            if not current_float_dtype_supported(inst.dtype):
                return Err(.CompileError("ir float unary dtype must be float"))
            continue

        if inst.op == IROP_SELECT:
            let cond_ref = validate_value_ref(value_count, inst.d1)?
            let true_ref = validate_value_ref(value_count, inst.d2)?
            let false_ref = validate_value_ref(value_count, inst.d3)?
            let cond = value_inst_by_ref(prog, cond_ref)?
            let on_true = value_inst_by_ref(prog, true_ref)?
            let on_false = value_inst_by_ref(prog, false_ref)?
            if cond.dtype != .Int32:
                return Err(.CompileError("ir select condition dtype must be i32"))
            if on_true.dtype != on_false.dtype or on_true.dtype != inst.dtype:
                return Err(.CompileError("ir select arm dtype mismatch"))
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir select dtype not supported by current interpreter"))
            continue

        if inst.op == IROP_CLAMP:
            let value_ref = validate_value_ref(value_count, inst.d1)?
            let lo_ref = validate_value_ref(value_count, inst.d2)?
            let hi_ref = validate_value_ref(value_count, inst.d3)?
            let value = value_inst_by_ref(prog, value_ref)?
            let lo = value_inst_by_ref(prog, lo_ref)?
            let hi = value_inst_by_ref(prog, hi_ref)?
            if value.dtype != lo.dtype or value.dtype != hi.dtype or value.dtype != inst.dtype:
                return Err(.CompileError("ir clamp dtype mismatch"))
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir clamp dtype not supported by current interpreter"))
            continue

        if inst.op == IROP_CAST:
            let value_ref = validate_value_ref(value_count, inst.d1)?
            let value = value_inst_by_ref(prog, value_ref)?
            if not cast_supported(value.dtype, inst.dtype):
                return Err(.CompileError("ir cast dtype not supported by current interpreter"))
            continue

        if inst.op == IROP_LOOP or inst.op == IROP_PARALLEL or inst.op == IROP_PARALLEL_GRID or inst.op == IROP_PARALLEL_WORKGROUP or inst.op == IROP_PARALLEL_SUBGROUP:
            if inst.d0 < 0 or inst.d0 >= IR_MAX_LOOP_SLOTS:
                return Err(.CompileError("ir loop slot out of range"))
            let start_ref = validate_scalar_ref(prog, value_count, inst.d1)?
            let end_ref = validate_scalar_ref(prog, value_count, inst.d2)?
            if not ir_is_loop_ref(start_ref):
                let start_value = value_inst_by_ref(prog, start_ref)?
                if start_value.dtype != .Int32 and start_value.dtype != .UInt32:
                    return Err(.CompileError("ir loop bounds must be i32 or u32"))
            if not ir_is_loop_ref(end_ref):
                let end_value = value_inst_by_ref(prog, end_ref)?
                if end_value.dtype != .Int32 and end_value.dtype != .UInt32:
                    return Err(.CompileError("ir loop bounds must be i32 or u32"))
            if inst.d3 < 0:
                return Err(.CompileError("ir block id is invalid"))
            let next_ip = ip + 1
            if next_ip >= prog.insts.len():
                return Err(.CompileError("ir loop body is missing"))
            let begin = prog.insts[next_ip]
            if begin.op != IROP_BLOCK_BEGIN or begin.d0 != inst.d3:
                return Err(.CompileError("ir loop body must start with matching block"))
            continue

        if inst.op == IROP_IF:
            let cond_ref = validate_value_ref(value_count, inst.d1)?
            let cond = value_inst_by_ref(prog, cond_ref)?
            if cond.dtype != .Int32:
                return Err(.CompileError("ir if condition dtype must be i32"))
            if inst.d2 < 0:
                return Err(.CompileError("ir if then block id is invalid"))
            if inst.d3 == inst.d2:
                return Err(.CompileError("ir if else block must differ from then block"))
            let _ = validate_if_layout(prog, ip, inst.d2, inst.d3)?
            continue

        if inst.op == IROP_REDUCE_SUM or inst.op == IROP_REDUCE_MAX or inst.op == IROP_REDUCE_MIN or inst.op == IROP_REDUCE_PROD:
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir reduction dtype not supported by current interpreter"))
            let start_reduce_ref = validate_scalar_ref(prog, value_count, inst.d1)?
            let end_reduce_ref = validate_scalar_ref(prog, value_count, inst.d2)?
            if not ir_is_loop_ref(start_reduce_ref):
                let start_reduce_value = value_inst_by_ref(prog, start_reduce_ref)?
                if start_reduce_value.dtype != .Int32 and start_reduce_value.dtype != .UInt32:
                    return Err(.CompileError("ir reduction bounds must be i32 or u32"))
            if not ir_is_loop_ref(end_reduce_ref):
                let end_reduce_value = value_inst_by_ref(prog, end_reduce_ref)?
                if end_reduce_value.dtype != .Int32 and end_reduce_value.dtype != .UInt32:
                    return Err(.CompileError("ir reduction bounds must be i32 or u32"))
            let reduce_base = validate_aux_window(prog, inst.d3, 3)?
            let reduce_slot = prog.aux[reduce_base]
            let reduce_block = prog.aux[reduce_base + 1]
            let reduce_value_ref = validate_value_ref(value_count, prog.aux[reduce_base + 2])?
            if reduce_slot < 0 or reduce_slot >= IR_MAX_LOOP_SLOTS:
                return Err(.CompileError("ir reduction loop slot out of range"))
            if reduce_block < 0:
                return Err(.CompileError("ir reduction block id is invalid"))
            let reduce_value = value_inst_by_ref(prog, reduce_value_ref)?
            if reduce_value.dtype != inst.dtype:
                return Err(.CompileError("ir reduction body dtype mismatch"))
            let reduce_begin = ip + 1
            let reduce_begin_inst = prog.insts[reduce_begin]
            if reduce_begin_inst.op != IROP_BLOCK_BEGIN or reduce_begin_inst.d0 != reduce_block:
                return Err(.CompileError("ir reduction body must start with matching block"))
            continue

        if inst.op == IROP_COLLECTIVE_ALLREDUCE_SUM or inst.op == IROP_COLLECTIVE_ALLREDUCE_MAX or inst.op == IROP_COLLECTIVE_ALLGATHER or inst.op == IROP_COLLECTIVE_BROADCAST or inst.op == IROP_COLLECTIVE_REDUCE_SCATTER:
            let collective_ref = validate_value_ref(value_count, inst.d1)?
            let collective_value = value_inst_by_ref(prog, collective_ref)?
            if collective_value.dtype != inst.dtype:
                return Err(.CompileError("ir collective operand dtype mismatch"))
            if not current_scalar_dtype_supported(inst.dtype):
                return Err(.CompileError("ir collective dtype not supported by current interpreter"))
            continue

        return Err(.CompileError(f"ir validator has no rule for opcode '{ir_op_name(inst.op)}'"))
    Ok(0)
