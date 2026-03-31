use crux.core
use crux.errors
use crux.ir
use crux.memory
use crux.view

type InterpValue {
    raw: u64,
    dtype: DType,
}

type LoopVars {
    v0: i32,
    v1: i32,
    v2: i32,
    v3: i32,
    v4: i32,
    v5: i32,
    v6: i32,
    v7: i32,
}

type InterpState {
    values: Vec[InterpValue],
    loops: LoopVars,
}

type BlockRange {
    body_start: i32,
    block_end: i32,
}

fn interp_value_zero -> InterpValue:
    InterpValue { raw: 0, dtype: .Int32 }

fn loop_vars -> LoopVars:
    LoopVars {
        v0: 0,
        v1: 0,
        v2: 0,
        v3: 0,
        v4: 0,
        v5: 0,
        v6: 0,
        v7: 0,
    }

fn loop_var_get(vars: LoopVars, slot: i32) -> i32:
    if slot == 0: return vars.v0
    if slot == 1: return vars.v1
    if slot == 2: return vars.v2
    if slot == 3: return vars.v3
    if slot == 4: return vars.v4
    if slot == 5: return vars.v5
    if slot == 6: return vars.v6
    if slot == 7: return vars.v7
    0

fn loop_var_set(vars: LoopVars, slot: i32, value: i32) -> LoopVars:
    if slot == 0: return { vars with v0: value }
    if slot == 1: return { vars with v1: value }
    if slot == 2: return { vars with v2: value }
    if slot == 3: return { vars with v3: value }
    if slot == 4: return { vars with v4: value }
    if slot == 5: return { vars with v5: value }
    if slot == 6: return { vars with v6: value }
    if slot == 7: return { vars with v7: value }
    vars

fn init_state(inst_count: i32) -> InterpState:
    let values: Vec[InterpValue] = Vec.new()
    for i in 0..inst_count:
        let _ = i
        values.push(interp_value_zero())
    InterpState {
        values,
        loops: loop_vars(),
    }

fn value_from_i32(value: i32) -> InterpValue:
    InterpValue { raw: value as u64, dtype: .Int32 }

fn value_from_f32(value: f32) -> InterpValue:
    let raw: u32 = unsafe: transmute[u32](value)
    InterpValue { raw: raw as u64, dtype: .Float32 }

fn value_as_i32(value: InterpValue) -> Result[i32, SubstrateError]:
    if value.dtype == .Int32 or value.dtype == .UInt32:
        return Ok(value.raw as i32)
    Err(.Unsupported("cpu interpreter expected i32-compatible value"))

fn value_ref_get(state: InterpState, ref: i32) -> Result[InterpValue, SubstrateError]:
    if ref < 0 or ref >= state.values.len() as i32:
        return Err(.CompileError("value ref out of range"))
    Ok(state.values[ref])

fn resolve_i32_ref(state: InterpState, ref: i32) -> Result[i32, SubstrateError]:
    if ir_is_loop_ref(ref):
        return Ok(loop_var_get(state.loops, ir_loop_index(ref)))
    let value = value_ref_get(state, ref)?
    value_as_i32(value)

fn indices_from_aux(state: InterpState, prog: IRProgram, aux_base: i32, rank: i32) -> Result[Shape, SubstrateError]:
    var out = shape_scalar()
    out.rank = rank
    var i: i32 = 0
    while i < rank:
        let ref = prog.aux[aux_base + i]
        let index = resolve_i32_ref(state, ref)?
        out = shape_set(out, i, index as Size)
        out.rank = rank
        i = i + 1
    Ok(out)

fn load_value(view: View, byte_offset: Size) -> Result[InterpValue, SubstrateError]:
    let base = memory_ptr(view.memory)
    if base == null:
        return Err(.InvalidView("view memory is null"))
    if view.dtype == .Int32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i32
        return Ok(value_from_i32(unsafe: *ptr))
    if view.dtype == .Float32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut f32
        return Ok(value_from_f32(unsafe: *ptr))
    Err(.Unsupported("cpu interpreter dtype not implemented"))

fn store_value(view: View, byte_offset: Size, value: InterpValue) -> Result[i32, SubstrateError]:
    let base = memory_ptr(view.memory)
    if base == null:
        return Err(.InvalidView("view memory is null"))
    if view.dtype == .Int32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i32
        unsafe:
            *ptr = value.raw as i32
        return Ok(0)
    if view.dtype == .Float32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut f32
        let decoded: f32 = unsafe: transmute[f32](value.raw as u32)
        unsafe:
            *ptr = decoded
        return Ok(0)
    Err(.Unsupported("cpu interpreter dtype not implemented"))

fn run_binop(op: i32, lhs: InterpValue, rhs: InterpValue) -> Result[InterpValue, SubstrateError]:
    if lhs.dtype != rhs.dtype:
        return Err(.DTypeMismatch("cpu interpreter operand dtype mismatch"))
    if lhs.dtype == .Int32:
        let a = lhs.raw as i32
        let b = rhs.raw as i32
        if op == IROP_ADD: return Ok(value_from_i32(a + b))
        if op == IROP_SUB: return Ok(value_from_i32(a - b))
        if op == IROP_MUL: return Ok(value_from_i32(a * b))
        if op == IROP_DIV: return Ok(value_from_i32(a / b))
        if op == IROP_MIN: return Ok(value_from_i32(if a < b then a else b))
        if op == IROP_MAX: return Ok(value_from_i32(if a > b then a else b))
    if lhs.dtype == .Float32:
        let a: f32 = unsafe: transmute[f32](lhs.raw as u32)
        let b: f32 = unsafe: transmute[f32](rhs.raw as u32)
        if op == IROP_ADD: return Ok(value_from_f32(a + b))
        if op == IROP_SUB: return Ok(value_from_f32(a - b))
        if op == IROP_MUL: return Ok(value_from_f32(a * b))
        if op == IROP_DIV: return Ok(value_from_f32(a / b))
        if op == IROP_MIN: return Ok(value_from_f32(if a < b then a else b))
        if op == IROP_MAX: return Ok(value_from_f32(if a > b then a else b))
    Err(.Unsupported("cpu interpreter binary op not implemented"))

fn run_cmp(op: i32, lhs: InterpValue, rhs: InterpValue) -> Result[InterpValue, SubstrateError]:
    if lhs.dtype != rhs.dtype:
        return Err(.DTypeMismatch("cpu interpreter operand dtype mismatch"))
    if lhs.dtype == .Int32:
        let a = lhs.raw as i32
        let b = rhs.raw as i32
        if op == IROP_EQ: return Ok(value_from_i32(if a == b then 1 else 0))
        if op == IROP_NE: return Ok(value_from_i32(if a != b then 1 else 0))
        if op == IROP_LT: return Ok(value_from_i32(if a < b then 1 else 0))
        if op == IROP_GT: return Ok(value_from_i32(if a > b then 1 else 0))
        if op == IROP_LE: return Ok(value_from_i32(if a <= b then 1 else 0))
        if op == IROP_GE: return Ok(value_from_i32(if a >= b then 1 else 0))
    if lhs.dtype == .Float32:
        let a: f32 = unsafe: transmute[f32](lhs.raw as u32)
        let b: f32 = unsafe: transmute[f32](rhs.raw as u32)
        if op == IROP_EQ: return Ok(value_from_i32(if a == b then 1 else 0))
        if op == IROP_NE: return Ok(value_from_i32(if a != b then 1 else 0))
        if op == IROP_LT: return Ok(value_from_i32(if a < b then 1 else 0))
        if op == IROP_GT: return Ok(value_from_i32(if a > b then 1 else 0))
        if op == IROP_LE: return Ok(value_from_i32(if a <= b then 1 else 0))
        if op == IROP_GE: return Ok(value_from_i32(if a >= b then 1 else 0))
    Err(.Unsupported("cpu interpreter compare op not implemented"))

fn run_fma(a: InterpValue, b: InterpValue, c: InterpValue) -> Result[InterpValue, SubstrateError]:
    if a.dtype != b.dtype or a.dtype != c.dtype:
        return Err(.DTypeMismatch("cpu interpreter operand dtype mismatch"))
    if a.dtype == .Int32:
        return Ok(value_from_i32((a.raw as i32) * (b.raw as i32) + (c.raw as i32)))
    if a.dtype == .Float32:
        let av: f32 = unsafe: transmute[f32](a.raw as u32)
        let bv: f32 = unsafe: transmute[f32](b.raw as u32)
        let cv: f32 = unsafe: transmute[f32](c.raw as u32)
        return Ok(value_from_f32(av * bv + cv))
    Err(.Unsupported("cpu interpreter fma not implemented"))

fn run_neg(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if value.dtype == .Int32:
        return Ok(value_from_i32(-(value.raw as i32)))
    if value.dtype == .Float32:
        let decoded: f32 = unsafe: transmute[f32](value.raw as u32)
        return Ok(value_from_f32(-decoded))
    Err(.Unsupported("cpu interpreter neg not implemented"))

fn run_select(cond: InterpValue, on_true: InterpValue, on_false: InterpValue) -> Result[InterpValue, SubstrateError]:
    if cond.dtype != .Int32:
        return Err(.DTypeMismatch("cpu interpreter select condition must be i32"))
    if on_true.dtype != on_false.dtype:
        return Err(.DTypeMismatch("cpu interpreter select arm dtype mismatch"))
    if cond.raw as i32 != 0:
        return Ok(on_true)
    Ok(on_false)

fn run_clamp(value: InterpValue, lo: InterpValue, hi: InterpValue) -> Result[InterpValue, SubstrateError]:
    if value.dtype != lo.dtype or value.dtype != hi.dtype:
        return Err(.DTypeMismatch("cpu interpreter operand dtype mismatch"))
    if value.dtype == .Int32:
        let v = value.raw as i32
        let lo_v = lo.raw as i32
        let hi_v = hi.raw as i32
        if v < lo_v:
            return Ok(value_from_i32(lo_v))
        if v > hi_v:
            return Ok(value_from_i32(hi_v))
        return Ok(value_from_i32(v))
    if value.dtype == .Float32:
        let v: f32 = unsafe: transmute[f32](value.raw as u32)
        let lo_v: f32 = unsafe: transmute[f32](lo.raw as u32)
        let hi_v: f32 = unsafe: transmute[f32](hi.raw as u32)
        if v < lo_v:
            return Ok(value_from_f32(lo_v))
        if v > hi_v:
            return Ok(value_from_f32(hi_v))
        return Ok(value_from_f32(v))
    Err(.Unsupported("cpu interpreter clamp not implemented"))

fn find_block_range(prog: IRProgram, loop_ip: i32, stop: i32, block_id: i32) -> Result[BlockRange, SubstrateError]:
    let begin = loop_ip + 1
    if begin >= stop:
        return Err(.CompileError("loop is missing a block body"))
    let begin_inst = prog.insts[begin]
    if begin_inst.op != IROP_BLOCK_BEGIN or begin_inst.d0 != block_id:
        return Err(.CompileError("loop body must start with block_begin"))
    var depth: i32 = 1
    var ip = begin + 1
    while ip < stop:
        let inst = prog.insts[ip]
        if inst.op == IROP_BLOCK_BEGIN:
            depth = depth + 1
        if inst.op == IROP_BLOCK_END:
            depth = depth - 1
            if depth == 0:
                if inst.d0 != block_id:
                    return Err(.CompileError("mismatched block id"))
                return Ok(BlockRange {
                    body_start: begin + 1,
                    block_end: ip,
                })
        ip = ip + 1
    Err(.CompileError("unterminated loop block"))

fn exec_range(prog: IRProgram, params: Vec[View], state: InterpState, start: i32, stop: i32) -> Result[InterpState, SubstrateError]:
    var out = state
    var ip = start
    while ip < stop:
        let inst = prog.insts[ip]

        if inst.op == IROP_CONST:
            out.values[ip] = InterpValue {
                raw: inst.d0 as u64,
                dtype: inst.dtype,
            }
            ip = ip + 1
            continue

        if inst.op == IROP_LOAD:
            let param_index = ir_param_index(inst.d0)
            let view = params[param_index]
            let indices = indices_from_aux(out, prog, inst.d1, view.shape.rank)?
            let offset = view_offset_of(view, indices)
            out.values[ip] = load_value(view, offset)?
            ip = ip + 1
            continue

        if inst.op == IROP_STORE:
            let param_index = ir_param_index(inst.d0)
            let view = params[param_index]
            let indices = indices_from_aux(out, prog, inst.d1, view.shape.rank)?
            let offset = view_offset_of(view, indices)
            let value = value_ref_get(out, inst.d2)?
            let _ = store_value(view, offset, value)?
            ip = ip + 1
            continue

        if inst.op == IROP_ADD or inst.op == IROP_SUB or inst.op == IROP_MUL or inst.op == IROP_DIV or inst.op == IROP_MIN or inst.op == IROP_MAX:
            let lhs = value_ref_get(out, inst.d0)?
            let rhs = value_ref_get(out, inst.d1)?
            out.values[ip] = run_binop(inst.op, lhs, rhs)?
            ip = ip + 1
            continue

        if inst.op == IROP_EQ or inst.op == IROP_NE or inst.op == IROP_LT or inst.op == IROP_GT or inst.op == IROP_LE or inst.op == IROP_GE:
            let lhs = value_ref_get(out, inst.d0)?
            let rhs = value_ref_get(out, inst.d1)?
            out.values[ip] = run_cmp(inst.op, lhs, rhs)?
            ip = ip + 1
            continue

        if inst.op == IROP_FMA:
            let a = value_ref_get(out, inst.d0)?
            let b = value_ref_get(out, inst.d1)?
            let c = value_ref_get(out, inst.d2)?
            out.values[ip] = run_fma(a, b, c)?
            ip = ip + 1
            continue

        if inst.op == IROP_NEG:
            let value = value_ref_get(out, inst.d0)?
            out.values[ip] = run_neg(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_SELECT:
            let cond = value_ref_get(out, inst.d0)?
            let on_true = value_ref_get(out, inst.d1)?
            let on_false = value_ref_get(out, inst.d2)?
            out.values[ip] = run_select(cond, on_true, on_false)?
            ip = ip + 1
            continue

        if inst.op == IROP_CLAMP:
            let value = value_ref_get(out, inst.d0)?
            let lo = value_ref_get(out, inst.d1)?
            let hi = value_ref_get(out, inst.d2)?
            out.values[ip] = run_clamp(value, lo, hi)?
            ip = ip + 1
            continue

        if inst.op == IROP_LOOP or inst.op == IROP_PARALLEL:
            let block = find_block_range(prog, ip, stop, inst.d3)?
            let start_value = resolve_i32_ref(out, inst.d1)?
            let end_value = resolve_i32_ref(out, inst.d2)?
            var iter = start_value
            while iter < end_value:
                out.loops = loop_var_set(out.loops, inst.d0, iter)
                out = exec_range(prog, params, out, block.body_start, block.block_end)?
                iter = iter + 1
            ip = block.block_end + 1
            continue

        if inst.op == IROP_BLOCK_BEGIN or inst.op == IROP_BLOCK_END:
            return Err(.CompileError("unexpected block marker during execution"))

        if inst.op == IROP_RETURN:
            return Ok(out)

        return Err(.Unsupported("cpu interpreter opcode not implemented"))
    Ok(out)

pub fn interp_dispatch(prog: IRProgram, params: Vec[View]) -> Result[i32, SubstrateError]:
    if params.len() as i32 != prog.num_params:
        return Err(.ShapeMismatch("binding count does not match program parameters"))
    let state = init_state(prog.insts.len() as i32)
    let _ = exec_range(prog, params, state, 0, prog.insts.len() as i32)?
    Ok(0)
