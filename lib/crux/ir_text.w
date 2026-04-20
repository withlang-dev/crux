use crux.core
use crux.ir

pub error IRTextError =
    ParseError(msg: str)

type LineScan {
    line: str,
    next: i32,
}

type TokenScan {
    token: str,
    next: i32,
}

type GroupScan {
    inner: str,
    next: i32,
}

type ParamSymbols {
    count: i32,
    name0: str,
    name1: str,
    name2: str,
    name3: str,
    name4: str,
    name5: str,
    name6: str,
    name7: str,
}

type ParseState {
    source: ProgramSource,
    params: ParamSymbols,
    locals: ParamSymbols,
    privates: ParamSymbols,
    next_value: i32,
}

type StringIntern {
    source: ProgramSource,
    index: i32,
}

pub fn parse_ir_text(text: str) -> Result[ProgramSource, IRTextError]:
    var state = ParseState {
        source: ProgramSource {
            ir: Vec.new(),
            aux: Vec.new(),
            strings: Vec.new(),
            entry: "main",
        },
        params: param_symbols(),
        locals: param_symbols(),
        privates: param_symbols(),
        next_value: 0,
    }
    var offset: i32 = 0
    let limit = text.len() as i32

    while offset < limit:
        let scan = next_line(text, offset)
        offset = scan.next
        let line = trim_ascii(scan.line)
        if line == "":
            continue
        if starts_with(line, "#") or starts_with(line, "//"):
            continue
        if starts_with(line, "spec_constant"):
            state = parse_spec_constant_line(line, state)?
            continue
        if starts_with(line, "param"):
            state = parse_param_line(line, state)?
            continue
        if starts_with(line, "local") or starts_with(line, "private"):
            state = parse_storage_line(line, state)?
            continue
        if starts_with(line, "store"):
            state = parse_store_line(line, state)?
            continue
        if starts_with(line, "parallel") or starts_with(line, "loop") or starts_with(line, "block_begin") or starts_with(line, "block_end") or starts_with(line, "if") or starts_with(line, "barrier"):
            state = parse_control_line(line, state)?
            continue
        if starts_with(line, "return"):
            state.source.ir.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))
            continue
        if starts_with(line, "%"):
            state = parse_value_line(line, state)?
            continue
        return Err(.ParseError("unknown IR text line"))

    Ok(state.source)

fn parse_spec_constant_line(line: str, state: ParseState) -> Result[ParseState, IRTextError]:
    var cursor = skip_ws(line, 0)
    cursor = expect_token(line, cursor, "spec_constant")?
    let name_scan = scan_token(line, cursor)
    if name_scan.token == "":
        return Err(.ParseError("spec_constant line missing name"))
    cursor = name_scan.next
    let dtype_scan = scan_token(line, cursor)
    let dtype = parse_dtype(dtype_scan.token)?
    cursor = dtype_scan.next
    let literal_scan = scan_token(line, cursor)
    let interned = intern_string(state.source, name_scan.token)
    let literal = parse_scalar_literal(dtype, literal_scan.token)?
    var next_state = state
    next_state.source = interned.source
    next_state.source.ir.push(ir_spec_constant_inst(interned.index, dtype, scalar_low_i32(literal.bits), scalar_high_i32(literal.bits)))
    Ok(next_state)

fn parse_param_line(line: str, state: ParseState) -> Result[ParseState, IRTextError]:
    var cursor = skip_ws(line, 0)
    cursor = expect_token(line, cursor, "param")?
    let name_scan = scan_token(line, cursor)
    if name_scan.token == "":
        return Err(.ParseError("param line missing name"))
    cursor = name_scan.next
    let mode_scan = scan_token(line, cursor)
    let mode = parse_param_mode(mode_scan.token)?
    cursor = mode_scan.next
    let shape_scan = scan_bracket_group(line, cursor)?
    let rank = count_rank(shape_scan.inner)
    cursor = shape_scan.next
    let dtype_scan = scan_token(line, cursor)
    let dtype = parse_dtype(dtype_scan.token)?
    let interned = intern_string(state.source, name_scan.token)
    var next_state = state
    next_state.source = interned.source
    next_state.source.ir.push(ir_param_inst(interned.index, mode, rank, dtype))
    next_state.params = param_symbols_add(state.params, name_scan.token)?
    Ok(next_state)

fn parse_storage_line(line: str, state: ParseState) -> Result[ParseState, IRTextError]:
    var cursor = skip_ws(line, 0)
    let op_scan = scan_token(line, cursor)
    let op_name = op_scan.token
    cursor = op_scan.next
    let name_scan = scan_token(line, cursor)
    if name_scan.token == "":
        return Err(.ParseError("storage line missing name"))
    cursor = name_scan.next
    let shape_scan = scan_bracket_group(line, cursor)?
    let rank = count_rank(shape_scan.inner)
    cursor = shape_scan.next
    let dtype_scan = scan_token(line, cursor)
    let dtype = parse_dtype(dtype_scan.token)?
    let interned = intern_string(state.source, name_scan.token)
    var next_state = state
    next_state.source = interned.source
    let aux_base = next_state.source.aux.len() as i32
    next_state.source.aux = append_index_refs(next_state.source.aux, shape_scan.inner)?
    let op = if op_name == "local" then IROP_LOCAL else IROP_PRIVATE
    next_state.source.ir.push(ir_inst(op, dtype, interned.index, rank, aux_base, 0))
    if op_name == "local":
        next_state.locals = param_symbols_add(state.locals, name_scan.token)?
    else:
        next_state.privates = param_symbols_add(state.privates, name_scan.token)?
    Ok(next_state)

fn parse_store_line(line: str, state: ParseState) -> Result[ParseState, IRTextError]:
    var cursor = skip_ws(line, 0)
    cursor = expect_token(line, cursor, "store")?
    let name_scan = scan_token(line, cursor)
    let storage_ref = resolve_named_ref(state, name_scan.token)?
    cursor = name_scan.next
    let group_scan = scan_bracket_group(line, cursor)?
    var next_state = state
    let aux_base = next_state.source.aux.len() as i32
    next_state.source.aux = append_index_refs(next_state.source.aux, group_scan.inner)?
    cursor = group_scan.next
    let value_scan = scan_token(line, cursor)
    let value_ref = parse_value_ref(value_scan.token)?
    let dtype = storage_dtype_by_ref(next_state.source, storage_ref)?
    next_state.source.ir.push(ir_inst(IROP_STORE, dtype, storage_ref, aux_base, value_ref, 0))
    Ok(next_state)

fn parse_control_line(line: str, state: ParseState) -> Result[ParseState, IRTextError]:
    var cursor = skip_ws(line, 0)
    let op_scan = scan_token(line, cursor)
    let op_name = op_scan.token
    cursor = op_scan.next

    if op_name == "block_begin" or op_name == "block_end":
        let block_scan = scan_token(line, cursor)
        let block_id = parse_i32_ascii(block_scan.token)?
        let op = if op_name == "block_begin" then IROP_BLOCK_BEGIN else IROP_BLOCK_END
        var next_state = state
        next_state.source.ir.push(ir_inst(op, .Int32, block_id, 0, 0, 0))
        return Ok(next_state)

    if op_name == "barrier":
        var next_state = state
        next_state.source.ir.push(ir_inst(IROP_BARRIER, .Int32, 0, 0, 0, 0))
        return Ok(next_state)

    if op_name == "if":
        let cond_scan = scan_token(line, cursor)
        let cond_ref = parse_value_ref(cond_scan.token)?
        cursor = cond_scan.next
        let then_scan = scan_token(line, cursor)
        let then_block = parse_i32_ascii(then_scan.token)?
        cursor = then_scan.next
        let else_scan = scan_token(line, cursor)
        let else_block = if else_scan.token == "" then -1 else parse_i32_ascii(else_scan.token)?
        var next_state = state
        next_state.source.ir.push(ir_inst(IROP_IF, .Int32, 0, cond_ref, then_block, else_block))
        return Ok(next_state)

    if op_name == "parallel" or op_name == "loop" or op_name == "parallel_grid" or op_name == "parallel_workgroup" or op_name == "parallel_subgroup":
        let slot_scan = scan_token(line, cursor)
        let slot = parse_i32_ascii(slot_scan.token)?
        cursor = slot_scan.next
        let start_scan = scan_token(line, cursor)
        let start_ref = parse_index_ref(start_scan.token)?
        cursor = start_scan.next
        let end_scan = scan_token(line, cursor)
        let end_ref = parse_index_ref(end_scan.token)?
        cursor = end_scan.next
        let block_scan = scan_token(line, cursor)
        let block_id = parse_i32_ascii(block_scan.token)?
        let op = if op_name == "parallel" then IROP_PARALLEL else if op_name == "loop" then IROP_LOOP else if op_name == "parallel_grid" then IROP_PARALLEL_GRID else if op_name == "parallel_workgroup" then IROP_PARALLEL_WORKGROUP else IROP_PARALLEL_SUBGROUP
        var next_state = state
        next_state.source.ir.push(ir_inst(op, .Int32, slot, start_ref, end_ref, block_id))
        return Ok(next_state)

    Err(.ParseError("unsupported control opcode"))

fn parse_value_line(line: str, state: ParseState) -> Result[ParseState, IRTextError]:
    var cursor = skip_ws(line, 0)
    let id_scan = scan_token(line, cursor)
    let value_id = parse_value_ref(id_scan.token)?
    if value_id != state.next_value:
        return Err(.ParseError("value ids must be sequential"))
    cursor = id_scan.next
    cursor = expect_token(line, cursor, "=")?
    let op_scan = scan_token(line, cursor)
    let op_name = op_scan.token
    cursor = op_scan.next

    if op_name == "const":
        let dtype_scan = scan_token(line, cursor)
        let dtype = parse_dtype(dtype_scan.token)?
        cursor = dtype_scan.next
        let literal_scan = scan_token(line, cursor)
        let literal = parse_scalar_literal(dtype, literal_scan.token)?
        var next_state = state
        next_state.source.ir.push(ir_const_scalar(value_id, literal))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "load":
        let name_scan = scan_token(line, cursor)
        let storage_ref = resolve_named_ref(state, name_scan.token)?
        cursor = name_scan.next
        let group_scan = scan_bracket_group(line, cursor)?
        var next_state = state
        let aux_base = next_state.source.aux.len() as i32
        next_state.source.aux = append_index_refs(next_state.source.aux, group_scan.inner)?
        let dtype = storage_dtype_by_ref(next_state.source, storage_ref)?
        next_state.source.ir.push(ir_inst(IROP_LOAD, dtype, value_id, storage_ref, aux_base, 0))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "add" or op_name == "sub" or op_name == "mul" or op_name == "div" or op_name == "mod" or op_name == "add_sat" or op_name == "sub_sat" or op_name == "and" or op_name == "or" or op_name == "xor" or op_name == "shl" or op_name == "shr" or op_name == "min" or op_name == "max":
        let lhs_scan = scan_token(line, cursor)
        let lhs = parse_value_ref(lhs_scan.token)?
        cursor = lhs_scan.next
        let rhs_scan = scan_token(line, cursor)
        let rhs = parse_value_ref(rhs_scan.token)?
        let op = parse_binop(op_name)?
        let dtype = value_dtype_by_ref(state.source, lhs)?
        var next_state = state
        next_state.source.ir.push(ir_inst(op, dtype, value_id, lhs, rhs, 0))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "eq" or op_name == "ne" or op_name == "lt" or op_name == "gt" or op_name == "le" or op_name == "ge":
        let lhs_scan = scan_token(line, cursor)
        let lhs = parse_value_ref(lhs_scan.token)?
        cursor = lhs_scan.next
        let rhs_scan = scan_token(line, cursor)
        let rhs = parse_value_ref(rhs_scan.token)?
        let op = parse_cmpop(op_name)?
        var next_state = state
        next_state.source.ir.push(ir_inst(op, .Int32, value_id, lhs, rhs, 0))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "fma":
        let a_scan = scan_token(line, cursor)
        let a = parse_value_ref(a_scan.token)?
        cursor = a_scan.next
        let b_scan = scan_token(line, cursor)
        let b = parse_value_ref(b_scan.token)?
        cursor = b_scan.next
        let c_scan = scan_token(line, cursor)
        let c = parse_value_ref(c_scan.token)?
        let dtype = value_dtype_by_ref(state.source, a)?
        var next_state = state
        next_state.source.ir.push(ir_inst(IROP_FMA, dtype, value_id, a, b, c))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "neg" or op_name == "abs" or op_name == "not" or op_name == "popcount" or op_name == "clz" or op_name == "ctz" or op_name == "exp" or op_name == "log" or op_name == "log2" or op_name == "sqrt" or op_name == "rsqrt" or op_name == "sin" or op_name == "cos" or op_name == "tanh" or op_name == "floor" or op_name == "ceil" or op_name == "round":
        let value_scan = scan_token(line, cursor)
        let value = parse_value_ref(value_scan.token)?
        let dtype = value_dtype_by_ref(state.source, value)?
        let op = if op_name == "neg" then IROP_NEG else if op_name == "abs" then IROP_ABS else if op_name == "not" then IROP_NOT else if op_name == "popcount" then IROP_POPCOUNT else if op_name == "clz" then IROP_CLZ else if op_name == "ctz" then IROP_CTZ else if op_name == "exp" then IROP_EXP else if op_name == "log" then IROP_LOG else if op_name == "log2" then IROP_LOG2 else if op_name == "sqrt" then IROP_SQRT else if op_name == "rsqrt" then IROP_RSQRT else if op_name == "sin" then IROP_SIN else if op_name == "cos" then IROP_COS else if op_name == "tanh" then IROP_TANH else if op_name == "floor" then IROP_FLOOR else if op_name == "ceil" then IROP_CEIL else IROP_ROUND
        var next_state = state
        next_state.source.ir.push(ir_inst(op, dtype, value_id, value, 0, 0))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "cast":
        let dtype_scan = scan_token(line, cursor)
        let target_dtype = parse_dtype(dtype_scan.token)?
        cursor = dtype_scan.next
        let value_scan = scan_token(line, cursor)
        let value = parse_value_ref(value_scan.token)?
        var next_state = state
        next_state.source.ir.push(ir_inst(IROP_CAST, target_dtype, value_id, value, 0, 0))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "select":
        let cond_scan = scan_token(line, cursor)
        let cond = parse_value_ref(cond_scan.token)?
        cursor = cond_scan.next
        let true_scan = scan_token(line, cursor)
        let on_true = parse_value_ref(true_scan.token)?
        cursor = true_scan.next
        let false_scan = scan_token(line, cursor)
        let on_false = parse_value_ref(false_scan.token)?
        let dtype = value_dtype_by_ref(state.source, on_true)?
        var next_state = state
        next_state.source.ir.push(ir_inst(IROP_SELECT, dtype, value_id, cond, on_true, on_false))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "clamp":
        let value_scan = scan_token(line, cursor)
        let value = parse_value_ref(value_scan.token)?
        cursor = value_scan.next
        let lo_scan = scan_token(line, cursor)
        let lo = parse_value_ref(lo_scan.token)?
        cursor = lo_scan.next
        let hi_scan = scan_token(line, cursor)
        let hi = parse_value_ref(hi_scan.token)?
        let dtype = value_dtype_by_ref(state.source, value)?
        var next_state = state
        next_state.source.ir.push(ir_inst(IROP_CLAMP, dtype, value_id, value, lo, hi))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "reduce_sum" or op_name == "reduce_max" or op_name == "reduce_min" or op_name == "reduce_prod":
        let start_scan = scan_token(line, cursor)
        let start_ref = parse_index_ref(start_scan.token)?
        cursor = start_scan.next
        let end_scan = scan_token(line, cursor)
        let end_ref = parse_index_ref(end_scan.token)?
        cursor = end_scan.next
        let group_scan = scan_bracket_group(line, cursor)?
        var next_state = state
        let parts = trim_ascii(group_scan.inner)
        let meta_base = next_state.source.aux.len() as i32
        next_state.source.aux = append_reduce_meta(next_state.source.aux, parts)?
        let body_ref = next_state.source.aux[meta_base + 2]
        let dtype = value_dtype_by_ref(next_state.source, body_ref)?
        let op = if op_name == "reduce_sum" then IROP_REDUCE_SUM else if op_name == "reduce_max" then IROP_REDUCE_MAX else if op_name == "reduce_min" then IROP_REDUCE_MIN else IROP_REDUCE_PROD
        next_state.source.ir.push(ir_inst(op, dtype, value_id, start_ref, end_ref, meta_base))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    if op_name == "collective_allreduce_sum" or op_name == "collective_allreduce_max" or op_name == "collective_allgather" or op_name == "collective_broadcast" or op_name == "collective_reduce_scatter":
        let value_scan = scan_token(line, cursor)
        let input_ref = parse_value_ref(value_scan.token)?
        let dtype = value_dtype_by_ref(state.source, input_ref)?
        let op = if op_name == "collective_allreduce_sum" then IROP_COLLECTIVE_ALLREDUCE_SUM else if op_name == "collective_allreduce_max" then IROP_COLLECTIVE_ALLREDUCE_MAX else if op_name == "collective_allgather" then IROP_COLLECTIVE_ALLGATHER else if op_name == "collective_broadcast" then IROP_COLLECTIVE_BROADCAST else IROP_COLLECTIVE_REDUCE_SCATTER
        var next_state = state
        next_state.source.ir.push(ir_inst(op, dtype, value_id, input_ref, 0, 0))
        next_state.next_value = next_state.next_value + 1
        return Ok(next_state)

    Err(.ParseError("unsupported IR text opcode"))

fn next_line(text: str, start: i32) -> LineScan:
    let limit = text.len() as i32
    if start >= limit:
        return LineScan { line: "", next: start }
    var end = start
    while end < limit and text.slice(end, end + 1) != "\n":
        end = end + 1
    var next = end
    if next < limit and text.slice(next, next + 1) == "\n":
        next = next + 1
    LineScan {
        line: text.slice(start, end),
        next,
    }

fn trim_ascii(text: str) -> str:
    var start: i32 = 0
    var stop = text.len() as i32
    while start < stop and is_ws(text.slice(start, start + 1)):
        start = start + 1
    while stop > start and is_ws(text.slice(stop - 1, stop)):
        stop = stop - 1
    text.slice(start, stop)

fn skip_ws(text: str, start: i32) -> i32:
    var pos = start
    let limit = text.len() as i32
    while pos < limit and is_ws(text.slice(pos, pos + 1)):
        pos = pos + 1
    pos

fn scan_token(text: str, start: i32) -> TokenScan:
    let begin = skip_ws(text, start)
    let limit = text.len() as i32
    var end = begin
    while end < limit:
        let ch = text.slice(end, end + 1)
        if is_ws(ch):
            break
        end = end + 1
    TokenScan {
        token: text.slice(begin, end),
        next: end,
    }

fn scan_bracket_group(text: str, start: i32) -> Result[GroupScan, IRTextError]:
    var pos = skip_ws(text, start)
    if text.slice(pos, pos + 1) != "[":
        return Err(.ParseError("expected '['"))
    pos = pos + 1
    let begin = pos
    let limit = text.len() as i32
    while pos < limit and text.slice(pos, pos + 1) != "]":
        pos = pos + 1
    if pos >= limit:
        return Err(.ParseError("unterminated '[' group"))
    Ok(GroupScan {
        inner: text.slice(begin, pos),
        next: pos + 1,
    })

fn expect_token(text: str, start: i32, expected: str) -> Result[i32, IRTextError]:
    let scan = scan_token(text, start)
    if scan.token != expected:
        return Err(.ParseError("unexpected token"))
    Ok(scan.next)

fn count_rank(text: str) -> i32:
    let trimmed = trim_ascii(text)
    if trimmed == "":
        return 0
    var count: i32 = 1
    var pos: i32 = 0
    let limit = trimmed.len() as i32
    while pos < limit:
        if trimmed.slice(pos, pos + 1) == ",":
            count = count + 1
        pos = pos + 1
    count

fn parse_param_mode(token: str) -> Result[ParamMode, IRTextError]:
    if token == "in":
        return Ok(.In)
    if token == "out":
        return Ok(.Out)
    if token == "inout":
        return Ok(.InOut)
    if token == "scratch":
        return Ok(.Scratch)
    Err(.ParseError("unknown param mode"))

fn parse_dtype(token: str) -> Result[DType, IRTextError]:
    if token == "i8": return Ok(.Int8)
    if token == "i16": return Ok(.Int16)
    if token == "i32": return Ok(.Int32)
    if token == "i64": return Ok(.Int64)
    if token == "u8": return Ok(.UInt8)
    if token == "u16": return Ok(.UInt16)
    if token == "u32": return Ok(.UInt32)
    if token == "u64": return Ok(.UInt64)
    if token == "f16": return Ok(.Float16)
    if token == "f32": return Ok(.Float32)
    if token == "f64": return Ok(.Float64)
    if token == "bf16": return Ok(.BFloat16)
    Err(.ParseError("unknown dtype"))

fn parse_binop(name: str) -> Result[i32, IRTextError]:
    if name == "add": return Ok(IROP_ADD)
    if name == "sub": return Ok(IROP_SUB)
    if name == "mul": return Ok(IROP_MUL)
    if name == "div": return Ok(IROP_DIV)
    if name == "mod": return Ok(IROP_MOD)
    if name == "add_sat": return Ok(IROP_ADD_SAT)
    if name == "sub_sat": return Ok(IROP_SUB_SAT)
    if name == "and": return Ok(IROP_AND)
    if name == "or": return Ok(IROP_OR)
    if name == "xor": return Ok(IROP_XOR)
    if name == "shl": return Ok(IROP_SHL)
    if name == "shr": return Ok(IROP_SHR)
    if name == "min": return Ok(IROP_MIN)
    if name == "max": return Ok(IROP_MAX)
    Err(.ParseError("unknown binary opcode"))

fn parse_cmpop(name: str) -> Result[i32, IRTextError]:
    if name == "eq": return Ok(IROP_EQ)
    if name == "ne": return Ok(IROP_NE)
    if name == "lt": return Ok(IROP_LT)
    if name == "gt": return Ok(IROP_GT)
    if name == "le": return Ok(IROP_LE)
    if name == "ge": return Ok(IROP_GE)
    Err(.ParseError("unknown compare opcode"))

fn parse_i32_ascii(token: str) -> Result[i32, IRTextError]:
    if token == "":
        return Err(.ParseError("expected integer literal"))
    var pos: i32 = 0
    let limit = token.len() as i32
    var sign: i32 = 1
    if token.slice(0, 1) == "-":
        sign = -1
        pos = 1
    if pos >= limit:
        return Err(.ParseError("invalid integer literal"))
    var out: i32 = 0
    while pos < limit:
        let ch = token.slice(pos, pos + 1)
        if ch != "0" and ch != "1" and ch != "2" and ch != "3" and ch != "4" and ch != "5" and ch != "6" and ch != "7" and ch != "8" and ch != "9":
            return Err(.ParseError("invalid integer literal"))
        let digit = if ch == "0" then 0 else if ch == "1" then 1 else if ch == "2" then 2 else if ch == "3" then 3 else if ch == "4" then 4 else if ch == "5" then 5 else if ch == "6" then 6 else if ch == "7" then 7 else if ch == "8" then 8 else 9
        out = out * 10 + digit
        pos = pos + 1
    Ok(out * sign)

fn parse_u64_ascii(token: str) -> Result[u64, IRTextError]:
    if token == "":
        return Err(.ParseError("expected unsigned integer literal"))
    var pos: i32 = 0
    let limit = token.len() as i32
    var out: u64 = 0u64
    while pos < limit:
        let ch = token.slice(pos, pos + 1)
        if ch != "0" and ch != "1" and ch != "2" and ch != "3" and ch != "4" and ch != "5" and ch != "6" and ch != "7" and ch != "8" and ch != "9":
            return Err(.ParseError("invalid unsigned integer literal"))
        let digit = if ch == "0" then 0u64 else if ch == "1" then 1u64 else if ch == "2" then 2u64 else if ch == "3" then 3u64 else if ch == "4" then 4u64 else if ch == "5" then 5u64 else if ch == "6" then 6u64 else if ch == "7" then 7u64 else if ch == "8" then 8u64 else 9u64
        out = out * 10u64 + digit
        pos = pos + 1
    Ok(out)

fn parse_i64_ascii(token: str) -> Result[i64, IRTextError]:
    if token == "":
        return Err(.ParseError("expected integer literal"))
    if token.slice(0, 1) == "-":
        let magnitude = parse_u64_ascii(token.slice(1, token.len() as i32))?
        if magnitude == 9223372036854775808u64:
            return Ok(-9223372036854775807 - 1)
        return Ok(-(magnitude as i64))
    let magnitude = parse_u64_ascii(token)?
    Ok(magnitude as i64)

fn parse_hex_u64(token: str) -> Result[u64, IRTextError]:
    if token == "":
        return Err(.ParseError("expected hex literal digits"))
    var pos: i32 = 0
    let limit = token.len() as i32
    var out: u64 = 0u64
    while pos < limit:
        let ch = token.slice(pos, pos + 1)
        let digit = if ch == "0" then 0u64 else if ch == "1" then 1u64 else if ch == "2" then 2u64 else if ch == "3" then 3u64 else if ch == "4" then 4u64 else if ch == "5" then 5u64 else if ch == "6" then 6u64 else if ch == "7" then 7u64 else if ch == "8" then 8u64 else if ch == "9" then 9u64 else if ch == "a" or ch == "A" then 10u64 else if ch == "b" or ch == "B" then 11u64 else if ch == "c" or ch == "C" then 12u64 else if ch == "d" or ch == "D" then 13u64 else if ch == "e" or ch == "E" then 14u64 else if ch == "f" or ch == "F" then 15u64 else return Err(.ParseError("invalid hex literal"))
        out = out * 16u64 + digit
        pos = pos + 1
    Ok(out)

fn parse_f64_ascii(token: str) -> Result[f64, IRTextError]:
    if token == "":
        return Err(.ParseError("expected float literal"))
    var pos: i32 = 0
    let limit = token.len() as i32
    var sign: f64 = 1.0
    if token.slice(0, 1) == "-":
        sign = -1.0
        pos = 1
    if pos >= limit:
        return Err(.ParseError("invalid float literal"))
    var whole: f64 = 0.0
    var saw_digit = false
    while pos < limit:
        let ch = token.slice(pos, pos + 1)
        if ch == ".":
            break
        if ch != "0" and ch != "1" and ch != "2" and ch != "3" and ch != "4" and ch != "5" and ch != "6" and ch != "7" and ch != "8" and ch != "9":
            return Err(.ParseError("invalid float literal"))
        let digit = if ch == "0" then 0.0 else if ch == "1" then 1.0 else if ch == "2" then 2.0 else if ch == "3" then 3.0 else if ch == "4" then 4.0 else if ch == "5" then 5.0 else if ch == "6" then 6.0 else if ch == "7" then 7.0 else if ch == "8" then 8.0 else 9.0
        whole = whole * 10.0 + digit
        saw_digit = true
        pos = pos + 1
    var frac: f64 = 0.0
    var scale: f64 = 1.0
    if pos < limit and token.slice(pos, pos + 1) == ".":
        pos = pos + 1
        while pos < limit:
            let ch = token.slice(pos, pos + 1)
            if ch != "0" and ch != "1" and ch != "2" and ch != "3" and ch != "4" and ch != "5" and ch != "6" and ch != "7" and ch != "8" and ch != "9":
                return Err(.ParseError("invalid float literal"))
            let digit = if ch == "0" then 0.0 else if ch == "1" then 1.0 else if ch == "2" then 2.0 else if ch == "3" then 3.0 else if ch == "4" then 4.0 else if ch == "5" then 5.0 else if ch == "6" then 6.0 else if ch == "7" then 7.0 else if ch == "8" then 8.0 else 9.0
            scale = scale * 10.0
            frac = frac + digit / scale
            saw_digit = true
            pos = pos + 1
    if not saw_digit or pos != limit:
        return Err(.ParseError("invalid float literal"))
    Ok(sign * (whole + frac))

fn dtype_bit_width(dtype: DType) -> i32:
    if dtype == .Int8 or dtype == .UInt8:
        return 8
    if dtype == .Int16 or dtype == .UInt16 or dtype == .Float16 or dtype == .BFloat16:
        return 16
    if dtype == .Int32 or dtype == .UInt32 or dtype == .Float32:
        return 32
    if dtype == .Int64 or dtype == .UInt64 or dtype == .Float64:
        return 64
    0

fn mask_bits(width: i32) -> u64:
    if width >= 64:
        return 18446744073709551615u64
    (1u64 << width) - 1u64

fn f32_to_f16_bits(value: f32) -> u16:
    let raw: u32 = unsafe: transmute[u32](value)
    let sign = (raw >> 16) & 32768u32
    let exp = ((raw >> 23) & 255u32) as i32
    let frac = raw & 8388607u32
    if exp == 255:
        if frac == 0u32:
            return (sign | 31744u32) as u16
        let payload = (frac >> 13) | 1u32
        return (sign | 31744u32 | payload) as u16
    let half_exp = exp - 127 + 15
    if half_exp >= 31:
        return (sign | 31744u32) as u16
    if half_exp <= 0:
        if half_exp < -10:
            return sign as u16
        let mant = frac | 8388608u32
        let shift = 14 - half_exp
        var rounded = mant >> shift
        let round_bit = (mant >> (shift - 1)) & 1u32
        if round_bit != 0u32:
            rounded = rounded + 1u32
        return (sign | rounded) as u16
    var half_frac = frac >> 13
    let round_bias = frac & 4095u32
    if round_bias > 2048u32 or (round_bias == 2048u32 and (half_frac & 1u32) != 0u32):
        half_frac = half_frac + 1u32
        if half_frac == 1024u32:
            return (sign | ((half_exp + 1) as u32 << 10)) as u16
    (sign | ((half_exp as u32) << 10) | half_frac) as u16

fn f32_to_bf16_bits(value: f32) -> u16:
    let raw: u32 = unsafe: transmute[u32](value)
    let lsb = (raw >> 16) & 1u32
    let rounded = raw + 32767u32 + lsb
    (rounded >> 16) as u16

fn parse_scalar_literal(dtype: DType, token: str) -> Result[Scalar, IRTextError]:
    if starts_with(token, "bits:0x"):
        let raw = parse_hex_u64(token.slice(7, token.len() as i32))?
        let width = dtype_bit_width(dtype)
        if width < 64 and raw > mask_bits(width):
            return Err(.ParseError("bit literal does not fit dtype width"))
        return Ok(Scalar { bits: raw, dtype })
    if dtype == .Int8:
        return Ok(scalar_i8(parse_i64_ascii(token)? as i8))
    if dtype == .Int16:
        return Ok(scalar_i16(parse_i64_ascii(token)? as i16))
    if dtype == .Int32:
        return Ok(scalar_i32(parse_i32_ascii(token)?))
    if dtype == .Int64:
        return Ok(scalar_i64(parse_i64_ascii(token)?))
    if dtype == .UInt8:
        return Ok(scalar_u8(parse_u64_ascii(token)? as u8))
    if dtype == .UInt16:
        return Ok(scalar_u16(parse_u64_ascii(token)? as u16))
    if dtype == .UInt32:
        return Ok(scalar_u32(parse_u64_ascii(token)? as u32))
    if dtype == .UInt64:
        return Ok(scalar_u64(parse_u64_ascii(token)?))
    if dtype == .Float16:
        return Ok(scalar_f16_bits(f32_to_f16_bits(parse_f64_ascii(token)? as f32)))
    if dtype == .Float32:
        return Ok(scalar_f32(parse_f64_ascii(token)? as f32))
    if dtype == .Float64:
        return Ok(scalar_f64(parse_f64_ascii(token)?))
    if dtype == .BFloat16:
        return Ok(scalar_bf16_bits(f32_to_bf16_bits(parse_f64_ascii(token)? as f32)))
    Err(.ParseError("unsupported scalar literal dtype"))

fn parse_value_ref(token: str) -> Result[i32, IRTextError]:
    if not starts_with(token, "%"):
        return Err(.ParseError("expected value ref"))
    parse_i32_ascii(token.slice(1, token.len() as i32))

fn parse_index_ref(token: str) -> Result[i32, IRTextError]:
    if starts_with(token, "@"):
        let loop_index = parse_i32_ascii(token.slice(1, token.len() as i32))?
        return Ok(ir_loop_ref(loop_index))
    parse_value_ref(token)

fn append_index_refs(aux: Vec[i32], inner: str) -> Result[Vec[i32], IRTextError]:
    let trimmed = trim_ascii(inner)
    if trimmed == "":
        return Ok(aux)
    var out = aux
    var start: i32 = 0
    let limit = trimmed.len() as i32
    while start < limit:
        var stop = start
        while stop < limit and trimmed.slice(stop, stop + 1) != ",":
            stop = stop + 1
        let token = trim_ascii(trimmed.slice(start, stop))
        if token == "":
            return Err(.ParseError("empty index token"))
        out.push(parse_index_ref(token)?)
        if stop >= limit:
            break
        start = stop + 1
    Ok(out)

fn append_reduce_meta(aux: Vec[i32], inner: str) -> Result[Vec[i32], IRTextError]:
    let trimmed = trim_ascii(inner)
    var parts: Vec[str] = Vec.new()
    var start: i32 = 0
    let limit = trimmed.len() as i32
    while start < limit:
        var stop = start
        while stop < limit and trimmed.slice(stop, stop + 1) != ",":
            stop = stop + 1
        let token = trim_ascii(trimmed.slice(start, stop))
        if token == "":
            return Err(.ParseError("empty reduce metadata token"))
        parts.push(token)
        if stop >= limit:
            break
        start = stop + 1
    if parts.len() != 3:
        return Err(.ParseError("reduce metadata must be [slot, block, %value]"))
    var out = aux
    out.push(parse_i32_ascii(parts[0])?)
    out.push(parse_i32_ascii(parts[1])?)
    out.push(parse_value_ref(parts[2])?)
    Ok(out)

fn param_symbols -> ParamSymbols:
    ParamSymbols {
        count: 0,
        name0: "",
        name1: "",
        name2: "",
        name3: "",
        name4: "",
        name5: "",
        name6: "",
        name7: "",
    }

fn param_symbols_add(params: ParamSymbols, name: str) -> Result[ParamSymbols, IRTextError]:
    if params.count >= 8:
        return Err(.ParseError("text parser currently supports at most 8 params"))
    if param_symbols_find(params, name) >= 0:
        return Err(.ParseError("duplicate param name"))
    var next = params
    if params.count == 0: next.name0 = name
    if params.count == 1: next.name1 = name
    if params.count == 2: next.name2 = name
    if params.count == 3: next.name3 = name
    if params.count == 4: next.name4 = name
    if params.count == 5: next.name5 = name
    if params.count == 6: next.name6 = name
    if params.count == 7: next.name7 = name
    next.count = params.count + 1
    Ok(next)

fn param_symbols_find(params: ParamSymbols, name: str) -> i32:
    if params.count > 0 and params.name0 == name: return 0
    if params.count > 1 and params.name1 == name: return 1
    if params.count > 2 and params.name2 == name: return 2
    if params.count > 3 and params.name3 == name: return 3
    if params.count > 4 and params.name4 == name: return 4
    if params.count > 5 and params.name5 == name: return 5
    if params.count > 6 and params.name6 == name: return 6
    if params.count > 7 and params.name7 == name: return 7
    -1

fn intern_string(source: ProgramSource, value: str) -> StringIntern:
    for i in 0..source.strings.len():
        if source.strings[i] == value:
            return StringIntern { source, index: i as i32 }
    var next = source
    next.strings.push(value)
    StringIntern {
        source: next,
        index: next.strings.len() as i32 - 1,
    }

fn param_dtype_by_index(source: ProgramSource, index: i32) -> Result[DType, IRTextError]:
    var seen: i32 = 0
    for ip in 0..source.ir.len():
        let inst = source.ir[ip]
        if inst.op == IROP_PARAM:
            if seen == index:
                return Ok(inst.dtype)
            seen = seen + 1
    Err(.ParseError("param dtype is unavailable"))

fn local_dtype_by_index(source: ProgramSource, index: i32) -> Result[DType, IRTextError]:
    var seen: i32 = 0
    for ip in 0..source.ir.len():
        let inst = source.ir[ip]
        if inst.op == IROP_LOCAL:
            if seen == index:
                return Ok(inst.dtype)
            seen = seen + 1
    Err(.ParseError("local dtype is unavailable"))

fn private_dtype_by_index(source: ProgramSource, index: i32) -> Result[DType, IRTextError]:
    var seen: i32 = 0
    for ip in 0..source.ir.len():
        let inst = source.ir[ip]
        if inst.op == IROP_PRIVATE:
            if seen == index:
                return Ok(inst.dtype)
            seen = seen + 1
    Err(.ParseError("private dtype is unavailable"))

fn resolve_named_ref(state: ParseState, name: str) -> Result[i32, IRTextError]:
    let param_index = param_symbols_find(state.params, name)
    if param_index >= 0:
        return Ok(ir_param_ref(param_index))
    let local_index = param_symbols_find(state.locals, name)
    if local_index >= 0:
        return Ok(ir_local_ref(local_index))
    let private_index = param_symbols_find(state.privates, name)
    if private_index >= 0:
        return Ok(ir_private_ref(private_index))
    Err(.ParseError("storage reference is unknown"))

fn storage_dtype_by_ref(source: ProgramSource, ref: i32) -> Result[DType, IRTextError]:
    if ir_is_param_ref(ref):
        return param_dtype_by_index(source, ir_param_index(ref))
    if ir_is_local_ref(ref):
        return local_dtype_by_index(source, ir_local_index(ref))
    if ir_is_private_ref(ref):
        return private_dtype_by_index(source, ir_private_index(ref))
    Err(.ParseError("storage dtype is unavailable"))

fn value_dtype_by_ref(source: ProgramSource, ref: i32) -> Result[DType, IRTextError]:
    for ip in 0..source.ir.len():
        let inst = source.ir[ip]
        if ir_is_value_producer(inst.op) and inst.d0 == ref:
            return Ok(inst.dtype)
    Err(.ParseError("value dtype is unavailable"))

fn starts_with(text: str, prefix: str) -> bool:
    let prefix_len = prefix.len() as i32
    if text.len() as i32 < prefix_len:
        return false
    text.slice(0, prefix_len) == prefix

fn is_ws(ch: str) -> bool:
    ch == " " or ch == "\t" or ch == "\n" or ch == "\r"
