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

pub fn parse_ir_text(text: str) -> Result[IRProgram, IRTextError]:
    var prog = ir_program()
    var params = param_symbols()
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
        if starts_with(line, "param"):
            let parsed = parse_param_line(line, prog, params)?
            prog = parsed.prog
            params = parsed.params
            continue
        if starts_with(line, "store"):
            prog = parse_store_line(line, prog, params)?
            continue
        if starts_with(line, "parallel") or starts_with(line, "loop") or starts_with(line, "block_begin") or starts_with(line, "block_end"):
            prog = parse_control_line(line, prog)?
            continue
        if starts_with(line, "return"):
            prog.insts.push(ir_inst(IROP_RETURN, .Int32, 0, 0, 0, 0))
            continue
        if starts_with(line, "%"):
            prog = parse_value_line(line, prog, params)?
            continue
        return Err(.ParseError("unknown IR text line"))

    Ok(prog)

type ParamParse {
    prog: IRProgram,
    params: ParamSymbols,
}

fn parse_param_line(line: str, prog: IRProgram, params: ParamSymbols) -> Result[ParamParse, IRTextError]:
    var cursor = skip_ws(line, 0)
    cursor = expect_token(line, cursor, "param")?
    let name_scan = scan_token(line, cursor)
    if name_scan.token == "":
        return Err(.ParseError("param line missing name"))
    let name = name_scan.token
    cursor = name_scan.next
    let mode_scan = scan_token(line, cursor)
    let mode = parse_param_mode(mode_scan.token)?
    cursor = mode_scan.next
    let shape_scan = scan_bracket_group(line, cursor)?
    let rank = count_rank(shape_scan.inner)
    cursor = shape_scan.next
    let dtype_scan = scan_token(line, cursor)
    let dtype = parse_dtype(dtype_scan.token)?
    var next_prog = prog
    next_prog.param_names.push(name)
    next_prog.param_modes.push(mode)
    next_prog.param_ranks.push(rank)
    next_prog.param_dtypes.push(dtype)
    next_prog.num_params = next_prog.num_params + 1
    let next_params = param_symbols_add(params, name)?
    Ok(ParamParse {
        prog: next_prog,
        params: next_params,
    })

fn parse_store_line(line: str, prog: IRProgram, params: ParamSymbols) -> Result[IRProgram, IRTextError]:
    var cursor = skip_ws(line, 0)
    cursor = expect_token(line, cursor, "store")?
    let name_scan = scan_token(line, cursor)
    let param_index = param_symbols_find(params, name_scan.token)
    if param_index < 0:
        return Err(.ParseError("store references unknown param"))
    cursor = name_scan.next
    let group_scan = scan_bracket_group(line, cursor)?
    var next_prog = prog
    let aux_base = next_prog.aux.len() as i32
    next_prog.aux = append_index_refs(next_prog.aux, group_scan.inner)?
    cursor = group_scan.next
    let value_scan = scan_token(line, cursor)
    let value_ref = parse_value_ref(value_scan.token)?
    next_prog.insts.push(ir_inst(IROP_STORE, .Int32, ir_param_ref(param_index), aux_base, value_ref, 0))
    Ok(next_prog)

fn parse_control_line(line: str, prog: IRProgram) -> Result[IRProgram, IRTextError]:
    var cursor = skip_ws(line, 0)
    let op_scan = scan_token(line, cursor)
    let op_name = op_scan.token
    cursor = op_scan.next

    if op_name == "block_begin" or op_name == "block_end":
        let block_scan = scan_token(line, cursor)
        let block_id = parse_i32_ascii(block_scan.token)?
        let op = if op_name == "block_begin" then IROP_BLOCK_BEGIN else IROP_BLOCK_END
        var next_prog = prog
        next_prog.insts.push(ir_inst(op, .Int32, block_id, 0, 0, 0))
        return Ok(next_prog)

    if op_name == "parallel" or op_name == "loop":
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
        let op = if op_name == "parallel" then IROP_PARALLEL else IROP_LOOP
        var next_prog = prog
        next_prog.insts.push(ir_inst(op, .Int32, slot, start_ref, end_ref, block_id))
        return Ok(next_prog)

    Err(.ParseError("unsupported control opcode"))

fn parse_value_line(line: str, prog: IRProgram, params: ParamSymbols) -> Result[IRProgram, IRTextError]:
    var cursor = skip_ws(line, 0)
    let id_scan = scan_token(line, cursor)
    let value_id = parse_value_ref(id_scan.token)?
    if value_id != prog.insts.len() as i32:
        return Err(.ParseError("value ids must be sequential"))
    cursor = id_scan.next
    cursor = expect_token(line, cursor, "=")?
    let op_scan = scan_token(line, cursor)
    let op_name = op_scan.token
    cursor = op_scan.next

    if op_name == "const":
        let dtype_scan = scan_token(line, cursor)
        if dtype_scan.token != "i32":
            return Err(.ParseError("text parser currently supports only const i32"))
        cursor = dtype_scan.next
        let literal_scan = scan_token(line, cursor)
        let literal = parse_i32_ascii(literal_scan.token)?
        var next_prog = prog
        next_prog.insts.push(ir_inst(IROP_CONST, .Int32, literal, 0, 0, 0))
        return Ok(next_prog)

    if op_name == "load":
        let name_scan = scan_token(line, cursor)
        let param_index = param_symbols_find(params, name_scan.token)
        if param_index < 0:
            return Err(.ParseError("load references unknown param"))
        cursor = name_scan.next
        let group_scan = scan_bracket_group(line, cursor)?
        var next_prog = prog
        let aux_base = next_prog.aux.len() as i32
        next_prog.aux = append_index_refs(next_prog.aux, group_scan.inner)?
        let dtype = param_dtype_by_index(next_prog, param_index)
        next_prog.insts.push(ir_inst(IROP_LOAD, dtype, ir_param_ref(param_index), aux_base, 0, 0))
        return Ok(next_prog)

    if op_name == "add" or op_name == "sub" or op_name == "mul" or op_name == "div" or op_name == "min" or op_name == "max":
        let lhs_scan = scan_token(line, cursor)
        let lhs = parse_value_ref(lhs_scan.token)?
        cursor = lhs_scan.next
        let rhs_scan = scan_token(line, cursor)
        let rhs = parse_value_ref(rhs_scan.token)?
        let op = parse_binop(op_name)?
        let dtype = value_dtype_by_ref(prog, lhs)?
        var next_prog = prog
        next_prog.insts.push(ir_inst(op, dtype, lhs, rhs, 0, 0))
        return Ok(next_prog)

    if op_name == "eq" or op_name == "ne" or op_name == "lt" or op_name == "gt" or op_name == "le" or op_name == "ge":
        let lhs_scan = scan_token(line, cursor)
        let lhs = parse_value_ref(lhs_scan.token)?
        cursor = lhs_scan.next
        let rhs_scan = scan_token(line, cursor)
        let rhs = parse_value_ref(rhs_scan.token)?
        let op = parse_cmpop(op_name)?
        var next_prog = prog
        next_prog.insts.push(ir_inst(op, .Int32, lhs, rhs, 0, 0))
        return Ok(next_prog)

    if op_name == "fma":
        let a_scan = scan_token(line, cursor)
        let a = parse_value_ref(a_scan.token)?
        cursor = a_scan.next
        let b_scan = scan_token(line, cursor)
        let b = parse_value_ref(b_scan.token)?
        cursor = b_scan.next
        let c_scan = scan_token(line, cursor)
        let c = parse_value_ref(c_scan.token)?
        let dtype = value_dtype_by_ref(prog, a)?
        var next_prog = prog
        next_prog.insts.push(ir_inst(IROP_FMA, dtype, a, b, c, 0))
        return Ok(next_prog)

    if op_name == "neg":
        let value_scan = scan_token(line, cursor)
        let value = parse_value_ref(value_scan.token)?
        let dtype = value_dtype_by_ref(prog, value)?
        var next_prog = prog
        next_prog.insts.push(ir_inst(IROP_NEG, dtype, value, 0, 0, 0))
        return Ok(next_prog)

    if op_name == "select":
        let cond_scan = scan_token(line, cursor)
        let cond = parse_value_ref(cond_scan.token)?
        cursor = cond_scan.next
        let true_scan = scan_token(line, cursor)
        let on_true = parse_value_ref(true_scan.token)?
        cursor = true_scan.next
        let false_scan = scan_token(line, cursor)
        let on_false = parse_value_ref(false_scan.token)?
        let dtype = value_dtype_by_ref(prog, on_true)?
        var next_prog = prog
        next_prog.insts.push(ir_inst(IROP_SELECT, dtype, cond, on_true, on_false, 0))
        return Ok(next_prog)

    if op_name == "clamp":
        let value_scan = scan_token(line, cursor)
        let value = parse_value_ref(value_scan.token)?
        cursor = value_scan.next
        let lo_scan = scan_token(line, cursor)
        let lo = parse_value_ref(lo_scan.token)?
        cursor = lo_scan.next
        let hi_scan = scan_token(line, cursor)
        let hi = parse_value_ref(hi_scan.token)?
        let dtype = value_dtype_by_ref(prog, value)?
        var next_prog = prog
        next_prog.insts.push(ir_inst(IROP_CLAMP, dtype, value, lo, hi, 0))
        return Ok(next_prog)

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

fn append_index_refs(base: Vec[i32], text: str) -> Result[Vec[i32], IRTextError]:
    let out = base
    var pos: i32 = 0
    let limit = text.len() as i32
    while pos < limit:
        pos = skip_group_sep(text, pos)
        if pos >= limit:
            break
        let scan = scan_group_token(text, pos)
        let value_ref = parse_index_ref(scan.token)?
        out.push(value_ref)
        pos = scan.next
    Ok(out)

fn skip_group_sep(text: str, start: i32) -> i32:
    var pos = start
    let limit = text.len() as i32
    while pos < limit:
        let ch = text.slice(pos, pos + 1)
        if ch != " " and ch != "\t" and ch != ",":
            return pos
        pos = pos + 1
    pos

fn scan_group_token(text: str, start: i32) -> TokenScan:
    let limit = text.len() as i32
    var end = start
    while end < limit:
        let ch = text.slice(end, end + 1)
        if ch == " " or ch == "\t" or ch == ",":
            break
        end = end + 1
    TokenScan {
        token: text.slice(start, end),
        next: end,
    }

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

fn parse_binop(token: str) -> Result[i32, IRTextError]:
    if token == "add": return Ok(IROP_ADD)
    if token == "sub": return Ok(IROP_SUB)
    if token == "mul": return Ok(IROP_MUL)
    if token == "div": return Ok(IROP_DIV)
    if token == "min": return Ok(IROP_MIN)
    if token == "max": return Ok(IROP_MAX)
    Err(.ParseError("unknown binary opcode"))

fn parse_cmpop(token: str) -> Result[i32, IRTextError]:
    if token == "eq": return Ok(IROP_EQ)
    if token == "ne": return Ok(IROP_NE)
    if token == "lt": return Ok(IROP_LT)
    if token == "gt": return Ok(IROP_GT)
    if token == "le": return Ok(IROP_LE)
    if token == "ge": return Ok(IROP_GE)
    Err(.ParseError("unknown compare opcode"))

fn parse_value_ref(token: str) -> Result[i32, IRTextError]:
    if not starts_with(token, "%"):
        return Err(.ParseError("expected value ref"))
    parse_i32_ascii(token.slice(1, token.len() as i32))

fn parse_index_ref(token: str) -> Result[i32, IRTextError]:
    if starts_with(token, "@"):
        let slot = parse_i32_ascii(token.slice(1, token.len() as i32))?
        return Ok(ir_loop_ref(slot))
    parse_value_ref(token)

fn parse_const_literal(token: str) -> Result[i32, IRTextError]:
    if token == "":
        return Err(.ParseError("missing const literal"))
    if contains_char(token, "."):
        return Ok(0)
    parse_i32_ascii(token)

fn parse_i32_ascii(token: str) -> Result[i32, IRTextError]:
    if token == "":
        return Err(.ParseError("expected integer"))
    var sign: i32 = 1
    var pos: i32 = 0
    if token.slice(0, 1) == "-":
        sign = -1
        pos = 1
    var value: i32 = 0
    let limit = token.len() as i32
    while pos < limit:
        let ch = token.slice(pos, pos + 1)
        let digit = digit_value(ch)
        if digit < 0:
            return Err(.ParseError("invalid integer"))
        value = value * 10 + digit
        pos = pos + 1
    Ok(value * sign)

fn digit_value(ch: str) -> i32:
    if ch == "0": return 0
    if ch == "1": return 1
    if ch == "2": return 2
    if ch == "3": return 3
    if ch == "4": return 4
    if ch == "5": return 5
    if ch == "6": return 6
    if ch == "7": return 7
    if ch == "8": return 8
    if ch == "9": return 9
    -1

fn contains_char(text: str, needle: str) -> bool:
    var pos: i32 = 0
    let limit = text.len() as i32
    while pos < limit:
        if text.slice(pos, pos + 1) == needle:
            return true
        pos = pos + 1
    false

fn starts_with(text: str, prefix: str) -> bool:
    if prefix.len() > text.len():
        return false
    text.slice(0, prefix.len() as i32) == prefix

fn is_ws(ch: str) -> bool:
    ch == " " or ch == "\t" or ch == "\r"

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
    if params.count == 0: return Ok({ params with count: 1, name0: name })
    if params.count == 1: return Ok({ params with count: 2, name1: name })
    if params.count == 2: return Ok({ params with count: 3, name2: name })
    if params.count == 3: return Ok({ params with count: 4, name3: name })
    if params.count == 4: return Ok({ params with count: 5, name4: name })
    if params.count == 5: return Ok({ params with count: 6, name5: name })
    if params.count == 6: return Ok({ params with count: 7, name6: name })
    if params.count == 7: return Ok({ params with count: 8, name7: name })
    Err(.ParseError("text parser supports at most 8 params"))

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

fn param_dtype_by_index(prog: IRProgram, index: i32) -> DType:
    if index == 0: return dtype_at(prog.param_dtypes, 0)
    if index == 1: return dtype_at(prog.param_dtypes, 1)
    if index == 2: return dtype_at(prog.param_dtypes, 2)
    if index == 3: return dtype_at(prog.param_dtypes, 3)
    if index == 4: return dtype_at(prog.param_dtypes, 4)
    if index == 5: return dtype_at(prog.param_dtypes, 5)
    if index == 6: return dtype_at(prog.param_dtypes, 6)
    if index == 7: return dtype_at(prog.param_dtypes, 7)
    .Int32

fn dtype_at(items: Vec[DType], index: i32) -> DType:
    if index < 0 or index >= items.len() as i32:
        return .Int32
    items[index as i64]

fn value_dtype_by_ref(prog: IRProgram, ref: i32) -> Result[DType, IRTextError]:
    if ref < 0 or ref >= prog.insts.len() as i32:
        return Err(.ParseError("value ref out of range"))
    Ok(prog.insts[ref].dtype)

fn push_inst(items: Vec[IRInst], inst: IRInst) -> Vec[IRInst]:
    let out = items
    out.push(inst)
    out

fn push_i32(items: Vec[i32], value: i32) -> Vec[i32]:
    let out = items
    out.push(value)
    out

fn push_str(items: Vec[str], value: str) -> Vec[str]:
    let out = items
    out.push(value)
    out

fn push_mode(items: Vec[ParamMode], value: ParamMode) -> Vec[ParamMode]:
    let out = items
    out.push(value)
    out

fn push_dtype(items: Vec[DType], value: DType) -> Vec[DType]:
    let out = items
    out.push(value)
    out
