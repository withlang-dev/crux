use crux.ir_text

fn smoke_text -> str:
    "param a in [N] f32\nparam b in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = add %1 %2\nstore out [%0] %3\nreturn\n"

fn test_parse_ir_text_smoke:
    let prog = match parse_ir_text(smoke_text())
        Ok(v) => v
        Err(_) =>
            assert(false)
            ir_program()
    let insts = prog.insts
    let names = prog.param_names
    assert(prog.num_params == 3)
    assert(insts.len() == 6)
    assert(names.len() == 3)

fn test_parse_rejects_unknown_dtype:
    let got_error = match parse_ir_text("param a in [N] f17\n")
        Err(.ParseError(_)) => true
        _ => false
    assert(got_error)

fn test_parse_rejects_unknown_param_reference:
    let text = "param a in [N] f32\n%0 = const i32 0\n%1 = load missing [%0]\n"
    let got_error = match parse_ir_text(text)
        Err(.ParseError(_)) => true
        _ => false
    assert(got_error)

fn test_parse_rejects_non_sequential_value_ids:
    let text = "param a in [N] f32\n%1 = const i32 0\n"
    let got_error = match parse_ir_text(text)
        Err(.ParseError(_)) => true
        _ => false
    assert(got_error)

fn test_parse_rejects_non_i32_const:
    let text = "%0 = const f32 0.0\n"
    let got_error = match parse_ir_text(text)
        Err(.ParseError(_)) => true
        _ => false
    assert(got_error)

fn test_parse_accepts_fma:
    let text = "param a in [N] i32\nparam b in [N] i32\nparam c in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = load c [%0]\n%4 = fma %1 %2 %3\nstore out [%0] %4\nreturn\n"
    let prog = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            ir_program()
    assert(prog.insts.len() == 7)
    assert(prog.insts[4].op == IROP_FMA)

fn test_parse_accepts_neg:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = neg %1\nstore out [%0] %2\nreturn\n"
    let prog = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            ir_program()
    assert(prog.insts.len() == 5)
    assert(prog.insts[2].op == IROP_NEG)

fn test_parse_accepts_compare_and_select:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 5\n%2 = load a [%0]\n%3 = lt %2 %1\n%4 = select %3 %1 %2\nstore out [%0] %4\nreturn\n"
    let prog = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            ir_program()
    assert(prog.insts.len() == 7)
    assert(prog.insts[3].op == IROP_LT)
    assert(prog.insts[4].op == IROP_SELECT)

fn test_parse_accepts_min_max_clamp:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 6\n%2 = load a [%0]\n%3 = max %2 %0\n%4 = min %3 %1\n%5 = clamp %2 %0 %1\nstore out [%0] %5\nreturn\n"
    let prog = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            ir_program()
    assert(prog.insts.len() == 8)
    assert(prog.insts[3].op == IROP_MAX)
    assert(prog.insts[4].op == IROP_MIN)
    assert(prog.insts[5].op == IROP_CLAMP)

fn main:
    test_parse_ir_text_smoke()
    test_parse_rejects_unknown_dtype()
    test_parse_rejects_unknown_param_reference()
    test_parse_rejects_non_sequential_value_ids()
    test_parse_rejects_non_i32_const()
    test_parse_accepts_fma()
    test_parse_accepts_neg()
    test_parse_accepts_compare_and_select()
    test_parse_accepts_min_max_clamp()
