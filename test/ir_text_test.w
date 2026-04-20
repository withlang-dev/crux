use crux.core
use crux.ir
use crux.ir_text

fn empty_source -> ProgramSource:
    ProgramSource {
        ir: Vec.new(),
        aux: Vec.new(),
        strings: Vec.new(),
        entry: "main",
    }

fn smoke_text -> str:
    "param a in [N] f32\nparam b in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = add %1 %2\nstore out [%0] %3\nreturn\n"

fn test_parse_ir_text_smoke:
    let source = match parse_ir_text(smoke_text())
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.entry == "main")
    assert(source.ir.len() == 9)
    assert(source.aux.len() == 3)
    assert(source.strings.len() == 3)
    assert(source.ir[0].op == IROP_PARAM)
    assert(source.ir[1].op == IROP_PARAM)
    assert(source.ir[2].op == IROP_PARAM)
    assert(source.ir[3].op == IROP_CONST)

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

fn test_parse_accepts_scalar_const_surface:
    let text = "%0 = const f32 0.0\n%1 = const u64 42\n%2 = const f16 bits:0x3c00\n%3 = const bf16 bits:0x3f80\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 4)
    assert(source.ir[0].dtype == .Float32)
    assert(source.ir[1].dtype == .UInt64)
    assert(source.ir[2].dtype == .Float16)
    assert(source.ir[3].dtype == .BFloat16)

fn test_parse_accepts_fma:
    let text = "param a in [N] i32\nparam b in [N] i32\nparam c in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = load c [%0]\n%4 = fma %1 %2 %3\nstore out [%0] %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 11)
    assert(source.ir[8].op == IROP_FMA)

fn test_parse_accepts_neg:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = neg %1\nstore out [%0] %2\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 7)
    assert(source.ir[4].op == IROP_NEG)

fn test_parse_accepts_abs:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = abs %1\nstore out [%0] %2\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 7)
    assert(source.ir[4].op == IROP_ABS)

fn test_parse_accepts_mod:
    let text = "param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = mod %1 %2\nstore out [%0] %3\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 9)
    assert(source.ir[6].op == IROP_MOD)

fn test_parse_accepts_sat_ops:
    let text = "param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = add_sat %1 %2\n%4 = sub_sat %3 %2\nstore out [%0] %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 10)
    assert(source.ir[6].op == IROP_ADD_SAT)
    assert(source.ir[7].op == IROP_SUB_SAT)

fn test_parse_accepts_i64_and_u32_dtypes:
    let i64_text = "param a in [] i64\nparam b in [] i64\nparam out out [] i64\n%0 = load a []\n%1 = load b []\n%2 = add %0 %1\nstore out [] %2\nreturn\n"
    let i64_source = match parse_ir_text(i64_text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(i64_source.ir.len() == 8)
    assert(i64_source.ir[0].dtype == .Int64)
    assert(i64_source.ir[5].dtype == .Int64)

    let u32_text = "param a in [] u32\nparam b in [] u32\nparam out out [] u32\n%0 = load a []\n%1 = load b []\n%2 = xor %0 %1\nstore out [] %2\nreturn\n"
    let u32_source = match parse_ir_text(u32_text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(u32_source.ir.len() == 8)
    assert(u32_source.ir[0].dtype == .UInt32)
    assert(u32_source.ir[5].dtype == .UInt32)

fn test_parse_accepts_spec_constant_header:
    let text = "spec_constant TILE i32 16\nparam a in [] i32\nparam out out [] i32\n%0 = load a []\nstore out [] %0\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 6)
    assert(source.ir[0].op == IROP_SPEC_CONSTANT)
    assert(source.ir[1].op == IROP_PARAM)
    assert(source.strings.len() == 3)
    assert(source.strings[0] == "TILE")
    assert(source.strings[1] == "a")
    assert(source.strings[2] == "out")

fn test_parse_accepts_storage_and_control_surface:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 1\nlocal tmp [%0] i32\nprivate scratch [%0] i32\n%1 = const i32 0\nparallel_grid 0 %1 %1 1\nblock_begin 1\nif %1 2\nblock_begin 2\nbarrier\nblock_end 2\nblock_end 1\n%2 = load a [%1]\nstore tmp [%1] %2\n%3 = load tmp [%1]\nstore scratch [%1] %3\n%4 = load scratch [%1]\n%5 = collective_allreduce_sum %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir[3].op == IROP_LOCAL)
    assert(source.ir[4].op == IROP_PRIVATE)
    assert(source.ir[6].op == IROP_PARALLEL_GRID)
    assert(source.ir[8].op == IROP_IF)
    assert(source.ir[10].op == IROP_BARRIER)
    assert(source.ir[source.ir.len() - 2].op == IROP_COLLECTIVE_ALLREDUCE_SUM)

fn test_parse_accepts_reduce_surface:
    let text = "param out out [] i32\n%0 = const i32 0\n%1 = const i32 4\nloop 0 %0 %1 1\nblock_begin 1\n%2 = const i32 3\nblock_end 1\n%3 = reduce_sum %0 %1 [0, 2, %2]\nblock_begin 2\n%4 = const i32 7\nblock_end 2\nstore out [] %3\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir[7].op == IROP_REDUCE_SUM)
    assert(source.aux.len() >= 3)

fn test_parse_accepts_nested_loop_indices:
    let text = "param a in [M,N] i32\nparam out out [M,N] i32\n%0 = const i32 0\n%1 = const i32 2\n%2 = const i32 3\nloop 0 %0 %1 1\nblock_begin 1\nloop 1 %0 %2 2\nblock_begin 2\n%3 = load a [@0, @1]\nstore out [@0, @1] %3\nblock_end 2\nblock_end 1\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 14)
    assert(source.ir[5].op == IROP_LOOP)
    assert(source.ir[7].op == IROP_LOOP)
    assert(source.ir[10].op == IROP_STORE)
    assert(source.aux.len() == 4)
    assert(source.aux[0] == ir_loop_ref(0))
    assert(source.aux[1] == ir_loop_ref(1))
    assert(source.aux[2] == ir_loop_ref(0))
    assert(source.aux[3] == ir_loop_ref(1))

fn test_parse_accepts_bitwise_ops:
    let text = "param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = and %1 %2\n%4 = or %1 %2\n%5 = xor %3 %4\n%6 = not %5\nstore out [%0] %6\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 12)
    assert(source.ir[6].op == IROP_AND)
    assert(source.ir[7].op == IROP_OR)
    assert(source.ir[8].op == IROP_XOR)
    assert(source.ir[9].op == IROP_NOT)

fn test_parse_accepts_shift_ops:
    let text = "param a in [N] i32\nparam b in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = load b [%0]\n%3 = shl %1 %2\n%4 = shr %3 %2\nstore out [%0] %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 10)
    assert(source.ir[6].op == IROP_SHL)
    assert(source.ir[7].op == IROP_SHR)

fn test_parse_accepts_bitcount_ops:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = popcount %1\n%3 = clz %1\n%4 = ctz %1\nstore out [%0] %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 9)
    assert(source.ir[4].op == IROP_POPCOUNT)
    assert(source.ir[5].op == IROP_CLZ)
    assert(source.ir[6].op == IROP_CTZ)

fn test_parse_accepts_exp_log_log2:
    let text = "param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = exp %1\n%3 = log %1\n%4 = log2 %1\nstore out [%0] %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 9)
    assert(source.ir[4].op == IROP_EXP)
    assert(source.ir[5].op == IROP_LOG)
    assert(source.ir[6].op == IROP_LOG2)

fn test_parse_accepts_sin_cos_tanh:
    let text = "param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = sin %1\n%3 = cos %1\n%4 = tanh %1\nstore out [%0] %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 9)
    assert(source.ir[4].op == IROP_SIN)
    assert(source.ir[5].op == IROP_COS)
    assert(source.ir[6].op == IROP_TANH)

fn test_parse_accepts_floor:
    let text = "param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = floor %1\nstore out [%0] %2\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 7)
    assert(source.ir[4].op == IROP_FLOOR)

fn test_parse_accepts_ceil:
    let text = "param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = ceil %1\nstore out [%0] %2\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 7)
    assert(source.ir[4].op == IROP_CEIL)

fn test_parse_accepts_round:
    let text = "param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = round %1\nstore out [%0] %2\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 7)
    assert(source.ir[4].op == IROP_ROUND)

fn test_parse_accepts_sqrt:
    let text = "param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = sqrt %1\nstore out [%0] %2\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 7)
    assert(source.ir[4].op == IROP_SQRT)

fn test_parse_accepts_rsqrt:
    let text = "param a in [N] f32\nparam out out [N] f32\n%0 = const i32 0\n%1 = load a [%0]\n%2 = rsqrt %1\nstore out [%0] %2\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 7)
    assert(source.ir[4].op == IROP_RSQRT)

fn test_parse_accepts_cast:
    let text = "param a in [] i32\nparam out out [] f32\n%0 = load a []\n%1 = cast f32 %0\nstore out [] %1\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 6)
    assert(source.ir[3].op == IROP_CAST)
    assert(source.ir[3].dtype == .Float32)

fn test_parse_accepts_compare_and_select:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 5\n%2 = load a [%0]\n%3 = lt %2 %1\n%4 = select %3 %1 %2\nstore out [%0] %4\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 9)
    assert(source.ir[5].op == IROP_LT)
    assert(source.ir[6].op == IROP_SELECT)

fn test_parse_accepts_min_max_clamp:
    let text = "param a in [N] i32\nparam out out [N] i32\n%0 = const i32 0\n%1 = const i32 6\n%2 = load a [%0]\n%3 = max %2 %0\n%4 = min %3 %1\n%5 = clamp %2 %0 %1\nstore out [%0] %5\nreturn\n"
    let source = match parse_ir_text(text)
        Ok(v) => v
        Err(_) =>
            assert(false)
            empty_source()
    assert(source.ir.len() == 10)
    assert(source.ir[5].op == IROP_MAX)
    assert(source.ir[6].op == IROP_MIN)
    assert(source.ir[7].op == IROP_CLAMP)

fn main:
    test_parse_ir_text_smoke()
    test_parse_rejects_unknown_dtype()
    test_parse_rejects_unknown_param_reference()
    test_parse_rejects_non_sequential_value_ids()
    test_parse_accepts_scalar_const_surface()
    test_parse_accepts_fma()
    test_parse_accepts_neg()
    test_parse_accepts_abs()
    test_parse_accepts_mod()
    test_parse_accepts_sat_ops()
    test_parse_accepts_i64_and_u32_dtypes()
    test_parse_accepts_spec_constant_header()
    test_parse_accepts_storage_and_control_surface()
    test_parse_accepts_reduce_surface()
    test_parse_accepts_nested_loop_indices()
    test_parse_accepts_bitwise_ops()
    test_parse_accepts_shift_ops()
    test_parse_accepts_bitcount_ops()
    test_parse_accepts_exp_log_log2()
    test_parse_accepts_sin_cos_tanh()
    test_parse_accepts_floor()
    test_parse_accepts_ceil()
    test_parse_accepts_round()
    test_parse_accepts_sqrt()
    test_parse_accepts_rsqrt()
    test_parse_accepts_cast()
    test_parse_accepts_compare_and_select()
    test_parse_accepts_min_max_clamp()
