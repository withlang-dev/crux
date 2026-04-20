use c_import("stdlib.h")

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

type ScratchSlot {
    ptr: *mut u8,
    shape: Shape,
    dtype: DType,
    live: bool,
    generation: i32,
}

type InterpState {
    values: Vec[InterpValue],
    loops: LoopVars,
    locals: Vec[ScratchSlot],
    privates: Vec[ScratchSlot],
    local_generation: i32,
    private_generation: i32,
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

fn scratch_slot -> ScratchSlot:
    ScratchSlot {
        ptr: null,
        shape: shape_scalar(),
        dtype: .Int32,
        live: false,
        generation: -1,
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

fn init_state(value_count: i32, local_count: i32, private_count: i32) -> InterpState:
    let values: Vec[InterpValue] = Vec.new()
    for i in 0..value_count:
        let _ = i
        values.push(interp_value_zero())
    let locals: Vec[ScratchSlot] = Vec.new()
    for i in 0..local_count:
        let _ = i
        locals.push(scratch_slot())
    let privates: Vec[ScratchSlot] = Vec.new()
    for i in 0..private_count:
        let _ = i
        privates.push(scratch_slot())
    InterpState {
        values,
        loops: loop_vars(),
        locals,
        privates,
        local_generation: 1,
        private_generation: 0,
    }

fn mask_bits(width: i32) -> u64:
    if width >= 64:
        return 18446744073709551615u64
    (1u64 << width) - 1u64

fn dtype_is_signed_int(dtype: DType) -> bool:
    dtype == .Int8 or dtype == .Int16 or dtype == .Int32 or dtype == .Int64

fn dtype_is_unsigned_int(dtype: DType) -> bool:
    dtype == .UInt8 or dtype == .UInt16 or dtype == .UInt32 or dtype == .UInt64

fn dtype_is_int(dtype: DType) -> bool:
    dtype_is_signed_int(dtype) or dtype_is_unsigned_int(dtype)

fn dtype_is_float(dtype: DType) -> bool:
    dtype == .Float16 or dtype == .Float32 or dtype == .Float64 or dtype == .BFloat16

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

fn truncate_raw(raw: u64, width: i32) -> u64:
    raw & mask_bits(width)

fn sign_extend_raw(raw: u64, width: i32) -> i64:
    if width >= 64:
        return unsafe: transmute[i64](raw)
    let shift = 64 - width
    let shifted: i64 = unsafe: transmute[i64](raw << shift)
    shifted >> shift

fn value_from_signed(dtype: DType, value: i64) -> InterpValue:
    let raw: u64 = unsafe: transmute[u64](value)
    InterpValue {
        raw: truncate_raw(raw, dtype_bit_width(dtype)),
        dtype,
    }

fn value_from_unsigned(dtype: DType, value: u64) -> InterpValue:
    InterpValue {
        raw: truncate_raw(value, dtype_bit_width(dtype)),
        dtype,
    }

fn f16_bits_to_f32(bits: u16) -> f32:
    let sign = (bits as u32 & 32768u32) << 16
    let exp = (bits as u32 >> 10) & 31u32
    let frac = bits as u32 & 1023u32
    if exp == 0u32:
        if frac == 0u32:
            return unsafe: transmute[f32](sign)
        var mant = frac
        var exponent: i32 = -14
        while (mant & 1024u32) == 0u32:
            mant = mant << 1
            exponent = exponent - 1
        mant = mant & 1023u32
        let exp32 = (exponent + 127) as u32
        let raw = sign | (exp32 << 23) | (mant << 13)
        return unsafe: transmute[f32](raw)
    if exp == 31u32:
        let raw = sign | 2139095040u32 | (frac << 13)
        return unsafe: transmute[f32](raw)
    let exp32 = exp + 112u32
    let raw = sign | (exp32 << 23) | (frac << 13)
    unsafe: transmute[f32](raw)

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

fn bf16_bits_to_f32(bits: u16) -> f32:
    let raw = (bits as u32) << 16
    unsafe: transmute[f32](raw)

fn f32_to_bf16_bits(value: f32) -> u16:
    let raw: u32 = unsafe: transmute[u32](value)
    let lsb = (raw >> 16) & 1u32
    let rounded = raw + 32767u32 + lsb
    (rounded >> 16) as u16

fn value_from_float_dtype(dtype: DType, value: f64) -> InterpValue:
    if dtype == .Float16:
        return InterpValue { raw: f32_to_f16_bits(value as f32) as u64, dtype }
    if dtype == .Float32:
        return value_from_f32(value as f32)
    if dtype == .Float64:
        return value_from_f64(value)
    if dtype == .BFloat16:
        return InterpValue { raw: f32_to_bf16_bits(value as f32) as u64, dtype }
    interp_value_zero()

fn value_from_i32(value: i32) -> InterpValue:
    value_from_signed(.Int32, value as i64)

fn value_from_i8(value: i8) -> InterpValue:
    value_from_signed(.Int8, value as i64)

fn value_from_i16(value: i16) -> InterpValue:
    value_from_signed(.Int16, value as i64)

fn value_from_i64(value: i64) -> InterpValue:
    value_from_signed(.Int64, value)

fn value_from_u8(value: u8) -> InterpValue:
    value_from_unsigned(.UInt8, value as u64)

fn value_from_u16(value: u16) -> InterpValue:
    value_from_unsigned(.UInt16, value as u64)

fn value_from_u32(value: u32) -> InterpValue:
    value_from_unsigned(.UInt32, value as u64)

fn value_from_u64(value: u64) -> InterpValue:
    value_from_unsigned(.UInt64, value)

fn value_from_f32(value: f32) -> InterpValue:
    let raw: u32 = unsafe: transmute[u32](value)
    InterpValue { raw: raw as u64, dtype: .Float32 }

fn value_from_f64(value: f64) -> InterpValue:
    let raw: u64 = unsafe: transmute[u64](value)
    InterpValue { raw, dtype: .Float64 }

fn value_from_f16_bits(bits: u16) -> InterpValue:
    InterpValue { raw: bits as u64, dtype: .Float16 }

fn value_from_bf16_bits(bits: u16) -> InterpValue:
    InterpValue { raw: bits as u64, dtype: .BFloat16 }

fn value_as_signed(value: InterpValue) -> Result[i64, SubstrateError]:
    if not dtype_is_signed_int(value.dtype):
        return Err(.Unsupported("cpu interpreter expected signed integer value"))
    Ok(sign_extend_raw(value.raw, dtype_bit_width(value.dtype)))

fn value_as_unsigned(value: InterpValue) -> Result[u64, SubstrateError]:
    if not dtype_is_unsigned_int(value.dtype):
        return Err(.Unsupported("cpu interpreter expected unsigned integer value"))
    Ok(truncate_raw(value.raw, dtype_bit_width(value.dtype)))

fn value_as_f64(value: InterpValue) -> Result[f64, SubstrateError]:
    if value.dtype == .Float16:
        return Ok(f16_bits_to_f32(value.raw as u16) as f64)
    if value.dtype == .Float32:
        let decoded: f32 = unsafe: transmute[f32](value.raw as u32)
        return Ok(decoded as f64)
    if value.dtype == .Float64:
        let decoded: f64 = unsafe: transmute[f64](value.raw)
        return Ok(decoded)
    if value.dtype == .BFloat16:
        return Ok(bf16_bits_to_f32(value.raw as u16) as f64)
    Err(.Unsupported("cpu interpreter expected float value"))

fn value_as_i32(value: InterpValue) -> Result[i32, SubstrateError]:
    if dtype_is_signed_int(value.dtype):
        return Ok(value_as_signed(value)? as i32)
    if dtype_is_unsigned_int(value.dtype):
        return Ok(value_as_unsigned(value)? as i32)
    Err(.Unsupported("cpu interpreter expected i32-compatible value"))

fn count_decl_ops(prog: IRProgram, op_kind: i32) -> i32:
    var count: i32 = 0
    for ip in 0..prog.insts.len():
        if prog.insts[ip].op == op_kind:
            count = count + 1
    count

fn decl_index_before(prog: IRProgram, ip: i32, op_kind: i32) -> i32:
    var count: i32 = 0
    var cursor: i32 = 0
    while cursor <= ip:
        if prog.insts[cursor].op == op_kind:
            count = count + 1
        cursor = cursor + 1
    count - 1

fn scratch_view(slot: ScratchSlot) -> View:
    View {
        memory: null_memory(),
        offset: 0usize,
        shape: slot.shape,
        strides: contiguous_strides(slot.shape, slot.dtype),
        dtype: slot.dtype,
    }

fn scratch_zero(ptr: *mut u8, bytes: Size):
    var i: Size = 0usize
    while i < bytes:
        unsafe:
            *(ptr + i as i64) = 0u8
        i = i + 1usize

fn scratch_alloc(slot: ScratchSlot, shape: Shape, dtype: DType, generation: i32) -> Result[ScratchSlot, SubstrateError]:
    if slot.ptr != null:
        let _ = realloc(slot.ptr as *mut c_void, 0)
    var bytes = shape_elem_count(shape) * dtype_size(dtype)
    if bytes == 0usize:
        bytes = 1usize
    let raw_opt = malloc(bytes)
    if raw_opt == None:
        return Err(.OutOfMemory)
    let raw = raw_opt.unwrap() as *mut u8
    scratch_zero(raw, bytes)
    Ok(ScratchSlot {
        ptr: raw,
        shape,
        dtype,
        live: true,
        generation,
    })

fn scratch_slot_view(slot: ScratchSlot) -> Result[View, SubstrateError]:
    if not slot.live or slot.ptr == null:
        return Err(.CompileError("scratch storage is not initialized"))
    Ok(scratch_view(slot))

fn resolve_load_view(state: InterpState, params: Vec[View], ref: i32) -> Result[View, SubstrateError]:
    if ir_is_param_ref(ref):
        return Ok(params[ir_param_index(ref)])
    if ir_is_local_ref(ref):
        let index = ir_local_index(ref)
        if index < 0 or index >= state.locals.len() as i32:
            return Err(.CompileError("local storage ref out of range"))
        return scratch_slot_view(state.locals[index])
    if ir_is_private_ref(ref):
        let index = ir_private_index(ref)
        if index < 0 or index >= state.privates.len() as i32:
            return Err(.CompileError("private storage ref out of range"))
        return scratch_slot_view(state.privates[index])
    Err(.CompileError("load/store ref is invalid"))

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

fn load_ptr(base: *mut u8, dtype: DType, byte_offset: Size) -> Result[InterpValue, SubstrateError]:
    if base == null:
        return Err(.InvalidView("view memory is null"))
    if dtype == .Int8:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i8
        return Ok(value_from_i8(unsafe: *ptr))
    if dtype == .UInt8:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u8
        return Ok(value_from_u8(unsafe: *ptr))
    if dtype == .Int16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i16
        return Ok(value_from_i16(unsafe: *ptr))
    if dtype == .UInt16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u16
        return Ok(value_from_u16(unsafe: *ptr))
    if dtype == .Int32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i32
        return Ok(value_from_i32(unsafe: *ptr))
    if dtype == .Int64:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i64
        return Ok(value_from_i64(unsafe: *ptr))
    if dtype == .UInt32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u32
        return Ok(value_from_u32(unsafe: *ptr))
    if dtype == .UInt64:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u64
        return Ok(value_from_u64(unsafe: *ptr))
    if dtype == .Float16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u16
        return Ok(value_from_f16_bits(unsafe: *ptr))
    if dtype == .Float32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut f32
        return Ok(value_from_f32(unsafe: *ptr))
    if dtype == .Float64:
        let ptr = unsafe: (base + byte_offset as i64) as *mut f64
        return Ok(value_from_f64(unsafe: *ptr))
    if dtype == .BFloat16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u16
        return Ok(value_from_bf16_bits(unsafe: *ptr))
    Err(.Unsupported("cpu interpreter dtype not implemented"))

fn load_value(view: View, byte_offset: Size) -> Result[InterpValue, SubstrateError]:
    let base = memory_ptr(view.memory)
    load_ptr(base, view.dtype, byte_offset)

fn store_ptr(base: *mut u8, dtype: DType, byte_offset: Size, value: InterpValue) -> Result[i32, SubstrateError]:
    if base == null:
        return Err(.InvalidView("view memory is null"))
    if dtype == .Int8:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i8
        unsafe:
            *ptr = value_as_signed(value)? as i8
        return Ok(0)
    if dtype == .UInt8:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u8
        unsafe:
            *ptr = value_as_unsigned(value)? as u8
        return Ok(0)
    if dtype == .Int16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i16
        unsafe:
            *ptr = value_as_signed(value)? as i16
        return Ok(0)
    if dtype == .UInt16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u16
        unsafe:
            *ptr = value_as_unsigned(value)? as u16
        return Ok(0)
    if dtype == .Int32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i32
        unsafe:
            *ptr = value_as_signed(value)? as i32
        return Ok(0)
    if dtype == .Int64:
        let ptr = unsafe: (base + byte_offset as i64) as *mut i64
        unsafe:
            *ptr = value_as_signed(value)?
        return Ok(0)
    if dtype == .UInt32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u32
        unsafe:
            *ptr = value_as_unsigned(value)? as u32
        return Ok(0)
    if dtype == .UInt64:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u64
        unsafe:
            *ptr = value_as_unsigned(value)?
        return Ok(0)
    if dtype == .Float16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u16
        unsafe:
            *ptr = if value.dtype == .Float16 then value.raw as u16 else f32_to_f16_bits(value_as_f64(value)? as f32)
        return Ok(0)
    if dtype == .Float32:
        let ptr = unsafe: (base + byte_offset as i64) as *mut f32
        unsafe:
            *ptr = value_as_f64(value)? as f32
        return Ok(0)
    if dtype == .Float64:
        let ptr = unsafe: (base + byte_offset as i64) as *mut f64
        unsafe:
            *ptr = value_as_f64(value)?
        return Ok(0)
    if dtype == .BFloat16:
        let ptr = unsafe: (base + byte_offset as i64) as *mut u16
        unsafe:
            *ptr = if value.dtype == .BFloat16 then value.raw as u16 else f32_to_bf16_bits(value_as_f64(value)? as f32)
        return Ok(0)
    Err(.Unsupported("cpu interpreter dtype not implemented"))

fn store_value(view: View, byte_offset: Size, value: InterpValue) -> Result[i32, SubstrateError]:
    let base = memory_ptr(view.memory)
    store_ptr(base, view.dtype, byte_offset, value)

fn signed_int_min(dtype: DType) -> i64:
    if dtype == .Int8:
        return -128
    if dtype == .Int16:
        return -32768
    if dtype == .Int32:
        return -2147483648
    if dtype == .Int64:
        return -9223372036854775807 - 1
    0

fn signed_int_max(dtype: DType) -> i64:
    if dtype == .Int8:
        return 127
    if dtype == .Int16:
        return 32767
    if dtype == .Int32:
        return 2147483647
    if dtype == .Int64:
        return 9223372036854775807
    0

fn shift_amount(value: InterpValue) -> Result[u32, SubstrateError]:
    if dtype_is_signed_int(value.dtype):
        return Ok((value_as_signed(value)? as u64 & 63u64) as u32)
    if dtype_is_unsigned_int(value.dtype):
        return Ok((value_as_unsigned(value)? & 63u64) as u32)
    Err(.Unsupported("cpu interpreter shift amount must be integer"))

fn run_binop(op: i32, lhs: InterpValue, rhs: InterpValue) -> Result[InterpValue, SubstrateError]:
    if lhs.dtype != rhs.dtype:
        return Err(.DTypeMismatch("cpu interpreter operand dtype mismatch"))
    if dtype_is_signed_int(lhs.dtype):
        let a = value_as_signed(lhs)?
        let b = value_as_signed(rhs)?
        let amount = shift_amount(rhs)?
        if op == IROP_ADD: return Ok(value_from_signed(lhs.dtype, a + b))
        if op == IROP_SUB: return Ok(value_from_signed(lhs.dtype, a - b))
        if op == IROP_MUL: return Ok(value_from_signed(lhs.dtype, a * b))
        if op == IROP_DIV: return Ok(value_from_signed(lhs.dtype, a / b))
        if op == IROP_MOD: return Ok(value_from_signed(lhs.dtype, a % b))
        if op == IROP_ADD_SAT:
            let max = signed_int_max(lhs.dtype)
            let min = signed_int_min(lhs.dtype)
            if b > 0 and a > max - b:
                return Ok(value_from_signed(lhs.dtype, max))
            if b < 0 and a < min - b:
                return Ok(value_from_signed(lhs.dtype, min))
            return Ok(value_from_signed(lhs.dtype, a + b))
        if op == IROP_SUB_SAT:
            let max = signed_int_max(lhs.dtype)
            let min = signed_int_min(lhs.dtype)
            if b > 0 and a < min + b:
                return Ok(value_from_signed(lhs.dtype, min))
            if b < 0 and a > max + b:
                return Ok(value_from_signed(lhs.dtype, max))
            return Ok(value_from_signed(lhs.dtype, a - b))
        if op == IROP_AND: return Ok(value_from_signed(lhs.dtype, a & b))
        if op == IROP_OR: return Ok(value_from_signed(lhs.dtype, a | b))
        if op == IROP_XOR: return Ok(value_from_signed(lhs.dtype, a ^ b))
        if op == IROP_SHL: return Ok(value_from_signed(lhs.dtype, a << amount))
        if op == IROP_SHR: return Ok(value_from_signed(lhs.dtype, a >> amount))
        if op == IROP_MIN: return Ok(value_from_signed(lhs.dtype, if a < b then a else b))
        if op == IROP_MAX: return Ok(value_from_signed(lhs.dtype, if a > b then a else b))
    if dtype_is_unsigned_int(lhs.dtype):
        let a = value_as_unsigned(lhs)?
        let b = value_as_unsigned(rhs)?
        let amount = shift_amount(rhs)?
        if op == IROP_ADD: return Ok(value_from_unsigned(lhs.dtype, a + b))
        if op == IROP_SUB: return Ok(value_from_unsigned(lhs.dtype, a - b))
        if op == IROP_MUL: return Ok(value_from_unsigned(lhs.dtype, a * b))
        if op == IROP_DIV: return Ok(value_from_unsigned(lhs.dtype, a / b))
        if op == IROP_MOD: return Ok(value_from_unsigned(lhs.dtype, a % b))
        if op == IROP_ADD_SAT:
            let max = mask_bits(dtype_bit_width(lhs.dtype))
            if max - a < b:
                return Ok(value_from_unsigned(lhs.dtype, max))
            return Ok(value_from_unsigned(lhs.dtype, a + b))
        if op == IROP_SUB_SAT:
            if a < b:
                return Ok(value_from_unsigned(lhs.dtype, 0))
            return Ok(value_from_unsigned(lhs.dtype, a - b))
        if op == IROP_AND: return Ok(value_from_unsigned(lhs.dtype, a & b))
        if op == IROP_OR: return Ok(value_from_unsigned(lhs.dtype, a | b))
        if op == IROP_XOR: return Ok(value_from_unsigned(lhs.dtype, a ^ b))
        if op == IROP_SHL: return Ok(value_from_unsigned(lhs.dtype, a << amount))
        if op == IROP_SHR: return Ok(value_from_unsigned(lhs.dtype, a >> amount))
        if op == IROP_MIN: return Ok(value_from_unsigned(lhs.dtype, if a < b then a else b))
        if op == IROP_MAX: return Ok(value_from_unsigned(lhs.dtype, if a > b then a else b))
    if dtype_is_float(lhs.dtype):
        let a = value_as_f64(lhs)?
        let b = value_as_f64(rhs)?
        if op == IROP_ADD: return Ok(value_from_float_dtype(lhs.dtype, a + b))
        if op == IROP_SUB: return Ok(value_from_float_dtype(lhs.dtype, a - b))
        if op == IROP_MUL: return Ok(value_from_float_dtype(lhs.dtype, a * b))
        if op == IROP_DIV: return Ok(value_from_float_dtype(lhs.dtype, a / b))
        if op == IROP_MIN: return Ok(value_from_float_dtype(lhs.dtype, if a < b then a else b))
        if op == IROP_MAX: return Ok(value_from_float_dtype(lhs.dtype, if a > b then a else b))
    Err(.Unsupported("cpu interpreter binary op not implemented"))

fn run_cmp(op: i32, lhs: InterpValue, rhs: InterpValue) -> Result[InterpValue, SubstrateError]:
    if lhs.dtype != rhs.dtype:
        return Err(.DTypeMismatch("cpu interpreter operand dtype mismatch"))
    if dtype_is_signed_int(lhs.dtype):
        let a = value_as_signed(lhs)?
        let b = value_as_signed(rhs)?
        if op == IROP_EQ: return Ok(value_from_i32(if a == b then 1 else 0))
        if op == IROP_NE: return Ok(value_from_i32(if a != b then 1 else 0))
        if op == IROP_LT: return Ok(value_from_i32(if a < b then 1 else 0))
        if op == IROP_GT: return Ok(value_from_i32(if a > b then 1 else 0))
        if op == IROP_LE: return Ok(value_from_i32(if a <= b then 1 else 0))
        if op == IROP_GE: return Ok(value_from_i32(if a >= b then 1 else 0))
    if dtype_is_unsigned_int(lhs.dtype):
        let a = value_as_unsigned(lhs)?
        let b = value_as_unsigned(rhs)?
        if op == IROP_EQ: return Ok(value_from_i32(if a == b then 1 else 0))
        if op == IROP_NE: return Ok(value_from_i32(if a != b then 1 else 0))
        if op == IROP_LT: return Ok(value_from_i32(if a < b then 1 else 0))
        if op == IROP_GT: return Ok(value_from_i32(if a > b then 1 else 0))
        if op == IROP_LE: return Ok(value_from_i32(if a <= b then 1 else 0))
        if op == IROP_GE: return Ok(value_from_i32(if a >= b then 1 else 0))
    if dtype_is_float(lhs.dtype):
        let a = value_as_f64(lhs)?
        let b = value_as_f64(rhs)?
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
    if dtype_is_signed_int(a.dtype):
        return Ok(value_from_signed(a.dtype, value_as_signed(a)? * value_as_signed(b)? + value_as_signed(c)?))
    if dtype_is_unsigned_int(a.dtype):
        return Ok(value_from_unsigned(a.dtype, value_as_unsigned(a)? * value_as_unsigned(b)? + value_as_unsigned(c)?))
    if dtype_is_float(a.dtype):
        return Ok(value_from_float_dtype(a.dtype, value_as_f64(a)? * value_as_f64(b)? + value_as_f64(c)?))
    Err(.Unsupported("cpu interpreter fma not implemented"))

fn run_neg(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if dtype_is_signed_int(value.dtype):
        return Ok(value_from_signed(value.dtype, -value_as_signed(value)?))
    if dtype_is_float(value.dtype):
        return Ok(value_from_float_dtype(value.dtype, -value_as_f64(value)?))
    Err(.Unsupported("cpu interpreter neg not implemented"))

fn run_abs(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if dtype_is_signed_int(value.dtype):
        let decoded = value_as_signed(value)?
        return Ok(value_from_signed(value.dtype, if decoded < 0 then -decoded else decoded))
    if dtype_is_float(value.dtype):
        let decoded = value_as_f64(value)?
        return Ok(value_from_float_dtype(value.dtype, if decoded < 0.0 then -decoded else decoded))
    Err(.Unsupported("cpu interpreter abs not implemented"))

fn run_not(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if dtype_is_signed_int(value.dtype):
        return Ok(value_from_signed(value.dtype, ~value_as_signed(value)?))
    if dtype_is_unsigned_int(value.dtype):
        return Ok(value_from_unsigned(value.dtype, ~value_as_unsigned(value)?))
    Err(.Unsupported("cpu interpreter not not implemented"))

fn run_bitcount(op: i32, value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if not dtype_is_int(value.dtype):
        return Err(.Unsupported("cpu interpreter bitcount op requires integer dtype"))
    let width = dtype_bit_width(value.dtype)
    let bits = truncate_raw(value.raw, width)
    if op == IROP_POPCOUNT:
        var count: u64 = 0u64
        var work = bits
        while work != 0u64:
            count = count + (work & 1u64)
            work = work >> 1
        if dtype_is_signed_int(value.dtype):
            return Ok(value_from_signed(value.dtype, count as i64))
        return Ok(value_from_unsigned(value.dtype, count))
    if op == IROP_CLZ:
        if bits == 0u64:
            if dtype_is_signed_int(value.dtype):
                return Ok(value_from_signed(value.dtype, width as i64))
            return Ok(value_from_unsigned(value.dtype, width as u64))
        var count: i32 = 0
        var mask = 1u64 << (width - 1)
        while (bits & mask) == 0u64:
            count = count + 1
            mask = mask >> 1
        if dtype_is_signed_int(value.dtype):
            return Ok(value_from_signed(value.dtype, count as i64))
        return Ok(value_from_unsigned(value.dtype, count as u64))
    if op == IROP_CTZ:
        if bits == 0u64:
            if dtype_is_signed_int(value.dtype):
                return Ok(value_from_signed(value.dtype, width as i64))
            return Ok(value_from_unsigned(value.dtype, width as u64))
        var count: i32 = 0
        var mask = 1u64
        while (bits & mask) == 0u64:
            count = count + 1
            mask = mask << 1
        if dtype_is_signed_int(value.dtype):
            return Ok(value_from_signed(value.dtype, count as i64))
        return Ok(value_from_unsigned(value.dtype, count as u64))
    Err(.Unsupported("cpu interpreter bitcount op not implemented"))

fn run_exp(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    let ln2_f32: f32 = 0.69314718056
    let ln2_f64: f64 = 0.6931471805599453
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        if decoded == 0.0:
            return Ok(value_from_float_dtype(value.dtype, 1.0))
        var reduced = decoded
        var shifts: i32 = 0
        while reduced > ln2_f32:
            reduced = reduced - ln2_f32
            shifts = shifts + 1
        while reduced < -ln2_f32:
            reduced = reduced + ln2_f32
            shifts = shifts - 1
        var sum: f32 = 1.0
        var term: f32 = 1.0
        var i: i32 = 1
        while i <= 12:
            term = term * (reduced / i as f32)
            sum = sum + term
            i = i + 1
        while shifts > 0:
            sum = sum * 2.0
            shifts = shifts - 1
        while shifts < 0:
            sum = sum * 0.5
            shifts = shifts + 1
        return Ok(value_from_float_dtype(value.dtype, sum as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        if decoded == 0.0:
            return Ok(value_from_f64(1.0))
        var reduced = decoded
        var shifts: i32 = 0
        while reduced > ln2_f64:
            reduced = reduced - ln2_f64
            shifts = shifts + 1
        while reduced < -ln2_f64:
            reduced = reduced + ln2_f64
            shifts = shifts - 1
        var sum: f64 = 1.0
        var term: f64 = 1.0
        var i: i32 = 1
        while i <= 18:
            term = term * (reduced / i as f64)
            sum = sum + term
            i = i + 1
        while shifts > 0:
            sum = sum * 2.0
            shifts = shifts - 1
        while shifts < 0:
            sum = sum * 0.5
            shifts = shifts + 1
        return Ok(value_from_f64(sum))
    Err(.Unsupported("cpu interpreter exp not implemented"))

fn run_log(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    let ln2_f32: f32 = 0.69314718056
    let ln2_f64: f64 = 0.6931471805599453
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        if decoded <= 0.0:
            return Err(.Unsupported("cpu interpreter log of nonpositive float is not implemented"))
        if decoded == 1.0:
            return Ok(value_from_float_dtype(value.dtype, 0.0))
        var reduced = decoded
        var shifts: i32 = 0
        while reduced >= 2.0:
            reduced = reduced * 0.5
            shifts = shifts + 1
        while reduced < 1.0:
            reduced = reduced * 2.0
            shifts = shifts - 1
        if reduced == 1.0:
            return Ok(value_from_f32(shifts as f32 * ln2_f32))
        let y = (reduced - 1.0) / (reduced + 1.0)
        let y2 = y * y
        var term = y
        var sum = y
        var denom: i32 = 3
        while denom <= 19:
            term = term * y2
            sum = sum + term / denom as f32
            denom = denom + 2
        return Ok(value_from_float_dtype(value.dtype, (2.0 * sum + shifts as f32 * ln2_f32) as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        if decoded <= 0.0:
            return Err(.Unsupported("cpu interpreter log of nonpositive f64 is not implemented"))
        if decoded == 1.0:
            return Ok(value_from_f64(0.0))
        var reduced = decoded
        var shifts: i32 = 0
        while reduced >= 2.0:
            reduced = reduced * 0.5
            shifts = shifts + 1
        while reduced < 1.0:
            reduced = reduced * 2.0
            shifts = shifts - 1
        if reduced == 1.0:
            return Ok(value_from_f64(shifts as f64 * ln2_f64))
        let y = (reduced - 1.0) / (reduced + 1.0)
        let y2 = y * y
        var term = y
        var sum = y
        var denom: i32 = 3
        while denom <= 27:
            term = term * y2
            sum = sum + term / denom as f64
            denom = denom + 2
        return Ok(value_from_f64(2.0 * sum + shifts as f64 * ln2_f64))
    Err(.Unsupported("cpu interpreter log not implemented"))

fn run_log2(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    let ln2_f32: f32 = 0.69314718056
    let ln2_f64: f64 = 0.6931471805599453
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        if decoded <= 0.0:
            return Err(.Unsupported("cpu interpreter log2 of nonpositive float is not implemented"))
        var reduced = decoded
        var shifts: i32 = 0
        while reduced >= 2.0:
            reduced = reduced * 0.5
            shifts = shifts + 1
        while reduced < 1.0:
            reduced = reduced * 2.0
            shifts = shifts - 1
        if reduced == 1.0:
            return Ok(value_from_float_dtype(value.dtype, shifts as f64))
        let reduced_log = run_log(value_from_f32(reduced))?
        let ln_value: f32 = unsafe: transmute[f32](reduced_log.raw as u32)
        return Ok(value_from_float_dtype(value.dtype, (ln_value / ln2_f32 + shifts as f32) as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        if decoded <= 0.0:
            return Err(.Unsupported("cpu interpreter log2 of nonpositive f64 is not implemented"))
        var reduced = decoded
        var shifts: i32 = 0
        while reduced >= 2.0:
            reduced = reduced * 0.5
            shifts = shifts + 1
        while reduced < 1.0:
            reduced = reduced * 2.0
            shifts = shifts - 1
        if reduced == 1.0:
            return Ok(value_from_f64(shifts as f64))
        let reduced_log = run_log(value_from_f64(reduced))?
        let ln_value: f64 = unsafe: transmute[f64](reduced_log.raw)
        return Ok(value_from_f64(ln_value / ln2_f64 + shifts as f64))
    Err(.Unsupported("cpu interpreter log2 not implemented"))

fn run_sin(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    let pi_f32: f32 = 3.14159265359
    let two_pi_f32: f32 = 6.28318530718
    let pi_f64: f64 = 3.141592653589793
    let two_pi_f64: f64 = 6.283185307179586
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        if decoded == 0.0:
            return Ok(value_from_float_dtype(value.dtype, 0.0))
        var reduced = decoded
        while reduced > pi_f32:
            reduced = reduced - two_pi_f32
        while reduced < -pi_f32:
            reduced = reduced + two_pi_f32
        let x2 = reduced * reduced
        let x3 = reduced * x2
        let x5 = x3 * x2
        let x7 = x5 * x2
        let x9 = x7 * x2
        return Ok(value_from_float_dtype(value.dtype, (reduced - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0) as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        if decoded == 0.0:
            return Ok(value_from_f64(0.0))
        var reduced = decoded
        while reduced > pi_f64:
            reduced = reduced - two_pi_f64
        while reduced < -pi_f64:
            reduced = reduced + two_pi_f64
        let x2 = reduced * reduced
        let x3 = reduced * x2
        let x5 = x3 * x2
        let x7 = x5 * x2
        let x9 = x7 * x2
        let x11 = x9 * x2
        return Ok(value_from_f64(reduced - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0 - x11 / 39916800.0))
    Err(.Unsupported("cpu interpreter sin not implemented"))

fn run_cos(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    let pi_f32: f32 = 3.14159265359
    let two_pi_f32: f32 = 6.28318530718
    let pi_f64: f64 = 3.141592653589793
    let two_pi_f64: f64 = 6.283185307179586
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        if decoded == 0.0:
            return Ok(value_from_float_dtype(value.dtype, 1.0))
        var reduced = decoded
        while reduced > pi_f32:
            reduced = reduced - two_pi_f32
        while reduced < -pi_f32:
            reduced = reduced + two_pi_f32
        let x2 = reduced * reduced
        let x4 = x2 * x2
        let x6 = x4 * x2
        let x8 = x6 * x2
        return Ok(value_from_float_dtype(value.dtype, (1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0) as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        if decoded == 0.0:
            return Ok(value_from_f64(1.0))
        var reduced = decoded
        while reduced > pi_f64:
            reduced = reduced - two_pi_f64
        while reduced < -pi_f64:
            reduced = reduced + two_pi_f64
        let x2 = reduced * reduced
        let x4 = x2 * x2
        let x6 = x4 * x2
        let x8 = x6 * x2
        let x10 = x8 * x2
        return Ok(value_from_f64(1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0 - x10 / 3628800.0))
    Err(.Unsupported("cpu interpreter cos not implemented"))

fn run_tanh(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        if decoded == 0.0:
            return Ok(value_from_float_dtype(value.dtype, 0.0))
        let doubled = value_from_f32(decoded * 2.0)
        let exp_value = run_exp(doubled)?
        let ev: f32 = unsafe: transmute[f32](exp_value.raw as u32)
        return Ok(value_from_float_dtype(value.dtype, ((ev - 1.0) / (ev + 1.0)) as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        if decoded == 0.0:
            return Ok(value_from_f64(0.0))
        let doubled = value_from_f64(decoded * 2.0)
        let exp_value = run_exp(doubled)?
        let ev: f64 = unsafe: transmute[f64](exp_value.raw)
        return Ok(value_from_f64((ev - 1.0) / (ev + 1.0)))
    Err(.Unsupported("cpu interpreter tanh not implemented"))

fn run_sqrt(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        if decoded < 0.0:
            return Err(.Unsupported("cpu interpreter sqrt of negative float is not implemented"))
        if decoded == 0.0 or decoded == 1.0:
            return Ok(value_from_float_dtype(value.dtype, decoded as f64))
        var guess: f32 = if decoded > 1.0 then decoded * 0.5 else 1.0
        var i: i32 = 0
        while i < 8:
            guess = 0.5 * (guess + decoded / guess)
            i = i + 1
        return Ok(value_from_float_dtype(value.dtype, guess as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        if decoded < 0.0:
            return Err(.Unsupported("cpu interpreter sqrt of negative f64 is not implemented"))
        if decoded == 0.0 or decoded == 1.0:
            return Ok(value_from_f64(decoded))
        var guess: f64 = if decoded > 1.0 then decoded * 0.5 else 1.0
        var i: i32 = 0
        while i < 12:
            guess = 0.5 * (guess + decoded / guess)
            i = i + 1
        return Ok(value_from_f64(guess))
    Err(.Unsupported("cpu interpreter sqrt not implemented"))

fn run_rsqrt(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    let root = run_sqrt(value)?
    if root.dtype == .Float16 or root.dtype == .Float32 or root.dtype == .BFloat16:
        let decoded = value_as_f64(root)? as f32
        if decoded == 0.0:
            return Err(.Unsupported("cpu interpreter rsqrt of zero float is not implemented"))
        return Ok(value_from_float_dtype(root.dtype, (1.0 / decoded) as f64))
    if root.dtype == .Float64:
        let decoded = value_as_f64(root)?
        if decoded == 0.0:
            return Err(.Unsupported("cpu interpreter rsqrt of zero f64 is not implemented"))
        return Ok(value_from_f64(1.0 / decoded))
    Err(.Unsupported("cpu interpreter rsqrt not implemented"))

fn run_floor(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        let truncated = decoded as i32
        let trunc_f = truncated as f32
        if trunc_f > decoded:
            return Ok(value_from_float_dtype(value.dtype, (truncated - 1) as f64))
        return Ok(value_from_float_dtype(value.dtype, trunc_f as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        let truncated = decoded as i64
        let trunc_f = truncated as f64
        if trunc_f > decoded:
            return Ok(value_from_f64((truncated - 1) as f64))
        return Ok(value_from_f64(trunc_f))
    Err(.Unsupported("cpu interpreter floor not implemented"))

fn run_ceil(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if value.dtype == .Float16 or value.dtype == .Float32 or value.dtype == .BFloat16:
        let decoded = value_as_f64(value)? as f32
        let truncated = decoded as i32
        let trunc_f = truncated as f32
        if trunc_f < decoded:
            return Ok(value_from_float_dtype(value.dtype, (truncated + 1) as f64))
        return Ok(value_from_float_dtype(value.dtype, trunc_f as f64))
    if value.dtype == .Float64:
        let decoded = value_as_f64(value)?
        let truncated = decoded as i64
        let trunc_f = truncated as f64
        if trunc_f < decoded:
            return Ok(value_from_f64((truncated + 1) as f64))
        return Ok(value_from_f64(trunc_f))
    Err(.Unsupported("cpu interpreter ceil not implemented"))

fn run_round(value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if dtype_is_float(value.dtype):
        let decoded = value_as_f64(value)?
        let shifted = if decoded >= 0.0 then decoded + 0.5 else decoded - 0.5
        if decoded >= 0.0:
            return run_floor(value_from_float_dtype(value.dtype, shifted))
        return run_ceil(value_from_float_dtype(value.dtype, shifted))
    Err(.Unsupported("cpu interpreter round not implemented"))

fn run_cast(value: InterpValue, target: DType) -> Result[InterpValue, SubstrateError]:
    if value.dtype == target:
        return Ok(value)
    if dtype_is_signed_int(value.dtype):
        let signed = value_as_signed(value)?
        if dtype_is_signed_int(target):
            return Ok(value_from_signed(target, signed))
        if dtype_is_unsigned_int(target):
            return Ok(value_from_unsigned(target, signed as u64))
        if dtype_is_float(target):
            return Ok(value_from_float_dtype(target, signed as f64))
    if dtype_is_unsigned_int(value.dtype):
        let unsigned = value_as_unsigned(value)?
        if dtype_is_signed_int(target):
            return Ok(value_from_signed(target, unsigned as i64))
        if dtype_is_unsigned_int(target):
            return Ok(value_from_unsigned(target, unsigned))
        if dtype_is_float(target):
            return Ok(value_from_float_dtype(target, unsigned as f64))
    if dtype_is_float(value.dtype):
        let decoded = value_as_f64(value)?
        if dtype_is_signed_int(target):
            return Ok(value_from_signed(target, decoded as i64))
        if dtype_is_unsigned_int(target):
            return Ok(value_from_unsigned(target, decoded as u64))
        if dtype_is_float(target):
            return Ok(value_from_float_dtype(target, decoded))
    Err(.Unsupported("cpu interpreter cast not implemented"))

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
    if dtype_is_signed_int(value.dtype):
        let v = value_as_signed(value)?
        let lo_v = value_as_signed(lo)?
        let hi_v = value_as_signed(hi)?
        if v < lo_v:
            return Ok(value_from_signed(value.dtype, lo_v))
        if v > hi_v:
            return Ok(value_from_signed(value.dtype, hi_v))
        return Ok(value_from_signed(value.dtype, v))
    if dtype_is_unsigned_int(value.dtype):
        let v = value_as_unsigned(value)?
        let lo_v = value_as_unsigned(lo)?
        let hi_v = value_as_unsigned(hi)?
        if v < lo_v:
            return Ok(value_from_unsigned(value.dtype, lo_v))
        if v > hi_v:
            return Ok(value_from_unsigned(value.dtype, hi_v))
        return Ok(value_from_unsigned(value.dtype, v))
    if dtype_is_float(value.dtype):
        let v = value_as_f64(value)?
        let lo_v = value_as_f64(lo)?
        let hi_v = value_as_f64(hi)?
        if v < lo_v:
            return Ok(value_from_float_dtype(value.dtype, lo_v))
        if v > hi_v:
            return Ok(value_from_float_dtype(value.dtype, hi_v))
        return Ok(value_from_float_dtype(value.dtype, v))
    Err(.Unsupported("cpu interpreter clamp not implemented"))

fn value_zero(dtype: DType) -> InterpValue:
    if dtype_is_signed_int(dtype):
        return value_from_signed(dtype, 0)
    if dtype_is_unsigned_int(dtype):
        return value_from_unsigned(dtype, 0u64)
    if dtype_is_float(dtype):
        return value_from_float_dtype(dtype, 0.0)
    interp_value_zero()

fn value_one(dtype: DType) -> InterpValue:
    if dtype_is_signed_int(dtype):
        return value_from_signed(dtype, 1)
    if dtype_is_unsigned_int(dtype):
        return value_from_unsigned(dtype, 1u64)
    if dtype_is_float(dtype):
        return value_from_float_dtype(dtype, 1.0)
    interp_value_zero()

fn reduce_identity(op: i32, dtype: DType) -> InterpValue:
    if op == IROP_REDUCE_PROD:
        return value_one(dtype)
    value_zero(dtype)

fn reduce_step(op: i32, accum: InterpValue, value: InterpValue) -> Result[InterpValue, SubstrateError]:
    if op == IROP_REDUCE_SUM:
        return run_binop(IROP_ADD, accum, value)
    if op == IROP_REDUCE_MAX:
        return run_binop(IROP_MAX, accum, value)
    if op == IROP_REDUCE_MIN:
        return run_binop(IROP_MIN, accum, value)
    if op == IROP_REDUCE_PROD:
        return run_binop(IROP_MUL, accum, value)
    Err(.Unsupported(f"cpu interpreter reduction step does not support '{ir_op_name(op)}'"))

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
    out.private_generation = out.private_generation + 1
    var ip = start
    while ip < stop:
        let inst = prog.insts[ip]

        if inst.op == IROP_PARAM or inst.op == IROP_SPEC_CONSTANT:
            ip = ip + 1
            continue

        if inst.op == IROP_CONST:
            out.values[inst.d0] = InterpValue {
                raw: scalar_bits_from_words(inst.d1, inst.d2),
                dtype: inst.dtype,
            }
            ip = ip + 1
            continue

        if inst.op == IROP_LOAD:
            let view = resolve_load_view(out, params, inst.d1)?
            let indices = indices_from_aux(out, prog, inst.d2, view.shape.rank)?
            let offset = view_offset_of(view, indices)
            if ir_is_param_ref(inst.d1):
                out.values[inst.d0] = load_value(view, offset)?
            else if ir_is_local_ref(inst.d1):
                let local_slot = out.locals[ir_local_index(inst.d1)]
                out.values[inst.d0] = load_ptr(local_slot.ptr, view.dtype, offset)?
            else:
                let private_slot = out.privates[ir_private_index(inst.d1)]
                out.values[inst.d0] = load_ptr(private_slot.ptr, view.dtype, offset)?
            ip = ip + 1
            continue

        if inst.op == IROP_STORE:
            let view = resolve_load_view(out, params, inst.d0)?
            let indices = indices_from_aux(out, prog, inst.d1, view.shape.rank)?
            let offset = view_offset_of(view, indices)
            let value = value_ref_get(out, inst.d2)?
            if ir_is_param_ref(inst.d0):
                let _ = store_value(view, offset, value)?
            else if ir_is_local_ref(inst.d0):
                let local_slot = out.locals[ir_local_index(inst.d0)]
                let _ = store_ptr(local_slot.ptr, view.dtype, offset, value)?
            else:
                let private_slot = out.privates[ir_private_index(inst.d0)]
                let _ = store_ptr(private_slot.ptr, view.dtype, offset, value)?
            ip = ip + 1
            continue

        if inst.op == IROP_LOCAL:
            let local_index = decl_index_before(prog, ip, IROP_LOCAL)
            let local_shape = indices_from_aux(out, prog, inst.d2, inst.d1)?
            let existing = out.locals[local_index]
            if not existing.live or existing.generation != out.local_generation:
                out.locals[local_index] = scratch_alloc(existing, local_shape, inst.dtype, out.local_generation)?
            ip = ip + 1
            continue

        if inst.op == IROP_PRIVATE:
            let private_index = decl_index_before(prog, ip, IROP_PRIVATE)
            let private_shape = indices_from_aux(out, prog, inst.d2, inst.d1)?
            let existing = out.privates[private_index]
            if not existing.live or existing.generation != out.private_generation:
                out.privates[private_index] = scratch_alloc(existing, private_shape, inst.dtype, out.private_generation)?
            ip = ip + 1
            continue

        if inst.op == IROP_ADD or inst.op == IROP_SUB or inst.op == IROP_MUL or inst.op == IROP_DIV or inst.op == IROP_MOD or inst.op == IROP_ADD_SAT or inst.op == IROP_SUB_SAT or inst.op == IROP_AND or inst.op == IROP_OR or inst.op == IROP_XOR or inst.op == IROP_SHL or inst.op == IROP_SHR or inst.op == IROP_MIN or inst.op == IROP_MAX:
            let lhs = value_ref_get(out, inst.d1)?
            let rhs = value_ref_get(out, inst.d2)?
            out.values[inst.d0] = run_binop(inst.op, lhs, rhs)?
            ip = ip + 1
            continue

        if inst.op == IROP_EQ or inst.op == IROP_NE or inst.op == IROP_LT or inst.op == IROP_GT or inst.op == IROP_LE or inst.op == IROP_GE:
            let lhs = value_ref_get(out, inst.d1)?
            let rhs = value_ref_get(out, inst.d2)?
            out.values[inst.d0] = run_cmp(inst.op, lhs, rhs)?
            ip = ip + 1
            continue

        if inst.op == IROP_FMA:
            let a = value_ref_get(out, inst.d1)?
            let b = value_ref_get(out, inst.d2)?
            let c = value_ref_get(out, inst.d3)?
            out.values[inst.d0] = run_fma(a, b, c)?
            ip = ip + 1
            continue

        if inst.op == IROP_NEG:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_neg(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_ABS:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_abs(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_NOT:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_not(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_POPCOUNT or inst.op == IROP_CLZ or inst.op == IROP_CTZ:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_bitcount(inst.op, value)?
            ip = ip + 1
            continue

        if inst.op == IROP_EXP:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_exp(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_LOG:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_log(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_LOG2:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_log2(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_SIN:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_sin(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_COS:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_cos(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_TANH:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_tanh(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_SQRT:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_sqrt(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_RSQRT:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_rsqrt(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_FLOOR:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_floor(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_CEIL:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_ceil(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_ROUND:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_round(value)?
            ip = ip + 1
            continue

        if inst.op == IROP_CAST:
            let value = value_ref_get(out, inst.d1)?
            out.values[inst.d0] = run_cast(value, inst.dtype)?
            ip = ip + 1
            continue

        if inst.op == IROP_SELECT:
            let cond = value_ref_get(out, inst.d1)?
            let on_true = value_ref_get(out, inst.d2)?
            let on_false = value_ref_get(out, inst.d3)?
            out.values[inst.d0] = run_select(cond, on_true, on_false)?
            ip = ip + 1
            continue

        if inst.op == IROP_CLAMP:
            let value = value_ref_get(out, inst.d1)?
            let lo = value_ref_get(out, inst.d2)?
            let hi = value_ref_get(out, inst.d3)?
            out.values[inst.d0] = run_clamp(value, lo, hi)?
            ip = ip + 1
            continue

        if inst.op == IROP_LOOP or inst.op == IROP_PARALLEL or inst.op == IROP_PARALLEL_GRID or inst.op == IROP_PARALLEL_WORKGROUP or inst.op == IROP_PARALLEL_SUBGROUP:
            let block = find_block_range(prog, ip, stop, inst.d3)?
            let start_value = resolve_i32_ref(out, inst.d1)?
            let end_value = resolve_i32_ref(out, inst.d2)?
            var iter = start_value
            while iter < end_value:
                out.loops = loop_var_set(out.loops, inst.d0, iter)
                if inst.op == IROP_PARALLEL_WORKGROUP:
                    out.local_generation = out.local_generation + 1
                out = exec_range(prog, params, out, block.body_start, block.block_end)?
                iter = iter + 1
            ip = block.block_end + 1
            continue

        if inst.op == IROP_IF:
            let cond = value_ref_get(out, inst.d1)?
            let then_block = find_block_range(prog, ip, stop, inst.d2)?
            var after_if = then_block.block_end + 1
            if cond.raw as i32 != 0:
                out = exec_range(prog, params, out, then_block.body_start, then_block.block_end)?
            if inst.d3 >= 0:
                let else_block = find_block_range(prog, then_block.block_end, stop, inst.d3)?
                if cond.raw as i32 == 0:
                    out = exec_range(prog, params, out, else_block.body_start, else_block.block_end)?
                after_if = else_block.block_end + 1
            ip = after_if
            continue

        if inst.op == IROP_REDUCE_SUM or inst.op == IROP_REDUCE_MAX or inst.op == IROP_REDUCE_MIN or inst.op == IROP_REDUCE_PROD:
            let loop_slot = prog.aux[inst.d3]
            let block_id = prog.aux[inst.d3 + 1]
            let body_ref = prog.aux[inst.d3 + 2]
            let block = find_block_range(prog, ip, stop, block_id)?
            let start_value = resolve_i32_ref(out, inst.d1)?
            let end_value = resolve_i32_ref(out, inst.d2)?
            var accum = reduce_identity(inst.op, inst.dtype)
            var seen = false
            var iter = start_value
            while iter < end_value:
                out.loops = loop_var_set(out.loops, loop_slot, iter)
                out = exec_range(prog, params, out, block.body_start, block.block_end)?
                let body_value = value_ref_get(out, body_ref)?
                if not seen:
                    accum = body_value
                    seen = true
                else:
                    accum = reduce_step(inst.op, accum, body_value)?
                iter = iter + 1
            if not seen:
                accum = reduce_identity(inst.op, inst.dtype)
            out.values[inst.d0] = accum
            ip = block.block_end + 1
            continue

        if inst.op == IROP_COLLECTIVE_ALLREDUCE_SUM or inst.op == IROP_COLLECTIVE_ALLREDUCE_MAX or inst.op == IROP_COLLECTIVE_ALLGATHER or inst.op == IROP_COLLECTIVE_BROADCAST or inst.op == IROP_COLLECTIVE_REDUCE_SCATTER:
            out.values[inst.d0] = value_ref_get(out, inst.d1)?
            ip = ip + 1
            continue

        if inst.op == IROP_BARRIER:
            ip = ip + 1
            continue

        if inst.op == IROP_BLOCK_BEGIN or inst.op == IROP_BLOCK_END:
            return Err(.CompileError("unexpected block marker during execution"))

        if inst.op == IROP_RETURN:
            return Ok(out)

        return Err(.Unsupported(f"cpu interpreter has no execution rule for opcode '{ir_op_name(inst.op)}'"))
    Ok(out)

fn destroy_scratch_slots(slots: Vec[ScratchSlot]):
    for i in 0..slots.len():
        let slot = slots[i]
        if slot.ptr != null:
            let _ = realloc(slot.ptr as *mut c_void, 0)

pub fn interp_dispatch(prog: IRProgram, sig: ProgramSig, params: Vec[View]) -> Result[i32, SubstrateError]:
    if params.len() as i32 != sig.params.len() as i32:
        return Err(.ShapeMismatch("binding count does not match program parameters"))
    let state = init_state(ir_value_count(prog), count_decl_ops(prog, IROP_LOCAL), count_decl_ops(prog, IROP_PRIVATE))
    let final_state = exec_range(prog, params, state, 0, prog.insts.len() as i32)?
    destroy_scratch_slots(final_state.locals)
    destroy_scratch_slots(final_state.privates)
    Ok(0)
