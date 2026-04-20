use crux.core
use crux.device
use crux.kernels
use crux.program

fn compile_sig(source: ProgramSource) -> ProgramSig:
    let prog = match compile(device_info(default_device()), source)
        Ok(v) => v
        Err(_) =>
            assert(false)
            null_program()
    let sig = program_sig(prog)
    program_destroy(prog)
    sig

fn assert_param(desc: ParamDesc, name: str, mode: ParamMode, rank: i32, dtype: DType):
    assert(desc.name == name)
    assert(desc.mode == mode)
    assert(desc.rank == rank)
    assert(desc.dtype == dtype)

fn test_kernel_source_signatures:
    let map_i32 = compile_sig(kernel_map_add_1d_i32_source())
    assert(map_i32.params.len() == 3)
    assert_param(map_i32.params[0], "a", .In, 1, .Int32)
    assert_param(map_i32.params[1], "b", .In, 1, .Int32)
    assert_param(map_i32.params[2], "out", .Out, 1, .Int32)

    let map_f32 = compile_sig(kernel_map_add_2d_f32_source())
    assert(map_f32.params.len() == 3)
    assert_param(map_f32.params[0], "a", .In, 2, .Float32)
    assert_param(map_f32.params[1], "b", .In, 2, .Float32)
    assert_param(map_f32.params[2], "out", .Out, 2, .Float32)

    let inplace = compile_sig(kernel_map_inplace_add_1d_i32_source())
    assert(inplace.params.len() == 2)
    assert_param(inplace.params[0], "a", .InOut, 1, .Int32)
    assert_param(inplace.params[1], "b", .In, 1, .Int32)

    let reduce_1d = compile_sig(kernel_reduce_sum_1d_i32_source())
    assert(reduce_1d.params.len() == 2)
    assert_param(reduce_1d.params[0], "a", .In, 1, .Int32)
    assert_param(reduce_1d.params[1], "out", .Out, 0, .Int32)

    let reduce_rows = compile_sig(kernel_reduce_sum_rows_2d_i32_source())
    assert(reduce_rows.params.len() == 2)
    assert_param(reduce_rows.params[0], "a", .In, 2, .Int32)
    assert_param(reduce_rows.params[1], "out", .Out, 1, .Int32)

    let reduce_max = compile_sig(kernel_reduce_max_rows_2d_f32_source())
    assert(reduce_max.params.len() == 2)
    assert_param(reduce_max.params[0], "a", .In, 2, .Float32)
    assert_param(reduce_max.params[1], "out", .Out, 1, .Float32)

    let transpose = compile_sig(kernel_transpose_2d_i32_source())
    assert(transpose.params.len() == 2)
    assert_param(transpose.params[0], "a", .In, 2, .Int32)
    assert_param(transpose.params[1], "out", .Out, 2, .Int32)

    let matmul_i32 = compile_sig(kernel_matmul_2d_i32_source())
    assert(matmul_i32.params.len() == 3)
    assert_param(matmul_i32.params[0], "a", .In, 2, .Int32)
    assert_param(matmul_i32.params[1], "b", .In, 2, .Int32)
    assert_param(matmul_i32.params[2], "out", .Out, 2, .Int32)

    let matmul_f32 = compile_sig(kernel_matmul_2d_f32_source())
    assert(matmul_f32.params.len() == 3)
    assert_param(matmul_f32.params[0], "a", .In, 2, .Float32)
    assert_param(matmul_f32.params[1], "b", .In, 2, .Float32)
    assert_param(matmul_f32.params[2], "out", .Out, 2, .Float32)

fn main:
    test_kernel_source_signatures()
