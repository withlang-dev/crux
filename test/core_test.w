use crux.core
use crux.memory

fn test_shape_elem_count:
    assert(shape_elem_count(shape_scalar()) == 1)
    assert(shape_elem_count(shape2(3, 4)) == 12)
    assert(shape_elem_count(shape3(2, 3, 4)) == 24)

fn test_shape_equal_and_set:
    let base = shape3(2, 3, 4)
    let same = shape_set(base, 1, 3)
    let changed = shape_set(base, 1, 5)
    assert(shape_equal(base, same))
    assert(not shape_equal(base, changed))

fn test_contiguous_strides_float32:
    let st = contiguous_strides(shape3(3, 4, 5), .Float32)
    assert(strides_get(st, 0) == 80)
    assert(strides_get(st, 1) == 20)
    assert(strides_get(st, 2) == 4)

fn test_contiguous_strides_float64:
    let st = contiguous_strides(shape2(3, 4), .Float64)
    assert(strides_get(st, 0) == 32)
    assert(strides_get(st, 1) == 8)

fn test_dtype_size_table:
    assert(dtype_size(.Int8) == 1)
    assert(dtype_size(.Float16) == 2)
    assert(dtype_size(.Float32) == 4)
    assert(dtype_size(.Float64) == 8)

fn test_memory_placement_helpers:
    let local = placement_local()
    let replicated = placement_replicated()
    let partitioned = placement_partitioned(8)
    assert(local.kind == PLACEMENT_LOCAL)
    assert(local.regions == 1)
    assert(replicated.kind == PLACEMENT_REPLICATED)
    assert(replicated.regions == 0)
    assert(partitioned.kind == PLACEMENT_PARTITIONED)
    assert(partitioned.regions == 8)

fn main:
    test_shape_elem_count()
    test_shape_equal_and_set()
    test_contiguous_strides_float32()
    test_contiguous_strides_float64()
    test_dtype_size_table()
    test_memory_placement_helpers()
