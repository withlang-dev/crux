use crux.core
use crux.view

fn test_view_smoke:
    let st = contiguous_strides(shape3(3, 4, 5), .Float32)
    assert(strides_get(st, 0) == 80)
    assert(strides_get(st, 1) == 20)
    assert(strides_get(st, 2) == 4)

    let base = view_contiguous(null_memory(), shape2(3, 4), .Float32)
    let sliced = view_slice(base, 0, 1, 3)
    assert(sliced.offset == 16)
    assert(shape_get(sliced.shape, 0) == 2)
    assert(shape_get(sliced.shape, 1) == 4)

    let transposed = view_transpose(base, 0, 1)
    assert(shape_get(transposed.shape, 0) == 4)
    assert(shape_get(transposed.shape, 1) == 3)
    assert(strides_get(transposed.strides, 0) == 4)
    assert(strides_get(transposed.strides, 1) == 16)

    let reshaped = match view_reshape(base, shape2(2, 6)):
        Ok(v) => v
        Err(_) =>
            assert(false)
            base
    assert(shape_get(reshaped.shape, 0) == 2)
    assert(shape_get(reshaped.shape, 1) == 6)

    let row = view_contiguous(null_memory(), shape2(1, 4), .Float32)
    let broadcast = match view_broadcast(row, shape2(3, 4)):
        Ok(v) => v
        Err(_) =>
            assert(false)
            row
    assert(view_is_broadcasted(broadcast))
    assert(strides_get(broadcast.strides, 0) == 0)

    assert(view_offset_of(base, shape2(1, 2)) == 24)

fn main:
    test_view_smoke()
