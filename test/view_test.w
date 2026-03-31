use crux.core
use crux.view

fn test_view_contiguous_layout:
    let v = view_contiguous(null_memory(), shape2(3, 4), .Float32)
    assert(view_is_contiguous(v))
    assert(not view_is_broadcasted(v))
    assert(view_elem_count(v) == 12)
    assert(view_byte_size(v) == 48)

fn test_view_constructor_validates_rank_mismatch:
    let desc = ViewDesc {
        shape: shape2(2, 3),
        strides: strides_zero(),
        dtype: .Float32,
    }
    let got_invalid = match view(null_memory(), desc)
        Err(.InvalidView(_)) => true
        _ => false
    assert(got_invalid)

fn test_view_slice_offset:
    let v = view_contiguous(null_memory(), shape2(10, 20), .Float32)
    let sliced = view_slice(v, 0, 2, 5)
    assert(sliced.offset == 160)
    assert(shape_get(sliced.shape, 0) == 3)
    assert(shape_get(sliced.shape, 1) == 20)

fn test_view_transpose_swaps_shape_and_stride:
    let v = view_contiguous(null_memory(), shape2(3, 4), .Float32)
    let transposed = view_transpose(v, 0, 1)
    assert(shape_get(transposed.shape, 0) == 4)
    assert(shape_get(transposed.shape, 1) == 3)
    assert(strides_get(transposed.strides, 0) == 4)
    assert(strides_get(transposed.strides, 1) == 16)

fn test_view_reshape_rejects_non_contiguous:
    let v = view_contiguous(null_memory(), shape2(3, 4), .Float32)
    let transposed = view_transpose(v, 0, 1)
    let got_unsupported = match view_reshape(transposed, shape2(2, 6))
        Err(.Unsupported(_)) => true
        _ => false
    assert(got_unsupported)

fn test_view_broadcast_sets_zero_stride:
    let row = view_contiguous(null_memory(), shape2(1, 4), .Float32)
    let expanded = match view_broadcast(row, shape2(3, 4))
        Ok(v) => v
        Err(_) =>
            assert(false)
            row
    assert(view_is_broadcasted(expanded))
    assert(strides_get(expanded.strides, 0) == 0)
    assert(strides_get(expanded.strides, 1) == 4)

fn test_view_broadcast_rejects_incompatible_dims:
    let left = view_contiguous(null_memory(), shape2(2, 4), .Float32)
    let got_mismatch = match view_broadcast(left, shape2(3, 4))
        Err(.ShapeMismatch(_)) => true
        _ => false
    assert(got_mismatch)

fn test_view_offset_of_uses_byte_strides:
    let v = view_contiguous(null_memory(), shape3(2, 3, 4), .Float32)
    assert(view_offset_of(v, shape3(1, 2, 3)) == 92)

fn main:
    test_view_contiguous_layout()
    test_view_constructor_validates_rank_mismatch()
    test_view_slice_offset()
    test_view_transpose_swaps_shape_and_stride()
    test_view_reshape_rejects_non_contiguous()
    test_view_broadcast_sets_zero_stride()
    test_view_broadcast_rejects_incompatible_dims()
    test_view_offset_of_uses_byte_strides()
