use crux.core
use crux.errors

pub fn view(mem: *mut Memory, desc: ViewDesc) -> Result[View, SubstrateError]:
    if not shape_is_valid(desc.shape):
        return Err(.InvalidView("shape rank must be between 0 and 8"))
    if not strides_is_valid(desc.strides):
        return Err(.InvalidView("stride rank must be between 0 and 8"))
    if desc.shape.rank != desc.strides.rank:
        return Err(.InvalidView("shape and stride ranks must match"))
    Ok(View {
        memory: mem,
        offset: desc.offset,
        shape: desc.shape,
        strides: desc.strides,
        dtype: desc.dtype,
    })

pub fn view_contiguous(mem: *mut Memory, shape: Shape, dtype: DType) -> View:
    View {
        memory: mem,
        offset: 0,
        shape,
        strides: contiguous_strides(shape, dtype),
        dtype,
    }

pub fn view_slice(v: View, dim: i32, start: Size, end: Size) -> View:
    let new_offset = v.offset + start * strides_get(v.strides, dim) as Size
    let new_shape = shape_set(v.shape, dim, end - start)
    View {
        memory: v.memory,
        offset: new_offset,
        shape: new_shape,
        strides: v.strides,
        dtype: v.dtype,
    }

pub fn view_transpose(v: View, dim0: i32, dim1: i32) -> View:
    let s0 = strides_get(v.strides, dim0)
    let s1 = strides_get(v.strides, dim1)
    let d0 = shape_get(v.shape, dim0)
    let d1 = shape_get(v.shape, dim1)
    let new_strides = strides_set(strides_set(v.strides, dim0, s1), dim1, s0)
    let new_shape = shape_set(shape_set(v.shape, dim0, d1), dim1, d0)
    View {
        memory: v.memory,
        offset: v.offset,
        shape: new_shape,
        strides: new_strides,
        dtype: v.dtype,
    }

pub fn view_reshape(v: View, shape: Shape) -> Result[View, SubstrateError]:
    if view_elem_count(v) != shape_elem_count(shape):
        return Err(.ShapeMismatch("reshape requires the same element count"))
    if not view_is_contiguous(v):
        return Err(.Unsupported("reshape currently requires a contiguous input view"))
    Ok(View {
        memory: v.memory,
        offset: v.offset,
        shape,
        strides: contiguous_strides(shape, v.dtype),
        dtype: v.dtype,
    })

pub fn view_broadcast(v: View, target: Shape) -> Result[View, SubstrateError]:
    if v.shape.rank != target.rank:
        return Err(.ShapeMismatch("broadcast currently requires matching ranks"))
    var out = v
    for i in 0..target.rank:
        let source_dim = shape_get(v.shape, i)
        let target_dim = shape_get(target, i)
        if source_dim == target_dim:
            continue
        if source_dim == 1 and target_dim > 1:
            out.shape = shape_set(out.shape, i, target_dim)
            out.strides = strides_set(out.strides, i, 0)
            continue
        return Err(.ShapeMismatch("broadcast dimensions are incompatible"))
    Ok(out)

pub fn view_is_contiguous(v: View) -> bool:
    strides_is_contiguous(v.strides, v.shape, v.dtype)

pub fn view_is_broadcasted(v: View) -> bool:
    strides_is_broadcasted(v.strides)

pub fn view_elem_count(v: View) -> Size:
    shape_elem_count(v.shape)

pub fn view_byte_size(v: View) -> Size:
    shape_elem_count(v.shape) * dtype_size(v.dtype)

pub fn view_offset_of(v: View, indices: Shape) -> Size:
    var offset: Size = v.offset
    for i in 0..v.shape.rank:
        offset = offset + shape_get(indices, i) * strides_get(v.strides, i) as Size
    offset
