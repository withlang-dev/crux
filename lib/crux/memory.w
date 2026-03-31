use c_import("stdlib.h")

use crux.core
use crux.device
use crux.errors

pub let PLACEMENT_LOCAL: i32 = 0
pub let PLACEMENT_REPLICATED: i32 = 1
pub let PLACEMENT_PARTITIONED: i32 = 2

let CPU_MEMORY_BYTES: usize = 64usize
let CPU_ARENA_BYTES: usize = 64usize

type CPUMemory {
    base: *mut c_void,
    data: *mut c_void,
    size: Size,
    device: *mut Device,
    owner_arena: *mut Arena,
}

type CPUArena {
    base: *mut c_void,
    data: *mut c_void,
    size: Size,
    used: Size,
    device: *mut Device,
}

fn cpu_release(ptr: *mut c_void):
    let _ = realloc(ptr, 0)

pub type MemoryPlacement {
    kind: i32,
    regions: Size,
}

pub fn placement_local -> MemoryPlacement:
    MemoryPlacement { kind: PLACEMENT_LOCAL, regions: 1usize }

pub fn placement_replicated -> MemoryPlacement:
    MemoryPlacement { kind: PLACEMENT_REPLICATED, regions: 0usize }

pub fn placement_partitioned(regions: Size) -> MemoryPlacement:
    MemoryPlacement { kind: PLACEMENT_PARTITIONED, regions }

pub fn alloc(device: *mut Device, size: Size) -> Result[*mut Memory, SubstrateError]:
    let actual_device = if device == null then default_device() else device
    let meta_opt = malloc(CPU_MEMORY_BYTES)
    if meta_opt == None:
        return Err(.OutOfMemory)
    let meta_raw = meta_opt.unwrap()
    let requested = if size == 0usize then 1usize else size
    let data_opt = malloc(requested)
    if data_opt == None:
        cpu_release(meta_raw)
        return Err(.OutOfMemory)
    let data_raw = data_opt.unwrap()
    let mem = meta_raw as *mut CPUMemory
    unsafe:
        (*mem).base = meta_raw
        (*mem).data = data_raw
        (*mem).size = size
        (*mem).device = actual_device
        (*mem).owner_arena = null
    Ok(mem as *mut Memory)

pub fn free(mem: *mut Memory):
    if mem == null:
        return
    let meta = mem as *mut CPUMemory
    let owner_arena = unsafe: (*meta).owner_arena
    let data = unsafe: (*meta).data
    let base = unsafe: (*meta).base
    if owner_arena == null and data != null:
        cpu_release(data)
    if base != null:
        cpu_release(base)

pub fn free_after(stream: *mut Stream, mem: *mut Memory):
    let _ = stream
    free(mem)

pub fn memory_size(mem: *mut Memory) -> Size:
    if mem == null:
        return 0
    let meta = mem as *mut CPUMemory
    unsafe: (*meta).size

pub fn memory_device(mem: *mut Memory) -> *mut Device:
    if mem == null:
        return default_device()
    let meta = mem as *mut CPUMemory
    unsafe: (*meta).device

pub fn memory_ptr(mem: *mut Memory) -> *mut u8:
    if mem == null:
        return null
    let meta = mem as *mut CPUMemory
    let data = unsafe: (*meta).data
    if data == null:
        return null
    data as *mut u8

fn align_up(value: Size, align: Size) -> Size:
    let actual = if align == 0usize then 1usize else align
    let rem = value % actual
    if rem == 0usize:
        return value
    value + (actual - rem)

pub fn arena_create(device: *mut Device, size: Size) -> Result[*mut Arena, SubstrateError]:
    let actual_device = if device == null then default_device() else device
    let meta_opt = malloc(CPU_ARENA_BYTES)
    if meta_opt == None:
        return Err(.OutOfMemory)
    let meta_raw = meta_opt.unwrap()
    let requested = if size == 0usize then 1usize else size
    let data_opt = malloc(requested)
    if data_opt == None:
        cpu_release(meta_raw)
        return Err(.OutOfMemory)
    let data_raw = data_opt.unwrap()
    let arena = meta_raw as *mut CPUArena
    unsafe:
        (*arena).base = meta_raw
        (*arena).data = data_raw
        (*arena).size = size
        (*arena).used = 0usize
        (*arena).device = actual_device
    Ok(arena as *mut Arena)

pub fn arena_destroy(arena: *mut Arena):
    if arena == null:
        return
    let raw = arena as *mut CPUArena
    let data = unsafe: (*raw).data
    let base = unsafe: (*raw).base
    if data != null:
        cpu_release(data)
    if base != null:
        cpu_release(base)

pub fn arena_alloc(arena: *mut Arena, size: Size, align: Size) -> Result[*mut Memory, SubstrateError]:
    if arena == null:
        return Err(.Unsupported("arena handle is null"))
    let raw_arena = arena as *mut CPUArena
    let start = align_up(unsafe: (*raw_arena).used, align)
    let next = start + size
    if next > unsafe: (*raw_arena).size:
        return Err(.OutOfMemory)
    let meta_opt = malloc(CPU_MEMORY_BYTES)
    if meta_opt == None:
        return Err(.OutOfMemory)
    let meta_raw = meta_opt.unwrap()
    let data = unsafe: ((*raw_arena).data as *mut u8 + start as i64) as *mut c_void
    let mem = meta_raw as *mut CPUMemory
    unsafe:
        (*mem).base = meta_raw
        (*mem).data = data
        (*mem).size = size
        (*mem).device = (*raw_arena).device
        (*mem).owner_arena = arena
        (*raw_arena).used = next
    Ok(mem as *mut Memory)

pub fn arena_reset(arena: *mut Arena):
    if arena == null:
        return
    let raw = arena as *mut CPUArena
    unsafe:
        (*raw).used = 0usize

pub fn arena_used(arena: *mut Arena) -> Size:
    if arena == null:
        return 0usize
    let raw = arena as *mut CPUArena
    unsafe: (*raw).used
