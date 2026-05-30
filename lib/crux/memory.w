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
    borrowed: bool,
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

fn alloc_meta -> Result[*mut CPUMemory, SubstrateError]:
    let meta_opt = malloc(CPU_MEMORY_BYTES)
    if meta_opt == None:
        return Err(.OutOfMemory)
    Ok(meta_opt.unwrap() as *mut CPUMemory)

pub fn alloc(device: *mut Device, size: Size) -> Result[*mut Memory, SubstrateError]:
    let actual_device = if device == null then default_device() else device
    let meta = alloc_meta()?
    let requested = if size == 0usize then 1usize else size
    let data_opt = malloc(requested)
    if data_opt == None:
        cpu_release(meta as *mut c_void)
        return Err(.OutOfMemory)
    let data_raw = data_opt.unwrap()
    unsafe:
        (*meta).base = meta as *mut c_void
        (*meta).data = data_raw
        (*meta).size = size
        (*meta).device = actual_device
        (*meta).owner_arena = null
        (*meta).borrowed = false
    Ok(meta as *mut Memory)

pub fn memory_from_ptr(device: *mut Device, ptr: *mut u8, size: Size) -> *mut Memory:
    if ptr == null:
        return null_memory()
    let actual_device = if device == null then default_device() else device
    let meta = match alloc_meta():
        Ok(v) => v
        Err(_) => return null_memory()
    unsafe:
        (*meta).base = meta as *mut c_void
        (*meta).data = ptr as *mut c_void
        (*meta).size = size
        (*meta).device = actual_device
        (*meta).owner_arena = null
        (*meta).borrowed = true
    meta as *mut Memory

pub fn free(mem: *mut Memory):
    if mem == null:
        return
    let meta = mem as *mut CPUMemory
    let owner_arena = unsafe meta.owner_arena
    let data = unsafe meta.data
    let base = unsafe meta.base
    let borrowed = unsafe meta.borrowed
    if owner_arena == null and not borrowed and data != null:
        cpu_release(data)
    if base != null:
        cpu_release(base)

pub fn free_after(stream: *mut Stream, mem: *mut Memory):
    let _ = stream
    free(mem)

pub fn memory_size(mem: *mut Memory) -> Size:
    if mem == null:
        return 0usize
    let meta = mem as *mut CPUMemory
    unsafe meta.size

pub fn memory_device(mem: *mut Memory) -> *mut Device:
    if mem == null:
        return default_device()
    let meta = mem as *mut CPUMemory
    unsafe meta.device

pub fn memory_ptr(mem: *mut Memory) -> *mut u8:
    if mem == null:
        return null
    let meta = mem as *mut CPUMemory
    let data = unsafe meta.data
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
    let data = unsafe raw.data
    let base = unsafe raw.base
    if data != null:
        cpu_release(data)
    if base != null:
        cpu_release(base)

pub fn arena_alloc(arena: *mut Arena, size: Size, align: Size) -> Result[*mut Memory, SubstrateError]:
    if arena == null:
        return Err(.Unsupported("arena handle is null"))
    let raw_arena = arena as *mut CPUArena
    let start = align_up(unsafe raw_arena.used, align)
    let capacity = unsafe raw_arena.size
    if start > capacity:
        return Err(.OutOfMemory)
    if size > capacity - start:
        return Err(.OutOfMemory)
    let next = start + size
    let meta = alloc_meta()?
    let data = unsafe { ((*raw_arena).data as *mut u8 + start as i64) as *mut c_void }
    unsafe:
        (*meta).base = meta as *mut c_void
        (*meta).data = data
        (*meta).size = size
        (*meta).device = (*raw_arena).device
        (*meta).owner_arena = arena
        (*meta).borrowed = false
        (*raw_arena).used = next
    Ok(meta as *mut Memory)

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
    unsafe raw.used
