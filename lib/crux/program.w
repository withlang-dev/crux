use c_import("stdlib.h")

use crux.core
use crux.device
use crux.errors
use crux.ir

let CPU_PROGRAM_BYTES: i32 = 32

type CPUProgram {
    base: *mut c_void,
    device: *mut Device,
}

type CPUProgramRecord {
    id: *mut Program,
    device: *mut Device,
    ir: IRProgram,
    sig: ProgramSig,
    live: bool,
}

var PROGRAMS: Vec[CPUProgramRecord] = Vec.new()

pub fn program_source(entry: str) -> ProgramSource:
    ProgramSource {
        ir: Vec.new(),
        aux: Vec.new(),
        strings: Vec.new(),
        entry,
    }

fn empty_sig -> ProgramSig:
    ProgramSig {
        params: Vec.new(),
    }

fn find_program_slot(id: *mut Program) -> i32:
    for i in 0..PROGRAMS.len():
        let rec = PROGRAMS[i]
        if rec.live and rec.id == id:
            return i
    -1

fn source_ir(source: ProgramSource) -> IRProgram:
    IRProgram {
        insts: source.ir,
        aux: source.aux,
    }

fn string_at(strings: Vec[str], index: i32) -> Result[str, SubstrateError]:
    if index < 0 or index >= strings.len() as i32:
        return Err(.CompileError("string pool index out of range"))
    Ok(strings[index])

fn has_name(names: Vec[str], name: str) -> bool:
    for i in 0..names.len():
        if names[i] == name:
            return true
    false

fn build_sig(source: ProgramSource) -> Result[ProgramSig, SubstrateError]:
    let params: Vec[ParamDesc] = Vec.new()
    let spec_names: Vec[str] = Vec.new()
    var phase: i32 = 0
    for ip in 0..source.ir.len():
        let inst = source.ir[ip]
        if inst.op == IROP_SPEC_CONSTANT:
            if phase != 0:
                return Err(.CompileError("spec constants must precede params and compute"))
            let name = string_at(source.strings, inst.d0)?
            if has_name(spec_names, name):
                return Err(.CompileError("duplicate spec constant name"))
            spec_names.push(name)
            continue
        if inst.op == IROP_PARAM:
            if phase == 2:
                return Err(.CompileError("param headers must precede compute"))
            phase = 1
            let name = string_at(source.strings, inst.d0)?
            let mode = ir_param_mode_from_code(inst.d1)?
            if inst.d2 < 0 or inst.d2 > MAX_RANK:
                return Err(.CompileError("param rank is invalid"))
            if has_name(spec_names, name):
                return Err(.CompileError("param name collides with spec constant"))
            var duplicate = false
            for pi in 0..params.len():
                if params[pi].name == name:
                    duplicate = true
                    break
            if duplicate:
                return Err(.CompileError("duplicate param name"))
            params.push(ParamDesc {
                name,
                mode,
                rank: inst.d2,
                dtype: inst.dtype,
            })
            continue
        phase = 2
    Ok(ProgramSig { params })

fn compatible_compile_info(live: DeviceInfo, requested: DeviceInfo) -> bool:
    live.kind == requested.kind and live.subgroup_size == requested.subgroup_size and live.max_shared_memory == requested.max_shared_memory and live.max_workgroup_size == requested.max_workgroup_size

fn resolve_compile_device(info: DeviceInfo) -> Result[*mut Device, SubstrateError]:
    let dev = default_device()
    if dev == null:
        return Err(.CompileError("no device is available for compilation"))
    let live = device_info(dev)
    if not compatible_compile_info(live, info):
        return Err(.CompileError("no compatible live device for DeviceInfo"))
    Ok(dev)

pub fn compile(device_info: DeviceInfo, source: ProgramSource) -> Result[*mut Program, SubstrateError]:
    let actual_device = resolve_compile_device(device_info)?
    let ir = source_ir(source)
    let sig = build_sig(source)?
    let _ = validate_ir(ir, sig)?
    let raw_opt = malloc(CPU_PROGRAM_BYTES)
    if raw_opt == None:
        return Err(.OutOfMemory)
    let raw = raw_opt.unwrap()
    let prog = raw as *mut CPUProgram
    unsafe:
        (*prog).base = raw
        (*prog).device = actual_device
    let id = prog as *mut Program
    PROGRAMS.push(CPUProgramRecord {
        id,
        device: actual_device,
        ir,
        sig,
        live: true,
    })
    Ok(id)

pub fn program_sig(prog: *mut Program) -> ProgramSig:
    let slot = find_program_slot(prog)
    if slot < 0:
        return empty_sig()
    PROGRAMS[slot].sig

pub fn program_ir(prog: *mut Program) -> Result[IRProgram, SubstrateError]:
    let slot = find_program_slot(prog)
    if slot < 0:
        return Err(.CompileError("unknown program handle"))
    Ok(PROGRAMS[slot].ir)

pub fn program_device(prog: *mut Program) -> Result[*mut Device, SubstrateError]:
    let slot = find_program_slot(prog)
    if slot < 0:
        return Err(.CompileError("unknown program handle"))
    Ok(PROGRAMS[slot].device)

pub fn program_destroy(prog: *mut Program):
    if prog == null:
        return
    let slot = find_program_slot(prog)
    if slot >= 0:
        let rec = PROGRAMS[slot]
        PROGRAMS[slot] = { rec with live: false }
    let raw = unsafe: (*(prog as *mut CPUProgram)).base
    if raw != null:
        let _ = realloc(raw, 0)
