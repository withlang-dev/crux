use c_import("stdlib.h")

use crux.core
use crux.device
use crux.errors
use crux.ir
use crux.ir_text

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

pub type ProgramSource {
    ir: IRProgram,
    ir_text: str,
    entry: str,
    spec_constants: Vec[ConstantDesc],
}

pub fn program_source(entry: str) -> ProgramSource:
    ProgramSource {
        ir: ir_program(),
        ir_text: "",
        entry,
        spec_constants: Vec.new(),
    }

fn empty_sig -> ProgramSig:
    ProgramSig {
        params: Vec.new(),
        constants: Vec.new(),
    }

fn find_program_slot(id: *mut Program) -> i32:
    for i in 0..PROGRAMS.len():
        let rec = PROGRAMS[i]
        if rec.live and rec.id == id:
            return i
    -1

fn source_ir(source: ProgramSource) -> Result[IRProgram, SubstrateError]:
    let direct = source.ir
    if direct.insts.len() > 0 or direct.num_params > 0:
        return Ok(direct)
    if source.ir_text != "":
        return match parse_ir_text(source.ir_text)
            Ok(ir) => Ok(ir)
            Err(.ParseError(msg)) => Err(.CompileError(msg))
    Ok(direct)

fn build_sig(ir: IRProgram) -> Result[ProgramSig, SubstrateError]:
    let count = ir.num_params
    if ir.param_names.len() as i32 != count:
        return Err(.CompileError("ir param_names length mismatch"))
    if ir.param_modes.len() as i32 != count:
        return Err(.CompileError("ir param_modes length mismatch"))
    if ir.param_ranks.len() as i32 != count:
        return Err(.CompileError("ir param_ranks length mismatch"))
    if ir.param_dtypes.len() as i32 != count:
        return Err(.CompileError("ir param_dtypes length mismatch"))
    let params: Vec[ParamDesc] = Vec.new()
    for i in 0..count:
        params.push(ParamDesc {
            name: ir.param_names[i],
            mode: ir.param_modes[i],
            rank: ir.param_ranks[i],
            dtype: ir.param_dtypes[i],
        })
    Ok(ProgramSig {
        params,
        constants: Vec.new(),
    })

pub fn compile(device: *mut Device, source: ProgramSource) -> Result[*mut Program, SubstrateError]:
    let actual_device = if device == null then default_device() else device
    let ir = source_ir(source)?
    let sig = build_sig(ir)?
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
