use c_import("stdlib.h")

use crux.core

let CPU_DEVICE_BYTES: i32 = 32

type CPUDevice {
    base: *mut c_void,
}

var default_cpu_device: *mut Device = null

fn ensure_default_device -> *mut Device:
    if default_cpu_device != null:
        return default_cpu_device
    let raw_opt = malloc(CPU_DEVICE_BYTES)
    if raw_opt == None:
        return null
    let raw = raw_opt.unwrap()
    let dev = raw as *mut CPUDevice
    unsafe:
        (*dev).base = raw
    default_cpu_device = dev as *mut Device
    default_cpu_device

pub fn devices() -> Vec[*mut Device]:
    let out: Vec[*mut Device] = Vec.new()
    let dev = default_device()
    if dev != null:
        out.push(dev)
    out

pub fn default_device() -> *mut Device:
    ensure_default_device()

pub fn device_info(device: *mut Device) -> DeviceInfo:
    let _ = device
    DeviceInfo {
        name: "cpu",
        kind: .CPU,
        memory_total: 0usize,
        memory_available: 0usize,
        max_workgroup_size: 1024,
        max_grid_dims: dim3(65535usize, 65535usize, 65535usize),
        max_shared_memory: 32768usize,
        memory_alignment: 64usize,
        memory_bandwidth_gbps: 0.0,
        preferred_vector_width: 1,
        subgroup_size: 1,
        unified_memory: true,
    }
