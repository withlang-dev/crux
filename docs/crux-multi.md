# Crux Multi-Device Extensions

## Design Document v0.2

---

## Overview

This spec defines multi-device extensions to Crux: device groups, resource placement, cross-device transfers, and transport discovery. These additions are **compute-general** — they serve ML training, simulation, and rendering equally. Any program using Crux on multiple GPUs benefits.

### Design Principles

1. **Discover, don't assume.** The substrate proves each transfer path works before using it.
2. **Measure, don't estimate.** Bandwidth and latency are measured at init, not hardcoded.
3. **Explicit ownership.** Every byte on every device has a known owner and a known generation.
4. **Transfers are work.** Cross-device data movement is a first-class scheduled operation, not a side-effect.
5. **Single-device is the degenerate case.** All multi-device abstractions collapse to zero overhead on one GPU.

---

## 1. Existing Crux Primitives (unchanged)

| Primitive | Role |
|-----------|------|
| `Device`  | One physical GPU. Capabilities, queues, limits. |
| `Memory`  | Owned allocation on a specific device. |
| `View`    | Typed window into Memory. Metadata only, no ownership. |
| `Program` | Compiled kernel. |
| `Stream`  | Ordered command sequence on one device. |
| `Event`   | Synchronization token across streams. |

These are not modified. Everything below extends Crux alongside them.

---

## 2. DeviceGroup

A set of Devices that Crux manages together. Not a "virtual GPU" — a real, queryable topology.

```
DeviceGroup {
    devices:        []Device
    display_owner:  ?Device          // owns swapchain (null if headless / compute-only)
    transfer_edges: []TransferEdge   // bandwidth/latency between pairs
    topology:       TopologyInfo     // physical interconnect structure
}
```

The DeviceGroup is the first thing created. Everything else is relative to it. A single-device DeviceGroup is the degenerate case and works identically — no special paths.

---

## 3. TransferEdge and Transport Types

```
TransferEdge {
    src:            Device
    dst:            Device
    bandwidth:      u64              // bytes/sec, measured at init
    latency_ns:     u64              // one-way latency, measured at init
    mechanism:      TransferKind
    bidirectional:  bool             // true if full bandwidth in both directions simultaneously
    validated:      bool             // true if probe transfer succeeded
}

TransferKind =
    | DirectP2P {                    // GPU writes directly to peer VRAM
        method: P2PMethod
    }
    | StagedBounce {                 // GPU → system RAM → GPU
        staging_pool: Memory         // pre-allocated host-visible staging
    }
    | SharedMemory {                 // integrated GPUs sharing system memory
    }

P2PMethod =
    | PCIeBAR1       // direct BAR1 mapping (NVIDIA w/ patch, or native)
    | NVLink         // NVIDIA NVLink
    | XGMI           // AMD Infinity Fabric / XGMI
    | PCIeNative     // kernel-mediated DMA-BUF P2P
```

---

## 4. Resource

Extends Crux's `Memory` for multi-device use. A Resource is an allocation with **placement policy**, **replica tracking**, and **generation counting**.

A single-device Resource degenerates to a normal Memory allocation. The abstraction costs nothing until you use multiple devices.

```
Resource {
    id:         ResourceId
    size:       u64
    usage:      ResourceUsage     // flags: storage, transfer_src, transfer_dst,
                                  //        plus opaque extension flags
                                  //        (e.g. CruxGFX adds: vertex, index, uniform,
                                  //         texture, render_target, accel_structure)
    placement:  Placement
    canonical:  Device            // authoritative owner
    replicas:   []Replica         // copies on other devices
    generation: u64               // incremented on write
}

Placement =
    | DeviceLocal { device: Device }
    | Replicated  { devices: []Device, consistency: Consistency }
    | Sharded     { chunks: []ShardChunk }

Consistency =
    | Eager       // transfer immediately after write
    | Lazy        // transfer before next read on stale device
    | Manual      // user/runtime controls when

Replica {
    device:     Device
    memory:     Memory
    generation: u64    // last known generation on this device
}

ResidencyState =
    | NotPresent                // no allocation on this device
    | Present { generation }    // allocated, data at this generation
    | Stale { generation }      // allocated, data outdated
```

**Key rule:** A Resource always has one canonical device. Replicas are explicitly tracked. Staleness is a number, not a guess.

**Relationship to Memory:** Resource wraps Memory. Each replica contains a Memory allocation on its device. Existing Crux code that works with raw Memory continues to work — Resource is an addition, not a replacement.

---

## 5. TransferOp

First-class scheduled work, not a side-effect. These are Crux operations — they move bytes between devices, with no knowledge of what those bytes represent.

```
TransferOp =
    | Copy        { src: Resource, dst: Resource, region: ByteRegion }
    | Replicate   { resource: Resource, dst_device: Device }
    | Gather      { sources: []ResourceRegion, dst: Resource }
    | Scatter     { src: Resource, destinations: []ResourceRegion }
```

Transfers sit in execution graphs as real nodes with real cost. Higher layers (e.g. CruxGFX) may define additional transfer types, but these four cover all compute-general use cases.

---

## 6. Transport Discovery

The DeviceGroup's `transfer_edges` are **discovered and measured** at init time. The substrate does not assume any transfer path exists. It proves each one works.

### 6.1 Why Discovery Is Necessary

GPU peer-to-peer transfer is not a portable capability. What actually works depends on:

**NVIDIA:**
- Consumer/workstation GPUs (GeForce, RTX Pro): P2P over PCIe is disabled by default in the driver. Enabling BAR1 P2P requires patching the open-source kernel module.
- NVLink-connected GPUs: P2P works natively, much higher bandwidth.
- Data center GPUs (A100, H100): P2P typically works out of the box.

**AMD:**
- CDNA/Instinct (MI250, MI300): P2P works natively via XGMI or PCIe. The amdgpu kernel driver exposes this through standard DMA-BUF interfaces.
- RDNA (gaming/pro): P2P support varies by kernel version and firmware.
- Resizable BAR (Smart Access Memory): Must be enabled in BIOS for full VRAM exposure.

**Cross-vendor:**
- P2P between NVIDIA and AMD GPUs does not work. Transfer must go through system memory.

The substrate cannot know any of this in advance. It discovers it.

### 6.2 Discovery Procedure

At `DeviceGroup` creation, before any work is scheduled:

```
discover_topology(devices: []Device) -> []TransferEdge:

    edges = []

    for each pair (src, dst) in devices × devices where src ≠ dst:

        // Phase 1: Capability check
        //   Query driver/API for P2P support indication.
        //   On Vulkan: check device group properties, peer memory features.
        //   On kernel level: check DMA-BUF, NVIDIA UVM, AMD KFD topology.
        //   This is a hint only — may be wrong in either direction.

        capability = query_p2p_capability(src, dst)

        // Phase 2: Probe transfer
        //   Allocate small buffer on src (4 MB).
        //   Attempt direct P2P copy to dst.
        //   If it succeeds: direct P2P is real.
        //   If it fails or hangs: fall back to staged.

        probe_result = attempt_p2p_transfer(src, dst, probe_size: 4MB, timeout: 500ms)

        if probe_result == .success:
            method = detect_p2p_method(src, dst)

            // Phase 3: Bandwidth measurement
            (bandwidth, latency) = measure_transfer(src, dst, sizes: [1MB, 16MB, 64MB, 256MB])

            edges.append(TransferEdge {
                src, dst,
                bandwidth,
                latency_ns: latency,
                mechanism:  .DirectP2P { method },
                bidirectional: measure_bidirectional(src, dst),
                validated: true,
            })

        else:
            // Phase 4: Staged fallback
            staging = allocate_staging(max(src.vram, dst.vram) / 64)
            (bandwidth, latency) = measure_staged_transfer(src, dst, staging)

            edges.append(TransferEdge {
                src, dst,
                bandwidth,
                latency_ns: latency,
                mechanism:  .StagedBounce { staging_pool: staging },
                bidirectional: true,
                validated: true,
            })

    return edges
```

### 6.3 PCIe Topology Detection

```
TopologyInfo {
    numa_node:       map[Device]u32       // which CPU socket
    pcie_bus:        map[Device]PCIeAddr  // bus/device/function
    switch_groups:   [][]Device           // GPUs behind same PCIe switch
    bar_sizes:       map[Device]BarInfo   // BAR1 size (full VRAM or windowed?)
}

BarInfo {
    bar1_size:       u64      // exposed BAR1 size in bytes
    vram_total:      u64      // actual VRAM size
    resizable_bar:   bool     // is rebar enabled?
    full_exposure:   bool     // bar1_size >= vram_total?
}
```

**Why BAR size matters:** If resizable BAR is not enabled, BAR1 may be only 256MB while VRAM is 96GB. P2P still works but requires remapping overhead for large transfers.

**Why NUMA matters:** On multi-socket workstations, staged transfers through the wrong CPU's memory controller add inter-socket latency.

**Why switch groups matter:** GPUs behind the same PCIe switch transfer without traversing the root complex. Lower latency, sometimes higher bandwidth.

### 6.4 Vendor-Specific Enablement

The one place where vendor differences are allowed to exist.

```
VendorInit =
    | NVIDIA {
        // Check if BAR1 P2P functions are real or stubs.
        // If stubs: log warning, P2P falls back to staged.
        // If patched driver: P2P probe will succeed naturally.
        // Check NVLink topology via NVML or sysfs.
    }
    | AMD {
        // Check amdgpu driver version.
        // Verify resizable BAR / Smart Access Memory enabled.
        // Query KFD topology for XGMI links.
        // Check DMA-BUF P2P support in kernel.
    }
    | Other {
        // Unknown vendor. Probe-only. Staged bounce expected.
    }
```

This code does not change Crux's abstractions. It increases the probability that probing succeeds.

### 6.5 Runtime Re-probing

Transport quality can change during operation (thermal throttling, driver bugs, power management). Crux supports periodic re-measurement:

```
DeviceGroup.reprobe(edge: TransferEdge) -> TransferEdge
    // Re-measures bandwidth and latency.
    // If a previously-working P2P path now fails,
    // downgrades to StagedBounce and logs a warning.
```

### 6.6 Diagnostic Logging

At init, Crux logs the full discovered topology:

```
[Crux] Device topology:
[Crux]   GPU 0: NVIDIA RTX PRO 6000 (96GB, PCIe 5.0 x16, BAR1 96GB, NUMA 0)
[Crux]   GPU 1: NVIDIA RTX PRO 6000 (96GB, PCIe 5.0 x16, BAR1 96GB, NUMA 0)
[Crux]   GPU 2: NVIDIA RTX PRO 6000 (96GB, PCIe 5.0 x16, BAR1 96GB, NUMA 1)
[Crux]   GPU 3: NVIDIA RTX PRO 6000 (96GB, PCIe 5.0 x16, BAR1 96GB, NUMA 1)
[Crux] Transfer edges:
[Crux]   GPU 0 → GPU 1: DirectP2P/BAR1, 63.2 GB/s, 1.4 µs latency (same switch)
[Crux]   GPU 0 → GPU 2: DirectP2P/BAR1, 51.7 GB/s, 2.1 µs latency (cross-socket)
[Crux]   GPU 2 → GPU 3: DirectP2P/BAR1, 63.4 GB/s, 1.3 µs latency (same switch)
[Crux]   ...
```

If P2P is not available:

```
[Crux]   GPU 0 → GPU 1: StagedBounce, 24.3 GB/s, 8.7 µs latency
[Crux]   WARNING: Direct P2P not available (GPU 0 → GPU 1).
[Crux]     NVIDIA driver may need BAR1 P2P patch.
[Crux]     Check: resizable BAR enabled in BIOS?
```

---

## 7. Memory Budgets

Each device in a DeviceGroup has a tracked memory budget. Crux enforces this.

```
DeviceBudget {
    device:          Device
    total_vram:      u64           // 96 GB per RTX PRO 6000
    reserved:        u64           // for system, swapchain, etc.
    persistent:      u64           // long-lived resources
    transient:       u64           // per-frame resources
    transfer_buffer: u64           // staging for copies
    available:       u64           // = total - reserved - persistent - transient - transfer
}
```

When a device is full, stale replicas are evicted first. If still full, work cannot be placed there.

---

## Summary

| Addition | Purpose |
|----------|---------|
| `DeviceGroup` | Multi-device topology. Hardware truth. |
| `TransferEdge` / `TransferKind` / `P2PMethod` | Transport characterization. Measured, not guessed. |
| `Resource` with `Placement` / `Replica` / `generation` | Multi-device data management. Wraps `Memory`. |
| `TransferOp` | Cross-device byte movement. First-class scheduled work. |
| Transport Discovery | Hardware capability detection. Probe, measure, fallback. |
| `TopologyInfo` / `BarInfo` | Physical interconnect structure. |
| `DeviceBudget` | Memory accounting per device. |
| `DeviceGroup.reprobe()` | Runtime transport adaptation. |

---

## Implementation Plan

**Months 1–4:**

- DeviceGroup with topology.
- Resource with placement, replicas, generation counters.
- TransferOp as first-class scheduled work.
- Transport discovery: vendor init, pairwise P2P probe, bandwidth measurement, fallback.
- PCIe topology detection (NUMA, switch groups, BAR sizes).
- Transfer engine: DirectP2P and StagedBounce paths.
- Cross-device Event sync.
- DeviceBudget tracking.
- Runtime re-probing.
- Diagnostic topology logging.

**Verification milestones:**

- Can dispatch compute on GPU 1, transfer result to GPU 0.
- Topology log accurately reflects real hardware capabilities.
- ML/compute workloads can use multi-device Resources without any graphics dependency.

---

## What Success Looks Like

A distributed compute job across 4 GPUs uses Crux's multi-device primitives:
- DeviceGroup discovers topology and measures real P2P bandwidth.
- Resources placed and replicated across devices with generation tracking.
- TransferOps scheduled as first-class work with measured cost.
- No graphics dependency needed.