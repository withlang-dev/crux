# Multi-GPU Execution System

## Design Document v0.2

---

## Overview

This system has two layers and two goals.

**Layer 1 — Crux Extensions:** Multi-device primitives added to Crux itself. Device groups, resource placement, cross-device transfers, transport discovery. These are compute-general — they serve ML training, simulation, and rendering equally. Any program using Crux on multiple GPUs benefits.

**Layer 2 — CruxGFX:** A graphics execution layer built on top of Crux. Pipelines, render targets, frame execution graphs, scheduling, composition, and a Vulkan compatibility translator. This is where rendering lives.

Two goals, served by both layers:

**Goal A — Transparent Mode:** Unmodified Vulkan games gain performance from multiple GPUs without code changes. (OpenGL support may be added later if demand warrants it.)

**Goal B — Explicit Mode:** Games and engines written against CruxGFX's native API get full, principled multi-GPU control.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Applications                          │
│                                                            │
│   ┌──────────────────┐            ┌──────────────────────┐ │
│   │ Unmodified       │            │ Native With Engine    │ │
│   │ Vulkan Game      │            │ (explicit multi-GPU)  │ │
│   └────────┬─────────┘            └──────────┬───────────┘ │
│            │                                 │             │
├────────────┼─────────────────────────────────┼─────────────┤
│            ▼                                 ▼             │
│   ┌──────────────────┐            ┌──────────────────────┐ │
│   │ Vulkan Compat    │            │ Native Front-End     │ │
│   │ Translator       │            │ (CruxGFX API)        │ │
│   │ (thin intercept) │            │                      │ │
│   └────────┬─────────┘            └──────────┬───────────┘ │
│            │                                 │             │
│            └──────────┬──────────────────────┘             │
│                       ▼                                    │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │              C r u x G F X                           │  │
│  │                                                      │  │
│  │  ExecGraph · Scheduler · Pipeline · RenderTarget     │  │
│  │  Compositor · Work Classifier · Offload Engine       │  │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
│                            │                               │
│  ┌─────────────────────────▼─────────────────────────────┐ │
│  │                    C r u x                             │ │
│  │                                                       │ │
│  │  Device · Memory · View · Program · Stream · Event    │ │
│  │  DeviceGroup · Resource · TransferOp                  │ │
│  │  Transport Discovery · Placement · Replicas           │ │
│  └──────┬────────┬────────┬────────┬─────────────────────┘ │
│         │        │        │        │                        │
│         ▼        ▼        ▼        ▼                        │
│      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                      │
│      │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│                       │
│      │(disp│ │     │ │     │ │     │                       │
│      │owner│ │     │ │     │ │     │                       │
│      └─────┘ └─────┘ └─────┘ └─────┘                      │
│         via Vulkan (primary) / Metal (Apple)               │
└────────────────────────────────────────────────────────────┘
```

---

# PART I — CRUX EXTENSIONS

Everything in this section becomes part of Crux. It is compute-general. A user doing distributed ML training across 4 GPUs uses these same primitives with no graphics involvement.

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
                                  //        plus opaque vendor-specific flags
                                  //        (CruxGFX adds: vertex, index, uniform,
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

Transfers sit in execution graphs as real nodes with real cost. CruxGFX adds a `Composite` transfer type for image composition, but the base types above are compute-general.

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

## Crux Extensions — Summary

What goes into Crux:

| Addition | Why Crux (not graphics layer) |
|----------|------|
| `DeviceGroup` | Multi-device topology is hardware truth. ML training needs it. |
| `TransferEdge` / `TransferKind` / `P2PMethod` | Transport characterization is compute-general. |
| `Resource` with `Placement` / `Replica` / `generation` | Multi-device data management. Wraps `Memory`. |
| `TransferOp` | Cross-device byte movement. No knowledge of what bytes represent. |
| Transport Discovery (probe, measure, vendor init) | Hardware capability detection. |
| `TopologyInfo` / `BarInfo` | Physical interconnect structure. |
| `DeviceBudget` | Memory accounting per device. |
| `DeviceGroup.reprobe()` | Runtime transport adaptation. |

What stays **out** of Crux (belongs in CruxGFX):

| Concept | Why NOT Crux |
|---------|------|
| Pipeline (graphics/RT) | Rasterization-specific. |
| RenderTarget | Framebuffer concept. |
| ExecGraph with Draw/TraceRays/Present | Rendering frame structure. |
| Scheduler (placement strategies) | Rendering-aware scheduling. |
| Compositor | Image assembly for presentation. |
| Vulkan translator | API compatibility layer. |
| Work classifier / offload engine | Rendering heuristics. |

---

# PART II — CruxGFX

Everything in this section is the graphics layer built on top of Crux. It uses Crux's multi-device primitives but adds rendering-specific concepts.

---

## 8. Pipeline

Full compiled GPU pipeline state. Graphics-specific. Uses Crux `Program` for shader stages.

```
Pipeline =
    | GraphicsPipeline {
        vertex:       Program
        fragment:     Program
        tessellation: ?TessPrograms
        geometry:     ?Program
        mesh:         ?MeshPrograms
        raster_state: RasterState
        depth_state:  DepthState
        blend_state:  BlendState
        vertex_layout: VertexLayout
        render_target_formats: []Format
    }
    | ComputePipeline {
        program: Program
    }
    | RayTracingPipeline {
        raygen:       Program
        miss:         []Program
        closest_hit:  []Program
        any_hit:      []Program
        intersection: []Program
        max_recursion: u32
    }
```

No hidden state. Immutable after creation. Cacheable.

Note: `ComputePipeline` is here rather than in Crux because Crux dispatches Programs directly via Streams. CruxGFX wraps them in Pipelines for uniform scheduling within the ExecGraph.

---

## 9. RenderTarget

A set of attachments that can be drawn to.

```
RenderTarget {
    color:   []Attachment
    depth:   ?Attachment
    stencil: ?Attachment
    width:   u32
    height:  u32
    layers:  u32
    samples: u32
}

Attachment {
    view:       View              // Crux View into a Resource
    load_op:    LoadOp            // clear, load, dont_care
    store_op:   StoreOp           // store, dont_care
    clear_val:  ?ClearValue
}
```

---

## 10. Execution Graph

The central data structure. Everything the GPU does in a frame is a node in a directed acyclic graph.

```
ExecNode =
    | Draw {
        pipeline:      GraphicsPipeline
        target:        RenderTarget
        bindings:      []Binding
        draw_call:     DrawCall
        device_hint:   ?Device         // null = scheduler decides
    }
    | Dispatch {
        pipeline:      ComputePipeline
        bindings:      []Binding
        group_count:   [3]u32
        device_hint:   ?Device
    }
    | TraceRays {
        pipeline:      RayTracingPipeline
        bindings:      []Binding
        dimensions:    [3]u32
        device_hint:   ?Device
    }
    | Transfer {
        op:            TransferOp      // Crux TransferOp
    }
    | Composite {
        layers:        []CompositeLayer
        dst:           RenderTarget
    }
    | Present {
        source:        RenderTarget
        swapchain:     Swapchain
        // always on display_owner device
    }
    | Barrier {
        scope:         BarrierScope
    }

ExecEdge {
    from:       ExecNode
    to:         ExecNode
    resource:   Resource         // the Crux Resource creating the dependency
    access:     AccessPair       // (write→read, write→write, etc.)
}

ExecGraph {
    nodes:      []ExecNode
    edges:      []ExecEdge
    frame_id:   u64
}
```

**Both front-ends produce ExecGraph instances.** This is the contract.

---

## 11. Scheduler

The scheduler takes an ExecGraph plus a Crux DeviceGroup and produces a **DevicePlan**: concrete assignments of nodes to devices, insertion of transfers, and synchronization.

### 11.1 Scheduling Pipeline

```
ExecGraph
    │
    ▼
┌──────────────────┐
│  Dependency       │   Validates the DAG. Detects hazards.
│  Analysis         │   Computes critical path.
│                   │   Identifies parallelism opportunities.
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Placement        │   Decides which device runs each node.
│  Assignment       │   Respects device_hints (native) or
│                   │   heuristics (compat).
│                   │   Accounts for resource residency.
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Transfer         │   Inserts Crux TransferOp nodes where
│  Insertion        │   a node's inputs are stale or absent
│                   │   on the assigned device.
│                   │   Updates replica state.
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Sync             │   Inserts Crux Events between nodes
│  Insertion        │   on different devices/streams.
│                   │   Minimizes wait points.
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Stream           │   Maps nodes to Crux Streams.
│  Assignment       │   One or more streams per device.
│                   │   Respects queue capabilities.
└────────┬─────────┘
         ▼
DevicePlan {
    per_device:  map[Device][]StreamPlan
    transfers:   []ScheduledTransfer
    sync_points: []ScheduledSync
    present:     PresentPlan
}
```

### 11.2 Placement Strategies

#### Strategy: DisplayOwnerPrimary (default for compat)

- All work defaults to GPU 0 (display owner).
- Work is offloaded only when the scheduler proves it is safe and beneficial.
- Conservative. Correct first.

#### Strategy: PipelineDecomposition (default for native)

- Assign work by function:
  - GPU 0: rasterization + present
  - GPU 1: ray tracing
  - GPU 2: compute/denoising/upscaling
  - GPU 3: simulation/streaming/procedural
- Developer can override with `device_hint`.

#### Strategy: SceneDecomposition

- Partition scene spatially or by content type.
- Each GPU owns a region or content class.
- Compositor merges on GPU 0.

#### Strategy: Hybrid

- Combine pipeline and scene decomposition.
- Scheduler evaluates per-frame cost models and adjusts.

### 11.3 Cost Model

Transfer costs are **derived from Crux's measured topology**, not estimated or hardcoded.

```
CostModel {
    // Per-node estimates (updated from profiling)
    node_cost:      map[ExecNode]Duration

    // Transfer costs (derived from DeviceGroup.transfer_edges)
    // Uses measured bandwidth and latency from transport discovery.
    //   DirectP2P:    cost = latency + (bytes / bandwidth)
    //   StagedBounce: cost = 2 * latency + (bytes / bandwidth)
    // Also accounts for current transfer queue saturation.
    transfer_cost:  fn(src: Device, dst: Device, bytes: u64) -> Duration

    // Device load (running total per frame)
    device_load:    map[Device]Duration

    // Transfer link saturation (running total per frame)
    link_load:      map[TransferEdge]Duration
}
```

The difference between DirectP2P and StagedBounce is often the difference between "offloading this shadow pass saves 3ms" and "offloading costs 1ms net." Crux's transport discovery determines which world you're in.

### 11.4 Profiling Feedback Loop

```
Frame N executes
    │
    ▼
Timing queries per node (Crux Stream timing)
    │
    ▼
CostModel updated
    │
    ▼
Frame N+1 scheduling uses updated costs
```

The scheduler learns the actual workload. First few frames are conservative. As profiling data accumulates, the scheduler offloads more aggressively.

---

## 12. Native Front-End (CruxGFX API)

This is the API that With game engines use directly. Explicit, multi-GPU-aware, closely mirrors the substrate.

### 12.1 Design Principles

1. **No lies.** Every GPU is visible. Every transfer is explicit or explicitly delegated.
2. **Hints, not commands.** `device_hint` suggests placement. The scheduler can override. `device_require` forces it.
3. **Frame graph is the API.** The developer builds an ExecGraph per frame. The runtime does the rest.
4. **Single-GPU is a trivial case.** A 1-GPU DeviceGroup works identically.

### 12.2 Frame Lifecycle

```with
frame := gfx.begin_frame(device_group, swapchain)

// Declare resources
gbuffer_color := frame.transient_resource(width * height * 4, .render_target | .texture)
gbuffer_depth := frame.transient_resource(width * height * 4, .render_target | .texture)
shadow_map    := frame.transient_resource(2048 * 2048 * 4,   .render_target | .texture)
gi_result     := frame.transient_resource(width * height * 4, .texture | .storage)

// Build execution graph
frame.pass("shadow", device_hint: gpu1) {
    .draw(shadow_pipeline, shadow_target, scene_geometry)
}

frame.pass("gbuffer", device_hint: gpu0) {
    .draw(gbuffer_pipeline, gbuffer_target, scene_geometry)
}

frame.pass("gi_trace", device_hint: gpu2) {
    .trace_rays(gi_pipeline, gi_bindings, width, height, 1)
}

frame.pass("lighting", device_hint: gpu0) {
    .reads(gbuffer_color, shadow_map, gi_result)
    .draw(lighting_pipeline, hdr_target, fullscreen_quad)
}

frame.pass("denoise", device_hint: gpu3) {
    .dispatch(denoise_pipeline, denoise_bindings, ceil(width/8), ceil(height/8), 1)
}

frame.pass("tonemap_present") {
    .draw(tonemap_pipeline, swapchain_target, fullscreen_quad)
}

frame.submit()
```

The runtime resolves the DAG, inserts Crux transfers, schedules across devices, executes, and presents.

### 12.3 Resource Lifecycle

```with
// Persistent resource: lives across frames
world_mesh := gfx.create_resource(mesh_data.len, .vertex | .storage,
    placement: .device_local(gpu0))

// Replicated resource: same data on multiple GPUs
material_table := gfx.create_resource(mat_data.len, .storage,
    placement: .replicated([gpu0, gpu1, gpu2], consistency: .lazy))

// Sharded resource: split across GPUs
terrain_heightmap := gfx.create_resource(huge_terrain.len, .texture,
    placement: .sharded([
        { device: gpu0, range: 0..quarter },
        { device: gpu1, range: quarter..half },
        { device: gpu2, range: half..three_quarter },
        { device: gpu3, range: three_quarter..total },
    ]))

// Transient resource: lives one frame, scheduler places freely
temp_buffer := frame.transient_resource(size, usage)
```

### 12.4 Explicit vs Delegated Control

```with
// Fully explicit: developer controls everything
frame.pass("work", device_require: gpu2) { ... }

// Hinted: developer suggests, scheduler may override
frame.pass("work", device_hint: gpu1) { ... }

// Delegated: scheduler decides entirely
frame.pass("work") { ... }

// Cost-annotated: developer provides estimated cost
frame.pass("work", estimated_cost: 2.ms) { ... }
```

---

## 13. Vulkan Compatibility Translator

The Vulkan translator intercepts unmodified Vulkan games and routes their work across multiple GPUs.

### 13.1 What It Is

A Vulkan ICD (Installable Client Driver) or layer. The game loads CruxGFX's Vulkan implementation instead of the vendor ICD. From the game's perspective, there is one `VkPhysicalDevice` with the combined capabilities of the device group.

### 13.2 Coverage

On Linux, the Vulkan translator effectively covers three APIs:

- **Native Vulkan games** → direct.
- **DirectX 11 games** (via DXVK / Proton) → already translated to Vulkan before reaching us.
- **DirectX 12 games** (via VKD3D-Proton) → same.

This is the vast majority of the modern game library.

### 13.3 Why It's Structurally Simple

Vulkan games already declare most of what CruxGFX needs:

| Vulkan Concept | CruxGFX Equivalent | Reconstruction Needed? |
|---|---|---|
| `VkRenderPass` / `vkCmdBeginRendering` | ExecNode (Draw) | No — direct mapping |
| `vkCmdPipelineBarrier` / `vkCmdPipelineBarrier2` | ExecEdge | No — direct mapping |
| `VkImage` / `VkBuffer` resource state | Crux Resource placement | No — already explicit |
| SPIR-V shaders | Crux Program (via Crux IR) | Minimal — already SPIR-V |
| `VkCommandBuffer` recording | Command stream | No — already batched |
| `VkSemaphore` / `VkFence` | Crux Event | No — direct mapping |
| Pipeline state | CruxGFX Pipeline | No — already explicit |
| `vkQueueSubmit` | Frame boundary | No — already explicit |

What CruxGFX must still do:

| Task | Why |
|---|---|
| Present a unified `VkPhysicalDevice` | Game must see one device |
| Virtualize `VkDeviceMemory` | Route allocations to real devices via Crux Resource |
| Intercept queue submission | Route command buffers to the right GPU |
| Insert cross-device transfers | Via Crux TransferOp when resources are stale |
| Handle presentation | Ensure final image reaches display owner |

### 13.4 Intercept Points

**Device-level intercept (critical):**

```
vkCreateDevice           → create per-GPU logical devices internally
vkAllocateMemory         → route to device via Crux Resource placement
vkCreateImage / Buffer   → track as Crux Resource, decide initial device
vkCmdBeginRenderPass     → record pass node for ExecGraph
vkCmdEndRenderPass       → finalize pass node
vkCmdBeginRendering      → same (dynamic rendering path)
vkCmdEndRendering        → same
vkCmdPipelineBarrier     → record dependency edge
vkCmdPipelineBarrier2    → same
vkCmdDraw*               → record draw within current pass
vkCmdDispatch*           → record compute node
vkCmdTraceRaysKHR        → record RT node
vkCmdCopyBuffer/Image    → record transfer node
vkQueueSubmit            → build ExecGraph, schedule, re-record per device
vkQueuePresent           → composite on display owner, present
vkCreateGraphicsPipelines → compile to CruxGFX Pipeline, cache, replicate
```

**Pass-through (no interception needed):**

```
vkCmdSetViewport         → state within a pass, stays with the pass
vkCmdSetScissor          → same
vkCmdBindPipeline        → same
vkCmdBindDescriptorSets  → same (but track resource references)
vkCmdPushConstants       → same
```

### 13.5 Virtual Physical Device

The game sees one `VkPhysicalDevice` with:

- **Memory:** unified heap reported as total across all GPUs (e.g. 384 GB for 4× 96 GB). Internally, `vkAllocateMemory` routes to a real device via Crux Resource placement.
- **Queues:** reported queue families are the superset of all devices. Submission routes internally.
- **Limits:** minimum across all devices (conservative).
- **Extensions:** intersection of all devices.

### 13.6 Resource Tracking

Every `VkImage` and `VkBuffer` maps to a Crux `Resource`.

```
VkResourceEntry {
    crux_resource:  Resource         // Crux Resource with placement/replicas
    initial_device: Device
    current_owner:  Device           // where most recently written
    read_set:       set[Device]      // devices with valid replicas
    usage_history:  []FrameUsage     // for speculation
}
```

When a command buffer on GPU 1 references a resource owned by GPU 2, the ExecGraph builder inserts a Crux TransferOp edge. The game never sees this.

### 13.7 Command Buffer Translation

```
game records command buffer:
    vkCmdBeginRenderPass(...)        → open new pass node
    vkCmdBindPipeline(...)           → associate pipeline with pass
    vkCmdDraw(...)                   → add draw to current pass
    vkCmdEndRenderPass(...)          → close pass node
    vkCmdPipelineBarrier(...)        → record dependency edge
    vkCmdBeginRenderPass(...)        → open new pass node
    ...

game calls vkQueueSubmit:
    CruxGFX builds ExecGraph from recorded pass nodes
    Scheduler assigns passes to devices (using Crux DeviceGroup topology)
    CruxGFX re-records real command buffers per device
    Crux TransferOps inserted for cross-device dependencies
    Submit to real device queues via Crux Streams
```

A single game command buffer may become multiple real command buffers on different GPUs, with Crux transfer commands inserted between them.

### 13.8 Work Classification

Once the ExecGraph is built, CruxGFX classifies each node for offload potential.

**Safe to offload (high confidence):**

| Work Type | Why Safe |
|-----------|----------|
| Shadow map rendering | Independent render target, clear boundary. |
| Reflection / environment map passes | Independent render target, early in frame. |
| Compute dispatches with declared barriers | Explicit sync. |
| Post-processing chain | Linear dependency chain, easy to move. |
| Texture decompression / streaming | Async, no frame-order dependency. |
| Secondary command buffers | Often independent work (shadow cascades, etc). |

**Risky to offload:**

| Work Type | Why Risky |
|-----------|-----------|
| Main scene geometry | May have feedback dependencies. Large resource footprint. |
| Subpasses with attachment dependencies | Order-dependent. |

**Do not offload (stay on GPU 0):**

| Work Type | Why |
|-----------|-----|
| Final present / swapchain render | Display owner only. |
| Readback targets | Need data on CPU. |
| Work with unknown side effects | Conservative default. |
| Host-mapped memory being written by CPU | Keep simple. |

### 13.9 Offload Decision

```
offload_benefit = estimated_node_cost - (transfer_cost + sync_cost + scheduler_overhead)

if offload_benefit > threshold:
    offload to best available GPU
else:
    keep on GPU 0
```

Early frames: threshold is high (conservative).
As profiling data accumulates: threshold drops, more work offloads.

Because Vulkan's dependency information is precise (the game declared it), the starting threshold can be relatively low.

### 13.10 Temporal Hazards

Modern rendering creates temporal dependencies (TAA, reprojection, frame generation, streaming). If resource X was written on GPU 2 last frame and needs to be read on GPU 0 this frame, there must be a Crux transfer.

```
begin_frame:
    for each Crux Resource with replicas:
        if any replica.generation < canonical.generation:
            schedule pre-frame Crux TransferOp if resource is likely needed
```

CruxGFX predicts which resources a pass will need based on the previous frame's access pattern. Most games are temporally coherent.

### 13.11 Correctness Guarantees

1. If the scheduler cannot prove a work item is safe to offload, it stays on GPU 0.
2. If a Crux transfer would miss a deadline (based on cost model), the work stays on GPU 0.
3. Floating-point determinism: same precision, same rounding. Identical GPUs produce identical results. Mixed models are out of scope for v1.
4. Vulkan semaphores and fences are translated into Crux Events.
5. Vulkan validation layers pass through.

### 13.12 Failure Modes and Fallback

```
Level 0: Full multi-GPU        All passes scheduled across all GPUs
Level 1: Partial offload        Some passes offloaded, most on GPU 0
Level 2: Async compute only     Only compute work offloaded
Level 3: Single GPU             All work on GPU 0 (guaranteed correct)
```

If CruxGFX detects instability, it drops to a lower level automatically. The user can also force a level via config.

---

## 14. Compositor

The final image must be assembled on the display owner (GPU 0).

### 14.1 Composition Modes

**Full composite:** Multiple GPUs each render part of the frame. GPU 0 merges them.

```
CompositeLayer {
    source:     RenderTarget    // on some GPU
    region:     Rect            // screen region or full
    blend:      CompositeBlend  // over, additive, replace
    depth_ref:  ?View           // for depth-aware compositing
}
```

**Single-source:** One GPU rendered the whole frame. Just present it.

**Overlay:** HUD/UI rendered on GPU 0, scene on another GPU. Composed via alpha.

### 14.2 Latency Management

1. Start offloaded work early in the frame.
2. Overlap Crux transfers with GPU 0's own work.
3. Composite as late as possible (just before present).
4. Use previous frame's result as fallback if transfer misses deadline (temporal reprojection).

---

## 15. Shader System

### 15.1 Shader Pipeline

```
  ┌─────────────┐                 ┌──────────────┐
  │  SPIR-V     │                 │  With shader  │
  │ (VK compat) │                 │  language     │
  └──────┬──────┘                 │ (native)     │
         │                        └──────┬───────┘
         ▼                               │
  ┌─────────────┐                        │
  │  Crux IR    │◄───────────────────────┘
  └──────┬──────┘
         ▼
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Vulkan │ │ Metal │
│SPIR-V │ │MSL/AIR│
└───────┘ └───────┘
```

The compat path is short: games already provide SPIR-V. The native path may bypass SPIR-V entirely if With compiles directly to Crux IR.

### 15.2 Spec Constant Injection

The scheduler can inject spec constants at compile time per device:

- `DEVICE_INDEX` — which GPU this variant runs on.
- `IS_DISPLAY_OWNER` — whether this device presents.
- `HAS_PEER_ACCESS` — whether peer memory is available.

Native front-end can use these for device-aware shader logic.

---

## 16. Configuration and Observability

### 16.1 Runtime Configuration

```toml
[cruxgfx]
display_device = 0              # which GPU owns the swapchain
mode = "auto"                   # auto | native | compat | single_gpu

[cruxgfx.compat]
offload_level = "auto"          # auto | full | partial | compute_only | none
aggressiveness = 0.7            # 0.0 = ultra conservative, 1.0 = offload everything
temporal_fallback = true        # use previous frame if transfer misses deadline
shader_cache = "~/.cruxgfx/cache"
pipeline_cache = "~/.cruxgfx/pipelines"

[cruxgfx.scheduler]
strategy = "auto"               # auto | display_owner_primary | pipeline_decomp | scene_decomp
profiling = true                # collect per-node timing
cost_model_warmup_frames = 30   # frames before aggressive offloading

[cruxgfx.devices]
gpu0.role = "display"
gpu1.role = "worker"
gpu2.role = "worker"
gpu3.role = "worker"
```

### 16.2 Observability

```
CruxGFX Overlay (opt-in):
    Per-GPU utilization bars
    Per-GPU memory usage (via Crux DeviceBudget)
    Transfer bandwidth utilization (via Crux TransferEdge)
    Frame time breakdown:
        GPU 0: [███████░░░] 7.2ms
        GPU 1: [████░░░░░░] 4.1ms (shadow)
        GPU 2: [██████░░░░] 6.0ms (GI trace)
        GPU 3: [███░░░░░░░] 3.3ms (denoise)
        Transfer: 0.8ms
        Composite: 0.3ms
        Frame total: 8.1ms (123 FPS)

    Offload decisions: which passes moved, why
    Stale replica warnings (Crux Resource generation mismatches)
    Fallback level changes
```

---

## 17. Implementation Order

### Phase 1: Crux Extensions (Months 1–4)

**Multi-device primitives added to Crux:**

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
- Verify: can dispatch compute on GPU 1, transfer result to GPU 0.
- Verify: topology log accurately reflects real hardware capabilities.
- Verify: ML/compute workloads can use multi-device Resources without graphics.

### Phase 2: CruxGFX Core (Months 4–7)

**Graphics layer on top of Crux:**

- Pipeline, RenderTarget, ExecGraph data structures.
- ExecGraph → DevicePlan scheduling pipeline.
- DisplayOwnerPrimary and PipelineDecomposition strategies.
- Cost model with profiling feedback (derived from Crux topology).
- Compositor.
- Single-device backend via Vulkan.
- Verify: can render a triangle through the full stack.
- Verify: can schedule a multi-pass frame across 4 GPUs.

### Phase 3: Native Front-End (Months 7–9)

- Frame graph API in With.
- Resource lifecycle management.
- Shader compilation pipeline (With → Crux IR → SPIR-V).
- Verify: a real multi-GPU renderer running a non-trivial scene.

### Phase 4: Vulkan Compatibility Translator (Months 9–12)

- Vulkan ICD / layer intercept.
- Virtual VkPhysicalDevice presenting unified device.
- VkDeviceMemory → Crux Resource routing.
- Render pass → ExecNode extraction.
- Pipeline barrier → ExecEdge extraction.
- Command buffer interception and re-recording per device.
- Work classifier and offload engine.
- Temporal hazard tracker.
- Fallback system.
- Pipeline cache (SPIR-V → Crux IR).
- Single-GPU correctness validation (pixel-match against real driver).
- Multi-GPU offloading (start with compute dispatches and independent render passes).
- Verify: Unmodified Vulkan games run correctly, some see measurable speedup.
- Verify: DXVK/VKD3D-Proton games work through the translator.

### Phase 5: Polish (Months 12–14)

- Advanced scheduling strategies (scene decomposition, hybrid).
- Observability overlay.
- Game-specific profiles (known offload patterns for popular titles).
- Documentation and With language integration.

---

## 18. What Success Looks Like

### For the native front-end

A With game engine renders a complex scene at 4K with:
- Rasterization on GPU 0
- Ray traced GI on GPU 1
- Neural denoising on GPU 2
- Volumetrics and particles on GPU 3
- All four GPUs utilized >70%
- No manual transfer management by the developer
- 2.5–3.5× the performance of a single GPU

### For the Vulkan compat translator

An unmodified Vulkan game runs and:
- Is pixel-correct compared to single GPU
- Independent render passes (shadows, reflections) offload to other GPUs automatically
- Compute dispatches route to worker GPUs when profitable
- Frame rate improves 20–50% depending on workload
- The game developer changed nothing
- DXVK/VKD3D-Proton games (DirectX → Vulkan) also benefit

### For Crux users (non-graphics)

A distributed ML training job or simulation across 4 GPUs uses Crux's multi-device primitives:
- DeviceGroup discovers topology and measures real P2P bandwidth
- Resources placed and replicated across devices with generation tracking
- TransferOps scheduled as first-class work with measured cost
- No CruxGFX dependency needed

### For all

One substrate. Two layers. One truth.