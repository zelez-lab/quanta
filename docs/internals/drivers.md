# Driver Internals

Quanta uses raw FFI to talk to GPU APIs. No wrapper crates. This gives zero
overhead, zero external dependencies, and full control over the ABI.

## Why raw FFI

| Approach | Deps | Binary size | Build time | Control |
|----------|------|-------------|------------|---------|
| `metal-rs` / `ash` | 5-20 crates | +200KB | slower | limited |
| Raw FFI | 0 crates | minimal | fast | total |

System calls go through hand-written `extern "C"` blocks with `#[repr(C)]`
struct layouts.
The driver code is the thinnest possible translation layer.

## Metal driver (`crates/gpu/quanta-core/src/driver/metal/`)

### FFI pattern

Metal is an Objective-C framework. Calling it from Rust requires:

```rust
// ffi.rs — raw ObjC runtime bindings
extern "C" {
    fn objc_msgSend(receiver: Id, selector: Sel, ...) -> Id;
    fn objc_getClass(name: *const u8) -> Class;
    fn sel_registerName(name: *const u8) -> Sel;
}
```

Helper macros/functions wrap the raw `objc_msgSend`:

```rust
// Selector helper
unsafe fn sel(name: &str) -> Sel {
    sel_registerName(name.as_ptr())
}

// Message send with typed return
unsafe fn msg_send<R>(obj: Id, sel: Sel) -> R {
    let f: unsafe extern "C" fn(Id, Sel) -> R = core::mem::transmute(objc_msgSend as *const ());
    f(obj, sel)
}
```

### Type aliases

All Metal objects are opaque `Id` pointers:

```rust
pub type Id = *mut c_void;       // Any ObjC object
pub type MTLDevice = Id;         // id<MTLDevice>
pub type MTLCommandQueue = Id;   // id<MTLCommandQueue>
pub type MTLBuffer = Id;         // id<MTLBuffer>
pub type MTLComputePipelineState = Id;
```

### Buffer creation

```rust
// memory.rs
fn field_alloc(&self, size: usize, _usage: FieldUsage) -> Result<u64, QuantaError> {
    unsafe {
        let buffer: Id = msg_send(
            self.device,
            sel("newBufferWithLength:options:"),
            size as NSUInteger,
            MTL_RESOURCE_STORAGE_MODE_SHARED,
        );
        if buffer == NIL {
            return Err(QuantaError::out_of_memory(size));
        }
        Ok(buffer as u64)
    }
}
```

### Compute dispatch

```rust
// compute.rs
fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
    unsafe {
        let cmd_buf: Id = msg_send(self.queue, sel("commandBuffer"));
        let encoder: Id = msg_send(cmd_buf, sel("computeCommandEncoder"));

        // Set pipeline state
        msg_send_void(encoder, sel("setComputePipelineState:"), wave.handle as Id);

        // Bind fields
        for binding in &wave.bindings {
            msg_send_void(encoder, sel("setBuffer:offset:atIndex:"),
                binding.field_handle as Id, 0u64, binding.slot as NSUInteger);
        }

        // Dispatch
        let grid = MTLSize { width: groups[0], height: groups[1], depth: groups[2] };
        let group = MTLSize { width: 64, height: 1, depth: 1 };
        msg_send_void(encoder, sel("dispatchThreads:threadsPerThreadgroup:"), grid, group);

        msg_send_void(encoder, sel("endEncoding"));
        msg_send_void(cmd_buf, sel("commit"));

        Ok(Pulse::new(cmd_buf as u64))
    }
}
```

## Vulkan driver (`crates/gpu/quanta-core/src/driver/vulkan/`)

### FFI pattern

Vulkan uses C function pointers loaded from the shared library:

```rust
// ffi.rs
#[cfg(target_os = "linux")]
#[link(name = "vulkan")]
extern "C" {
    pub fn vkCreateInstance(
        create_info: *const VkInstanceCreateInfo,
        allocator: *const c_void,
        instance: *mut VkInstance,
    ) -> VkResult;

    pub fn vkCreateBuffer(
        device: VkDevice,
        create_info: *const VkBufferCreateInfo,
        allocator: *const c_void,
        buffer: *mut VkBuffer,
    ) -> VkResult;

    // ... ~50 more functions
}
```

### Struct definitions

```rust
#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: VkDeviceSize,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}
```

### Platform gating

The ~100 entry-point signatures live in one macro (`vk_fns!` in
`ffi/extern_fns.rs`); a per-platform emitter chooses the binding form:

```rust
// Linux / Android / macOS (MoltenVK under `vulkan-portability`):
// one link-time extern block.
#[link(name = "vulkan")]
unsafe extern "C" { pub fn vkCreateInstance(...) -> VkResult; ... }

// Windows: a function-pointer table resolved at runtime
// (LoadLibraryA("vulkan-1.dll") + GetProcAddress) behind
// identically-named shims — call sites are the same on every platform.
pub unsafe fn vkCreateInstance(...) -> VkResult {
    (fns().vkCreateInstance)(...)
}
```

Windows deliberately does **not** link `vulkan-1.lib`: the import
library ships only with the Vulkan SDK (which would make the SDK a
build dependency), and a link-time-bound app fails at *process load*
on a machine without `vulkan-1.dll` — before init can engage the
software fallback. Instead, `discover()` gates on
`ffi::ensure_loaded()`: when the DLL (or an export — a Vulkan 1.3
loader is required) is missing, it prints a loud `quanta vulkan:` line
naming the missing piece and returns no devices, letting the software
last resort engage. `QUANTA_VULKAN_LOADER` overrides the DLL name (see
[Environment](../reference/environment.md)).

### Memory management

Vulkan requires explicit memory type selection:

```rust
// memory.rs
fn find_memory_type(&self, type_filter: u32, properties: u32) -> Option<u32> {
    let mem_props = self.memory_properties;
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (mem_props.memory_types[i].property_flags & properties) == properties
        {
            return Some(i);
        }
    }
    None
}
```

### Synchronization

```rust
// sync.rs
fn barrier_texture(&self, image: VkImage, old_layout: u32, new_layout: u32) {
    let barrier = VkImageMemoryBarrier {
        s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        old_layout,
        new_layout,
        image,
        subresource_range: VkImageSubresourceRange { ... },
        src_access_mask: layout_to_access(old_layout),
        dst_access_mask: layout_to_access(new_layout),
        ..zeroed()
    };

    unsafe {
        vkCmdPipelineBarrier(
            self.cmd_buffer,
            layout_to_stage(old_layout),
            layout_to_stage(new_layout),
            0, 0, null(), 0, null(), 1, &barrier,
        );
    }
}
```

## WebGPU driver (`crates/gpu/quanta-core/src/driver/webgpu/`)

Browser-only (`target_arch = "wasm32"` + `webgpu` feature). Same
no-wrapper-crate rule as Metal/Vulkan, applied to wasm32: no
`web-sys`, no `wgpu`, no `wasm-bindgen` runtime.

### B⁰ — Quanta-owned wasm ↔ JS ABI

Step `1d2fc65` (2026-04-28). Replaces `wasm-bindgen` (~30-60 KB
third-party runtime) with hand-authored boundaries on both sides:

```
Rust side (quanta-core)       JS side
─────────────────────────     ──────
src/driver/webgpu/ffi.rs      web/src/quanta.ts (entry)
  unsafe extern "C" {           + handles.ts (handle table)
    fn quanta_*(...);           + tasks.ts (Promise → wasm callback)
  }                             + webgpu.ts (FFI imports)
src/driver/webgpu/             + strings.ts (TextDecoder)
  executor.rs (futures)         + codes.ts (enum string tables)
```

Total: ~500 LOC of project-local code on each side. Cargo wasm
output ships zero npm runtime deps; `quanta.js` is the only JS the
browser loads. ABI conventions:

- Long-lived JS objects (devices, buffers, pipelines, …) cross as
  `u32` handles into a JS-side handle table. Handle 0 = null.
- Strings cross as `(ptr: *const u8, len: usize)`; JS reads them via
  `TextDecoder` against the wasm linear memory.
- `u64` sizes cross as `f64` (exact up to 2^53, larger than any
  WebGPU resource).
- Async ops take a `task: u32` argument. JS resolves the underlying
  Promise and calls back into wasm exports `quanta_resolve(task,
  handle)` / `quanta_reject(task)`. Rust executor (~150 LOC) turns
  those into `Future::poll → Ready`.

### B′ — WebIDL → Rust + TS code tables

Step `1d93e78` (2026-04-28). Eliminates the lockstep hazard between
the Rust `mod format`/`mod blend_factor`/… in `ffi.rs` and the
parallel TypeScript arrays in `web/src/codes.ts`.

```
web/webgpu.idl                          (vendored, sha256-pinned)
        │
        ▼
crates/lang/quanta-codegen  ──────────────► weedle parse
        │                                  │
        ├─► src/webgpu_generated_codes.rs  (Rust spec tables)
        │       + cargo test:    quanta_strings_are_spec_subsets
        │
        └─► web/src/generated/codes.ts     (TS spec tables)
                + module init: assertSpecSubset()
```

`quanta codegen webgpu` regenerates both files from one parsed AST.
Rust side: a unit test on every `cargo test --lib` checks every
Quanta-side enum string is a member of the spec table. TS side:
`assertSpecSubset()` runs at module-init in the browser; page load
throws if any string drifted out of spec — visible before the first
FFI call.

### Build / serve

```sh
quanta codegen webgpu      # regenerate spec tables (when webgpu.idl changes)
quanta build web           # compile glue.ts + cargo build wasm32 + stage
quanta serve web_add_one   # embedded HTTP server on 127.0.0.1:8000
```

All three are dev-only — `quanta-cli` and `quanta-codegen` never
ship to user codebases. The user-facing `quanta` library crate has
zero transitive deps; the browser sees only `quanta.js` + the wasm.

## Capability discovery

Each driver populates a per-device capability cache at discovery time
so `gpu.supports_*()` queries are O(1) reads, never re-probing the
hardware. The cache is a simple `Caps` struct on the device:

```rust
struct Caps {
    has_vrs: bool,
    has_ray_tracing: bool,
    has_mesh_shaders: bool,
    has_tessellation: bool,
    has_sparse_residency: bool,
    supported_shading_rates: Vec<(u32, u32)>,
    // ...
}
```

### Vulkan capability discovery

At `vkCreateDevice` time the driver enumerates extensions
(`vkEnumerateDeviceExtensionProperties`) and chains
`VkPhysicalDeviceFeatures2 → ...Features extensions` on the create
chain so each feature is *enabled* alongside being *enumerated*:

| Capability | Extension(s) | Feature struct |
|------------|--------------|----------------|
| Sparse residency | (core) | `VkPhysicalDeviceFeatures.sparseBinding` |
| VRS | `VK_KHR_fragment_shading_rate` | `VkPhysicalDeviceFragmentShadingRateFeaturesKHR` |
| Mesh shaders | `VK_EXT_mesh_shader` | `VkPhysicalDeviceMeshShaderFeaturesEXT` |
| Tessellation | (core) | `VkPhysicalDeviceFeatures.tessellationShader` |
| Ray tracing | `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure` + `VK_KHR_buffer_device_address` + `VK_KHR_deferred_host_operations` | `accelerationStructure` + `bufferDeviceAddress` |

`bufferDeviceAddress` is required for ray tracing because the
acceleration-structure build inputs reference vertex / index buffers
by GPU device address. When enabled, `field_alloc_impl` augments the
buffer-creation usage with `SHADER_DEVICE_ADDRESS_BIT` +
`AS_BUILD_INPUT_READ_ONLY_BIT_KHR` so user-allocated `Field`s can be
fed into the AS build without a re-create.

VRS supported rates are enumerated via
`vkGetPhysicalDeviceFragmentShadingRatesKHR` and cached as
`supported_shading_rates: Vec<(u32, u32)>`.

### Metal capability discovery

Metal exposes capabilities via family-feature gates:

| Capability | Gate |
|------------|------|
| Sparse residency | `[device supportsFamily:Apple7]` (M1 / A15+) |
| Ray tracing | `[device supportsFamily:Apple6]` |
| Mesh shaders | `[device supportsFamily:Mac2]` or Apple9 |
| Tessellation | always (all Metal 2 devices) |
| VRS | `[device supportsFamily:Apple6]` + rasterization-rate-map API present |

The Metal driver caches `MTLDevice.supportsFamily:` answers once at
device creation rather than calling them per-allocation.

### WebGPU capability discovery

WebGPU has no fine-grained capability bits in the spec for ray tracing
/ mesh shaders / sparse residency (none of which are in W3C). The
WebGPU driver reports them all as `false`; constructors return
`NotSupported` with feature-named messages.

The W3C spec exposes a few features (e.g. `timestamp-query`,
`depth-clip-control`) via `device.features`; the driver translates
those into the corresponding `Caps` bits at discovery time.

### CPU software device

Returns `false` for every capability except the always-software
ones (basic compute, plain texture write, etc.). All advanced-feature
constructors return `NotSupported` with a software-CPU-specific
message.

## Feature support queries

Public API for the cache:

```rust
gpu.supports_vrs()              // bool
gpu.supports_ray_tracing()      // bool
gpu.supports_mesh_shaders()     // bool
gpu.supports_tessellation()     // bool
gpu.supports_sparse_residency() // bool
gpu.supported_shading_rates()   // Vec<(u32, u32)>
```

These read the cached `Caps` and never re-probe — cheap to call in
hot loops. The convention across the codebase is *capability-check
first, then construct*: callers branch on `supports_*` and either
construct the typed wrapper or take a fallback path. Constructors
themselves still gate (returning `NotSupported`) so callers who skip
the up-front check fail explicitly rather than silently.

## Native feature lowerings (v0.1)

| Feature | Vulkan | Metal | WebGPU | CPU |
|---------|--------|-------|--------|-----|
| Compute / draw / blit | ✅ native | ✅ native | ✅ native | ✅ software |
| Async copy | ✅ `MTLBlitCommandEncoder` |  | `GPUQueue.copyBufferToBuffer` | software memcpy |
| Multi-queue | ✅ per family | ✅ per family | single queue | software FIFO |
| Tessellation | ✅ device-feature gated; software MVP, native render-pipeline pending | ✅ MTLBuffer-backed; native draw pending | `NotSupported` | ✅ full software |
| Mesh shaders | ✅ extension-gated; software MVP, native pipeline pending | ✅ family-gated; software MVP, native pipeline pending | `NotSupported` | ⚠️ software MVP lifecycle only (`mesh_pipeline_create` / `mesh_dispatch` record limits + dispatch order, no rasterization); a render `PipelineDesc` with a `mesh_shader` desc is rejected `NotSupported` ("CPU has no rasterizer"), and the `dispatch_mesh` render entry point returns `NotSupported` as well |
| Variable rate shading | ✅ native (rate enumeration + `vkCmdSetFragmentShadingRateKHR`) | ✅ native (`MTLRasterizationRateMap`) | `NotSupported` | software lifecycle |
| Sparse residency | ✅ native (`vkQueueBindSparse`, 2D / single-mip) | ✅ native (`MTLHeap` placement, 2D / single-mip) | `NotSupported` | software lifecycle |
| Ray tracing | ⚠️ AS proc-addr foundation; build dispatch returns `NotSupported` (lavapipe segfault, awaiting AMDGPU runner) | ⚠️ family-gated; intersector dispatch pending | `NotSupported` | software lifecycle |
| Indirect command buffer | ✅ native (`MTLIndirectCommandBuffer`, compute + render-bundle draw via `executeCommandsInBuffer`) | ✅ native (secondary command buffers + `vkCmdExecuteCommands`; render bundles record in `RENDER_PASS_CONTINUE` mode) | `NotSupported` (render bundles are a separate path) | ✅ full software |
| Occlusion queries | ✅ native | ✅ native | ✅ native (async read via `mapAsync`; sync `occlusion_query_read` returns `NotSupported`) | ✅ software |
| Compute textures (storage image) | ✅ native storage load + write + sample (emitter bakes an R32f image; sampled `&Texture2D` slots bind as `COMBINED_IMAGE_SAMPLER` with a cached per-device compute sampler — nearest, clamp-to-edge, unnormalized coords — matching the CPU executor) | ✅ native (storage load + write + sample) | `NotSupported` (`wave_dispatch` rejects texture bindings loudly) | ✅ software |

`supports_compute_textures()` reports this row: `true` on Metal, Vulkan, and
CPU; `false` on WebGPU.

The Vulkan driver keeps a per-device render-path **sampler cache** — a
`HashMap<SamplerDesc, VkSampler>` keyed on the *whole* `SamplerDesc` — so a
given descriptor maps to exactly one `VkSampler` for the device's lifetime;
repeated `.sampler(slot, desc)` binds with equal descs reuse the same object,
and every cached sampler is destroyed when the device drops.

### Occlusion queries on WebGPU

Native backends expose `gpu.occlusion_query_read(query) -> Result<Vec<u64>>`
synchronously. WebGPU has no synchronous readback — query results must
go through `GPUQuerySet.resolveQuerySet` into a staging buffer with
`COPY_SRC | COPY_DST | QUERY_RESOLVE | MAP_READ` usage, then
`buffer.mapAsync(GPUMapMode.READ)` resolves to a `Promise<void>` the
host awaits. The WebGPU driver therefore:

1. Returns `NotSupported` from the synchronous `occlusion_query_read`
   with a message pointing at the async path.
2. Provides `pub async fn occlusion_query_read_async(&self, query: u64)
   -> Result<Vec<u64>>` as a driver-level extension method that drives
   the resolve + mapAsync sequence and decodes the per-slot 8-byte
   little-endian `u64` count.

The render encoder pre-walks `pass.ops` to attach the query set to
the `RenderPassDescriptor.occlusionQuerySet` field before
`beginRenderPass`, since WebGPU validates the attachment up front
rather than letting `begin_occlusion_query` discover the missing set.

## Validation layer (`crates/gpu/quanta-core/src/driver/validation.rs`)

Wraps any `GpuDevice` and adds runtime checks:

```rust
pub struct ValidationDevice {
    inner: Box<dyn GpuDevice>,
    allocated_fields: HashSet<u64>,
    field_sizes: HashMap<u64, usize>,
}
```

Enabled with `QUANTA_VALIDATE=1`:
- Checks field handles are valid before use.
- Checks bind slots match kernel expectations.
- Checks dispatch sizes don't exceed device limits.
- Reports use-after-free on dropped fields.
- Prints warnings for suboptimal patterns.

## Adding a new driver operation

1. Add the method to the `GpuDevice` trait in
   `crates/gpu/quanta-core/src/api/device.rs`.
2. Add a default implementation (or make it required).
3. Add the public API method on `Gpu` in
   `crates/gpu/quanta-core/src/api/gpu.rs` (render-face methods go on the
   sealed `RenderGpu` trait in `crates/gpu/quanta-render/src/gpu_ext.rs`
   instead).
4. Implement in `crates/gpu/quanta-core/src/driver/metal/` (Metal version).
5. Implement in `crates/gpu/quanta-core/src/driver/vulkan/` (Vulkan version).
6. Implement in `crates/gpu/quanta-core/src/driver/webgpu/` if it's exposable to the browser
   (and add the corresponding TS plumbing in `web/src/webgpu.ts`).
   If a new WebGPU IDL enum is touched, add it to
   `crates/lang/quanta-codegen/src/parse.rs`'s `PROJECT_RELEVANT_ENUMS`
   and re-run `quanta codegen webgpu`.
7. Add validation checks in `crates/gpu/quanta-core/src/driver/validation.rs`.
8. Add an integration test under `tests/` (see `tests/gpu_compute.rs`
   and friends for the pattern).

## Adding a new driver (e.g., DirectX 12)

1. Create `crates/gpu/quanta-core/src/driver/dx12/` with `mod.rs`, `ffi.rs`, etc.
2. Implement `GpuDevice` for `Dx12Device`.
3. Add `pub fn discover() -> Vec<Box<dyn GpuDevice>>`.
4. Add a feature flag in `crates/gpu/quanta-core/Cargo.toml`: `dx12 = ["std"]`
   (and forward it from the facade's and `quanta-render`'s `Cargo.toml`,
   like the existing backend features).
5. Wire into `crates/gpu/quanta-core/src/lib.rs`:
   ```rust
   #[cfg(feature = "dx12")]
   devs.extend(driver::dx12::discover());
   ```
