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

## Metal driver (`src/driver/metal/`)

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

## Vulkan driver (`src/driver/vulkan/`)

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

```rust
#[cfg(target_os = "linux")]
#[link(name = "vulkan")]
extern "C" { ... }

#[cfg(target_os = "windows")]
#[link(name = "vulkan-1")]
extern "C" { ... }

#[cfg(target_os = "android")]
#[link(name = "vulkan")]
extern "C" { ... }
```

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

## WebGPU driver (`src/driver/webgpu/`)

Browser-only (`target_arch = "wasm32"` + `webgpu` feature). Same
no-wrapper-crate rule as Metal/Vulkan, applied to wasm32: no
`web-sys`, no `wgpu`, no `wasm-bindgen` runtime.

### B⁰ — Quanta-owned wasm ↔ JS ABI

Step `1d2fc65` (2026-04-28). Replaces `wasm-bindgen` (~30-60 KB
third-party runtime) with hand-authored boundaries on both sides:

```
Rust side                     JS side
─────────                     ──────
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
crates/quanta-codegen  ──────────────► weedle parse
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

## Validation layer (`src/driver/validation.rs`)

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

1. Add the method to the `GpuDevice` trait in `src/api/device.rs`.
2. Add a default implementation (or make it required).
3. Add the public API method on `Gpu` in `src/api/gpu.rs`.
4. Implement in `src/driver/metal/` (Metal version).
5. Implement in `src/driver/vulkan/` (Vulkan version).
6. Implement in `src/driver/webgpu/` if it's exposable to the browser
   (and add the corresponding TS plumbing in `web/src/webgpu.ts`).
   If a new WebGPU IDL enum is touched, add it to
   `crates/quanta-codegen/src/parse.rs`'s `PROJECT_RELEVANT_ENUMS`
   and re-run `quanta codegen webgpu`.
7. Add validation checks in `src/driver/validation.rs`.
8. Add a conformance test in `tests/conformance/`.

## Adding a new driver (e.g., DirectX 12)

1. Create `src/driver/dx12/` with `mod.rs`, `ffi.rs`, etc.
2. Implement `GpuDevice` for `Dx12Device`.
3. Add `pub fn discover() -> Vec<Box<dyn GpuDevice>>`.
4. Add a feature flag in `Cargo.toml`: `dx12 = ["std"]`.
5. Wire into `lib.rs`:
   ```rust
   #[cfg(feature = "dx12")]
   devs.extend(driver::dx12::discover());
   ```
