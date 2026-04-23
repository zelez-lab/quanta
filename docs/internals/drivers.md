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
6. Add validation checks in `src/driver/validation.rs`.
7. Add a conformance test in `tests/conformance/`.

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
