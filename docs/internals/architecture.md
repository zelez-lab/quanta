# Architecture

## Crate structure

```
quanta/                         Main library (no_std, zero external deps)
+-- src/
|   +-- lib.rs                  Entry points: init(), devices()
|   +-- api/
|   |   +-- gpu.rs              Gpu struct — all user-facing operations
|   |   +-- field.rs            Field<T>, MappedField<T>
|   |   +-- wave.rs             Wave (bound kernel + bindings)
|   |   +-- pulse.rs            Pulse (GPU completion signal)
|   |   +-- texture.rs          Texture, TextureDesc, TextureView
|   |   +-- pipeline.rs         Pipeline, PipelineDesc
|   |   +-- render_pass.rs      RenderPass, draw commands
|   |   +-- ray_tracing.rs      Acceleration structures, RT pipelines
|   |   +-- device.rs           GpuDevice trait (driver interface)
|   |   +-- types.rs            Caps, Vendor, Format, FieldUsage, etc.
|   |   +-- error.rs            QuantaError
|   +-- driver/
|   |   +-- metal/              Metal backend (macOS/iOS)
|   |   |   +-- mod.rs          MetalDevice (implements GpuDevice)
|   |   |   +-- ffi.rs          Raw ObjC FFI bindings
|   |   |   +-- compute.rs      Compute dispatch implementation
|   |   |   +-- memory.rs       Buffer/field allocation
|   |   |   +-- render.rs       Render pass encoding
|   |   |   +-- texture.rs      Texture creation and operations
|   |   +-- vulkan/             Vulkan backend (Linux/Android/Windows)
|   |   |   +-- mod.rs          VulkanDevice (implements GpuDevice)
|   |   |   +-- ffi.rs          Raw Vulkan FFI bindings
|   |   |   +-- compute.rs      Compute dispatch implementation
|   |   |   +-- memory.rs       Buffer allocation + memory management
|   |   |   +-- render.rs       Render pass encoding
|   |   |   +-- texture.rs      Image creation and transitions
|   |   |   +-- sync.rs         Fences, semaphores, barriers
|   |   +-- validation.rs       Debug validation layer (wraps any driver)
|   +-- kernel.rs               KernelBinary, ShaderBinary, GpuType trait

crates/quanta-ir/               Shared IR definition (zero external deps)
+-- src/
|   +-- lib.rs                  KernelOp (36 variants), KernelDef, CompilerOutput
|   +-- wire/                   Binary serialization (custom format, no serde)
|       +-- encode.rs           KernelDef -> bytes
|       +-- decode.rs           bytes -> KernelDef
|       +-- tests.rs            Roundtrip tests

crates/quanta-macros/           Proc macros (depends on syn + quote)
+-- src/
|   +-- lib.rs                  #[kernel], #[device], #[shared],
|   |                           #[vertex], #[fragment],
|   |                           #[tess_control], #[tess_eval],
|   |                           #[task], #[mesh],
|   |                           #[ray_gen], #[closest_hit], #[miss]
|   +-- parse.rs                Rust AST -> KernelDef (with parse/expr.rs, parse/stmt.rs)
|   +-- compiler.rs             Orchestrates compilation, shader emitters
|   +-- validate.rs             Pre-parse validation rules

crates/quanta-compiler/         LLVM compiler binary (depends on inkwell/LLVM 22)
+-- src/
|   +-- main.rs                 CLI entry: stdin KernelDef -> stdout CompilerOutput
|   +-- to_llvm.rs              KernelOp -> LLVM IR module builder
|   +-- to_llvm/emit.rs         Per-op LLVM IR emission
|   +-- to_llvm/metadata.rs     GPU-specific metadata (nvvm annotations, etc.)
|   +-- targets.rs              GpuTarget enum, GpuIntrinsics trait
|   +-- targets/nvptx.rs        NVIDIA: thread IDs, barriers, atomics
|   +-- targets/amdgpu.rs       AMD: workitem IDs, barriers, atomics
|   +-- targets/spirv.rs        SPIR-V: OpAccessChain, OpControlBarrier
|   +-- emit_msl.rs             KernelOp -> MSL source text
|   +-- emit_wgsl.rs            KernelOp -> WGSL source text
|   +-- rustc_compile.rs        Alternative path: Rust source -> rustc -> LLVM IR
+-- build.rs                    LLVM detection and linking
```

## Build-time compilation flow

```
                    BUILD TIME (cargo build)
                    ========================

Source:
    #[quanta::kernel]
    fn my_kernel(input: &[f32], output: &mut [f32]) {
        let i = quark_id();
        output[i] = input[i] * 2.0;
    }

Step 1: Parse (quanta-macros/parse.rs)
    Rust AST (syn) -> KernelDef {
        name: "my_kernel",
        params: [FieldRead("input", slot=0), FieldWrite("output", slot=1)],
        body: [QuarkId{r0}, Load{r1,field=0,r0}, Const{r2,2.0}, BinOp{r3,r1*r2}, Store{field=1,r0,r3}],
    }

Step 2: Serialize + invoke compiler (quanta-macros/compiler.rs)
    KernelDef -> quanta_ir::serialize_kernel() -> [u8]
    Pipe bytes to `quanta-compiler` binary via stdin

Step 3: Compile to all targets (quanta-compiler)
    KernelOp -> LLVM IR -> PTX (NVIDIA)
    KernelOp -> LLVM IR -> GCN ELF (AMD)
    KernelOp -> LLVM IR -> SPIR-V (Vulkan)
    KernelOp -> MSL source -> xcrun metal -> metallib (Apple)
    KernelOp -> WGSL source (browsers)

Step 4: Embed (quanta-macros/lib.rs)
    CompilerOutput -> proc_macro TokenStream:

    pub static MY_KERNEL_BINARY: KernelBinary = KernelBinary {
        nvidia: Some(b"...ptx..."),
        amd: Some(b"...elf..."),
        spirv: Some(b"..."),
        metallib: Some(b"..."),
        msl: Some("..."),
        wgsl: Some("..."),
        llvm_ir: None,
    };

    pub fn my_kernel(device: &Gpu) -> Result<Wave, QuantaError> {
        let binary = MY_KERNEL_BINARY.for_vendor(device.caps().vendor)?;
        device.wave(binary)
    }
```

```
                    RUNTIME
                    =======

    let gpu = quanta::init()?;
    //  -> discovers Metal or Vulkan device
    //  -> wraps in Gpu struct

    let wave = my_kernel(&gpu)?;
    //  -> selects pre-compiled binary for this vendor
    //  -> creates pipeline state object (Metal) or compute pipeline (Vulkan)

    wave.bind(0, &input_field);
    //  -> records binding (slot -> field handle)

    gpu.dispatch(&wave, 1024)?;
    //  -> Metal: encodes dispatch to command buffer, commits
    //  -> Vulkan: records dispatch, submits to queue
```

## Driver interface

All backend operations go through the `GpuDevice` trait:

```rust
pub trait GpuDevice {
    fn caps(&self) -> &Caps;
    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError>;
    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError>;
    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError>;
    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError>;
    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError>;
    fn pulse_wait(&self, pulse: &Pulse) -> Result<(), QuantaError>;
    // ... ~40 more methods
}
```

The `Gpu` struct wraps a `Box<dyn GpuDevice>`. Metal and Vulkan each implement this trait.
The validation layer wraps any `GpuDevice` and adds runtime checks.

## Feature flags

```toml
[features]
default = ["std", "metal"]
std = []
metal = ["std"]        # macOS/iOS only
vulkan = ["std"]       # Linux/Android/Windows
```

Without `std`, only types and the kernel language are available (for `no_std` environments).
