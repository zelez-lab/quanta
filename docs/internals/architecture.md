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
|   +-- gpu_type.rs             #[quanta::gpu_type] runtime support

crates/quanta-ir/               Shared IR definition (zero external deps)
+-- src/
|   +-- lib.rs                  KernelOp (47 variants), KernelDef, CompilerOutput
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
|   |                           #[ray_gen], #[closest_hit], #[miss],
|   |                           #[gpu_type]
|   +-- gpu_type.rs             #[gpu_type] expansion (repr(C), GpuType impl, MSL/WGSL)
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

crates/quanta-render/            Rendering front door (depends on quanta)
+-- src/
|   +-- lib.rs                  Re-exports render types + shader macros
|   +-- gpu_ext.rs              RenderGpu marker trait over quanta::Gpu
```

## Compute / render split (the `render` feature)

Quanta has two faces — GPU **compute** (CUDA-like) and **rendering**
(Metal/Vulkan/WGSL-like) — that share the `#[quanta::kernel]` family, the
IR, and the LLVM backend. They are separated by the **`render` Cargo
feature** on the root `quanta` crate, not by splitting the crate.

Why a feature and not a separate compute crate: the compute/render boundary
cuts *through* the driver line, not above it. The `GpuDevice` trait speaks
render types (`pipeline_create(&PipelineDesc) -> Pipeline`,
`render_begin -> RenderPass`, …) and all four backend drivers execute
render ops, so the render code can't live in a sibling crate without a
dependency cycle. Instead the render API modules, the render `GpuDevice`
methods, the render driver code, and the render `Gpu` methods all carry
`#[cfg(feature = "render")]` in `quanta`.

- **Headless compute** (Thiaba, `ai_project`, any GPGPU app) depends on
  `quanta` with `default-features = false`: zero render code compiled, no
  render type on the surface (~37% less of the `quanta` crate to build).
- **Graphical** consumers add `crates/quanta-render`, which turns the
  feature on, re-exports the render types + shader macros, and offers the
  `RenderGpu` marker. The render methods stay inherent on `quanta::Gpu`.

Two CI invariants keep the boundary from regressing (step 085):
surface-disjointness (render-off `quanta` exposes no render type name,
checked from rustdoc JSON) and dependency-acyclicity (no
`quanta -> quanta-render` edge). See `scripts/check-render-boundary.sh`.

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

Step 3: Compile to all targets (quanta-compiler) — binary-only
    KernelOp -> LLVM IR -> PTX (NVIDIA)
    KernelOp -> LLVM IR -> GCN ELF (AMD)
    KernelOp -> LLVM IR -> SPIR-V (Vulkan)
    KernelOp -> LLVM IR -> SPIR-V -> xcrun metal -> metallib (Apple)
    No text output (MSL/WGSL) in the build path.
    Text emitters (emit_msl.rs, emit_wgsl.rs) are reserved for the JIT path.

Step 4: Embed (quanta-macros/lib.rs)
    CompilerOutput -> proc_macro TokenStream:

    pub static MY_KERNEL_BINARY: KernelBinary = KernelBinary {
        nvidia: Some(b"...ptx..."),
        amd: Some(b"...elf..."),
        spirv: Some(b"..."),
        metallib: Some(b"..."),
        llvm_ir: None,
    };

    For JIT kernels (#[quanta::kernel(jit)]), the KernelDef IR is
    serialized and embedded instead:
    pub static MY_KERNEL_KERNEL_DEF: &[u8] = b"...serialized IR...";

    pub fn my_kernel(device: &Gpu) -> Result<Wave, QuantaError> {
        let binary = MY_KERNEL_BINARY.for_vendor(device.caps().vendor)?;
        device.wave(binary)
    }
```

```
                    RUNTIME
                    =======

    // Standard path: GPU hardware
    let gpu = quanta::init()?;
    //  -> discovers Metal or Vulkan device
    //  -> wraps in Gpu struct

    // Alternative: CPU software executor (feature = "software")
    let gpu = quanta::init_cpu();
    //  -> creates CPU IR interpreter with barrier-correct workgroup execution
    //  -> or set QUANTA_CPU=1 to make init() return the CPU executor

    let wave = my_kernel(&gpu)?;
    //  -> selects pre-compiled binary for this vendor
    //  -> creates pipeline state object (Metal) or compute pipeline (Vulkan)

    // JIT path (feature = "jit"): compile IR at runtime
    let wave = gpu.wave_jit(&MY_KERNEL_KERNEL_DEF)?;
    //  -> deserializes KernelDef, compiles to current vendor's format
    //  -> creates pipeline state object

    wave.bind(0, &input_field);
    //  -> records binding (slot -> field handle)

    gpu.dispatch(&wave, 1024)?;
    //  -> Metal: encodes dispatch to command buffer, commits
    //  -> Vulkan: records dispatch, submits to queue
    //  -> CPU: interprets IR sequentially per workgroup
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
default = ["metal", "render"]
std = []
metal = ["std"]                    # macOS/iOS only
vulkan = ["std"]                   # Linux/Android/Windows
software = ["std", "jit"]          # CPU software executor (IR interpreter)
jit = ["quanta-ir/jit", "quanta-ir/op-matrix-cases"]  # runtime JIT via wave_jit()
webgpu = ["std", "jit"]            # browser (wasm32)
render = ["quanta-macros/render"]  # the rendering face (see compute/render split)
```

Without `std`, only types and the kernel language are available (for `no_std` environments).
A **headless compute** consumer builds with `default-features = false` to drop the
`render` (and `metal`) defaults — see the compute/render split above.

The `software` feature adds a full CPU executor that interprets kernel IR with
barrier-correct workgroup execution. The `jit` feature enables `gpu.wave_jit()`
for runtime compilation of serialized KernelDef IR.

## Dependency philosophy

The `quanta` runtime crate has **zero external dependencies** (aside from its own
workspace crates `quanta-macros` and `quanta-ir`). GPU drivers use raw FFI:

- **Metal**: `objc_msgSend` / `sel_registerName` / `objc_getClass` via `extern "C"`
- **Vulkan**: `vkCreateInstance` / `vkCreateBuffer` / etc. via `#[link(name = "vulkan")]`

No `metal-rs`, no `ash`, no `objc` crate. Raw FFI gives minimal binary size,
fast builds, and total ABI control.

Only `quanta-macros` depends on `syn` + `quote` (proc-macro requirements), and
`quanta-compiler` depends on `inkwell` / LLVM 22 (build-time only).

## Math-first internals

The runtime follows a math-first discipline:

- **Zero heap allocation on hot paths** -- dispatch, bind, and wait touch no allocator
- **AtomicU64 handles** -- GPU resource handles are atomic integers, not Arc/Rc
- **RwLock for device state** -- readers (dispatch) never block each other
- **Inline arrays** -- binding slots, push constants stored in fixed-size arrays
- **Result propagation** -- all driver calls return `Result`, no mutex unwrap panics

## Line count

~45,000 lines of Rust across all workspace crates (runtime + IR + macros + compiler).
