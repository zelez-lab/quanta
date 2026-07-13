# Architecture

## Crate structure

Quanta ships as **three consumable surfaces over one shared substrate**:
`quanta-core` (the device line), `quanta-render` (the render face), and
the `quanta` facade (which adds the compute face and re-exports the
other two behind features).

```
crates/quanta-core/             Shared GPU substrate (no_std, zero external deps)
+-- src/
|   +-- lib.rs                  Entry points: init(), devices(), init_cpu()
|   +-- api/
|   |   +-- gpu.rs              Gpu handle — shared operations + capability queries
|   |   +-- gpu/compute.rs      Compute-face Gpu methods (feature = "compute")
|   |   +-- field.rs            Field<T>, MappedField<T>
|   |   +-- wave.rs             Wave (bound kernel + bindings)
|   |   +-- pulse.rs            Pulse (GPU completion signal)
|   |   +-- texture.rs          Texture, TextureDesc, TextureView, Sampler,
|   |   |                       NativeTextureHandle (zero-copy export)
|   |   +-- pipeline.rs         Pipeline, PipelineDesc, ShaderSource
|   |   +-- render_pass.rs      RenderPass, ColorTarget, DepthTarget
|   |   +-- surface.rs          SurfaceConfig, SurfaceTarget, PresentMode
|   |   +-- ray_tracing.rs      GeometryDesc, RayTracingPipelineDesc
|   |   +-- device.rs           GpuDevice trait (driver interface)
|   |   +-- types.rs            Caps, Vendor, Format, CompareOp, FieldUsage, …
|   |   +-- error.rs            QuantaError (implements core::error::Error)
|   +-- driver/
|       +-- metal/              Metal backend (macOS/iOS) — raw ObjC FFI
|       +-- vulkan/             Vulkan backend (Linux/Windows) — raw Vulkan FFI
|       +-- cpu/                CPU software executor (feature = "software")
|       +-- webgpu/             WebGPU backend (wasm32, feature = "webgpu")
|       +-- validation.rs       Debug validation layer (wraps any driver)

quanta/                         The facade (root crate)
+-- src/
|   +-- lib.rs                  Re-exports quanta-core wholesale; gates the
|   |                           compute face behind `compute` and re-exports
|   |                           quanta-render behind `render`
|   +-- kernel.rs               KernelBinary, GpuType (compute face)
|   +-- scan.rs                 Scan library (compute face)
|   +-- intrinsics.rs           GPU intrinsics for the wasm32 kernel route

crates/quanta-render/           The render face — a real crate. Depends on
+-- src/                        quanta-core (+ the render DSL macros) ONLY;
|   +-- lib.rs                  never on the compute / IR-JIT / wasm-lowering
|   |                           stack. Re-exports quanta-core wholesale.
|   +-- gpu_ext.rs              RenderGpu — sealed extension trait on Gpu
|   |                           (pipeline, render, render_target, create_surface, …)
|   +-- render_builder.rs       Chainable RenderBuilder (draw recording + pulse)
|   +-- surface_wrap.rs         Surface / SurfaceFrame (acquire → render → present)
|   +-- mesh_shader.rs          MeshPipeline (typed wrapper)
|   +-- tessellation.rs         TessellationPipeline (typed wrapper)
|   +-- ray_tracing_wrap.rs     AccelerationStructure, RayTracingPipeline
|   +-- vrs_wrap.rs             VrsState

crates/quanta-ir/               Shared IR definition (zero external deps)
+-- src/
|   +-- lib.rs                  KernelOp, KernelDef, CompilerOutput
|   +-- wire/                   Binary serialization (custom format, no serde)

crates/quanta-compute-dsl/      Compute-face proc macros (depends on syn + quote)
+-- src/
|   +-- lib.rs                  #[kernel], #[device], #[gpu_type],
|   |                           derive(Fields), derive(Uniforms), import_devices!
|   +-- kernel_macro.rs         #[kernel] expansion + auto-dispatch wrapper
|   +-- compile_via_wasm.rs     Kernel body → rustc → wasm32 → KernelDef
|   +-- kernel_type_inference.rs  Per-field scalar-type inference

crates/quanta-render-dsl/       Render-face proc macros (depends on syn + quote)
+-- src/
|   +-- lib.rs                  #[vertex], #[fragment], #[tess_control],
|   |                           #[tess_eval], #[task], #[mesh],
|   |                           #[ray_gen], #[closest_hit], #[miss], derive(Vertex)
|   +-- shader_macro.rs         Render-stage macro bodies

crates/quanta-dsl-core/         Shared behind both DSL faces (quanta-ir + syn/quote)
+-- src/
|   +-- binary.rs               Compiler-binary discovery, rev handshake, invocation
|   +-- shader_types.rs         Shader-parameter parsing + body extraction
|   +-- emit_msl.rs, emit_wgsl.rs, shader_emit.rs   JIT-path text emitters

crates/quanta-wasm-lowering/    WASM → KernelOps translator (kernel route)

crates/quanta-compiler/         LLVM compiler binary (inkwell / LLVM 22)
+-- src/
|   +-- main.rs                 CLI entry: stdin KernelDef -> stdout CompilerOutput
|   +-- to_llvm.rs              KernelOp -> LLVM IR module builder
|   +-- targets/…               NVPTX / AMDGPU / SPIR-V intrinsics
|   +-- emit_msl.rs             KernelOp -> MSL source text
|   +-- emit_wgsl.rs            KernelOp -> WGSL source text
```

## The crate split (`quanta-core` / `quanta-render` / the facade)

Quanta has two faces — GPU **compute** (CUDA-like) and **rendering**
(Metal/Vulkan/WGSL-like). Each face is a Cargo feature on the facade
(`compute` and `render`, both default-on), and a consumer pulls in only
what it uses:

- **`quanta-core`** is the substrate both faces share: the `Gpu` handle,
  the sealed `GpuDevice` trait, the four drivers, fields, textures,
  errors, and capability queries. The compute/render boundary cuts
  *through* the driver line — the `GpuDevice` trait itself speaks the
  render data model (`PipelineDesc`, `RenderPass`, `RenderOp`) and the
  compute data model (`Wave`, `Batch`, queues) — so both data models
  live in `quanta-core`, each behind the matching feature
  (`quanta-core/render`, `quanta-core/compute`).
- **`quanta-render`** is the render face: the `RenderGpu` extension
  trait, the chainable `RenderBuilder`, the typed
  mesh/tessellation/VRS/ray-tracing wrappers, and `Surface`
  presentation. It depends on `quanta-core` (render feature on) plus
  the render-stage DSL macros — **never** on the compute stack: a
  render-only consumer's dependency graph contains no kernel lowering,
  no JIT, no WASM machinery.
- **The `quanta` facade** re-exports `quanta-core` wholesale, compiles
  the compute face (`#[quanta::kernel]`, waves, the scan library) in
  behind `compute`, and re-exports `quanta-render` behind `render`.

Because `quanta-render` is a foreign crate to `quanta-core`, the render
methods that were historically inherent on `Gpu` are now a **sealed
extension trait**, `RenderGpu` (a foreign crate cannot add inherent
methods across a crate boundary). It is implemented only for
`quanta_core::Gpu` through a `#[doc(hidden)]` device-handle hook, so
methods can still be added without a breaking change — same policy as
the sealed `GpuDevice`. Callers write `use quanta::RenderGpu;` (or the
`use quanta::*;` glob, which covers it).

Two CI invariants keep the boundary from regressing:
surface-disjointness (render-off `quanta` exposes no render type name,
checked from rustdoc JSON) and dependency-acyclicity (no
`quanta -> quanta-render` edge in a render-off build; render builds on
the substrate, never the reverse). See
`scripts/check-render-boundary.sh` and the four-combo build matrix in
CI (core-only / render / compute / both).

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

Step 1: Kernel body -> IR (quanta-compute-dsl/compile_via_wasm.rs)
    The macro renders the body as a wasm32 `extern "C"` function,
    compiles it with `rustc --target wasm32-unknown-unknown`, and
    lowers the WASM to KernelOps via `quanta-wasm-lowering`:
    KernelDef {
        name: "my_kernel",
        params: [FieldRead("input", slot=0), FieldWrite("output", slot=1)],
        body: [QuarkId{r0}, Load{r1,field=0,r0}, Const{r2,2.0}, BinOp{r3,r1*r2}, Store{field=1,r0,r3}],
    }

Step 2: Serialize + invoke compiler (quanta-dsl-core/binary.rs)
    KernelDef -> quanta_ir::serialize_kernel() -> [u8]
    Pipe bytes to `quanta-compiler` binary via stdin

Step 3: Compile to all targets (quanta-compiler) — binary-only
    KernelOp -> LLVM IR -> PTX (NVIDIA)
    KernelOp -> LLVM IR -> GCN ELF (AMD)
    KernelOp -> LLVM IR -> SPIR-V (Vulkan)
    KernelOp -> LLVM IR -> SPIR-V -> xcrun metal -> metallib (Apple)
    No text output (MSL/WGSL) in the build path.
    Text emitters (emit_msl.rs, emit_wgsl.rs) are reserved for the JIT path.

Step 4: Embed (quanta-compute-dsl)
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

All backend operations go through the `GpuDevice` trait (in
`quanta-core`):

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

The `Gpu` struct wraps an `Arc<dyn GpuDevice>`. Metal, Vulkan, CPU, and
WebGPU each implement this trait. The validation layer wraps any
`GpuDevice` and adds runtime checks. Typed wrappers (textures,
pipelines, samplers, views, the render wrappers) hold a clone of the
device `Arc` and release their driver resource on `Drop`, exactly once
(guarded by a `live` flag).

## Feature flags (the `quanta` facade)

```toml
[features]
default = ["metal", "vulkan", "render", "compute"]
std = ["quanta-core/std", "quanta-render?/std"]
metal = ["std", "quanta-core/metal", "quanta-render?/metal"]
vulkan = ["std", "quanta-core/vulkan", "quanta-render?/vulkan"]
vulkan-portability = ["vulkan", "quanta-core/vulkan-portability", "quanta-render?/vulkan-portability"]
render = ["quanta-core/render", "dep:quanta-render"]
compute = ["quanta-core/compute", "dep:quanta-compute-dsl"]
software = ["std", "jit", "compute", "quanta-core/software", "quanta-render?/software"]
jit = ["quanta-ir/jit", "quanta-ir/op-matrix-cases", "quanta-core/jit"]
webgpu = ["std", "jit", "compute", "quanta-core/webgpu", "quanta-render?/webgpu"]
```

Cargo features are **target-independent**, but the two backend faces
that are always on by default (`metal`, `vulkan`) are made portable by
`cfg(target_os)`-gating their **code**, not by juggling features per
target:

- The **Metal** face (`driver::metal`, its discovery arm, the
  `objc_msgSend` link) is gated to `any(target_os = "macos", "ios")`.
- The **Vulkan** face (`driver::vulkan`, its discovery arm, the
  `#[link(name = "vulkan")]` stanza) is gated to
  `any(target_os = "linux", "android", "windows")`. On Apple targets it
  is compiled out entirely, so a plain macOS build with `vulkan` on
  needs **no** Vulkan loader — this is what makes both backends safe to
  default-enable (the "MoltenVK link trap": default-enabling the
  *feature* alone, without the code gate, would break every Mac build at
  link time).
- `vulkan-portability` extends the Vulkan code gate to include macOS
  (against MoltenVK's `libvulkan`). It is additive over `vulkan` and is
  **not** in the defaults — it forces a link dependency a plain Mac
  build must not carry.

The result: one dependency line with default features compiles the
right face on every OS. Other notes on the faces:

- `render` — the render face. Pulls in `quanta-render` and re-exports
  it (the `RenderGpu` trait, builders, typed wrappers, render-stage
  macros).
- `compute` — the compute face: the `#[quanta::kernel]` /
  `#[quanta::device]` macros, `Wave` dispatch, batches, queues, the
  scan library. Symmetric to `render`.
- `software` and `webgpu` imply `compute` (those drivers' whole job is
  executing kernels).
- A **headless compute** consumer builds with
  `default-features = false, features = ["metal", "vulkan", "compute", "jit"]`:
  zero render code compiled, no render type on the surface. (List the
  backends you target; both stay portable across OSes.)
- A **render-only** consumer builds with
  `default-features = false, features = ["metal", "vulkan", "render"]` —
  or depends on `quanta-render` directly — and compiles zero kernel
  machinery.

Without `std`, only types and the kernel language are available (for
`no_std` environments). The `software` feature adds a full CPU executor
that interprets kernel IR with barrier-correct workgroup execution. The
`jit` feature enables `gpu.wave_jit()` for runtime compilation of
serialized KernelDef IR.

## Dependency philosophy

The runtime crates (`quanta-core`, `quanta`, `quanta-render`) have
**zero external dependencies** (aside from the workspace DSL crates and
`quanta-ir`). GPU drivers use raw FFI:

- **Metal**: `objc_msgSend` / `sel_registerName` / `objc_getClass` via `extern "C"`
- **Vulkan**: `vkCreateInstance` / `vkCreateBuffer` / etc. via `#[link(name = "vulkan")]` (the link stanza is `cfg(target_os)`-gated to Linux/Android/Windows, plus macOS only under `vulkan-portability`)

No `metal-rs`, no `ash`, no `objc` crate. Raw FFI gives minimal binary size,
fast builds, and total ABI control.

Only the DSL crates (`quanta-compute-dsl`, `quanta-render-dsl`, and the
shared `quanta-dsl-core`) depend on `syn` + `quote` (proc-macro
requirements), and `quanta-compiler` depends on `inkwell` / LLVM 22
(build-time only).

## Math-first internals

The runtime follows a math-first discipline:

- **Zero heap allocation on hot paths** -- dispatch, bind, and wait touch no allocator
- **AtomicU64 handles** -- GPU resource handles are atomic integers, not Arc/Rc
- **RwLock for device state** -- readers (dispatch) never block each other
- **Inline arrays** -- binding slots, push constants stored in fixed-size arrays
- **Result propagation** -- all driver calls return `Result`, no mutex unwrap panics

## Line count

~45,000 lines of Rust across all workspace crates (runtime + IR + macros + compiler).
