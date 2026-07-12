# Compiler Internals

The `quanta-compiler` binary reads a `KernelDef` from stdin and writes a
`CompilerOutput` to stdout. It produces 5 GPU targets from one kernel definition.

## Compilation pipeline

```
KernelDef
    |
    +-- emit_msl.rs ----------> MSL source --> xcrun metal --> metallib (Apple)
    |
    +-- emit_spirv.rs ---------> SPIR-V binary (Vulkan, custom emitter)
    |
    +-- emit_wgsl.rs ----------> WGSL source (WebGPU, embedded as string)
    |
    +-- to_llvm.rs (subprocess) --> LLVM Module
                                       |
                                       +-- NVPTX target --> PTX binary
                                       +-- AMDGPU target --> GCN ELF binary
    |
    +-- (JIT only) emit_msl.rs --> MSL source string
    +-- (JIT only) emit_wgsl.rs -> WGSL source string
```

### Vertex/fragment SPIR-V

The SPIR-V emitter handles all execution models, not just `GLCompute`.
Vertex shaders emit with `OpEntryPoint Vertex`, fragment shaders with
`OpEntryPoint Fragment`. The same SPIR-V target generates both compute
and graphics pipelines. On Apple, the SPIR-V output is compiled to
metallib via `xcrun metal`, so vertex/fragment shaders also produce
native metallib binaries.

### LLVM subprocess isolation

LLVM's fatal error handler calls `abort()` on unsupported ops (e.g. `fsin` on
the SPIR-V target). To prevent this from killing the compiler before metallib +
SPIR-V + WGSL are serialized, LLVM compilation runs in a subprocess via
`--llvm-only <target>`. The parent process emits metallib/SPIR-V/WGSL first,
then spawns children for PTX/GCN. If a child crashes, the parent still succeeds.

### KernelOp coverage

50 ops across 5 emitters. Key additions:

| Op | SPIR-V | MSL | WGSL | LLVM |
|---|---|---|---|---|
| SatAdd/SatSub | OpULessThan + OpSelect | ternary | scalar | compare + select |
| CooperativeMMA | scalar fallback | simdgroup_matrix | scalar | unsupported |
| WaveShuffle/Ballot | OpGroupNonUniform* | simd_shuffle | subgroup* | unsupported |
| TextureLoad2D | OpImageFetch | tex.read() | textureLoad | unsupported |
| TextureWrite2D | OpImageWrite | tex.write() | textureStore | unsupported |

## LLVM IR emission (`to_llvm/emit.rs`)

The emitter uses an **alloca-based register file**. Each virtual register (`Reg(n)`)
maps to an `alloca` in the entry block. LLVM's `mem2reg` pass promotes these to SSA
registers during optimization.

```
KernelOp::BinOp { dst: Reg(3), a: Reg(1), b: Reg(2), op: Add, ty: F32 }

Emits:
    %r1 = load float, float* %reg1
    %r2 = load float, float* %reg2
    %r3 = fadd float %r1, %r2
    store float %r3, float* %reg3
```

After `mem2reg`:
```
    %r3 = fadd float %r1, %r2
```

### Kernel function signature

Each kernel becomes a void function with pointer parameters:

```llvm
define void @my_kernel(float* %field0, float* %field1, i32 %constant0) {
    ; ... body ...
}
```

- `FieldRead` / `FieldWrite` params become pointer arguments.
- `Constant` params become scalar arguments (push constants).

### Thread indexing intrinsics

Handled by the `GpuIntrinsics` trait. Each target provides its own:

```
QuarkId:
  NVPTX:  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
           %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
           %bsz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
           %gid = mul i32 %bid, %bsz
           %id  = add i32 %gid, %tid

  AMDGPU: %id = call i32 @llvm.amdgcn.workitem.id.x()
           %goff = call i32 @llvm.amdgcn.workgroup.id.x()
           ; ... similar computation

  SPIR-V: Uses OpLoad from gl_GlobalInvocationID built-in variable
```

## GpuIntrinsics trait

```rust
pub trait GpuIntrinsics {
    fn emit_quark_id(&self, builder: &Builder, module: &Module) -> IntValue;
    fn emit_local_id(&self, builder: &Builder, module: &Module) -> IntValue;
    fn emit_group_id(&self, builder: &Builder, module: &Module) -> IntValue;
    fn emit_group_size(&self, builder: &Builder, module: &Module) -> IntValue;
    fn emit_barrier(&self, builder: &Builder, module: &Module);
    fn emit_atomic_add(&self, ...) -> BasicValueEnum;
    fn emit_wave_shuffle(&self, ...) -> BasicValueEnum;
    // ...
}
```

Implementations:
- `NvptxTarget` (`targets/nvptx.rs`): uses `@llvm.nvvm.*` intrinsics
- `AmdgpuTarget` (`targets/amdgpu.rs`): uses `@llvm.amdgcn.*` intrinsics
- `SpirvTarget` (`targets/spirv.rs`): uses SPIR-V built-in variable conventions

## KernelOp -> LLVM IR mapping

| KernelOp | LLVM IR |
|----------|---------|
| `Load { field, index }` | `getelementptr` + `load` |
| `Store { field, index, src }` | `getelementptr` + `store` |
| `BinOp { Add, F32 }` | `fadd` |
| `BinOp { Add, I32 }` | `add` |
| `BinOp { Mul, F32 }` | `fmul` |
| `BinOp { Div, F32 }` | `fdiv` |
| `BinOp { Shl }` | `shl` |
| `BinOp { BitAnd }` | `and` |
| `Cmp { Lt, F32 }` | `fcmp olt` |
| `Cmp { Eq, I32 }` | `icmp eq` |
| `Cast { F32 -> I32 }` | `fptosi` |
| `Cast { I32 -> F32 }` | `sitofp` |
| `Cast { U32 -> F32 }` | `uitofp` |
| `Branch { cond, then, else }` | `br i1` + basic blocks |
| `Loop { count, body }` | `br` loop with phi + icmp + br |
| `Barrier` | target-specific intrinsic |
| `AtomicOp { Add }` | `atomicrmw add` |
| `MathCall { Sin }` | `call @llvm.sin.f32` |
| `MathCall { Sqrt }` | `call @llvm.sqrt.f32` |
| `SharedDecl` | `alloca` in address space 3 (shared/local) |
| `SharedLoad` | `load` from address space 3 |
| `SharedStore` | `store` to address space 3 |

## Optimization passes

The optimization level is set per-kernel via the proc macro attribute:

```rust
#[quanta::kernel]              // default: O3
#[quanta::kernel(opt = "O2")]  // explicit O2
#[quanta::kernel(opt = "O0")]  // no optimization (debug)
```

Maps to LLVM optimization levels:
- `O0`: `OptimizationLevel::None` — no passes, useful for debugging IR
- `O1`: `OptimizationLevel::Less` — basic cleanup
- `O2`: `OptimizationLevel::Default` — standard optimization
- `O3`: `OptimizationLevel::Aggressive` — full optimization (default)

Key passes that run:
- `mem2reg`: promotes allocas to SSA (eliminates our register file overhead)
- `instcombine`: algebraic simplification
- `loop-vectorize`: auto-vectorization within a quark (rare but possible)
- `gvn`: common subexpression elimination
- `dce`: dead code elimination

## MSL emitter (`emit_msl.rs`)

Translates `KernelOp` directly to Metal Shading Language text. Used by both
the JIT path **and** the standard build pipeline — the build-time path
goes through `crates/quanta-compiler/src/metallib.rs::compile_msl_to_metallib`,
which writes the emitted MSL to a temp `.metal` file and shells out to
`xcrun metal` + `xcrun metallib` to produce the metallib that ships in
the binary. (SPIR-V is also emitted, but the Metal backend prefers the
direct-MSL artifact when present.)

```
KernelDef { name: "foo", params: [FieldRead("a", 0, F32), ...], body: [...] }

Emits:
    #include <metal_stdlib>
    using namespace metal;

    kernel void foo(
        device const float* a [[buffer(0)]],
        ...
        uint quark_id [[thread_position_in_grid]]
    ) {
        ...
    }
```

### Metal atomic-order clamp

MSL's `device` address space only supports `memory_order_relaxed` for
atomic ops and fences. `memory_order_seq_cst` and `memory_order_acquire` /
`memory_order_release` are valid in the C++ standard but rejected by the
Metal compiler with `error: atomic operation must have memory_order_relaxed`.

The MSL emitter therefore **ignores the per-op `MemoryOrder`** on
`AtomicOp` / `AtomicCas` / `Fence` and unconditionally emits
`memory_order_relaxed`. The emitter for SPIR-V / LLVM / WGSL preserves
the requested order; only Metal clamps. This is documented in the
emitter source and was the root cause of the dev-Mac `gpu_atomics`
breakage closed in commit `d37e3ab`.

The behavioral consequence: cross-threadgroup ordering on Metal relies
on the implicit barriers from `Fence` / `Barrier` ops rather than on
the relaxed atomic itself. For most GPU-style workloads this matches
how Vulkan / D3D programs are written anyway (relaxed atomics +
explicit fences), so the clamp does not change correctness for any
shipped Quanta kernel.

## WGSL emitter (`emit_wgsl.rs`)

Direct text generation from `KernelOp`, same shape as the MSL emitter.
Used by the WebGPU backend on every platform — `quanta-compute-dsl` embeds
the WGSL string in the binary via `embed_wgsl`, and the runtime hands
it to `device.createShaderModule({ code })` at pipeline-build time.

```wgsl
@group(0) @binding(0) var<storage, read> a: array<f32>;
@compute @workgroup_size(64)
fn foo(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quark_id = gid.x;
    ...
}
```

## Alternative path: rustc compilation (`rustc_compile.rs`)

For kernels with `body_source` (raw Rust source captured by the proc macro),
an alternative path uses `rustc` to compile Rust to LLVM IR, which is then
fed into the LLVM pipeline for GPU target emission.

```
Rust source -> rustc (with custom target) -> LLVM IR -> retarget -> GPU binary
```

This handles complex Rust features (generics, traits, closures) that the
KernelOp parser does not yet support.

## Device function inlining

When a kernel calls `#[quanta::device]` functions or defines inner `fn` items,
the compiler emits them as real function definitions in the target:

**SPIR-V**: each device function becomes an `OpFunction` with its own
`OpFunctionParameter` entries. Call sites emit `OpFunctionCall` referencing
the function's `%id`. The SPIR-V optimizer (`OptimizerPass::Inline`) may
inline them, but the unoptimized module preserves the call structure.

```
; Device function
%relu = OpFunction %float None %relu_type
%x = OpFunctionParameter %float
       OpLabel
       ; ... body ...
       OpReturnValue %result
       OpFunctionEnd

; Call site inside kernel
%val = OpFunctionCall %float %relu %input_val
```

**LLVM (PTX, GCN)**: device functions are emitted as `always_inline` functions.
LLVM guarantees they are inlined during optimization, so the final binary has
no function call overhead.

**Metal**: since metallib is compiled from SPIR-V, the same OpFunction structure
applies before Metal's optimizer runs.

## Toolchain: discovery, rev handshake, and release packaging

The proc macros locate the `quanta-compiler` binary through a fixed
search chain (`quanta-dsl-core/src/binary.rs`): `QUANTA_COMPILER`,
then the workspace `target/{release,debug}` dirs, then `PATH`, then the
`~/.quanta/bin/` cache, then a download from GitHub Releases (unless
`QUANTA_NO_DOWNLOAD=1`).

Once resolved, the binary is probed **once** with `--rev` (null stdin)
before any kernel or shader is piped to it. That single probe classifies
it three ways:

- **Usable** — its embedded `QUANTA_BUILD_REV` matches this build, *or*
  it predates rev stamping (older binary, no `--rev`: a loud
  `[quanta] WARNING`, still used), *or* a proven mismatch was accepted
  via `QUANTA_ACCEPT_STALE_COMPILER`.
- **RevMismatch** — it printed a *different* rev. Fatal by default: a
  stale compiler has shipped `spirv-val`-invalid modules that segfault
  some drivers (v3dv), so the macro returns a build error rather than
  JIT-fall-back silently.
- **NotLoadable** — the loader killed it (a downloaded release build
  whose bundled libLLVM isn't present: dyld "Library not loaded" /
  ld.so exit 127 / `STATUS_DLL_NOT_FOUND`). Soft: kernels JIT, shaders
  ship no binaries.

Probing with null stdin first is deliberate — spawning a loader-killed
binary with piped stdin races its death and a broken-pipe write can
`SIGPIPE` the host `rustc` process on macOS.

### Release archive layout

`.github/workflows/release-compiler.yml` builds `quanta-compiler` for
`aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`,
`aarch64-unknown-linux-gnu`, and `x86_64-pc-windows-msvc` (the Windows
asset is built through the GNU triple but named for MSVC so the
downloader's target lookup matches). Each archive
(`quanta-compiler-<target>.tar.gz`, or `.zip` on Windows, plus a
`.sha256`) is **self-contained**: the binary at the root, its full
libLLVM dependency closure beside it. libLLVM alone is not enough — it
transitively links z3, zstd, tinfo, xml2, ffi and more, so the workflow
bundles everything `ldd` / `otool -L` reports except a glibc / loader /
compiler-runtime baseline. Unix bakes an `$ORIGIN/lib` (Linux) or
`@loader_path/lib` (macOS) run path so the binary finds the bundled
libraries beside itself; Windows places the DLLs flat next to the `.exe`
(the first place the loader looks). macOS re-signs ad-hoc after
`install_name_tool` rewrites, or arm64 refuses to launch it.

A separate `verify` job downloads each archive onto a **clean runner**
with no LLVM and no Rust toolchain, extracts it as a user would, checks
the sha256, and runs `--rev`. If bundling regressed, the loader kills it
there and the release is blocked.

## Adding a new KernelOp

1. Add the variant to `KernelOp` enum in `quanta-ir/src/lib.rs`.
2. Add serialization in `wire/encode.rs` and `wire/decode.rs`.
3. Add LLVM emission in `quanta-compiler/src/to_llvm/emit.rs`.
4. Add MSL emission in `emit_msl.rs`.
5. Add WGSL emission in `emit_wgsl.rs`.
6. Extend the WASM lowering in `quanta-wasm-lowering` so the op is
   recognised on the `rustc → wasm32 → KernelOps` route (the hand-written
   AST parser it replaced is gone).
7. Add roundtrip test in `quanta-ir/tests/roundtrip.rs`.
