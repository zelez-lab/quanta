# Getting Started

GPU compute in 3 minutes. You know Rust. You may not know GPUs.

## Add dependency

Quanta is not on crates.io yet, so add it from the git repository:

```sh
cargo add quanta --git https://github.com/zelez-lab/quanta
```

That produces the following line in your `Cargo.toml`:

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta" }
```

**That one line is all you need on every OS.** The defaults
(`["metal", "vulkan", "render", "compute"]`) compile the right GPU face
for the target with zero backend knowledge on your part: **Metal** on
macOS / iOS, **Vulkan** on Linux / Android / Windows. There is no
per-target feature table to maintain — each backend's code is
`cfg(target_os)`-gated, so enabling both `metal` and `vulkan` is safe:
the one that can't run on the target is pruned at compile time. In
particular a plain macOS build needs **no** Vulkan loader — the Vulkan
face is compiled out on Apple targets.

Pin to a specific revision with `--rev <sha>` (or `--tag <tag>` once
tagged releases exist) for reproducible builds.

### Compute, rendering, or both?

Quanta has two faces: **GPU compute** (the CUDA-like side — kernels,
fields, dispatch) and **rendering** (render passes, graphics pipelines,
mesh/tessellation/ray-tracing, presentation). Each face is a Cargo
feature — `compute` and `render`, both on by default
(`default = ["metal", "vulkan", "render", "compute"]`) — so a lean
consumer can compile only the face it uses. **Backend selection is
automatic** (see above); the opt-out below is about trimming *faces*,
not about picking a GPU driver. There are no negative features — you
turn defaults off and name what you want back.

- **Compute only** (a database, an ML runtime, any GPGPU app) — turn the
  defaults off and keep the compute face:

  ```toml
  [dependencies]
  quanta = { git = "...", default-features = false, features = ["metal", "vulkan", "compute", "jit"] }
  ```

  Your build has **no rendering types on its surface** and skips
  compiling the render stack entirely. Listing both `metal` and
  `vulkan` keeps the build portable across OSes; drop one if you only
  ever target the other, or use `software` for a CPU-only build.

- **Render only** (a UI toolkit, a compositor) — keep the render face:

  ```toml
  [dependencies]
  quanta = { git = "...", default-features = false, features = ["metal", "vulkan", "render"] }
  ```

  This compiles **zero kernel machinery**: no compute face, and the
  kernel-lowering/JIT crates never enter your dependency graph. (A
  render-only consumer can equivalently depend on the `quanta-render`
  crate directly.)

- **Compute + rendering** (a graphical app) — keep the defaults:

  ```toml
  [dependencies]
  quanta = { git = "..." }    # metal + vulkan + render + compute, right face per OS
  ```

> **MoltenVK on macOS?** Vulkan is compiled out on Apple targets by
> default (so a plain Mac build needs no Vulkan loader). To run Vulkan
> on macOS through MoltenVK, add the `vulkan-portability` feature — it
> extends the Vulkan face to macOS and links MoltenVK's `libvulkan` at
> build time (which must then be on the loader path):
>
> ```toml
> quanta = { git = "...", features = ["vulkan-portability"] }
> ```

The `render` feature pulls in the `quanta-render` crate and re-exports
it: render types (`RenderBuilder`, `Pipeline`, `Surface`, …), the
shader-stage macros (`#[quanta::vertex]`, `#[quanta::fragment]`, …), and
the **`RenderGpu` extension trait** that carries the render methods on
`Gpu` (`gpu.render_target(...)`, `gpu.render(...)`,
`gpu.create_surface(...)`). Bring it into scope with
`use quanta::RenderGpu;` — or `use quanta::*;`, which covers it.

The rest of this guide is pure compute and works either way.

## The ahead-of-time compiler (git-dependency consumers)

The `#[quanta::kernel]` / `#[quanta::vertex]` / `#[quanta::fragment]`
macros shell out to the **`quanta-compiler`** binary at build time to
embed GPU binaries (SPIR-V, metallib, PTX). Inside the quanta workspace
the freshly built `target/release/quanta-compiler` is found
automatically — but a project consuming quanta as a **git dependency**
builds from a pristine checkout with no `target/`, so the macros search
`$QUANTA_COMPILER`, then `PATH`, then a cached download. Install it
once, from the SAME rev your dependency pins:

```sh
cargo install --git https://github.com/zelez-lab/quanta quanta-compiler --locked
```

(Building it needs LLVM installed — on macOS `brew install llvm`;
point `LLVM_PREFIX` at it if it isn't in a default location.)

You often don't have to install it at all: for every supported host
triple, Quanta publishes a pre-built `quanta-compiler` as a GitHub
Release asset (`quanta-compiler-<target>.tar.gz`, or `.zip` on Windows,
each with a `.sha256` sidecar) and downloads the matching one to
`~/.quanta/bin/` on first build. The archive is **self-contained**: the
binary sits at the root with its entire libLLVM dependency closure
bundled beside it (in `lib/` on Unix via an `$ORIGIN/lib` /
`@loader_path/lib` run path, flat next to the `.exe` on Windows), so it
runs on a machine with no system LLVM. A CI job extracts and runs every
archive on a clean runner with no toolchain before it can be released.
Set `QUANTA_NO_DOWNLOAD=1` to disable the download.

**Pinning a branch-tip rev** (`quanta = { git = "…", rev = "…" }`) instead
of a release tag? The exact rev you pin usually has no tagged release, so
there is no version-keyed binary for it — but a maintainer can publish a
**rev-exact** compiler binary on demand, and the downloader fetches it
automatically (it asks for the rev-exact asset first, then falls back to
the version-keyed one). If your build reports the compiler isn't present
for your rev, ask a maintainer to publish it (Actions → *Compiler Dev
Binary* → Run workflow → your ref), or build the compiler locally with the
`cargo install` command above.

Two things to know:

- **Keep it in sync — a mismatch is a hard error.** Re-run the install
  whenever you bump the quanta git rev. The macros probe the binary's
  embedded build rev (`quanta-compiler --rev`) and compare it to the
  quanta crate you're compiling against. A **proven** mismatch **fails
  the build** — a stale compiler can emit invalid SPIR-V that crashes
  some drivers, so it stops rather than silently ships bad codegen. The
  error names both revs and the reinstall command. To proceed anyway on
  a rig that deliberately pins a known-compatible compiler, set
  `QUANTA_ACCEPT_STALE_COMPILER=1`, which downgrades the mismatch to a
  note. (An *older* compiler that predates rev stamping can't be proven
  mismatched, so it only prints a loud `[quanta] WARNING` — the fatal
  path fires only on a proven difference.)
- **Without a compiler at all**, the build still succeeds: compute
  kernels JIT at runtime, but render shaders ship with **no binaries**
  and fail at pipeline creation. Watch for the `[quanta] note: …
  compiler not present` line in the build log. A compiler that can't
  *load* here (a downloaded release build whose bundled libLLVM is
  missing) is treated the same way — soft, JIT covers it.

### Offline rigs (no network, no git-remote SSH key)

For a machine that can't reach the git remote at all (a test rig with
no network, or no SSH key provisioned for a private remote), point the
`quanta` dependency at a local checkout instead. Edit the consuming
crate's `Cargo.toml` directly and replace the `git` dep with a `path`
dep:

```toml
# Cargo.toml on the rig, BEFORE
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta" }

# Cargo.toml on the rig, AFTER
[dependencies]
quanta = { path = "/home/rig/quanta" }
```

If the crate also pulls `quanta` in under a
`[target.'cfg(...)'.dependencies]` table (common for platform-specific
consumers), repeat the same edit there — every family that names
`quanta` as a git dep needs the path swap, not just the default one:

```toml
[target.'cfg(target_os = "linux")'.dependencies]
quanta = { path = "/home/rig/quanta" }
```

`/home/rig/quanta` is a checkout kept in sync with the rev your build
expects (same rule as the compiler binary above). The other fully
offline option is `cargo vendor`, which snapshots the whole dependency
graph, including `quanta`, into a local directory and repoints
`.cargo/config.toml` at it — heavier to set up but doesn't require
hand-editing every consumer's `Cargo.toml`.

> **Don't use a `[patch]` section for this.** A workspace-root
> `[patch]` override looks like it should replace the git dependency
> before cargo ever touches the network, but under cargo 1.97 it
> doesn't: cargo still resolves (and therefore fetches) the original
> `git` dependency during dependency resolution, `[patch]` or not. On a
> keyless rig that fetch fails outright —
> `ssh://git@github.com/zelez-lab/quanta.git?branch=main: no
> authentication methods succeeded` — before the build even starts.
> This was verified on a Raspberry Pi in two separate workspaces. The
> direct path-dependency edit above is what actually gets an offline
> rig building.

**PATH footnote for scripted/rig builds:** a non-interactive `ssh`
session does not source the shell profile that puts `~/.cargo/bin` on
`PATH`, so a bare `cargo` or `quanta-compiler` invocation in a script
or `nohup` job fails with "command not found" before anything else
runs. Call them by absolute path instead:

```sh
nohup ~/.cargo/bin/cargo build --release > build.log 2>&1 &
```

## Write a kernel

A kernel is a function that runs on the GPU. Thousands of copies run in parallel,
each on a different element.

```rust
use quanta::*;

#[derive(quanta::Fields)]
struct VecAdd {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

#[quanta::kernel]
fn vector_add(data: &VecAdd) {
    let i = quark_id();
    data.result[i] = data.a[i] + data.b[i];
}
```

`quark_id()` returns this quark's index. If you dispatch 1024 quarks,
`quark_id()` ranges from 0 to 1023.

`#[derive(quanta::Fields)]` tells the framework how to map your struct
to the GPU. `Vec<T>` fields become GPU storage buffers. Scalar fields
become push constants. No manual slot numbers, no manual binding.

`#[quanta::kernel]` compiles the function to all 5 GPU targets at build
time: metallib (Apple), SPIR-V (Vulkan), PTX (NVIDIA), GCN (AMD), and
WGSL (WebGPU). All are embedded in your binary. At runtime, the right
one runs on whatever GPU is present.

> **Targeting iOS?** iOS rejects a macOS-platform metallib, so the Apple
> binary is emitted as up to three variants — macOS, iOS device, iOS
> simulator — and the runtime selects the one matching your build target.
> The iOS variants need the iOS SDKs (full Xcode); a Command-Line-Tools
> -only mac builds macOS-only. See
> [Macro Reference — platform-targeted metallibs](reference/macros.md#platform-targeted-metallibs-apple).

## Run it

```rust
use quanta::*;

#[derive(quanta::Fields)]
struct VecAdd {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

#[quanta::kernel]
fn vector_add(data: &VecAdd) {
    let i = quark_id();
    data.result[i] = data.a[i] + data.b[i];
}

fn main() -> Result<(), QuantaError> {
    let gpu = init()?;

    // Per-lane inputs so the output varies across the 1024 quarks —
    // result[i] = i + 2*i = 3*i.
    let mut data = VecAdd {
        a: (0..1024).map(|i| i as f32).collect(),
        b: (0..1024).map(|i| (i * 2) as f32).collect(),
        result: vec![0.0f32; 1024],
    };

    // One line: upload, bind, dispatch, readback
    vector_add(&gpu, &mut data, 1024)?.wait()?;

    assert_eq!(data.result[0], 0.0);
    assert_eq!(data.result[1023], 3069.0);
    println!("GPU computed: {} elements", data.result.len());
    println!("first 5: {:?}", &data.result[..5]);
    println!("last 5:  {:?}", &data.result[data.result.len() - 5..]);
    Ok(())
}
```

The dispatch call is one line:

```rust
vector_add(&gpu, &mut data, 1024)?.wait()?;
```

That is the entire GPU interaction. Define your data, call the kernel, read
the results back from the same struct.

> **Note:** `main` returns `Result<(), QuantaError>` so the `?` operator can
> propagate errors out of the program. If you copy only the body of `main`,
> keep the signature too — `?` does not work inside a `fn main()` that
> returns `()`.

## What happened

1. **Build time**: `#[quanta::kernel]` compiled `vector_add` to native GPU
   binaries for all 5 targets. `#[derive(quanta::Fields)]` generated the
   metadata that maps struct fields to GPU buffer slots and push constant
   slots. Both happen at `cargo build` -- zero runtime compilation.

2. **`init()`**: Discovered the first available GPU and returned a `Gpu` handle.

3. **`vector_add(&gpu, &mut data, 1024)?`**: This single call did everything:
   - Allocated GPU storage buffers for each `Vec<T>` field
   - Uploaded CPU data to the GPU
   - Selected the right pre-compiled binary for your GPU vendor
   - Created a Wave (a kernel ready to dispatch) and bound all fields
   - Dispatched 1024 quarks on the GPU
   - Read results back into `data.result`

4. **`.wait()?`**: Blocked until the GPU finished. Returns a `Pulse`
   (a completion signal) that you wait on.

No shader files. No intermediate representations. No runtime compilation.
No manual slot numbers. No `gpu.write_field`. The GPU binary is baked into
your Rust binary at `cargo build`.

Quanta has **zero runtime dependencies** -- no `metal-rs`, no `ash`, no
`objc` crate. Drivers use raw FFI (`objc_msgSend`, `extern "C"` Vulkan
functions) for minimal binary size and maximum build speed.

## Workgroup sizes

Control how quarks are grouped for optimal GPU utilization:

```rust
#[quanta::kernel(workgroup = [256, 1, 1])]
fn vector_add(data: &VecAdd) {
    let i = quark_id();
    data.result[i] = data.a[i] + data.b[i];
}
```

See [Compute basics](computation/tutorials/compute-basics.md) for details on 1D/2D/3D
workgroup sizes.

## Structured GPU data

For kernels that operate on multi-field data (particles, vertices, game
entities), use `#[quanta::gpu_type]` to define a GPU-compatible struct:

```rust
#[quanta::gpu_type]
struct Particle {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}
```

The macro generates `#[repr(C)]`, `Copy`, `Clone`, `GpuType` impl, and
backend struct declarations (MSL/WGSL). Then use it as an element type
in your Fields struct:

```rust
#[derive(quanta::Fields)]
struct Simulation {
    particles: Vec<Particle>,
    dt: f32,
}
```

See [Fields and types](computation/tutorials/fields-and-types.md) for the full reference.

## Manual API

For power users who need fine-grained control over GPU memory, binding,
and dispatch, Quanta exposes the full manual API:

```rust
let a = gpu.field::<f32>(1024)?;
a.write(&data_a)?;
// ... see Expert: Manual API for the full story
```

See [Expert: Manual API](expert/manual-api.md) for manual field allocation,
explicit wave binding, double-buffering, and raw handles.

## Capability queries and graceful fallback

Some features only exist on certain backends or device families: ray tracing,
mesh shaders, variable rate shading, sparse residency. Query before you build:

```rust
let gpu = quanta::init()?;

if gpu.supports_ray_tracing() {
    // build BLAS, dispatch rays
} else if gpu.supports_mesh_shaders() {
    // mesh-shader culling fast path
} else {
    // classic vertex/fragment fallback
}
```

When a feature is not implemented for the active backend, the call returns
`QuantaError { kind: NotSupported(reason), .. }` rather than panicking. Branch on
the error kind to fall back without giving up:

```rust
match gpu.sparse_texture(&desc) {
    Ok(tex) => /* use sparse */,
    Err(e) if matches!(e.kind, QuantaErrorKind::NotSupported(_)) => {
        // fall back to a regular texture
    }
    Err(e) => return Err(e),
}
```

`NotFound` is the partner variant — it means a handle no longer points at a
live resource (typically a double-free or use-after-drop).

See [Compute basics: Error handling](computation/tutorials/compute-basics.md#error-handling)
and [Reference: Errors](reference/errors.md) for the full kind list.

## Next

- [Compute basics](computation/tutorials/compute-basics.md) -- execution model, workgroups, optimization
- [Fields and types](computation/tutorials/fields-and-types.md) -- GPU memory management
- [Multi-queue](rendering/tutorials/multi-queue.md), [Indirect commands](rendering/tutorials/indirect-commands.md),
  [Tessellation](rendering/tutorials/tessellation.md), [Mesh shaders](rendering/tutorials/mesh-shaders.md),
  [Ray tracing](rendering/tutorials/ray-tracing.md), [VRS](rendering/tutorials/variable-rate-shading.md),
  [Presentation](rendering/tutorials/presentation.md),
  [Async copy + printf](computation/tutorials/async-copy-and-printf.md) -- v0.1 advanced surface
