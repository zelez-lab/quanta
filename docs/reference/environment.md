# Environment Variables

Every environment variable Quanta reads, what it changes, and whether it
takes effect at **build time** (inside the proc macros / the
`quanta-compiler` binary, while `cargo build` runs) or at **runtime**
(inside the driver, while your program runs).

Most of these are diagnostics you will never set. The toolchain knobs at
the top of the table (`QUANTA_COMPILER`, `QUANTA_ACCEPT_STALE_COMPILER`,
`QUANTA_NO_DOWNLOAD`, `QUANTA_SKIP_METALLIB`) are the ones a
git-dependency consumer or an offline rig occasionally needs; see
[Getting Started — the ahead-of-time compiler](../getting-started.md#the-ahead-of-time-compiler-git-dependency-consumers).

| Variable | Effect | Scope | Notes |
|----------|--------|-------|-------|
| `QUANTA_COMPILER` | Absolute path to a `quanta-compiler` binary. Tried first in the discovery chain, ahead of the workspace `target/` dirs, `PATH`, the `~/.quanta/bin/` cache, and the GitHub-Release download. | Build time | Point it at a custom or cross-built compiler. The path must exist or it is ignored. |
| `QUANTA_ACCEPT_STALE_COMPILER` | Downgrades a **proven** compiler/crate revision mismatch from a fatal build error to a note, letting the build proceed with the mismatched compiler. | Build time | Any non-empty value enables it. Only for rigs deliberately pinning a known-compatible compiler — a mismatched compiler can emit invalid SPIR-V that crashes some drivers. |
| `QUANTA_NO_DOWNLOAD` | Set to `1` to disable the automatic download of the compiler binary from GitHub Releases. Discovery still uses `QUANTA_COMPILER`, the workspace dirs, `PATH`, and the cache. | Build time | For CI and offline environments. Without a compiler the build stays soft (kernels JIT, shaders ship no binaries). |
| `QUANTA_SKIP_METALLIB` | Set to `1` to skip `xcrun` metallib compilation and emit no Apple binary. | Build time | For a Mac cross-compiling to a non-Apple target without the Metal toolchain. On macOS, a *missing* `xcrun` is otherwise a hard error (axiom A1: never silently drop the metallib). |
| `QUANTA_METAL_PLATFORMS` | Comma list selecting which Apple metallib variants the compiler attempts: `macos`, `ios`, `ios-sim` (e.g. `macos,ios`). Unset means all three. | Build time | By default each shader/kernel embeds a macOS metallib plus, when their SDK is installed, iOS-device and iOS-simulator variants; the runtime picks the one matching the build target. Use this to trim output to the platforms you ship. The override only limits what is *attempted* — an iOS variant whose SDK is absent still soft-skips (no error). |
| `QUANTA_CPU` | Set to `1` so `quanta::init()` includes the CPU software executor in device discovery, without calling `init_cpu()`. | Runtime | Requires the `software` feature. |
| `QUANTA_BACKEND` | Force device discovery to a single backend: `metal`, `vulkan`, or `cpu` (case-insensitive). `quanta::devices()` then returns **only** that backend's devices and never falls through to another; a forced-but-unavailable backend yields an empty list, so `quanta::init()` fails with an error naming the env var rather than silently picking a different backend. `cpu` includes the software device regardless of `QUANTA_CPU`. | Runtime | For deterministic backend selection in CI. Unset means the normal per-OS order (Apple: Metal; Linux/Android/Windows: Vulkan; then CPU under `QUANTA_CPU`), with the CPU software device engaging as a **loud last resort** when no GPU backend is found and nothing is forced — each unavailable lane and the fallback itself print a `quanta` line to stderr. An unrecognized value fails `init()` with a message listing the accepted values. |
| `QUANTA_VULKAN_LOADER` | Windows only: the Vulkan loader DLL name passed to `LoadLibraryA` (default `vulkan-1.dll`). Quanta resolves the loader at runtime on Windows — there is no link-time `vulkan-1.lib` dependency, so the Vulkan SDK is not needed to build. | Runtime | Point it at an alternate loader, or at a nonexistent name to deterministically exercise the missing-loader path (loud `quanta vulkan:` line + software fallback) on a machine that has Vulkan. |
| `QUANTA_VALIDATE` | Set to `1` to wrap the driver in the validation layer: it checks binding counts, field sizes, and use-after-free before work reaches the GPU, and **panics** on a violation (e.g. a write to a freed handle). | Runtime | A debugging aid — leave it off in production. See [Errors — Validation layer](errors.md#validation-layer). |
| `QUANTA_VALIDATE_VERBOSE` | Set to `1` to re-enable the per-kernel "skipped this backend" lines the compiler prints when a kernel fails a backend's capability check. | Build time | Build-time skip messages are silent by default; the runtime `NotSupported` error already carries the full report. |
| `QUANTA_SPIRV_VAL_STRICT` | Set to `1` to turn a `spirv-val` failure on emitted SPIR-V into a hard build error instead of a loud warning. | Build time | Recommended in CI. A no-op when `spirv-val` is not on `PATH`. |
| `QUANTA_LOWER_DEBUG` | Set to a kernel name (or `*` for all) to dump the WASM-route lowering's per-local get/set events, with loop depth and the stable-register assignment each local rebases onto. | Build time | Diagnoses register-lifetime and loop-carried-address lowering. |
| `QUANTA_LOWER_DUMP_INSTRS` | Set to a kernel name (or `*`) to dump the decoded WASM instruction stream with block nesting, before lowering. | Build time | The input side of the lowering pass. |
| `QUANTA_LOWER_DUMP_OPS` | Set to a kernel name (or `*`) to dump the final lowered `KernelOp` tree. | Build time | The output side of the lowering pass. |
| `QUANTA_DUMP_KERNEL` | Set to a kernel name to dump the lowered kernel definition once its scope check passes. | Build time | Complements `QUANTA_SCOPE_DUMP`. |
| `QUANTA_SCOPE_DUMP` | Set (any value) to additionally dump the scope-validity analysis alongside `QUANTA_DUMP_KERNEL`. | Build time | For chasing an emitter scope-check rejection. |

Kernels lower at build time inside the `quanta-compiler` binary. Touch a
kernel's source to force it to re-lower — the diagnostics above only fire
while a kernel is actually being compiled.

## `QUANTA_BUILD_REV`

`QUANTA_BUILD_REV` is **not a user knob**. It is a build stamp: the
`quanta-dsl-core` and `quanta-compiler` build scripts bake the source revision
into each artifact, and `quanta-compiler --rev` prints it. The macros
compare the two to detect a stale compiler (the mismatch handshake behind
`QUANTA_ACCEPT_STALE_COMPILER` above). Do not set it by hand.
