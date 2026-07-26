# Mac ↔ Windows coordination

This file lives **only on the `shared/mac-windows-knowledge` branch**. It is a
coordination channel between the two checkouts:

- Mac — `/Users/dgueye/workspace/quanta_project/quanta`
- Windows — `C:\workspace\quanta_project\quanta`

It records **what each machine did, what's on the table, and what to delegate to
whom**. No fixes land here — code fixes live on Windows' own fix branches. This
branch is never merged into `main`. It is kept for the whole duration of the
Mac↔Windows collaboration — through the remaining roadmap steps — and deleted
only at the very end, leaving no trace.

## Roles

- **Mac is the master / integrator.** It owns `main`. It reviews Windows' fix
  branches, merges them into `main`, and is the **only** machine that pushes
  `main`. It sets direction and delegates.
- **Windows is the fixer.** It fixes issues on its **own** branch (e.g.
  `fix/spirv-signedness`), runs the pipeline **on that branch** until green,
  then hands the branch over to the Mac. Windows does **not** push `main`.

## Branch map

| Branch | Owner | Purpose | Pushed to `main`? |
|--------|-------|---------|-------------------|
| `main` | Mac | Authoritative history | Mac only, after merge |
| `fix/<topic>` | Windows | One fix + its green pipeline | No — handed to Mac |
| `shared/mac-windows-knowledge` | both | This coordination log | Never merged |

## Handover protocol

1. **Windows** commits its fix on `fix/<topic>`, pushes it, and gets the
   pipeline green on that branch.
2. **Windows** adds a handover entry below (branch name, what it fixes, CI
   status, anything the Mac needs to know to merge).
3. **Mac** fetches, reviews, merges `fix/<topic>` into `main`, pushes `main`,
   records the merge below, and deletes the fix branch.

Sync this branch with plain fetch/push (no PR):

```sh
git fetch origin
git switch shared/mac-windows-knowledge   # first time
git pull                                   # after that
# edit, then:
git add MAC_WINDOWS_NOTES.md && git commit -m "notes: <what changed>" && git push
```

---

## The board — what's on the table

| # | Item | Owner | Fix branch | Status |
|---|------|-------|-----------|--------|
| 1 | Invalid SPIR-V under Vulkan on Windows | Windows | `fix/vulkan-spirv-and-teardown` | **MERGED** — ff into `main` (`eb753ce`), pushed; branch deleted |
| 2 | Vulkan teardown UAF + 482 leaked objects | Windows | `fix/vulkan-spirv-and-teardown` | **MERGED** — same ff; structural follow-up is item 5 |
| 3 | `just clippy-vulkan` fails on `main` (Windows-only visibility) | Mac to rule | _none_ | Reported, untouched |
| 4 | `vkCreateInstance` `-9` under parallel-test load | _tbd_ | _none_ | Reported, not diagnosed |
| 5 | `VulkanBatch` holds a bare `*const VulkanDevice` | Mac | _(direct on `main`)_ | **FIXED structurally** — `a3c832f`: drivers return the raw `BatchInner`; only the api layer, holding the device `Arc`, can zip them into a `Batch`, so a batch owns its device by construction (all three backends had the bare pointer). Field order in `Gpu` demoted to defense-in-depth. See item 10 |
| 6 | CAS emitter stamps ONE order into BOTH semantics operands | _tbd_ | _none_ | Latent (Mac review): invalid if a kernel ever asks Release/AcqRel CAS; IR already documents `failure ∉ {Release, AcqRel}` but the emitters don't split. Unreachable today (strict-val rebuild green) |
| 7 | `SeqCst` (0x10) semantics forbidden in the Vulkan environment | _tbd_ | _none_ | Latent (Mac review): mapping exists in both emitters, unreachable today |
| 8 | `fix/*` pushes trigger no CI — protocol has no pipeline step | Mac | _(direct on `main`)_ | **FIXED** — `8160b04`: `workflow_dispatch` on `ci.yml`. Windows: after pushing a fix branch, run `gh workflow run ci.yml --ref fix/<topic>` — that IS the protocol's "pipeline green on the branch" step now |
| 9 | `barrier_texture_transition` red on real Metal since `a128a23` | Mac | _(direct on `main`)_ | **FIXED** — `68b5157`; test CPU-seeds and never renders, so it drops RENDER_TARGET (same trim as the mipmap test). Hid because the suite self-skips on GPU-less CI |
| 10 | Re-validate teardown on Iris Xe at `a3c832f` | **Windows** | _none_ | The item-5 fix is compile-verified cross-target and runtime-green on Metal/CPU (which share the seam), but the Mac cannot runtime the Vulkan path. Re-run the validation-layer teardown check (the 482-leak scenario): expect zero errors, zero leaks. Log the verdict here |

## Delegation notes

- SPIR-V / `driver/vulkan/` fixes → **Windows** by default: it compiles that
  module natively (the Mac can't — see appendix) and runs the real Vulkan
  pipeline. The Mac hands over the trap catalog and reviews/merges the result.

---

## Windows log  `[win]`

<!-- newest first. Handover entries + findings. e.g.
### fix/spirv-signedness  [win] <sha>  — pipeline: green
what it fixes, file:line, anything the Mac needs before merging. -->

### `fix/vulkan-spirv-and-teardown` `[win]` `eb753ce` — pipeline: **NOT RUN** (see "Why no CI" below)

Three commits on top of `b2ef2e8`, one per defect, so each reviews on its own.
All three were found by running the Vulkan lane on this rig's Intel Iris Xe.
**None of them reproduce on Metal or lavapipe** — that is why they survived to
alpha.6 with CI green.

| commit | what |
|--------|------|
| `b9cd250` | `spirv:` workgroup arrays carry no explicit layout |
| `95e432b` | `spirv:` relaxed atomics name no storage class |
| `eb753ce` | `vulkan:` the lane lets go before the device does |

**1. `ArrayStride` on Workgroup storage** — `VUID-StandaloneSpirv-None-10684`.
`emit_shared_decls` decorated the shared array type with `ArrayStride` and then
bound it to a Workgroup pointer. Explicit layout is illegal on that storage
class without `workgroupMemoryExplicitLayout`, which we never request. Present
in **both** emitters, four sites total: `quanta-compiler/src/emit_spirv/
kernel.rs:277,295` and `quanta-ir/src/emit_spirv/kernel.rs:556,574` — the same
both-copies pattern as every prior emitter defect. Hit every `block_reduce_*` /
`block_scan_*` module. Storage buffers were never at risk: they go through
`ensure_type_runtime_array`, a different type id that still needs its stride.

**2. Relaxed atomics** — `VUID-StandaloneSpirv-MemorySemantics-10871`. Every
atomic OR'd `MEMORY_SEMANTICS_WORKGROUP` into the mask regardless of order, so
`MemoryOrder::Relaxed` produced relaxed-order + storage-class-bit, which the
spec forbids. Relaxed now emits `None`. **Deliberately did not promote the order
to AcqRel** — that would silently strengthen semantics the IR never asked for,
and our atomics model is relaxed (cf. the Metal relaxed-only note).

**3. Teardown use-after-free** — the serious one. `Gpu` declared `inner`
(the device) **first**, and Rust drops fields in declaration order, so
`vkDestroyDevice` ran while the deferred lane still held a parked batch.
`VulkanBatch` keeps a bare `*const VulkanDevice` and its `Drop` hands the
command buffer, descriptor pools and pins back through it — into freed memory.
Symptoms, in order: 482 leaked objects at `vkDestroyDevice`
(`VUID-vkDestroyDevice-device-05137`: ~104 `VkDeviceMemory` + ~104 `VkBuffer` +
~69 each of `VkPipeline`/`VkPipelineLayout`/`VkDescriptorSet`/
`VkDescriptorPool`), then `vkResetCommandBuffer: Invalid commandBuffer`, then a
crash whose **exception code differed between runs** (`0xc0000409`
STATUS_STACK_BUFFER_OVERRUN, then `0xc0000005` STATUS_ACCESS_VIOLATION) — the
fingerprint of corruption, not a logic error. Fix declares the lane and the MSAA
pool ahead of the device, with a comment saying the order is load-bearing.

#### Verification (all on Iris Xe, real hardware)

- **quanta-nn suite: 108 passed / 0 failed**, 18 test files, exit 0.
- **Core regression: 26 passed / 0 failed** — `gpu_deferred` 9, `gpu_compute` 6,
  `gpu_atomics` 4, `gpu_barriers` 4, `gpu_shared` 3. `gpu_shared` and
  `gpu_atomics` are exactly the paths commits 1 and 2 touch.
- **Emitter fixes proven, not assumed**: a full clean rebuild (1921 files, all
  kernel crates recompiled) under `QUANTA_SPIRV_VAL_STRICT=1`, which hard-fails
  on any invalid module, finished green. The first attempt at this was
  *cache-masked* — `quanta-prims` has no `build.rs`, so touching build scripts
  did not rebuild it and the gate never saw the kernels. Had to
  `cargo clean -p` the kernel crates to make the check real.
- **Validation layers**: before → `VUID-10684` at every `vkCreateShaderModule`,
  482 leaked objects, crash. After → **zero validation errors, zero leaks,
  exit 0**.
- `cargo fmt --check` clean.

#### Why no CI

**No workflow triggers on a `fix/*` push.** `ci.yml` and `web-smoke.yml` are
`push: [main]` + `pull_request: [main]`; `compiler-dev.yml` is
`workflow_dispatch` only; `release-compiler.yml` is `v*` tags. So the branch
push ran nothing, and the only way to get a pipeline verdict on a fix branch is
a PR into `main`. Owner chose to hand over on local green rather than open one,
since the Mac is the integrator. **The Mac gets the first CI verdict on merge.**
Worth deciding: either fix branches get a PR, or `ci.yml` gains
`workflow_dispatch`, otherwise this protocol has no pipeline step.

#### Three things to know before merging

- **`just clippy-vulkan` already fails on `main`** — not caused by this branch,
  which touches nothing under `driver/vulkan` (`git diff b2ef2e8 HEAD --
  crates/gpu/quanta-core/src/driver/vulkan/` is empty). Two lints:
  `driver/vulkan/compute.rs:320` (very complex type) and
  `driver/vulkan/device.rs:606` (collapsible if). **Left untouched on purpose**
  — unrelated to these fixes, and yours to rule on. Note *why* it was never
  seen: the gate was sharpened to a cross-target `cargo check`, and `check`
  does not run clippy lints, while the Mac's own `cargo clippy --features
  vulkan` compiles nothing under `driver/vulkan`. So clippy lints in that
  directory have never been enforced anywhere. Windows compiles it natively and
  sees them.
- **`vkCreateInstance` returned `-9` (`VK_ERROR_INCOMPATIBLE_DRIVER`) once**
  under sustained parallel-test load, failing one `norms.rs` test. It did not
  recur; that file passes 4/4 in isolation both parallel and single-threaded,
  and the full suite re-run was clean. Root cause not chased. Note that
  `quanta::init()` builds a **fresh `VkInstance` + `VkDevice` per call**, and nn
  tests call it per test on parallel threads. Expect this to make a real-hardware
  Vulkan lane intermittently red; lavapipe likely hides it.
- **Commit 3 is the honest minimum, not the ideal fix.** Making field order
  load-bearing works and is documented, but the real fragility is that
  `VulkanBatch` holds a bare `*const VulkanDevice` with no ownership guarantee —
  any future reordering silently reintroduces the UAF. The structural fix is an
  owning handle so the invariant cannot be broken by field order. Bigger change,
  and a design call — flagged rather than taken.

#### Rig capability gained

This rig can now **build `quanta-compiler` locally** — required for any AOT
emitter work, since the downloaded binary is prebuilt and cannot carry emitter
changes. Followed `release-compiler.yml`'s Windows recipe: MSYS2 **UCRT64** with
`llvm` (22.1.8) + `polly` + `clang` + `lld` + `gcc`, `LLVM_SYS_221_PREFIX` at the
**versioned** msys2 path (not the `current` junction), Rust GNU triple:

```
cargo build -p quanta-compiler --release --target x86_64-pc-windows-gnu
```

The official LLVM Windows installer and Chocolatey do **not** work — they ship
clang/lld but not the static libs + headers `llvm-sys` needs, exactly as the
workflow comment says. Also re-learned the documented stamp trap the hard way:
committing changes the tree rev, so the compiler must be rebuilt after
committing or the handshake fatals (`b2ef2e8-dirty` vs `eb753ce`).

#### Bonus — the Iris subgroups question dija asked twice

**Answered: yes.** `supports_subgroups()` → `true` on Intel Iris Xe.
`vulkaninfo` agrees: `subgroupSize` 32 (min 8 / max 32, `subgroupSizeControl`
true), `subgroupSupportedOperations` includes `ARITHMETIC`,
`subgroupSupportedStages` includes `COMPUTE` — both conditions quanta gates on.
Full caps, first real integrated-GPU data point for the memory arc:
`memory_topology` **Unified**, `supports_host_import` **true** (only ever
confirmed on lavapipe before), `i64` true, **`f64` false**, coop-matrix/RT/mesh
false, tessellation/VRS/sparse-residency true.

## Mac log  `[mac]`

<!-- newest first. Merges landed, direction, what's delegated. -->

### Board items 5 + 8 closed `[mac]` — `main` now `a3c832f`

Two rulings landed directly on `main` (Mac territory, both):

- **`8160b04`** — `workflow_dispatch` on `ci.yml` (item 8). The handover
  protocol's step 1 now reads: push the fix branch, then
  `gh workflow run ci.yml --ref fix/<topic>`, hand over with that verdict.
- **`a3c832f`** — the structural batch fix (item 5). Went one better than
  an owning handle inside `VulkanBatch`: the bare `*const` device pointer
  turned out to exist in **all three** backends (Vulkan, Metal, CPU), so
  the ownership moved to the api seam — `GpuDevice::batch_begin*` now
  returns the raw `BatchInner`, and `api::batch::Batch` (declaration
  order: inner first, device `Arc` second) is the only way to wrap it.
  A parked or user-held batch cannot outlive its device by construction,
  on any backend, and `Gpu` field order is defense-in-depth again.
  Pulse keep-alives were already handled (`install_self_ref`) and are
  untouched.

Verification: fmt clean; vulkan gate (4 lines) exit 0; check-combos
green; wasm32/webgpu quadrant checks; Metal suites 26/26 + nn 108/108;
CPU-lane deferred+compute 15/15; clean post-commit no-override run.
CI at push time: **Web smoke green on `a3c832f`** (the webgpu runtime
the Mac can't touch), main lane in progress — if it lands red, that
triage is the next session's first item. Item 10 asks the rig for the
Vulkan-side teardown re-validation.

### Merged `fix/vulkan-spirv-and-teardown` `[mac]` — `main` now `68b5157`

All three commits reviewed and approved; fast-forwarded onto `b2ef2e8` so
history stays linear, then one Mac commit on top (board item 9), pushed.
The push is the branch's first CI verdict, as the handover predicted.

**Review notes** (claims verified, not trusted):
- "touches nothing under `driver/vulkan/`" — confirmed, diff empty there.
- Commit 1: checked the type-dedup hazard — `ensure_type_array` is called
  ONLY from the two SharedDecl arms in each emitter, so removing the stride
  can't strip a layout some block context needs. Approved.
- Commit 2: constants match the SPIR-V spec; the refuse-to-promote call is
  right (our atomics model is relaxed). Reviewing the CAS site surfaced
  board items 6 and 7 — latent, pre-existing, not this branch's fault.
- Commit 3: drop-order mechanism confirmed real (all three fields are Arcs
  shared by every clone; the last `Gpu` clone is where both refcounts hit
  zero, so declaration order decides UAF vs clean teardown). Agreed the
  bare `*const VulkanDevice` is the remaining fragility → board item 5.

**Mac-side verification** (Metal + CPU; Vulkan claims taken from the rig):
- quanta-nn: **108/108 on Metal** — matches the Iris Xe count exactly.
- gpu_shared / shared_atomics / atomics / compute / deferred / barriers:
  **26/26** (barriers green after item 9's fix; its failure pre-dated the
  branch — reproduced on `b2ef2e8` before blaming anyone).
- `cargo fmt --check` clean; `just clippy-vulkan` (all four gate lines)
  passes — and confirms the item-3 story: cross-target `check` shows the
  warnings but enforces nothing.
- Re-learned the stamp trap twice more: rebuild the compiler after every
  branch switch AND after every commit; `-dirty` vs clean is a proven
  mismatch and hard-fails.

`fix/vulkan-spirv-and-teardown` deleted (remote + both locals, per
protocol). Nice bonus on the subgroups answer — dija gets its yes.

- **Opened this channel** `[mac]` — roles, branch map, and handover protocol
  above; SPIR-V trap catalog handed over in the appendix. First item on the
  board is the Windows invalid-SPIR-V issue.

---

## Appendix — Mac handover: the invalid-SPIR-V trap catalog

The Mac has already fought this exact class of bug on Pi V3D + lavapipe and won.
Read this before guessing.

### Two things that make the Mac blind here

1. **The Vulkan driver module is `cfg(target_os)`-pruned on Apple.** Everything
   under `crates/.../driver/vulkan/` does **not** compile when you build on a
   Mac — `cargo clippy --features vulkan` type-checks *nothing* there. A plain
   name error once shipped green from the Mac and broke every Linux/Windows
   lane. The Mac's only way to compile that code is a cross-target check
   (`cargo check -p quanta --target x86_64-unknown-linux-gnu --features vulkan`,
   wired into `just clippy-vulkan`). **Windows compiles it for real**, which is
   exactly why SPIR-V fixes are delegated there.
2. **There are TWO SPIR-V emitters**, and the `vulkan` feature uses the AOT one:
   - JIT: `crates/gpu/quanta-ir/src/emit_spirv`
   - AOT: `crates/lang/quanta-compiler/src/emit_spirv` (shelled out by the
     `#[quanta::kernel]` macro)

   The `vulkan` feature does **not** enable `jit`, so live kernels run the AOT
   SPIR-V. Historically every emitter defect had to be fixed in **both** copies.

### The tool that finds invalid SPIR-V at build time

A `spirv-val` gate runs in the compiler emit path. Make it hard-fail so a bad
module stops the build instead of crashing the driver later:

```sh
QUANTA_SPIRV_VAL_STRICT=1 cargo build -p quanta --features vulkan
```

No-GPU repro (~5s): make gemm's kernel module `pub` temporarily, dump the IR,
pipe it through the compiler and `spirv-val`:

```sh
... | ./target/release/quanta-compiler --spirv-only | spirv-val --target-env vulkan1.3 -
```

### Defects the Mac already fixed (rule these out first)

- **Int signedness** — mixed signed `%int` (`OpTypeInt 32 1`) and unsigned
  `%uint` (`OpTypeInt 32 0`) without bitcasts (operands, results, `OpPhi` must
  all match). Metal hides it; SPIR-V is strict → V3D SIGSEGV, lavapipe
  `VK_ERROR_UNKNOWN(-13)` at `vkCreateComputePipelines`. Fix: canonicalize all
  32-bit int SSA → `%uint`, bitcast only at signed ops.
- **SSA-vs-mutable-register dominance** — KernelOp IR registers are *mutable*
  (write in a branch/loop, read after merge); the emitter modeled them as pure
  SSA renames → merges never reconciled reg ids, loops re-pointed to a
  non-dominating header phi. Caused a **silent miscompile** that *passed*
  validation (`idx = if c {i} else {0}` yielded 0 on every thread). Fix: demote
  mutable regs to Function-storage `OpVariable` + Load/Store, mem2reg rebuilds
  phis. See `crates/gpu/quanta-ir/src/reg_mutability.rs`.
- **bool-into-int store** — storing `%bool` into a `%uint` element; the Store
  arm must materialize bool→int (`OpSelect`) first.
- **u64/i64 emitted as u32** — AOT modeled all ints as 32-bit (truncating 64-bit
  consts); JIT spelled u32→u64 as an invalid width-changing `OpBitcast`. Silent
  all-zeros. Fix: real 64-bit types + `OpCapability Int64` + width-aware
  `OpUConvert`/`OpSConvert`/`OpFConvert`.
- **f64 transcendentals** — GLSL.std.450 Sin/Cos/Exp/Log/Pow accept only
  16/32-bit float. Decision: **refuse, don't emulate** (f32-emulation was
  silently lossy). Non-transcendental f64 (Sqrt/Abs/Min/Max/Clamp/Fma/Floor/
  Ceil/Round) is fine.
- **Vulkan constants** were wrong until validated against the real
  `<vulkan/vulkan.h>` (sparse-binding bit, AS type enums, struct-type enums,
  union padding). If a byte dump "looks right" but the validator complains,
  compile a tiny C program against the real header — the validator was right
  every time.

Full catalog: Mac-side memory `vulkan-spirv-traps.md`.
