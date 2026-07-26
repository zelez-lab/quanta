# Mac ↔ Windows coordination

This file lives **only on the `shared/mac-windows-knowledge` branch**. It is a
coordination channel between the two checkouts:

- Mac — `/Users/dgueye/workspace/quanta_project/quanta`
- Windows — `C:\workspace\quanta_project\quanta`

It records **what each machine did, what's on the table, and what to delegate to
whom**. No fixes land here — code fixes live on Windows' own fix branches. This
branch is never merged into `main`, so when the work it tracks has landed it can
be deleted with no trace.

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
| 1 | Invalid SPIR-V under Vulkan on Windows | Windows | _tbd_ | Investigating |

## Delegation notes

- SPIR-V / `driver/vulkan/` fixes → **Windows** by default: it compiles that
  module natively (the Mac can't — see appendix) and runs the real Vulkan
  pipeline. The Mac hands over the trap catalog and reviews/merges the result.

---

## Windows log  `[win]`

<!-- newest first. Handover entries + findings. e.g.
### fix/spirv-signedness  [win] <sha>  — pipeline: green
what it fixes, file:line, anything the Mac needs before merging. -->

_(nothing yet — Windows adds here)_

## Mac log  `[mac]`

<!-- newest first. Merges landed, direction, what's delegated. -->

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
