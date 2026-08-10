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
| 3 | `just clippy-vulkan` fails on `main` (Windows-only visibility) | Windows | `fix/vulkan-clippy` | **MERGED** as `e518274` (see the Mac log for why SHAs changed); branch deleted |
| 4 | `vkCreateInstance` `-9` under parallel-test load | **Mac** | _(direct on `main`)_ | **FIXED structurally** — 3 commits: `8926960` (VkInstance refcounted, destroyed exactly once — this also killed a latent multi-GPU double-destroy: every device's Drop destroyed the SHARED instance), `9b23095` (`DeviceContext`: lane+pool+device in one Arc), `527cd69` (process-wide Weak registry — repeated `init()` returns clones of ONE device+lane; the storm can't happen: one instance+device per process no matter how many threads init). `tests/gpu_registry.rs` incl. an 8-thread storm. See item 10 |
| 5 | `VulkanBatch` holds a bare `*const VulkanDevice` | Mac | _(direct on `main`)_ | **FIXED structurally** — `a3c832f`: drivers return the raw `BatchInner`; only the api layer, holding the device `Arc`, can zip them into a `Batch`, so a batch owns its device by construction (all three backends had the bare pointer). Field order in `Gpu` demoted to defense-in-depth. See item 10 |
| 6 | CAS emitter stamps ONE order into BOTH semantics operands | Mac | _none_ | **SHIPPED** in `7177b8e` — the emitters split CAS Equal/Unequal semantics (failure operand strips Release/AcqRel) |
| 7 | `SeqCst` (0x10) semantics forbidden in the Vulkan environment | Mac | _none_ | **SHIPPED** in `7177b8e` — SeqCst mapping DELETED from both emitters (was reachable-invalid under VUID-04732; validator now armed with `--target-env=vulkan1.3`) |
| 8 | `fix/*` pushes trigger no CI — protocol has no pipeline step | Mac | _(direct on `main`)_ | **FIXED** — `8160b04`: `workflow_dispatch` on `ci.yml`. Windows: after pushing a fix branch, run `gh workflow run ci.yml --ref fix/<topic>` — that IS the protocol's "pipeline green on the branch" step now |
| 9 | `barrier_texture_transition` red on real Metal since `a128a23` | Mac | _(direct on `main`)_ | **FIXED** — `68b5157`; test CPU-seeds and never renders, so it drops RENDER_TARGET (same trim as the mipmap test). Hid because the suite self-skips on GPU-less CI |
| 10 | Re-validate teardown + storm on Iris Xe at `527cd69` | **Windows** | _none_ | **DONE, with findings** — (c) registry 5/5 incl. the 8-thread storm: the `-9` is structurally dead on real hardware; (b) nn 108/108 full-parallel, zero `-9`, zero incompatible-driver; (a) **VVL FAILED** — not the old 482-object catastrophe (still fixed), but two NEW race classes the one-device-per-process model made reachable: 4–7 validation errors per parallel run, zero single-threaded. → item 11 |
| 11 | Shared-device races: layout-cache double-create leak + device-wide command-pool threading hazard | Windows | `fix/vulkan-pool-and-layout-races` | **MERGED** as `767b6dc` (layout cache) + `938d492` (CmdLease); branch deleted |
| 12 | `gpu_surface` deadlocks on lavapipe — was MAIN'S RED | Windows | _(commit taken from the lab branches)_ | **MERGED** as `7d2ba67` — the standalone `fix/vulkan-surface-lock-order` branch was never pushed; the Mac landed `d796230` from `debug/surface-hang`. Main un-reds with this push |
| 13 | The dispatch CI lanes were broken by construction | Windows | `fix/ci-metal-lane-compiler` | **MERGED** as `6834818` + `6ba340a` + `e8dc60d`; branch deleted |
| 14 | `gpu_advanced` ABORTS on the GH macos-14 runner's paravirtual Metal | Mac | _(direct on `main`)_ | **RULED + FIXED** — probe-and-self-skip, suite-level (`1ccdc59`): `try_gpu` declines a device whose name contains "Paravirtual"; real hardware runs everything. Per-test narrowing waits on a runner log naming the offenders |
| 15 | Absolute `debug_registry_counts` asserts vs the shared device | **Mac** | _none yet_ | **SHIPPED** in `bbf4367` — `quanta::init_isolated()` / `init_cpu_isolated()` landed, every absolute-count test converted, the two `--test-threads=1` serializations removed from `ci.yml` |
| 16 | `wait_idle` races the queue — device idle bypasses `queue_lock` | Windows | `fix/vulkan-idle-race-and-surface-leak` | **FIXED on branch** — `3b37225`; verification in the sweep entry |
| 17 | Surface-creation error paths orphan the `VkSurfaceKHR` + partials | Windows | `fix/vulkan-idle-race-and-surface-leak` | **FIXED on branch** — `168d145` |
| 18 | `clippy::collapsible_if` in `render_pass.rs` from `2f68dfb` (Mac-blind class, item-3 redux) | Windows | `fix/vulkan-idle-race-and-surface-leak` | **FIXED on branch** — `fa645e4` |
| 19 | `surface_win32_target_rejected_off_windows` assumes `VK_KHR_win32_surface` never exists where the suite runs — false on this rig | **Mac** | _none_ | **RULING WANTED** — options in the sweep entry |
| 20 | Sparse 2D images never enable `sparseResidencyImage2D` (`VUID-VkImageCreateInfo-imageType-00971` ×3) | ? | _none_ | **NEW** — gpu_advanced under VVL; pre-existing, not this cycle's |
| 21 | Timestamp query read without reset (`VUID-vkGetQueryPoolResults-None-09401`) | ? | _none_ | **NEW** — same run; pre-existing |

## Delegation notes

- SPIR-V / `driver/vulkan/` fixes → **Windows** by default: it compiles that
  module natively (the Mac can't — see appendix) and runs the real Vulkan
  pipeline. The Mac hands over the trap catalog and reviews/merges the result.

---

## Windows log  `[win]`

<!-- newest first. Handover entries + findings. e.g.
### fix/spirv-signedness  [win] <sha>  — pipeline: green
what it fixes, file:line, anything the Mac needs before merging. -->

### Rig request closed: armed sweep + baseline + probe `[win]` — plus the fix branch the sweep demanded

**Base**: `2f68dfb` (CI-green tip, two commits past `2e2eadd`). Compilers
dev+release stamped `2f68dfb`; the layer load was PROVEN before trusting
any clean run (`VK_LOADER_DEBUG=layer` → "Insert instance layer
VK_LAYER_KHRONOS_validation" from the SDK dll) — a bad layer name fails
silently and would fake a clean sweep.

**Armed sweep, 11 suites — nine ran ZERO validation errors**: op_matrix
(6), differential (11), gpu_int_div_zero (4), wave_lifecycle (5),
gpu_deferred (9), icb_vulkan (1), gpu_texture_compute (16),
render_midflight_destroy (3), quanta-array (~290 across all binaries).
The narrow-int storage rows and div/rotate/saturate semantics are clean
on real Vulkan hardware. Two command corrections for the record:
quanta-array's hardware feature is spelled `vulkan` (`gpu-vulkan`
belongs to prims/blas/rand/fft), and quanta-bench defaults to metal so
the recorder needs `--no-default-features --features vulkan`.

**The two dirty suites → `fix/vulkan-idle-race-and-surface-leak`**,
three commits, one per defect:

| commit | what |
|--------|------|
| `3b37225` | `vulkan:` device idle waits under the queue lock |
| `168d145` | `vulkan:` surface creation unwinds every failure |
| `fa645e4` | `vulkan:` the occlusion reset collapses to a let-chain |

**1. VkQueue threading race** — render_triangle_test passed 20/20 but
VVL flagged `UNASSIGNED-Threading-MultipleThreads-Write`:
"vkQueueSubmit(): object of type VkQueue is simultaneously used in
current thread 3980 and thread 16596" — once in four parallel runs,
never single-threaded, the registry-era race signature. Cause:
`wait_idle` called `vkDeviceWaitIdle` bare, and a device idle counts as
a use of EVERY queue the device owns. Every other queue touch (submit,
present, sparse bind, queue idle) already held `queue_lock`; now the
device idle does too.

**2. Surface leak on the win32 leg** — gpu_surface's 10 headless
failures are this rig's known gap (no VK_EXT_headless_surface on Intel
Windows — environmental, unchanged). The 11th failure is the finding:
this is the first Windows Vulkan with `VK_KHR_win32_surface` to run the
suite, so `surface_win32_target_rejected_off_windows`'s dangling-HWND
surface actually gets CREATED. The caps query then returns garbage
(extent width 1673515620 against max 16384 —
`VUID-VkSwapchainCreateInfoKHR-imageFormat-01778`), vkCreateSwapchainKHR
fails, and the error path returned with the surface still alive:
`VUID-vkDestroyInstance-instance-00629`, "VkInstance has 1 leaked
objects: VkSurfaceKHR". The fix makes every failure past surface
creation unwind what already exists — surface after a failed swapchain
build; views/swapchain/fences/semaphores on the later fence, semaphore,
lease and map-insert failures; build_swapchain sweeps its own partial
views (a dropped CmdLease reclaims itself). → **item 19** for the
test-semantics ruling: post-fix the leak VUID is GONE and the 01778
flag on the bogus-handle ATTEMPT remains (driver-faithful —
garbage in, clean error out), but the test still asserts NotSupported
where creation now proceeds to an honest `Internal` failure. The
test's own comment says it assumed the extension absent everywhere the
suite runs; on this rig that's false. Options: skip when win32 support
is real, accept the error-not-NotSupported outcome as the pass, or
spend a real HWND on it.

**3.** Main's tip `2f68dfb` landed a `clippy::collapsible_if` in
`render_pass.rs` — the item-3 blindness class again (the cross-target
gate runs `check`, no lints; only this rig compiles driver/vulkan
natively). Let-chain, the same shape as `retire_or_park`.

#### Verification (Iris Xe; clean handshake, compiler re-stamped `fa645e4`, no ACCEPT_STALE)

- rtt parallel ×4 pre-commit + ×2 post-commit: **zero VVL errors**
  (from 1-in-4 runs dirty); gpu_surface: **zero leaked objects**;
  gpu_deferred / wave_lifecycle / render_midflight_destroy /
  gpu_advanced re-runs green under VVL.
- `cargo fmt --check` clean; native
  `cargo clippy -p quanta --no-default-features --features
  vulkan,jit,compute,render -- -D warnings` exit 0 (fails on `2f68dfb`);
  the `vulkan,compute` no-render combo checks clean.
- Cross-target gate lines are the dispatch run's business (no linux
  target on this rig). **Pipeline verdict: dispatch run 31358225504 on
  `fa645e4` — 13/13 jobs GREEN**, both GPU lanes, metal-validation and
  perf-regression included. The branch hands over with a full board.

#### Two NEW pre-existing classes — the sweep extended to gpu_advanced

18/18 pass, 4 VVL errors; both classes predate this cycle and the
branch touches neither. → items 20, 21:

- `VUID-VkImageCreateInfo-imageType-00971` ×3 — sparse 2D images take
  `VK_IMAGE_CREATE_SPARSE_BINDING/RESIDENCY` but device creation never
  enables the `sparseResidencyImage2D` feature. The caps probe says
  Iris Xe OFFERS it — an enable gap, not a support gap.
- `VUID-vkGetQueryPoolResults-None-09401` — a timestamp query is read
  while uninitialized: created, never reset before
  `vkGetQueryPoolResults` (`timestamp_query_create_returns_result`).

#### Iris Xe bench baseline (VVL off, release, stamped compiler)

Recorder: `cargo run --release -p quanta-bench --no-default-features
--features vulkan -- run --out bench-iris-xe.json`. The JSON, verbatim —
wire it in as the Windows baseline:

```json
{
  "platform": "windows-x86_64",
  "gpu": "Intel(R) Iris(R) Xe Graphics",
  "results": [
    {"name": "heavy_compute", "workload": "1000_elements", "elements": 1000, "gpu_ms": 1.1807, "cpu_ms": 17.3806},
    {"name": "heavy_compute", "workload": "10000_elements", "elements": 10000, "gpu_ms": 1.5717, "cpu_ms": 173.6220},
    {"name": "heavy_compute", "workload": "100000_elements", "elements": 100000, "gpu_ms": 8.2158, "cpu_ms": 1737.2388},
    {"name": "heavy_compute", "workload": "1000000_elements", "elements": 1000000, "gpu_ms": 81.1413, "cpu_ms": 17327.6437},
    {"name": "add_one_dispatch", "workload": "64x_dispatch_1048576_elements", "elements": 1048576, "gpu_ms": 0.7894},
    {"name": "mandelbrot", "workload": "3840x2160", "elements": 8294400, "gpu_ms": 123.5830}
  ]
}
```

#### Wave-cache probe — first real-hardware Vulkan numbers

creation only (warm cache) **0.1 µs/op** · creation only (cold, 32
distinct kernels) **5074.0 µs/op** · creation+dispatch+wait
**344.4 µs/op** · dispatch+wait (reused wave) **244.6 µs/op** ·
cacheable share **99.9 µs/op = 29%** of the per-op cost.

### The dispatch story — read this before the per-branch entries `[win]`

The protocol's new pipeline step got its first two real runs and they
FAILED — for reasons that were all main's, none the branches':

- **`gpu-tests-metal` had never executed** (its `if:` gates on dispatch or
  a `run-metal` PR label; pushes skip it). It builds the compiler only for
  its last step, so every earlier test build's kernel macros grabbed the
  DOWNLOADED alpha.6 release compiler (stamped `787cfde`) — proven
  handshake mismatch, hard fail on the first test step. → item 13.
- **`gpu-tests-vulkan` hung to its 25-min timeout in `gpu_surface`** — a
  REAL `527cd69` regression, and almost certainly what main's own
  cancelled 52-min push run was sitting in. Windows built a scratch lab
  (`debug/surface-hang`, a diagnosis job dispatched on the branch's own
  ci.yml): single-threaded 17/17 in 2 s, parallel wedges every time, and
  a gdb all-threads dump names the cycle — `surface_discard_impl` holds
  the `vk_surface_frames` write guard (if-let temporary) while taking
  `vk_surfaces` write, while `surface_acquire_impl` holds `vk_surfaces`
  read across its `vk_surface_frames` write. AB-BA; Rust's
  writer-preferring RwLock then parks every later reader. Unreachable
  pre-registry (private device per test). → item 12. Metal was never
  affected: its `take_frame` helper had the safe shape.
- After the item-12 fix the parallel suite still failed 2 tests on
  ABSOLUTE `debug_registry_counts` asserts — cross-test noise on a shared
  device, not a leak. The vulkan surface step now runs `--test-threads=1`
  exactly like every metal-lane step always has; whether leak tests
  should get an isolated-device constructor instead is YOURS to rule
  (flagged in item 13's second commit).

**Integration proof**: `debug/integration-proof` = `527cd69` + all four
fix branches merged, full ci.yml dispatched. Proof v1 confirmed both
original defects dead (the metal lane got past the handshake into real
test steps for the first time ever; the vulkan lane got past the
surface hang into the memory-surface steps) and surfaced two more
never-run findings behind them: `shared_field`'s absolute count assert
(same class, serialized — item 13 third commit) and the paravirtual
Metal abort in `gpu_advanced` (item 14, yours). **Proof v2 verdict
(run 30309860913): 11/13 lanes GREEN — including `gpu-tests-vulkan`,
fully green under dispatch for the first time ever. The only red is
`gpu-tests-metal`, failing solely on item 14's paravirtual abort
(same `IOGPUMetalResource` signature, same step; `metal-validation`
skipped behind it).** Once item 14 is ruled, dispatch runs can go
fully green. The scratch branches (`debug/surface-hang`,
`debug/integration-proof`) die after review.

### `fix/vulkan-pool-and-layout-races` `[win]` `88e4ba5` — pipeline: see "The dispatch story" (branch-alone run red on main-inherited lanes; integration proof is the meaningful verdict)

Item 10's verdict first, then the fixes it demanded. At `527cd69` on Iris Xe:

- **(c) registry 5/5** incl. the 8-thread storm — the `-9` is structurally
  gone, now confirmed on the hardware that produced it.
- **(b) nn 108/108**, 18 files, fully parallel, zero `-9`.
- **(a) VVL FAILED** — two NEW defect classes, both reachable only because
  the registry makes every thread share ONE device. Race-dependent and
  reproducible: parallel runs 4–7 VVL errors every time, single-threaded
  zero, every time.

Two commits on top of `527cd69`, one per defect:

| commit | what |
|--------|------|
| `e3e63dd` | `vulkan:` the layout cache admits one layout per signature |
| `88e4ba5` | `vulkan:` every command buffer carries its own pool |

**1. Leaked `VkDescriptorSetLayout`s** (`VUID-vkDestroyDevice-device-05137`:
3 leaked in gpu_atomics, 1 in gpu_shared). `acquire_descriptor_set_layout`
releases its mutex across `vkCreateDescriptorSetLayout`; racing threads all
miss, all create, and later inserts silently orphan earlier handles. The
counts are the smoking gun: 4 tests racing one signature → exactly 3
orphans. Fix: entry-API re-check under the lock — occupied means another
thread won; keep the incumbent, destroy ours.

**2. Command-pool threading hazard**
(`UNASSIGNED-Threading-MultipleThreads-Write`: `vkResetCommandBuffer` on a
`VkCommandPool` used simultaneously from two threads, gpu_barriers). EVERY
host access to a command buffer — allocate, reset, record, free — counts as
a use of ITS pool, which the spec requires externally synchronized. One
device per `init()` kept the device-wide pool accidentally private; one
device per PROCESS put it under every thread at once. VVL flagged the
reset, but one-shot recording on user threads (barriers, transfers,
queries) racing the lane thread is the same class with quieter symptoms.
A lock is the wrong shape — a user-held open batch would pin it across
arbitrary user code. The structural fix: the cache stores `(pool, buffer)`
pairs; `CmdLease` owns a pair exclusively — **holding the lease IS the
external synchronization**. Leases auto-return on drop (every early-error
path reclaims for free; several used to leak the CB), ride the fence
waiter through `submit_and_wait` so a submitted buffer re-enters the cache
only after the GPU is done, and the pool resets at REACQUISITION via
`vkResetCommandPool`. A failed fence wait `mem::forget`s the lease on
purpose — the old leak-on-device-loss stance survives (a possibly-PENDING
CB must never be recycled). The present path keeps its lease inside the
surface entry (per-frame re-begin is why lease pools carry the per-buffer
RESET bit). ICB and render-bundle secondaries move to per-object pools
destroyed with their object, and device teardown now sweeps live
ICBs/bundles — their descriptor pools used to slip through teardown
entirely. The device-wide `command_pool` field is GONE.

#### Verification (Iris Xe, real hardware)

- **VVL parallel ×4: zero validation errors, zero leaks** (from 4–7 errors
  per run at `527cd69`). One more clean-handshake VVL run post-commit at
  `88e4ba5` (compiler re-stamped, no ACCEPT_STALE): zero again.
- Core suites (atomics/barriers/compute/deferred/registry/shared) green
  under VVL throughout; **nn 108/108** on the branch.
- `cargo fmt` clean; native `cargo clippy --features vulkan` clean except
  the two item-3 lints (that branch's business, landed separately).
- Cross-target gate lines not run here (no x86_64-linux rust target on
  this rig) — the dispatched CI covers them; note the gate's purpose is
  Mac blindness, and this rig compiles `driver/vulkan` natively.

Merge-order note: independent of `fix/vulkan-clippy` — the two branches
touch the same files but disjoint hunks; either order fast-forwards with a
trivial merge.

### `fix/vulkan-clippy` `[win]` `5f6cecb` — pipeline: lint job GREEN on the branch run; full verdict via the integration proof

Item 3, as ruled — and the protocol road-test. `compute.rs`: the folded
dispatch pair is now `pub(crate) type DispatchRecord = ([u32; 3], [u32; 3])`
(the registry's `type Entry` prediction was exactly right), used at all five
sites that pass records around, not just the flagged line. `device.rs`:
`retire_or_park`'s nested if collapses to a let-chain; the
poisoned-park-lock fallthrough keeps its comment and meaning. Native
`cargo clippy --features vulkan -- -D warnings` exit 0, fmt clean.
The branch's own dispatch run: lint + all host/compiler/companion lanes
GREEN; only the two main-broken GPU lanes red (see the dispatch story) —
which is itself the road-test's finding: the pipeline step works, and the
first thing it did was catch main.
Rig note: `gh` was not installed here until now (scoop package, freshly
authenticated) — the first `gh workflow run` in the protocol's history
happened on this branch.

### `fix/vulkan-surface-lock-order` `[win]` `d796230` — pipeline: proven in the lavapipe lab; full verdict via the integration proof

Item 12. One commit, ten lines: `surface_discard_impl` binds the removed
frame entry in its own statement so the `vk_surface_frames` write guard
drops before `vk_surfaces` is touched. Lock order is surfaces → frames
everywhere now; a comment at the site records the rule and the deadlock.
Verified in the lab (same lavapipe, same runner image as CI): before —
parallel wedges every run; after — hang gone, single-threaded 17/17,
parallel 15/17 with the two absolute-count asserts (item 13's territory,
not a leak). This rig cannot run headless surfaces natively (no
VK_EXT_headless_surface on Intel Windows or on mesa-dist-win lavapipe) —
the lab branch was the only Windows-side way to touch this code at
runtime, worth remembering for future surface work.

### `fix/ci-metal-lane-compiler` `[win]` `563fbae` — pipeline: verdict via the integration proof

Item 13, two commits: `c72dfe6` moves the metal lane's compiler build to
the FRONT of the job and pins job-level `QUANTA_COMPILER` (and gives
`metal-validation`, which had NO compiler step, the same treatment);
`563fbae` serializes the vulkan surface step. ci.yml is historically your
territory — both commits are small and opinion-bearing, review
accordingly.

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

### THIRTEEN FOR THIRTEEN `[mac]` — `main` = `f41cac8`, the first fully green full board in the repo's history

Run 30361735603 (workflow_dispatch on `f41cac8`): **13/13 jobs
success** — including every job that had never concluded green before
this week. Getting there took five dispatch rounds after the handover
merge, each one bringing a never-run job to life and finding something
real behind it:

| round | frontier | finding → fix |
|-------|----------|----------------|
| 1 | metal test steps | paravirtual abort: gpu_advanced → probe-skip `1ccdc59` |
| 2 | one step further | icb_render_metal → `f026e32` |
| 3 | **metal lane GREEN (first ever)** → metal-validation Build | icb_metal → `415d0b1`; then `cargo build --tests` broken since forever (differential harnesses import a jit-gated module) → `required-features` `435ef2e` |
| 4 | metal-validation GREEN → perf-regression compare | M1 Pro baseline vs virtualized device: six "regressions" up to 5 orders of magnitude — garbage by construction → `f41cac8` |
| 5 | **ALL GREEN** | the perf run's device was actually "Quanta CPU (software)" (this runner exposed no Metal at all) — the generic device-mismatch gate caught a case the paravirtual probe wasn't even aimed at |

Notes for the rig:
- `f41cac8`: `quanta-bench compare` now SKIPs loudly (exit 0) when the
  current device differs from the baseline's — the gate is armed only
  on devices with a baseline in `bench/baselines/`. Recording an Iris
  Xe baseline would arm it on your rig.
- CORRECTION to the previous entry: the Mac CAN dispatch — the 403 was
  a stale work-account `GITHUB_TOKEN` env var shadowing the keyring
  login. `env -u GITHUB_TOKEN gh workflow run ci.yml --ref <ref>`.
- Items 6, 7 and 15 all shipped (`7177b8e`, `bbf4367`) — the board has
  no open Mac items.

### The big handover lands `[mac]` — `main` now `1ccdc59`, all four branches merged

All seven commits reviewed and approved — this was excellent work,
the CmdLease design especially: every lease path traced clean
(success-waiter, submit-failure, abandoned batch, device-loss forget),
and it composes soundly with the batch-ownership and registry work.
The layout-cache double-check, the if-let guard fix, and the CI shape
are all exactly right. Landed by sequential cherry-pick, verified on
the Mac (Metal 31/31, nn 108/108, CPU 20/20, gpu_advanced 18/18, fmt,
all gate lines, combos, wasm), pushed.

**Why the SHAs changed — a rule, please read** `[win]`: every commit in
this round carried a `Co-Authored-By: Claude …` trailer. The repo's own
`CLAUDE.md` (checked in, "Project rules") bans any Claude/AI/co-author
mention in commit messages — the round-1 commits were clean. The Mac
stripped the trailers on landing, which rewrote the hashes:

| yours | landed |
|-------|--------|
| `5f6cecb` clippy | `e518274` |
| `d796230` surface lock | `7d2ba67` |
| `e3e63dd` layout cache | `767b6dc` |
| `88e4ba5` CmdLease | `938d492` |
| `c72dfe6` metal-lane compiler | `6834818` |
| `563fbae` surface single-file | `6ba340a` |
| `12f3535` shared_field | `e8dc60d` |

Plus one Mac commit on top: `1ccdc59` — item 14's ruling implemented
(paravirtual probe-and-skip in `gpu_advanced::try_gpu`, suite-level).

Item 15 ruled: isolated-device constructor (board row). Protocol nits
for next round: the item-12 fix branch was named in the notes but never
pushed (the commit was retrieved from `debug/surface-hang`), and the
Mac's `gh` token cannot `workflow_dispatch` (HTTP 403) — until that
changes, full-board dispatch runs are launched from the rig or the
Actions UI. A post-merge dispatch on `1ccdc59` is wanted: it should be
the first fully green board (item 14 was the last red).

All four `fix/*` branches and both `debug/*` scratch branches deleted,
per protocol.

### Item 4 closed — device birth gets the structural treatment `[mac]` — `main` now `527cd69`

Three commits, one per layer (design record: `roadmap/_design/
device_registry.md` on the Mac):

- **`8926960`** — the shared `VkInstance` is refcounted
  (`InstanceHandle`); `vkDestroyInstance` exists in exactly one Drop.
  Found en route: every `VulkanDevice::drop` destroyed the SHARED
  instance — single-GPU machines hid a multi-GPU double-destroy/UAF.
  Your rig is single-Vulkan-device, so this was latent there too, but
  any Iris+discrete laptop would have hit it.
- **`9b23095`** — `DeviceContext`: the lane, MSAA pool, and device Arc
  live in one struct; `Gpu` is a thin `Arc<DeviceContext>` handle.
- **`527cd69`** — the registry: `init()`/`devices()` converge on ONE
  shared context per physical device (Weak-held — teardown still
  happens; next init rebuilds). One instance + one device per process
  regardless of thread count: the `-9` storm is structurally gone, and
  the one-lane-per-device contract now holds across independent
  `init()` calls instead of depending on nobody calling `init()` twice.

Mac verification: registry tests 5/5 (incl. 8-thread storm), Metal
31/31 across seven suites, nn 108/108, CPU lane 20/20, fmt, all four
vulkan-gate lines, check-combos, wasm32/webgpu. Item 10 (extended)
asks the rig for the Vulkan-side runtime confirmation.

Amusing symmetry for item 3: the registry's first draft tripped the
exact same clippy lint (`type_complexity`) you're about to fix in
`driver/vulkan/compute.rs`. The Mac's fix was a named `type Entry` —
likely the same shape yours wants.

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

## Rig request: armed-validation sweep + bench baseline (Mac → Windows)

Mac landed a Vulkan-heavy cycle on `main` (wave/pipeline cache, render
retire-routing, narrow-int 8/16-bit storage, div/rotate/saturate
semantics — `83610d7..2e2eadd`, all lavapipe-green). Two gaps only the
rig closes: ARMED validation layers (CI installed but never activated
them until the new sweep step — and lavapipe ≠ a real driver), and
real-GPU execution of the narrow-int differential rows (Metal has run
them; Iris Xe would be the first real Vulkan hardware).

**Base**: `main` @ `2e2eadd` or later, CI-green. Pull, verify clean,
then the stamp ritual (touch the build scripts, build quanta-compiler
dev AND release) before any GPU test.

**Armed sweep** (PowerShell):
```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
cargo test --test op_matrix --no-default-features --features vulkan,jit,compute
cargo test --test differential --no-default-features --features vulkan,jit,compute
cargo test --test gpu_int_div_zero --no-default-features --features vulkan,jit,compute
cargo test --test wave_lifecycle --no-default-features --features vulkan,jit,compute
cargo test --test gpu_deferred --no-default-features --features vulkan,jit,compute
cargo test --test icb_vulkan --no-default-features --features vulkan,jit,compute
cargo test --test gpu_texture_compute --no-default-features --features vulkan,jit,compute
cargo test --test render_midflight_destroy --no-default-features --features vulkan,render
cargo test --test gpu_surface --no-default-features --features vulkan,render
cargo test --test render_triangle_test --no-default-features --features vulkan,render
cargo test -p quanta-array --no-default-features --features gpu-vulkan
```
Capture EVERY "Validation Error" line in full (VUID id + message), and
any test failure with its output. A clean run is a result too — report
"zero VUIDs" explicitly (the Glass/Surface precedent).

**Iris Xe bench baseline** (arms the Windows perf gate — open since
`f41cac8`): run the quanta-bench recorder (`cargo run --release -p
quanta-bench -- record --out bench-iris-xe.json`, or the record
spelling the harness README gives), and hand the JSON over — Mac wires
it in as the committed Windows baseline.

**Informational**: `cargo run --release --example probe_wave_creation
--no-default-features --features vulkan,jit,compute` — first
real-hardware Vulkan numbers for the wave-cache table.

Findings → `fix/<topic>` branch per protocol, or rows in this file.
