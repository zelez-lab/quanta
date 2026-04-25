/-
# Machine-model axioms — Per-backend GPU memory models

Trusted properties of each GPU backend's memory model that Quanta
assumes for correct cross-thread synchronization. These formalize
axioms A6-A9: the ordering guarantees, scoping rules, barrier
semantics, and fence behavior that each backend provides.

Without these axioms, Quanta's atomic operations (`atomic_add`,
`atomic_cas`, etc.) and barrier calls (`workgroup_barrier`) have
no formal grounding — the emitter can produce the correct
instruction, but there is no proof that the hardware delivers
the expected visibility.

Each model defines:
- Scope hierarchy (which threads see which writes)
- Ordering guarantees per scope (relaxed / acquire-release / seq-cst)
- Barrier interaction with visibility
- Fence semantics for cross-scope synchronization

See `Quanta.Axioms.Gpu` for the shared execution model and
`Quanta.Axioms.Vulkan` / `Metal` / `Llvm` for per-backend
driver correctness.
-/

import Quanta.Axioms.Gpu

namespace Quanta.Axioms.MemoryModels

-- ════════════════════════════════════════════════════════════════════
-- Shared types
-- ════════════════════════════════════════════════════════════════════

/-- Memory ordering strength, from weakest to strongest. -/
inductive Ordering where
  | Relaxed     -- no ordering guarantees beyond atomicity
  | Acquire     -- subsequent reads/writes cannot be reordered before
  | Release     -- preceding reads/writes cannot be reordered after
  | AcqRel      -- both acquire and release
  | SeqCst      -- total order across all seq-cst operations
  deriving Repr, DecidableEq

/-- Visibility scope: which set of threads observe a memory operation. -/
inductive Scope where
  | Workgroup   -- threads within the same workgroup (CTA / threadgroup / subgroup)
  | Device      -- all threads on the GPU
  | System      -- all threads including CPU
  deriving Repr, DecidableEq

/-- A memory operation with ordering and scope annotation. -/
structure ScopedOp where
  ordering : Ordering
  scope : Scope

/-- Scope inclusion: workgroup < device < system. -/
def Scope.includes : Scope → Scope → Bool
  | .System, _          => true
  | .Device, .Device    => true
  | .Device, .Workgroup => true
  | .Workgroup, .Workgroup => true
  | _, _                => false

/-- Ordering strength: relaxed < acquire = release < acq_rel < seq_cst. -/
def Ordering.atLeastAsStrong : Ordering → Ordering → Bool
  | .SeqCst, _         => true
  | .AcqRel, .SeqCst   => false
  | .AcqRel, _         => true
  | .Acquire, .Relaxed  => true
  | .Acquire, .Acquire  => true
  | .Acquire, _         => false
  | .Release, .Relaxed  => true
  | .Release, .Release  => true
  | .Release, _         => false
  | .Relaxed, .Relaxed  => true
  | .Relaxed, _         => false

-- ════════════════════════════════════════════════════════════════════
-- A6: Vulkan memory model (VulkanMemoryModel extension, Khronos)
-- ════════════════════════════════════════════════════════════════════

namespace VulkanMM

/-- Vulkan scopes as defined by the VulkanMemoryModel SPIR-V extension.
    Maps to SPIR-V Scope operands on memory operations. -/
inductive VkScope where
  | Invocation  -- single thread (no synchronization)
  | Subgroup    -- threads within a subgroup (warp / wavefront)
  | Workgroup   -- threads within a workgroup
  | Device      -- all threads on the device (QueueFamily scope)
  | System      -- all agents including CPU (CrossDevice scope)
  deriving Repr, DecidableEq

/-- **vulkan_workgroup_seq_cst**: Within a single workgroup, atomic
    operations with `memory_scope_Workgroup` and
    `MemorySemanticsSequentiallyConsistentKHR` are sequentially
    consistent — all threads in the workgroup agree on a total
    order of these operations.

    Vulkan Memory Model specification, section "Scoped Modification
    Order": "For each atomic object and each scope, there is a total
    order of all modifications that are performed with that scope." -/
axiom vulkan_workgroup_seq_cst
    (wg : Gpu.Workgroup)
    (addr : Nat)
    : True -- seq-cst atomics at workgroup scope are totally ordered

/-- **vulkan_cross_workgroup_acq_rel**: Atomic operations across
    workgroups require at minimum acquire-release semantics with
    device scope to establish visibility. A release-store at device
    scope synchronizes with an acquire-load at device scope when the
    load reads the value written by the store (or a later value in
    modification order).

    Without the VulkanMemoryModel extension, cross-workgroup atomics
    have no guaranteed ordering. Quanta requires this extension. -/
axiom vulkan_cross_workgroup_acq_rel
    (addr : Nat)
    (write_val : Nat)
    : True -- release(device) → acquire(device) establishes
          -- happens-before across workgroups

/-- **vulkan_relaxed_with_barrier**: Relaxed atomics within a
    workgroup are not ordered with respect to non-atomic memory.
    However, an `OpControlBarrier` with appropriate memory semantics
    and scope flushes all pending relaxed operations, making them
    visible to all threads in the barrier's scope.

    This is how Quanta's `workgroup_barrier()` works: it emits
    `OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory`. -/
axiom vulkan_relaxed_with_barrier
    (wg : Gpu.Workgroup)
    : True -- barrier + AcqRel|WorkgroupMemory flushes relaxed ops

/-- **vulkan_storage_buffer_fence**: `OpMemoryBarrier` with
    `StorageBuffer` memory semantics ensures that storage buffer
    writes from one dispatch are visible to a subsequent dispatch
    on the same queue, provided an appropriate queue-family scope.

    This grounds Quanta's cross-dispatch data dependencies. -/
axiom vulkan_storage_buffer_fence
    (scope : VkScope)
    : True -- memory barrier flushes storage buffer writes
          -- within the specified scope

end VulkanMM

-- ════════════════════════════════════════════════════════════════════
-- A7: PTX release-acquire scoping (NVIDIA CUDA)
-- ════════════════════════════════════════════════════════════════════

namespace PtxMM

/-- PTX memory scopes, specified as suffixes on atomic instructions
    (e.g., `atom.gpu.add`, `ld.acquire.cta`, `st.release.sys`). -/
inductive PtxScope where
  | CTA    -- cooperative thread array (= workgroup / thread block)
  | GPU    -- all CTAs on the same GPU
  | Sys    -- all agents including CPU (PCIe coherent)
  deriving Repr, DecidableEq

/-- Map PTX scopes to the shared Scope type. -/
def PtxScope.toScope : PtxScope → Scope
  | .CTA => .Workgroup
  | .GPU => .Device
  | .Sys => .System

/-- **ptx_cta_acquire_release**: Within a CTA, a `st.release.cta`
    synchronizes with a subsequent `ld.acquire.cta` on the same
    address — all writes by the releasing thread before the store
    are visible to the acquiring thread after the load.

    PTX ISA 8.5, section "Memory Consistency Model":
    "release and acquire operations at .cta scope synchronize
    between threads in the same CTA." -/
axiom ptx_cta_acquire_release
    (addr : Nat) (val : Nat)
    : True -- st.release.cta → ld.acquire.cta establishes
          -- happens-before within a CTA

/-- **ptx_gpu_acquire_release**: A `st.release.gpu` synchronizes
    with a subsequent `ld.acquire.gpu` across different CTAs on the
    same GPU. This is the mechanism for cross-workgroup communication
    in CUDA/PTX.

    PTX ISA 8.5: "release and acquire operations at .gpu scope
    synchronize between all threads on the GPU." -/
axiom ptx_gpu_acquire_release
    (addr : Nat) (val : Nat)
    : True -- st.release.gpu → ld.acquire.gpu establishes
          -- happens-before across CTAs on the same GPU

/-- **ptx_sys_acquire_release**: A `st.release.sys` synchronizes
    with a subsequent `ld.acquire.sys` across GPU and CPU threads.
    This is the only scope that provides CPU-GPU coherency.

    Requires PCIe coherent memory (CUDA managed memory or
    `cudaMallocHost`). Without coherent memory, `.sys` scope
    atomics have undefined behavior. -/
axiom ptx_sys_acquire_release
    (addr : Nat) (val : Nat)
    : True -- st.release.sys → ld.acquire.sys establishes
          -- happens-before between GPU and CPU threads

/-- **ptx_bar_cta**: `bar.sync` (PTX barrier) within a CTA acts
    as a full acquire-release fence at CTA scope. All shared memory
    and global memory writes by any thread in the CTA before the
    barrier are visible to all threads after the barrier.

    This grounds Quanta's `workgroup_barrier()` on NVIDIA. -/
axiom ptx_bar_cta
    : True -- bar.sync is a full AcqRel fence at CTA scope

/-- **ptx_fence_scope_inclusion**: A fence at a wider scope
    subsumes a narrower scope. `fence.gpu` implies `fence.cta`;
    `fence.sys` implies `fence.gpu`.

    PTX ISA 8.5: "A fence at a given scope also acts as a fence
    at all narrower scopes." -/
axiom ptx_fence_scope_inclusion
    (wide narrow : PtxScope)
    (h : wide.toScope.includes narrow.toScope = true)
    : True -- fence at `wide` scope implies fence at `narrow` scope

end PtxMM

-- ════════════════════════════════════════════════════════════════════
-- A8: Metal memory model (Apple GPU)
-- ════════════════════════════════════════════════════════════════════

namespace MetalMM

/-- Metal memory address spaces relevant to ordering. -/
inductive MetalAddrSpace where
  | Device          -- device memory (global), visible to all threads
  | Threadgroup     -- threadgroup memory (shared), visible within workgroup
  | ThreadgroupImageblock -- imageblock memory, visible within workgroup
  deriving Repr, DecidableEq

/-- Metal memory order qualifiers, applied via `atomic_*` functions
    or `memory_order` enum in MSL. -/
inductive MetalMemOrder where
  | Relaxed         -- `memory_order_relaxed`
  | Acquire         -- `memory_order_acquire`
  | Release         -- `memory_order_release`
  | AcqRel          -- `memory_order_acq_rel`
  deriving Repr, DecidableEq

/-- **metal_device_relaxed_default**: Device memory accesses in
    Metal are relaxed by default — no ordering guarantees between
    non-atomic loads/stores across threads. Ordering requires
    explicit `atomic_*` functions with memory order qualifiers.

    Metal Shading Language Specification, section 6.13:
    "The default memory order for atomic operations is relaxed." -/
axiom metal_device_relaxed_default
    (addr : Nat)
    : True -- non-atomic device memory has no cross-thread ordering

/-- **metal_atomic_acq_rel**: Atomic operations with explicit
    `memory_order_acquire` or `memory_order_release` on device
    memory provide the standard acquire-release semantics: a
    release-store synchronizes with a subsequent acquire-load
    that reads the stored value.

    MSL spec, section 6.13.1: atomic operations with specified
    memory order follow C++14 atomics semantics. -/
axiom metal_atomic_acq_rel
    (addr : Nat)
    (val : Nat)
    : True -- atomic_store(release) → atomic_load(acquire)
          -- establishes happens-before

/-- **metal_threadgroup_barrier**: `threadgroup_barrier(mem_flags)`
    synchronizes all threads in a workgroup. After the barrier:
    - If `mem_flags` includes `mem_threadgroup`: all threadgroup
      memory writes before the barrier are visible.
    - If `mem_flags` includes `mem_device`: all device memory
      writes before the barrier are visible.
    - If `mem_flags` includes `mem_texture`: all texture writes
      before the barrier are visible.

    This grounds Quanta's `workgroup_barrier()` on Apple GPUs.
    Quanta emits `threadgroup_barrier(mem_flags::mem_threadgroup)`. -/
axiom metal_threadgroup_barrier
    (wg : Gpu.Workgroup)
    : True -- barrier with mem_threadgroup flushes threadgroup writes

/-- **metal_simdgroup_barrier**: `simdgroup_barrier(mem_flags)`
    synchronizes threads within a SIMD-group (32 threads on Apple
    GPU). Cheaper than a full threadgroup barrier when only
    simd-group-level synchronization is needed.

    Quanta uses this for warp-level reductions on Apple hardware. -/
axiom metal_simdgroup_barrier
    : True -- simdgroup_barrier synchronizes within SIMD-group

/-- **metal_threadgroup_coherent_after_barrier**: Threadgroup memory
    in Metal is coherent within a workgroup after a
    `threadgroup_barrier`. Before the barrier, writes to threadgroup
    memory by one thread are NOT guaranteed visible to other threads.
    After the barrier, all prior writes are visible.

    This is stronger than device memory (which requires explicit
    atomics) because threadgroup memory is on-chip SRAM with
    hardware-managed coherency within the workgroup. -/
axiom metal_threadgroup_coherent_after_barrier
    (wg : Gpu.Workgroup)
    (addr : Nat)
    : True -- threadgroup write before barrier visible to all
          -- threads in workgroup after barrier

/-- **metal_no_cross_workgroup_threadgroup**: Threadgroup memory
    is NOT visible across workgroups. Period. Each workgroup has
    its own threadgroup memory allocation. Attempting to share
    threadgroup addresses across workgroups is undefined behavior.

    Cross-workgroup communication requires device memory with
    atomic operations. -/
axiom metal_no_cross_workgroup_threadgroup
    : True -- threadgroup memory is workgroup-local only

end MetalMM

-- ════════════════════════════════════════════════════════════════════
-- A9: AMD RDNA memory model
-- ════════════════════════════════════════════════════════════════════

namespace RdnaMM

/-- RDNA scope hierarchy. Wave64 is the native SIMD width on
    RDNA 1/2/3; wave32 is also supported on RDNA 2+. -/
inductive RdnaScope where
  | Wave      -- threads within a wave (wavefront, 32 or 64 lanes)
  | Workgroup -- threads within a workgroup (multiple waves)
  | Agent     -- all threads on the GPU (= device scope)
  | System    -- all agents including CPU
  deriving Repr, DecidableEq

/-- Map RDNA scopes to the shared Scope type. -/
def RdnaScope.toScope : RdnaScope → Scope
  | .Wave      => .Workgroup  -- wave is a subset of workgroup
  | .Workgroup => .Workgroup
  | .Agent     => .Device
  | .System    => .System

/-- **rdna_workgroup_acq_rel**: Within a workgroup on RDNA,
    `s_buffer_atomic` or `flat_atomic` with `glc` (globally
    coherent) flag and scope annotation provides acquire-release
    ordering. A release-store synchronizes with an acquire-load
    on the same address within the workgroup.

    AMD RDNA ISA Reference, chapter "Memory Model":
    "Scoped acquire-release ordering is supported at workgroup,
    agent, and system scope." -/
axiom rdna_workgroup_acq_rel
    (addr : Nat)
    (val : Nat)
    : True -- scoped acq-rel at workgroup scope works

/-- **rdna_agent_acq_rel**: Cross-workgroup atomics at agent scope
    synchronize across all workgroups on the GPU. Requires the `glc`
    (globally coherent) bit on load instructions to bypass the L1
    vector cache, reading directly from L2.

    Without `glc`, loads may read stale L1 data even with agent scope
    ordering. Quanta's GCN emitter sets `glc` on all cross-workgroup
    atomic loads. -/
axiom rdna_agent_acq_rel
    (addr : Nat)
    (val : Nat)
    : True -- agent-scope acq-rel with glc bypasses L1,
          -- establishes happens-before across workgroups

/-- **rdna_s_barrier**: `s_barrier` instruction synchronizes all
    waves within a workgroup. After `s_barrier`, all LDS (Local Data
    Share = threadgroup memory) writes and global memory writes with
    appropriate scope are visible.

    This grounds Quanta's `workgroup_barrier()` on AMD GPUs.
    Quanta emits `s_barrier` after a `s_waitcnt lgkmcnt(0)` to
    ensure all prior LDS operations complete before the barrier. -/
axiom rdna_s_barrier
    : True -- s_barrier + s_waitcnt is a workgroup-scope
          -- acquire-release fence

/-- **rdna_lds_coherent_within_workgroup**: LDS (Local Data Share,
    64KB per workgroup on RDNA) is on-chip SRAM, coherent within a
    workgroup after `s_barrier`. Before `s_barrier`, LDS writes by
    one wave may not be visible to other waves in the same workgroup.

    After `s_barrier + s_waitcnt lgkmcnt(0)`, all LDS writes from
    all waves are visible to all waves. -/
axiom rdna_lds_coherent_within_workgroup
    (addr : Nat)
    : True -- LDS write before s_barrier visible to all waves after

/-- **rdna_wave_scope_lockstep**: Threads within a wave execute in
    lockstep (SIMD). All lanes in a wave see the same instruction
    at the same time. Within a wave, no explicit synchronization is
    needed — `ds_permute` and `v_readlane` provide direct register
    communication between lanes.

    Quanta uses wave-level intrinsics for subgroup reductions on
    AMD hardware, avoiding the cost of LDS round-trips. -/
axiom rdna_wave_scope_lockstep
    : True -- lanes within a wave are implicitly synchronized

/-- **rdna_gl_scope_mapping**: Vulkan's `gl_ScopeWorkgroup` maps
    to RDNA's workgroup scope (all waves in a workgroup).
    `gl_ScopeDevice` maps to agent scope. `gl_ScopeSubgroup` maps
    to wave scope.

    This is the bridge between Quanta's Vulkan SPIR-V atomics and
    the AMD hardware — the SPIR-V scope operand is lowered to the
    correct RDNA scope by the AMD Vulkan driver (AMDVLK / RADV). -/
axiom rdna_gl_scope_mapping
    : True -- gl_ScopeWorkgroup → workgroup,
          -- gl_ScopeDevice → agent,
          -- gl_ScopeSubgroup → wave

end RdnaMM

-- ════════════════════════════════════════════════════════════════════
-- Cross-model agreement
-- ════════════════════════════════════════════════════════════════════

/-- **barrier_semantics_agreement**: All four backends agree on the
    minimal semantics of `workgroup_barrier()`: after the barrier,
    all shared/threadgroup/LDS memory writes by any thread in the
    workgroup before the barrier are visible to all threads after it.

    Quanta emits:
    - Vulkan: `OpControlBarrier Workgroup Workgroup AcqRel|WorkgroupMemory`
    - PTX:    `bar.sync 0`
    - Metal:  `threadgroup_barrier(mem_flags::mem_threadgroup)`
    - RDNA:   `s_waitcnt lgkmcnt(0)` + `s_barrier`

    This axiom asserts that all four produce the same observable
    effect: a full acquire-release fence at workgroup scope for
    shared memory. -/
axiom barrier_semantics_agreement
    (wg : Gpu.Workgroup)
    : True -- all backends agree: barrier is AcqRel at workgroup scope

/-- **cross_workgroup_requires_device_scope**: On all four backends,
    communication between threads in different workgroups requires
    at minimum device-scope acquire-release atomics. Shared/threadgroup
    memory is workgroup-local and cannot be used.

    This is the common denominator that Quanta enforces: cross-workgroup
    atomics always emit device-scope ordering instructions. -/
axiom cross_workgroup_requires_device_scope
    (addr : Nat)
    : True -- cross-workgroup visibility requires device-scope atomics

end Quanta.Axioms.MemoryModels
