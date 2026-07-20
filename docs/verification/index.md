# Verification Dashboard

Live status of every theorem and axiom in Quanta. This is the
public-facing **theorem inventory** the roadmap calls out as step 078:
no other Rust GPU library publishes its proof obligations openly.
Reviewers can cross-check every claim here against the source files
and the verifier output.

## Headline numbers

|                            |  Count |
|---------------------------:|-------:|
| **Proven Lean theorems**   |  ~300  |
| **Lean sorrys**            |   0    |
| **Lean TCB axioms** (narrow) | 15 (11 FFI + 2 WGSL spec + 1 opaque float + 1 step-level `stmt_heap_step_helper`) |
| **Verus theorems**         |  87 / 87 |
| **Verus `external_body`**  |   1 (hazard tracking) |
| **Tools used**             |   5    |
| **Backends covered**       |   5    |
| **Source preservation (E)** |  proven (T590-T5B0) |
| **Headless smoke tests** | 3 in CI (per-PR) — `web_triangle` + `web_textured` also assert framebuffer SHA-256 against vendored golden bytes |
| **Differential CI kernels** | 4 (saxpy, reduce_sum, counter, race) × {software, WGSL, Metal*, Vulkan*, AMDGPU**} |
| **Memory-order primitives** | 5 (Relaxed, Acquire, Release, AcqRel, SeqCst) × {AtomicOp, AtomicCas, Fence} |
| **Verified Tier-A tracks** | 9 (ICB, tessellation, mesh shaders, ray tracing, VRS, sparse textures, multi-queue, async copy, printf) |
| **Companion-crate numerics** | `quanta-blas` Higham `(1+δ)` forward-error bounds (Level-1/2/3 + mixed-precision); `quanta-autograd` VJP rules proven = analytic derivatives (`HasDerivAt`); `quanta-fft` Cooley-Tukey radix-2 proven = direct DFT; `quanta-nn` online-softmax attention proven = two-pass softmax (T9200–T9209) plus norm/rotary/optimizer/activation/loss/dropout identities (T9210–T9233) — 37 theorems, 0 axioms |

**Sustainment state (2026-04-30).** The post-E finalization closed
`kernel_body_compose` from a single monolithic axiom to a body-level
*theorem* (`kernel_body_compose_nil`, `kernel_body_compose_cons`,
`kernel_body_compose`, `t5b0_kernel_preservation` all closed). The
residual `stmt_heap_step_helper` axiom is narrower — single-stmt
translation shape + one-step heap projection. See
`memory/verification_finalized_2026-04-30.md` for the audit.

Verifiers in active use: **Lean 4** (semantics + axioms), **Verus**
(code-matches-spec), **Kani** (bounded model checking), **herd7**
(memory-model litmus tests), **Miri** (UB detection on native).

## Companion-crate numerics

The math companion crates carry their own proof obligations, in the same
"theorem not axiom" spirit:

* **`quanta-blas`** — every op ships a mechanically-proven Higham `(1+δ)`
  forward-error bound (`specs/verify/lean/Quanta/Blas/`), and the
  mixed-precision GEMM splits the entry error into the proven f32 bound plus a
  per-dtype input-quantisation term (`gemmEntry_narrow_error_split`). The
  numeric bound abstracts away *where* element `i` lives, so
  `Quanta.Dtype.StorageAddressing` closes that gap separately: it models the
  bf16/fp8 addressing of each carrier (host upload, CPU executor, MSL and
  SPIR-V emitters at native 2-/1-byte stride; WGSL at its u32-slot layout
  with host repack) and proves every backend's element `i` reads source
  element `i` — plus a negative theorem pinning the old cross-stride bug,
  which agreed at element 0 and diverged from element 1 on.
* **`quanta-autograd`** — every reverse-mode VJP multiplier is proven *equal to
  the analytic derivative* via Mathlib's `HasDerivAt`
  (`specs/verify/lean/Quanta/Autograd/`): elementwise + activations
  (`Vjp.lean`, `ActivationVjp.lean` — relu/sigmoid/tanh), `matmul` (`G·Bᵀ`,
  `Aᵀ·G` in `MatmulVjp.lean`), reductions (`ReduceVjp.lean`), `conv2d`
  (`ConvVjp.lean`), and pooling (`PoolVjp.lean`). These are not restatements —
  `sigmoid' = σ(1−σ)`, `tanh' = 1−tanh²`, and the matmul Jacobian fall out of the
  inverse/quotient/chain rules. `conv2d` is im2col → matmul → reshape, so its
  backward *is* the matmul VJP plus the `col2im` step; `ConvVjp.lean` proves the
  one fact that step rests on — that **col2im is the transpose of im2col**
  (`⟨im2col x, y⟩ = ⟨x, col2im y⟩`, for any gather), so the `∂x` it computes is
  the true reverse-mode gradient. Pooling is the same shape: `PoolVjp.lean`
  proves avgpool's backward is its adjoint (`⟨avgpool x, y⟩ = ⟨x, avgpoolBack
  y⟩`) and that maxpool's subgradient routes each output to its window argmax.
  The Rust crate cross-checks every rule against finite differences on real GPU
  execution, and an MLP trains end-to-end. 0 sorry, no new axioms (rests on
  Mathlib calculus).
* **`quanta-fft`** — Cooley-Tukey radix-2 is proven *equal to the direct DFT*,
  end to end (`specs/verify/lean/Quanta/Fft/`): the butterfly identity
  `X[k] = Xe[k] + ω_{2M}^k·Xo[k]` (`dft_radix2`) and its `log₂N` iteration to
  the full transform (`fftRec_eq_dftN`), built from scratch over an ℕ-indexed
  DFT (Mathlib has the DFT but no radix-2 decomposition). The Rust crate is
  differential-tested against the direct DFT on the software lane and real
  Metal. 0 sorry, rests on Mathlib. The arbitrary-N Bluestein path is
  **differential-tested only** (primes and composites up to N = 1000) — the
  Lean proof models the radix-2 recursion, not the chirp-z reformulation.
* **`quanta-nn`** — the fused (FlashAttention-style) attention kernel's
  online-softmax recurrence is proven *equal to the textbook two-pass
  softmax-weighted sum*, end to end (`specs/verify/lean/Quanta/Nn/OnlineSoftmax.lean`,
  T9200–T9209). The load-bearing result is a single generalised fold invariant
  (T9203) that simultaneously gives the prefix invariant, the block-merge /
  append law (T9206 — so a kernel may process K/V in blocks of *any* schedule),
  and the equivalence with the two-pass form (T9204/T9205); it closes to the
  textbook `Σⱼ softmax(x)ⱼ·vⱼ` (T9205′). Numerical stability is proven too: every
  `exp` argument the streaming step or the block-merge evaluates is `≤ 0`
  (T9207/T9209), so every weight lies in `(0, 1]` (T9208) — no overflow. Exact
  over `ℝ`, no rounding model; **0 sorry, 0 axioms**. The Rust crate
  differential-tests the fused `f32` kernel against an f64 two-pass reference
  (many shapes, causal + padding masks), cross-checks it against the composed
  autograd forward, and asserts finiteness under ±80 logits. See
  [Fused attention](../computation/tutorials/fused-attention.md) for the tutorial.
  The same recipe now covers the whole stack — norms, rotary, optimizers,
  activations, losses, dropout (T9210–T9233; details in the
  [fused attention section](#the-rest-of-the-quanta-nn-proof-foundation-t9210t9230)
  below).

## Verification chain

The chain runs top-down: each level depends on the levels below.
This is the **CompCert shape** — every property below ground level
is named as an axiom; nothing is silently trusted.

```
   Source preservation         (T590-T5B0 ✅ — Lean, route a, step E.
                                Body-level induction now a closed
                                theorem; residual TCB is the narrow
                                `stmt_heap_step_helper` axiom on
                                single-stmt heap projection.)
            │
   Race freedom                (T606/T607 — Verus; step 057.
                                Level 2 may-happen-in-parallel: open)
            │
   Memory ordering             (T1600-T1622 — Lean axioms + herd7;
                                055+056 ✅)
            │
   WebIDL conformance          (T1710-T1713 ✅ — Lean, step B″)
            │
   Emitter correctness         (T1xx-T4xx — Lean+Verus+Kani; +T410 ✅
                                lifted to theorem via A12+A13+T420,
                                step B)
            │
   API invariants              (T720-T754 ✅ — Verus; step 075)
            │
   Render-path no-silent-drops (T417 ✅ — Kani; step C wired set:
                                SetTexture/Sampler/Value/ClearDepth/
                                ClearStencil/SetStencilRef)
            │
   Driver host                 (A10 + A11 ✅ narrowed — irreducible
                                WebIDL-surface floor; step 050+079+B⁰
                                +B′+B″+B+C)
            │
   Hardware                    (A1-A5 axioms — Lean; existing)
```

## Operational regression net (B‴, 2026-04-29)

Three smoke tests run on every PR + main push via Playwright +
headless Chromium with `--enable-unsafe-webgpu`:

| Test | Path | Validates |
|------|------|-----------|
| `web_add_one` | compute | buffer = `[1, 2, …, 64]` byte-equal |
| `web_triangle` | render | center pixel ≈ triangle blue **and** framebuffer SHA-256 matches vendored golden (`ded669be…22650d`) |
| `web_textured` | render w/ texture | 16 pixels match source `rgba(255,165,64,255)`; SetTexture+SetSampler wiring (step C); framebuffer SHA-256 matches vendored golden (`0aa80fb0…050910`) |

The SHA assertions catch sub-pixel drift — e.g. a half-pixel shift in
a viewport scissor, or a one-bit tonemap change — that the rgb-tolerance
spot check would otherwise smooth over.

Local: `cd web && npm run smoke` — 3 passed, 3.3s.
CI: `.github/workflows/web-smoke.yml`.

## Differential CI (step D / step 077, 2026-04-29)

Empirical complement to the memory-model axioms (055/056) and the
emitter correctness theorems. Runs the same KernelDef on every
backend lane and asserts each candidate matches the pure-Rust
reference oracle within tolerance.

| Kernel | Type | Tolerance | Exercises |
|--------|------|-----------|-----------|
| `saxpy` | f32, N=1024 | ≤ 1 ULP | mul-then-add, no FMA contraction |
| `reduce_sum` | u32, N=64 | bit-exact | shared memory + barrier + thread-0 accumulator |
| `counter` | u32, N=128 | bit-exact (final value = N) | atomic_add, lost-update detection |
| `race` | u32, N=2 | trace-membership (permitted set) | atomic_exchange race, ordering-non-determinism gate |

| Lane | Triggered by | Workflow |
|------|--------------|----------|
| Reference (pure Rust) | every test invocation | — |
| Software (CpuDevice) | every PR + main push | `.github/workflows/ci.yml` |
| WGSL (Chrome+Dawn) | every PR + main push | `.github/workflows/web-smoke.yml` |
| Metal (macOS) | nightly cron + `run-full-diff` PR label + manual | `.github/workflows/diff-full.yml` |
| Vulkan (lavapipe) | nightly cron + `run-full-diff` PR label + manual | `.github/workflows/diff-full.yml` |
| AMDGPU (Mesa RADV, real silicon) | manual + `run-amd-diff` PR label; needs self-hosted runner tagged `[self-hosted, linux, gpu-amd]` | `.github/workflows/diff-full.yml` |

Local:
```sh
# software lane (per-PR slice)
cargo test --test differential --features software --no-default-features
# software + metal (macOS, nightly slice)
cargo test --test differential --features software,metal
# vulkan via lavapipe (Linux, nightly slice; needs libvulkan-dev + mesa-vulkan-drivers)
cargo test --test differential --no-default-features --features vulkan,jit,compute
# WGSL lane (browser)
cd web && npm run smoke    # exercises web_diff page
```

The `counter` kernel — N atomic_adds against a single cell, expected
final value = N — is the empirical gate on backend atomic semantics.
A non-atomic implementation produces a value < N; the backend
disagrees with the reference and the test fails.

The `race` kernel (D-ext.3b.2) is the first kernel where backends
are *allowed* to disagree within a model-permitted set. Two quarks
each `atomic_exchange(&cell, quark_id)` with `MemoryOrder::Relaxed`
and store the prior value into `out[i]`. Output layout
`[cell_final, out_0, out_1]`; permitted set = {`[1, 0, 0]`,
`[0, 1, 0]`}. Comparator is `compare_u32_in_set` — candidate must
equal at least one element of the permitted set. Sets the
template for future MP / SB / IRIW litmus kernels.

The `*` lanes (Metal, Vulkan, AMDGPU) run on opt-in workflows. The
AMDGPU lane is wired up but stays inert until a self-hosted runner
matching `[self-hosted, linux, gpu-amd]` is registered (setup steps
inline in `diff-full.yml`).

## Memory-order IR primitives (D-ext.3a / D-ext.3b)

Five `MemoryOrder` variants — `Relaxed`, `Acquire`, `Release`,
`AcqRel`, `SeqCst` — wired through every backend on three opcodes:

  * `KernelOp::Fence { order }` — explicit memory fence. Lowering:
    WGSL `storageBarrier()` (non-Relaxed only — WGSL atomics are
    SC-by-spec); MSL `atomic_thread_fence(mem_flags::mem_device, …)`;
    SPIR-V `OpMemoryBarrier` with the matching `MemorySemantics`;
    LLVM AOT no-op (parked); CPU interpreter no-op (sequential).
  * `KernelOp::AtomicOp { …, order }` — atomic RMW. SPIR-V derives
    semantics; LLVM threads through `atomicrmw <ordering>`. MSL is
    `device`-RELAXED-only by spec, so `order` is ignored on Metal
    storage-buffer atomics — surrounding fences carry the ordering.
    WGSL is SC-only by spec; ignored.
  * `KernelOp::AtomicCas { …, order }` — atomic CAS. LLVM clamps the
    failure-path ordering (Release→Monotonic, AcqRel→Acquire) per
    libstdc++ recipe.

Proc-macro frontend: `fence(MemoryOrder::Release)` (or just
`fence(Release)`) inside a `#[quanta::kernel]` body emits the new
opcode; `atomic_*()` builtins still default to `SeqCst`.

## Theorem chains by area

### IR → backend emitters

The longest-running verification work in the project. Five emitters
covered: SPIR-V, MSL, WGSL (build-time + JIT), LLVM, CPU.

* **T100–T119** — SPIR-V structural correctness, 20 theorems.
* **T200–T217** — Wire format roundtrips, 18 theorems.
* **T300+** — MSL emitter, 8 theorems.
* **T400–T416** — WGSL emitter, JIT path, 7 theorems (T414 chains
  through to a runtime guarantee via A10).
* **T500+** — LLVM emitter, 5 theorems.
* **T1000–T1003** — Cross-emitter exhaustiveness (every KernelOp
  variant handled in every backend).

### Render path

* **T417 / T418 / T419** — every `RenderOp` variant in `render_end`
  is wired XOR rejected, on **WebGPU + Metal + Vulkan**. Same Kani
  template applied to all three drivers; the discipline caught the
  same VRS silent-drop bug on each.

### API surface (step 075)

* **T720–T722** — Pulse lifecycle. Wait closure (FnOnce) consumed
  exactly once: no double-fire.
* **T730–T733** — Field handle invariants. Drop frees exactly once,
  no use-after-free at the mirror level.
* **T740–T743** — Wave bindings. **Capability monotonicity**:
  `binding_count` and `texture_count` never shrink.
* **T750–T754** — Submission queue. No double-submit, fence ordering,
  raw_hazard_free as a per-backend obligation.

### Memory models (steps 055–056)

A6–A9 formalized in Lean (T1600–T1622), cross-checked by herd7
litmus tests in `specs/verify/herd7/`:
* **A6** — Vulkan memory model
* **A7** — PTX release-acquire scoping
* **A8** — Metal memory model
* **A9** — AMD RDNA memory model

Four litmus tests (`message_passing.litmus`, `store_buffer.litmus`,
`store_buffer_sc.litmus`, `atomic_add_visibility.litmus`) check the
message-passing and store-buffer patterns under release-acquire (and
SeqCst for the SB pair) on a Cat-language model. They run under a
vendored LISA model (`vmm.bell` + `vmm.cat`, RC11 axioms re-expressed
for LISA) via one command — `just litmus` — which asserts each herd7
verdict matches the expected outcome (MP Never, SB rel/acq Sometimes,
SB seqcst Never, AtomicAdd Never). `just litmus` runs in the nightly
`diff-full.yml` workflow and skips cleanly if herd7 is absent.

Empirical companion: `tests/litmus.rs` runs the same MP / SB shapes as
real GPU kernels (packing 10^5+ instances per dispatch into an outcome
histogram) on the software / Metal / Vulkan lanes. These are empirical
falsifiers, not proofs — see that module's doc and the herd7 README.

### WebGPU host (step 050 + 079 + B⁰ + B′ + B″)

**A10** (WebGPU host correctness) + **A11** (Quanta wasm ↔ JS ABI
faithfulness, post-B⁰/B′) make the WebGPU driver a peer of the
Metal/Vulkan drivers in the verification scheme:
* **T1700–T1706** — A10 sub-axioms (shader module acceptance,
  pipeline creation, dispatch executes the kernel, queue ordering,
  Promise resolution semantics, mapAsync visibility, writeBuffer
  atomicity).
* **T1707–T1709** — A11 sub-axioms after step B⁰ (2026-04-28). The
  surface narrowed from "wasm-bindgen runtime" to Quanta's own
  `extern "C"` block + hand-authored `quanta.ts` (~500 LOC owned and
  audited). Sub-axioms: ABI faithfulness, async callback lossless,
  handle-table identity uniqueness.
* **B′ refinement (2026-04-28)** — every WebGPU enum string Quanta
  hands the JS side (texture format, blend factor, …) is now derived
  from the W3C `webgpu.idl` by `crates/lang/quanta-codegen` into spec
  tables on both sides:
  - `src/webgpu_generated_codes.rs` (Rust spec tables + a `cargo
    test`-time `quanta_strings_are_spec_subsets` check).
  - `web/src/generated/codes.ts` (TS spec tables + an
    `assertSpecSubset()` runtime check on every page load).
  The two-edits-without-enforcement lockstep hazard between the Rust
  and TS sides collapses to a single parsed AST.
* **B″ (2026-04-28)** — the same parsed IDL AST now also emits a
  Lean view, `Quanta.Idl.WebGpuSpec` in
  `specs/verify/lean/Quanta/Idl/WebGpuSpec.lean`. Four conformance
  theorems discharge by `native_decide`:
  - **T1710 — `quanta_strings_in_spec`**: every WebGPU enum string
    Quanta uses is declared by the spec.
  - **T1711 — `quanta_methods_in_spec`**: every WebGPU method
    `quanta.ts`/`webgpu.ts` calls is declared by the spec on the
    right interface, with WebIDL `includes` mixins (e.g.
    `GPUBindingCommandsMixin`, `GPURenderCommandsMixin`) flattened.
  - **T1712 — `quanta_call_arities_in_spec`**: at every call site,
    Quanta's argument count falls within some declared overload's
    `[requiredArity, maxArity]` range (variadics admitted).
  - **T1713 — `quanta_call_types_in_spec`**: at every call site,
    some declared overload's leading param type names equal the
    spec-canonical types Quanta supplies (typedefs preserved
    verbatim — `GPUSize64` stays `GPUSize64`).

  Together these *replace* the enum-string, method-presence, arity,
  and parameter-type components of T1707. The remaining A11 surface
  is the irreducible WebIDL-surface floor: wasm-linker faithfulness
  for `extern "C"` (libc-equivalent trust on the calling convention
  itself) plus typedef-stability (f64 ≡ `GPUSize64` etc. in the JS
  engine). Both are below what Lean can discharge against
  `webgpu.idl` alone.
* **T414** — first end-to-end conditional theorem: given A10.1+A10.2
  and T410 (emitter exhaustiveness), `wave_jit` always succeeds.
* **Step C (2026-04-28)** — closed the WebGPU render-parity gap.
  SetTexture, SetSampler, SetValue (per-call uniform-buffer
  fallback for push constants), ClearDepth (via rpass `clearValue`),
  ClearStencil (absorbed), and SetStencilRef are now in T417's
  *wired* set. Kani re-verifies T417 SUCCESSFUL. The remaining
  `rejected` set is the architectural floor: indirect draws (Tier
  A 032+033, deferred), occlusion queries (M3.3), and variable-rate
  shading (not in the WebGPU spec).
* **B (2026-04-28)** — WGSL grammar mirror in
  `specs/verify/lean/Quanta/Wgsl/{Grammar,Serialize,OpPatterns}.lean`.
  Models the WGSL fragment Quanta's emitter produces (enable
  directives, bindings, fn decls, statements, expressions, types)
  and a structural `Source.wellFormed : Source → Bool`. Two named
  bridge axioms in `Quanta/Axioms/Wgsl.lean`:
  - **A12 — `wgsl_serializer_preserves_grammar`**: structural
    `Source.wellFormed` ⇒ string `wgsl_string_well_formed` (W3C
    WGSL §3 lex + §4 validate).
  - **A13 — `emit_wgsl_jit_factors`**: the JIT emitter factors
    through *some* structurally well-formed `Source` (operational
    backing in Verus per-tag exhaustiveness + Kani BMC).
  - **T420 — `wgsl_op_patterns_well_formed`**: every per-`KernelOpTag`
    representative pattern in `OpPatterns.lean` is structurally
    well-formed (40-tag enumeration, `native_decide`).
  - **T410** is now a Lean theorem chained from A12 + A13 (replacing
    the axiom that previously imported the Verus claim into Lean).

### Fused attention — online softmax (`quanta-nn`)

The proof foundation for the `quanta-nn` fused attention kernel: the
online (streaming, block-wise) softmax-weighted sum computes exactly the
textbook two-pass result and never materialises the N² score matrix.
Exact over `ℝ`, no rounding model; proven in
`specs/verify/lean/Quanta/Nn/OnlineSoftmax.lean`. **11 theorems, 0 sorry,
0 axioms.**

* **T9200 / T9201** — softmax basics: coordinates sum to 1; each
  coordinate is independent of the reference/shift `m` (why subtracting
  the running max is safe).
* **T9202** — the rescale law: lowering the reference `m→m'` scales the
  running normaliser/accumulator by `exp(m−m')` — the algebraic heart of
  the merge.
* **T9203** — the generalised fold invariant: an online `step` on a
  summarised prefix summarises `pre ++ [q]`. This one contract, iterated
  (T9204), gives the whole equivalence.
* **T9204 / T9205 / T9205′** — the online fold summarises the whole list;
  its `acc/l` equals the direct two-pass output, which equals
  `Σⱼ softmax(x)ⱼ·vⱼ` (the textbook definition).
* **T9206** — merge = append: combining two block summaries equals
  appending the raw blocks, so *any* block schedule is admissible.
* **T9207 / T9208 / T9209** — stability: every `exp` argument in a step
  or a merge is `≤ 0`, so every weight lies in `(0, 1]` — no overflow.

The Rust crate makes these empirical: the fused `f32` kernel is
differential-tested against an f64 two-pass reference across many shapes
(causal + padding masks), cross-checked against the composed autograd
forward, gradient-checked through `sdpa_var` by finite differences, and
asserted finite under ±80 logits.

### The rest of the `quanta-nn` proof foundation (T9210–T9233)

Every later fused kernel family in `quanta-nn` follows the same recipe —
Lean identities first, then the kernels, then differential tests that run
the theorems empirically. All files under `specs/verify/lean/Quanta/Nn/`;
**0 sorry, 0 axioms** throughout.

* **Norm VJPs** (`NormVjp.lean`, T9210–T9215) — the closed-form
  three-term LayerNorm/RMSNorm backwards ARE the adjoints of the
  normalizations' linearizations; `√ε ≤ σ` lower bounds mean the kernels
  need no division guards.
* **Rotation VJP** (`RotationVjp.lean`, T9216–T9218) — the RoPE backward
  is the rotation by `−θ`, so ONE sign-flagged kernel serves both
  directions; every frequency pair is an isometry (the stability story).
* **Optimizer steps** (`OptimStep.lean`, T9219–T9222) — momentum unrolls
  to a geometric gradient memory; Adam's bias correction is EXACT at
  every step under a constant gradient; the two AdamW decay spellings
  are equal (one kernel serves both); the Adam step magnitude is `lr`
  regardless of gradient scale.
* **Activations** (`ActivationVjp.lean`, T9223–T9227) — max-subtraction
  in softmax is exact, not approximate; the softmax/log-softmax fused
  backwards are the proven adjoints of `diag(p) − ppᵀ` and `I − 1pᵀ`;
  the sech² identity lets the GeLU backward reuse the forward's tanh
  (no cosh is ever evaluated); σ′ algebra powers the SwiGLU backward.
* **Losses** (`LossVjp.lean`, T9228–T9230) — stable cross-entropy
  `lse(x) − x_y` is nonnegative on every row; the overflow-free
  BCE-with-logits spelling equals the textbook form for all logits; the
  Huber gradient is globally `clamp(r, −δ, δ)`, continuous across the
  knee.
* **Dropout** (`DropoutVjp.lean`, T9231–T9233) — inverted dropout is
  EXACTLY unbiased at the quantized rate the kernel implements (summed
  over all 2³² equally-likely Philox words); the mask-scale map is
  self-adjoint, so the backward is the forward kernel run on the
  cotangent with the mask regenerated from the key — never stored; the
  floor threshold undershoots the requested rate by less than 2⁻³².

## Trusted Computing Base

Stated explicitly so reviewers know what is trusted vs proven:

| Axiom | Source                                     | Backend       |
|-------|--------------------------------------------|---------------|
| A1    | Apple Metal Programming Guide              | macOS / iOS   |
| A2    | Vulkan 1.3 + sync2 spec                    | Linux/Win/And |
| A3    | GPU hardware execution model (general)     | All           |
| A4    | LLVM 22 codegen for nvptx64 + amdgcn       | All           |
| A5    | rustc as a black box for kernel source     | (kernels)     |
| A6    | Vulkan Memory Model extension              | Vulkan        |
| A7    | NVIDIA PTX ISA 8.5                         | NVIDIA        |
| A8    | Metal Shading Language §6.13               | Apple         |
| A9    | AMD RDNA ISA Reference                     | AMD           |
| **A10** | **W3C WebGPU spec, §6 Devices + §10 Queue** | **WebGPU**    |
| **A11** | **Quanta wasm ↔ JS ABI** (`crates/gpu/quanta-core/src/driver/webgpu/ffi.rs` + `web/src/quanta.ts`, B⁰; enum strings codegen'd from `web/webgpu.idl`, B′; enum-string + method-presence + call-arity + param-type conformance proven against `Quanta.Idl.WebGpuSpec` as T1710 + T1711 + T1712 + T1713, B″ complete; residue = `extern "C"` linker faithfulness + typedef stability) | **wasm32**    |

If a hardware/driver/browser violates these, the bug is upstream of
Quanta. The proof boundary is named explicitly.

## How to read this dashboard

1. **`specs/THEOREMS.md`** is the source of truth — the ID-by-ID list.
2. **`specs/machine_model.md`** has the full prose A1–A11 with citations
   to upstream specs.
3. **`specs/verify/lean/Quanta/Axioms/`** has the Lean axiom files.
4. **`specs/verify/verus/`** has the Verus mirror crates per project.
5. **`specs/verify/kani/`** has the Kani harnesses.
6. **`specs/verify/herd7/`** has the litmus tests.

Each theorem lists the tool that proves it. Re-running any verifier
locally:

```sh
# Lean axioms + theorems
cd specs/verify/lean && lake build

# Verus mirrors (per file, no workspace)
verus --crate-type=lib specs/verify/verus/quanta-api/pulse_lifetime.rs

# Kani harnesses
kani specs/verify/kani/emitter_exhaustiveness.rs --harness <name>

# herd7 litmus (all four, with verdict assertions)
just litmus
# or one by hand:
herd7 -bell specs/verify/herd7/vmm.bell -model specs/verify/herd7/vmm.cat \
      specs/verify/herd7/<test>.litmus

# empirical GPU litmus kernels (MP / SB histograms)
cargo test --test litmus --no-default-features --features software,metal,jit,compute
```

## Roadmap toward more verification

Closed in this v0.1 cycle:

* ✅ **WGSL grammar mirror** (step B) — `Quanta/Wgsl/{Grammar,Serialize,
  OpPatterns}.lean`; A12 + A13 lift T410 to a Lean theorem.
* ✅ **WebIDL conformance** (step B″) — T1710-T1713 from `webgpu.idl`
  alone, by `native_decide`.
* ✅ **Source preservation** (step E, route a) — T590-T5B0 + post-E
  sustainment closing the body-induction axiom.
* ✅ **Differential CI** (step D + D-extended) — 4 kernels × 5 lanes,
  full memory-order primitive set wired across all 5 emitters.
* ✅ **Golden-image SHA** — vendored framebuffer canaries on
  `web_triangle` + `web_textured`, asserted per-PR.
* ✅ **Narrow-dtype storage addressing** —
  `Quanta.Dtype.StorageAddressing` proves the bf16/fp8 addressing contract
  across host / CPU / MSL / SPIR-V (native stride) and WGSL (u32-slot):
  every backend's element `i` denotes source element `i`.
* ✅ **spirv-val as a test assertion** — the SPIR-V emitter test suites
  (bool-mask, narrow-storage, subgroup, texture-compute, plus the
  compiler-output validators) now *assert* `spirv-val` cleanliness when the
  tool is installed, instead of only logging invalid modules at build time.
  The tests skip the check silently on machines without SPIRV-Tools.

Still open, by priority:

1. **AMDGPU runner activation** — infra-only; the `diff-amdgpu`
   workflow stays inert until a self-hosted runner with labels
   `[self-hosted, linux, gpu-amd]` is registered. Once activated,
   the Vulkan ray-tracing build dispatch (`NotSupported` today)
   can be flipped on with empirical confidence.
2. **W3C CTS lane** — pull the upstream WebGPU Conformance Test
   Suite into the per-PR matrix to broaden the WebGPU host check
   beyond Quanta's own smoke tests.
3. **Firefox lane** — second WebGPU implementation in CI for
   cross-vendor agreement on the WGSL emitter output.
4. **Race-freedom Level 2** — may-happen-in-parallel analysis
   beyond the Level 1 single-source-binding gate. Genuinely novel
   research, multi-month estimate.
5. **MP / SB litmus kernels** — DONE (race-freedom L2, Phase 1).
   `tests/litmus.rs` runs message-passing and store-buffer shapes as
   real GPU kernels, 10^5+ instances/dispatch, on software / Metal /
   Vulkan; `just litmus` runs the herd7 cross-checks. These are
   empirical falsifiers, not the L2 proof. Findings surfaced: (a) an
   MSL emitter bug — `atomic_thread_fence` only accepts relaxed /
   seq_cst, so acquire/release/acqrel now promote to seq_cst; (b) on
   Metal, a device fence between *plain* accesses does not forbid MP /
   SB — the synchronizing accesses must be atomic. IRIW and the L2
   analyzer + theorem remain open.

Every shipped theorem above moves us further along the
`hardware → IR → user source` chain. Every named axiom names something
we trust instead of silently relying on it. That's the discipline.
