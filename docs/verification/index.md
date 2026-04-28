# Verification Dashboard

Live status of every theorem and axiom in Quanta. This is the
public-facing **theorem inventory** the roadmap calls out as step 078:
no other Rust GPU library publishes its proof obligations openly.
Reviewers can cross-check every claim here against the source files
and the verifier output.

## Headline numbers

|                            |  Count |
|---------------------------:|-------:|
| **Proven theorems**        |  176   |
| **TCB axioms (A1–A11)**    |   33   |
| **Tools used**             |   5    |
| **Backends covered**       |   5    |

Verifiers in active use: **Lean 4** (semantics + axioms), **Verus**
(code-matches-spec), **Kani** (bounded model checking), **herd7**
(memory-model litmus tests), **Miri** (UB detection on native).

## Verification chain

The chain runs top-down: each level depends on the levels below.
This is the **CompCert shape** — every property below ground level
is named as an axiom; nothing is silently trusted.

```
   Source preservation         (T59x — Rust→WASM→KernelOps; 058+059)
            │
   Race freedom                (T606/T607 — Verus; 057)
            │
   Memory ordering             (T1600-T1622 — Lean axioms + herd7; 055+056)
            │
   Emitter correctness         (T1xx-T4xx — Lean+Verus+Kani; existing)
            │
   API invariants              (T720-T754 — Verus; step 075)
            │
   Driver host                 (A10/A11 axioms — Lean; step 050+079)
            │
   Hardware                    (A1-A5 axioms — Lean; existing)
```

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

Three litmus tests (`message_passing.litmus`, `store_buffer.litmus`,
`atomic_add_visibility.litmus`) check the message-passing and
store-buffer patterns under release-acquire on a Cat-language model.

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
  from the W3C `webgpu.idl` by `crates/quanta-codegen` into spec
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
| **A11** | **Quanta wasm ↔ JS ABI** (`src/driver/webgpu/ffi.rs` + `web/src/quanta.ts`, B⁰; enum strings codegen'd from `web/webgpu.idl`, B′; enum-string + method-presence + call-arity + param-type conformance proven against `Quanta.Idl.WebGpuSpec` as T1710 + T1711 + T1712 + T1713, B″ complete; residue = `extern "C"` linker faithfulness + typedef stability) | **wasm32**    |

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

# herd7 litmus
herd7 specs/verify/herd7/<test>.litmus
```

## Roadmap toward more verification

Open items the verification track is working on, in priority order:

1. **WGSL grammar mirror** — promotes T410 from "every variant
   handled" to "produces parseable WGSL." Lifts T414 from depending
   on the T410 axiom-in-Lean to a fully proved chain.
2. **T416 end-to-end round-trip** — composes T410 + T1003 (cross-
   emitter agreement) + A10.6 (mapAsync visibility) to prove
   `field_read_bytes_async ∘ wave_dispatch ∘ field_write_bytes`
   round-trips a kernel `f` to `f(input)`.
3. **Step 058 + 059** — full Rust → WASM → KernelOps semantic
   preservation proof. Closes the entire source-to-ISA gap.
4. **Step 077** — differential testing in CI: bit-exact integer,
   ≤1 ULP float, atomic visibility across all 5 backends. Composes
   with the memory-model axioms.

Every shipped theorem above moves us further along the
`hardware → IR → user source` chain. Every named axiom names something
we trust instead of silently relying on it. That's the discipline.
