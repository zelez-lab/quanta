# herd7 litmus tests for Quanta atomics

Four Cat-language litmus tests that empirically corroborate the
release-acquire axioms (A6-A9) declared in
`specs/verify/lean/Quanta/Axioms/MemoryModels.lean`.

These are **cross-checks**, not proofs. They check that the canonical
message-passing (MP) and store-buffer (SB) shapes that Quanta's emitters
produce are forbidden / allowed in the way the axioms claim, under a
release/acquire Cat-language model compatible with the Vulkan Memory
Model.

## Files

| File | Pattern | Bad outcome | Expected | Grounds axiom (T-id) |
|------|---------|-------------|----------|----------------------|
| `message_passing.litmus` | MP+rel+acq | `flag=1 ∧ data=0` | Never | T1601 / T1605 / T1610 / T1616 |
| `store_buffer.litmus` | SB+rel+acq | both read `0` | **Sometimes** | T1600 / T1607 |
| `store_buffer_sc.litmus` | SB+sc | both read `0` | Never | T1600 / T1607 |
| `atomic_add_visibility.litmus` | AtomicAdd+rel+acq | `counter=1 ∧ data=0` | Never | T1601 / T1605 / T1610 / T1616 |

Each `.litmus` file contains a header comment that names the SPIR-V / PTX /
MSL / RDNA instructions Quanta emits for the modeled pattern.

The model files:

- `vmm.bell` — LISA annotation declarations (`rlx`/`acq`/`rel`/`acq_rel`/`sc`).
- `vmm.cat` — the release/acquire consistency model (RC11 axioms
  re-expressed for LISA).

## Running

These tests use the [LISA](https://diy.inria.fr/doc/lisa.html) input
language for [herd7](https://diy.inria.fr/doc/herd.html), part of the
[diy](https://diy.inria.fr/) toolsuite by Alglave & Maranget.

Install (macOS):

```sh
opam install herdtools7
```

One command, from the repo root, runs all four and asserts the verdicts:

```sh
just litmus
```

`just litmus` skips cleanly (exit 0, install hint) if `herd7` is not on
`PATH`, so it never blocks a machine without the toolsuite. It runs in
the nightly `Differential CI (full lanes)` workflow (`diff-full.yml`),
never on regular PR CI.

To run a single test by hand:

```sh
herd7 -bell specs/verify/herd7/vmm.bell -model specs/verify/herd7/vmm.cat \
      specs/verify/herd7/message_passing.litmus
```

## Which model, and the vmm.cat / license story

`vmm.cat` is a **release/acquire consistency model** whose axiom bodies
are RC11 ("Repaired C11", Lahav, Vafeiadis, Kang, Hur & Dreyer, PLDI
2017), re-expressed for herd7's LISA architecture.

There is deliberately **no vendored Khronos file**. The Khronos
[Vulkan-MemoryModel](https://github.com/KhronosGroup/Vulkan-MemoryModel)
repository publishes an **Alloy** formalization (`alloy/spirv.als`, under
CC-BY-4.0), *not* a herd7 `.cat`. So there is no upstream `vmm.cat` to
vendor — the name in older revisions of this README referred to a file
that does not exist upstream. Rather than convert Alloy → Cat (a
substantial, error-prone port), we use RC11, whose acquire-release
fragment agrees with the Vulkan Memory Model on exactly the MP / SB /
AtomicAdd shapes these tests exercise: rel/acq forbids MP and allows SB;
SeqCst forbids SB.

`vmm.bell` and `vmm.cat` are authored for Quanta. The RC11 relation
algebra they reuse is itself part of herdtools7 (CeCILL-B); no
third-party file is copied into the tree. herd7 loads the RC11-shipped
`cos.cat` (coherence generation) from its own install directory at run
time.

If a faithful Khronos VMM `.cat` is wanted later, the honest path is to
port `alloy/spirv.als` and vendor it under this directory with a
CC-BY-4.0 attribution header — tracked as future work, not needed for
the axiom cross-check these tests provide.

## Expected outcomes

| Test | Model | Outcome |
|------|-------|---------|
| `message_passing.litmus` | rel/acq | `Never` (bad outcome forbidden) |
| `atomic_add_visibility.litmus` | rel/acq | `Never` |
| `store_buffer.litmus` | rel/acq | `Sometimes` (SB allowed) |
| `store_buffer_sc.litmus` | SeqCst | `Never` (SB forbidden) |

The SB pair documents the boundary: pure release-acquire is **not**
strong enough to forbid the SB anomaly, but SeqCst is. Quanta therefore
promotes to SeqCst when two-sided cross-workgroup synchronization is
required (rare; most patterns are MP-shaped).

## Empirical companion: the GPU litmus kernel suite

`tests/litmus.rs` runs the *same* MP and SB shapes as real Quanta
kernels on the local GPU (Metal / Vulkan / software), packing 10^5+
independent instances into a single dispatch and building an outcome
histogram. Those tests are **empirical falsifiers** with the same
epistemics as this directory: they can catch a driver or emitter that
lets the forbidden outcome through, but a clean run is corroboration,
not proof. See the module doc in `tests/litmus.rs`.

## Why this is a cross-check, not a proof

The litmus tests show that the *abstract pattern* Quanta emits has the
expected behavior under the *abstract memory model*. They do not prove:

- that Quanta's emitter actually emits that pattern (that is theorems
  T100-T119 for SPIR-V, T300-T307 for MSL, etc., proven in Lean / Verus),
- that the GPU vendor's driver implements the abstract model faithfully
  (that is axiom A6 / A7 / A8 / A9 — assumed),
- that the GPU silicon implements the driver-promised model faithfully
  (that is axiom A3 — assumed).

What they buy us: confidence that the *axioms themselves* are coherent —
that the rel/acq guarantees we name in `MemoryModels.lean` actually
forbid the bad outcomes we expect them to forbid, on an independently
maintained machine-checked memory-model formalization.
