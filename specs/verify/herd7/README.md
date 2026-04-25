# herd7 litmus tests for Quanta atomics

Three Cat-language litmus tests that empirically corroborate the
release-acquire axioms (A6-A9) declared in
`specs/verify/lean/Quanta/Axioms/MemoryModels.lean`.

These are **cross-checks**, not proofs. They check that the canonical
message-passing (MP) and store-buffer (SB) shapes that Quanta's emitters
produce are forbidden / allowed in the way the axioms claim, under a
Cat-language model compatible with the Vulkan Memory Model.

## Files

| File | Pattern | Forbidden outcome | Grounds axiom (T-id) |
|------|---------|-------------------|----------------------|
| `message_passing.litmus` | MP+rel+acq | `flag=1 ∧ data=0` | T1601 / T1605 / T1610 / T1616 |
| `store_buffer.litmus` | SB+rel+acq | (allowed under rel/acq, forbidden only under SeqCst) | T1600 / T1607 |
| `atomic_add_visibility.litmus` | AtomicAdd+rel+acq | `counter=1 ∧ data=0` | T1601 / T1605 / T1610 / T1616 |

Each `.litmus` file contains a header comment that names the SPIR-V / PTX /
MSL / RDNA instructions Quanta emits for the modeled pattern.

## Running

These tests use the [LISA](https://diy.inria.fr/doc/lisa.html) input
language for [herd7](https://diy.inria.fr/doc/herd.html), part of the
[diy](https://diy.inria.fr/) toolsuite by Alglave & Maranget.

Install (macOS):

```sh
opam install herdtools7
```

Run all three under a Vulkan-compatible Cat model:

```sh
herd7 -model vmm.cat specs/verify/herd7/message_passing.litmus
herd7 -model vmm.cat specs/verify/herd7/store_buffer.litmus
herd7 -model vmm.cat specs/verify/herd7/atomic_add_visibility.litmus
```

`vmm.cat` is the Vulkan Memory Model Cat-language formalization
(Hadarean et al., "A Concurrency Semantics for Relaxed Atomics that
Permits Optimisation and Avoids Thin-Air Executions", PLDI 2017,
adapted by the Khronos working group).

A generic acquire-release Cat model (e.g. RC11 or `aarch64.cat`) is
also acceptable for the MP and AtomicAdd tests; SB requires the
Vulkan Memory Model semantics to distinguish rel/acq from SeqCst.

## Expected outcomes

| Test | Cat model | Outcome |
|------|-----------|---------|
| `message_passing.litmus` | rel/acq or VMM | `Never` (bad outcome forbidden) |
| `atomic_add_visibility.litmus` | rel/acq or VMM | `Never` |
| `store_buffer.litmus` | rel/acq | `Sometimes` (SB allowed) |
| `store_buffer.litmus` | SeqCst | `Never` |

The SB result documents the boundary: pure release-acquire is **not** strong
enough to forbid the SB anomaly. Quanta therefore promotes to SeqCst when
two-sided cross-workgroup synchronization is required (rare; most patterns
are MP-shaped).

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
