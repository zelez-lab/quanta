# Quanta Formal Specifications

Formal verification specs for the Quanta GPU framework.

## Structure

```
specs/
├── README.md                ← this file
├── machine_model.md         ← GPU hardware + driver axioms (A1-A5)
├── properties/              ← what we prove (theorems T1-T8)
│   ├── opcode_correctness.md
│   ├── wire_roundtrip.md
│   └── layout_consistency.md
└── proofs/
    ├── lean/                ← mathematical proofs (Lean 4)
    │   ├── lakefile.toml
    │   ├── Quanta.lean
    │   ├── Quanta/
    │   │   ├── Opcodes.lean       ← SPIR-V opcode bijection
    │   │   ├── WireFormat.lean    ← encoding injectivity
    │   │   └── VaryingCoord.lean  ← vertex→fragment location consistency
    ├── verus/               ← Rust code proofs (Verus)
    │   ├── emit_spirv.rs    ← opcode emission matches Lean spec
    │   ├── wire_encode.rs   ← roundtrip correctness
    │   └── parse_params.rs  ← slot assignment uniqueness
    └── kani/                ← model checking (Kani)
        ├── opcodes.rs       ← all 50 opcode constants match SPIR-V spec
        ├── wire_roundtrip.rs← BinOp/UnaryOp/CmpOp tag roundtrip
        └── push_layout.rs   ← push constant offset calculation
```

## Verification chain

```
Axioms (A1-A5)          ← hardware/driver guarantees (assumed correct)
    │
    ▼
Lean specs              ← mathematical properties (the WHAT)
    │
    ▼
Verus proofs            ← code matches spec (the HOW)
    │
    ▼
Kani harnesses          ← exhaustive edge cases (the WHEN)
    │
    ▼
Integration tests       ← real hardware validation (the WHERE)
```

## Tool roles

| Tool | Role | Proves |
|------|------|--------|
| **Lean 4** | Mathematical truth | The spec itself is consistent and correct |
| **Verus** | Code-level proof | The Rust implementation matches the Lean spec |
| **Kani** | Bounded model check | No counterexample exists for bounded inputs |
| **Tests** | Hardware validation | Real GPUs produce the expected output |

## Axiom boundary

Quanta's proof boundary stops at the GPU API call. We prove that:
- Our IR-to-SPIR-V translation emits correct opcodes
- Our wire format roundtrips without data loss
- Our push constant layout matches CPU and GPU sides
- Our vertex→fragment varying locations are consistent

We do NOT prove that:
- The Metal/Vulkan driver correctly executes the shader
- The GPU hardware implements SPIR-V correctly
- LLVM emits correct machine code

These are axioms. If they fail, the fault is in the hardware/driver, not in Quanta.
