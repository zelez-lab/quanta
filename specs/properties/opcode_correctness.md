# Property: SPIR-V Opcode Correctness

> **ID:** T2
> **Layer:** Compiler (emit_spirv.rs)
> **Status:** Specification complete, Lean proof written, Kani harness written

## Statement

Every KernelOp maps to the SPIR-V opcode defined by the SPIR-V 1.6 specification.
No two different KernelOps produce the same opcode.

## Formal

```
∀ op ∈ KernelOp, ∀ ty ∈ ScalarType:
  emit_spirv_opcode(op, ty) = spv_spec_opcode(op, ty)
```

Specifically, the bitwise opcodes must be:
```
BitAnd → 199 (OpBitwiseAnd)
BitOr  → 197 (OpBitwiseOr)
BitXor → 198 (OpBitwiseXor)
```

## Historical bug

Opcodes were swapped (AND=197, OR=198, XOR=199) causing `x & 255` to compute
`x | 255` on Vulkan. Fixed in commit a4fbab0. This property exists to prevent
regression.

## Proof

- **Lean:** `specs/proofs/lean/Quanta/Opcodes.lean` — `bitand_is_199`, `bitor_is_197`, `bitxor_is_198`
- **Kani:** `specs/proofs/kani/opcodes.rs` — `verify_bitwise_opcodes`, `verify_no_opcode_collisions`

## Depends on

- `machine_model.md` § A3 (GPU executes SPIR-V correctly)
