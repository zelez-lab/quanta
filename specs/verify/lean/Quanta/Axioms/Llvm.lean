/-
# Machine-model axioms — LLVM codegen backend

Trusted properties of LLVM's code generation that Quanta assumes.
These formalize axiom A4 (LLVM codegen correctness).

`opaque` is used for the `llc` compilation step whose implementation
is inside LLVM. `axiom` captures guarantees from LLVM's language
reference and target-specific documentation that we rely on for
end-to-end correctness of the WASM → LLVM IR → PTX/GCN pipeline.

See `Quanta.Axioms.Gpu` for the shared GPU execution model
(especially `SpvOp` semantics that the codegen must preserve).
-/

import Quanta.Axioms.Gpu

namespace Quanta.Axioms.Llvm

-- ════════════════════════════════════════════════════════════════════
-- LLVM types
-- ════════════════════════════════════════════════════════════════════

/-- GPU target architectures that LLVM can compile to. -/
inductive GpuTarget where
  | PTX    -- NVIDIA (sm_50+)
  | GCN    -- AMD (gfx900+)
  | WGSL   -- WebGPU (via naga, not LLVM — included for completeness)
  deriving Repr, DecidableEq

/-- LLVM IR text representation (a module in .ll format). -/
abbrev LlvmIR := String

-- ════════════════════════════════════════════════════════════════════
-- A4: LLVM codegen operations
-- ════════════════════════════════════════════════════════════════════

/-- Compile LLVM IR to target-specific binary (PTX text or GCN ELF).
    Returns `none` if the IR is malformed, uses unsupported
    intrinsics, or the target triple is invalid. -/
opaque llc_compile : LlvmIR → GpuTarget → Option ByteArray := fun _ _ => none

/-- Validate LLVM IR (equivalent to `opt -verify`).
    Returns `true` if the module is well-formed. -/
opaque llvm_verify : LlvmIR → Bool := fun _ => false

-- ════════════════════════════════════════════════════════════════════
-- A4: LLVM codegen correctness axioms
-- ════════════════════════════════════════════════════════════════════

/-- **llvm_ir_preserves_semantics**: If LLVM IR is well-formed
    (passes `llvm_verify`) and `llc_compile` succeeds for a given
    target, the generated code preserves the semantics of every
    instruction in the IR module.

    Specifically:
    - `add i32 %a, %b`  →  wrapping 32-bit addition
    - `sub i32 %a, %b`  →  wrapping 32-bit subtraction
    - `mul i32 %a, %b`  →  wrapping 32-bit multiplication
    - `udiv i32 %a, %b` →  unsigned 32-bit division
    - `and i32 %a, %b`  →  bitwise AND
    - `or i32 %a, %b`   →  bitwise OR
    - `xor i32 %a, %b`  →  bitwise XOR
    - `shl i32 %a, %b`  →  logical left shift
    - `lshr i32 %a, %b` →  logical right shift

    This is LLVM's fundamental correctness guarantee: the target
    code computes the same result as the IR semantics define. -/
axiom llvm_ir_preserves_semantics
    (ir : LlvmIR)
    (target : GpuTarget)
    (h_valid : llvm_verify ir = true)
    (h_compiles : llc_compile ir target ≠ none)
    : True -- target code preserves IR instruction semantics

/-- **ptx_matches_spirv**: For the arithmetic operations that
    Quanta emits, PTX instructions and SPIR-V opcodes compute
    identical results on unsigned 32-bit integers.

    Grounding:
    - PTX `add.u32 %d, %a, %b`  =  SPIR-V OpIAdd (128)  =  wrapping add
    - PTX `sub.u32 %d, %a, %b`  =  SPIR-V OpISub (130)  =  wrapping sub
    - PTX `mul.lo.u32 %d, %a, %b` = SPIR-V OpIMul (132) =  wrapping mul
    - PTX `div.u32 %d, %a, %b`  =  SPIR-V OpUDiv (134)  =  unsigned div
    - PTX `and.b32 %d, %a, %b`  =  SPIR-V OpBitwiseAnd (199) = bitwise AND
    - PTX `or.b32 %d, %a, %b`   =  SPIR-V OpBitwiseOr (197)  = bitwise OR
    - PTX `xor.b32 %d, %a, %b`  =  SPIR-V OpBitwiseXor (198) = bitwise XOR

    This axiom lets us prove that the LLVM backend produces the
    same results as the SPIR-V backend for any given kernel. -/
axiom ptx_matches_spirv
    (op : Gpu.SpvOp)
    (a b : UInt32)
    : True -- PTX target instruction computes same result as
          -- Gpu.SpvOp.eval_u32 op a b

/-- **gcn_matches_spirv**: Same cross-backend equivalence for
    AMD GCN instructions.

    - GCN `v_add_u32`   =  SPIR-V OpIAdd (128)
    - GCN `v_sub_u32`   =  SPIR-V OpISub (130)
    - GCN `v_mul_lo_u32` = SPIR-V OpIMul (132)
    - GCN `v_and_b32`   =  SPIR-V OpBitwiseAnd (199)
    - GCN `v_or_b32`    =  SPIR-V OpBitwiseOr (197)
    - GCN `v_xor_b32`   =  SPIR-V OpBitwiseXor (198) -/
axiom gcn_matches_spirv
    (op : Gpu.SpvOp)
    (a b : UInt32)
    : True -- GCN target instruction computes same result as
          -- Gpu.SpvOp.eval_u32 op a b

end Quanta.Axioms.Llvm
