/-
# WGSL serializer — `Source` to spec-conforming string

Step **B.2** of the FFI TCB shrink track. `serialize : Source → String`
is the printer that takes the structural AST in `Quanta.Wgsl.Grammar`
and produces a WGSL string in the form Quanta's runtime ships.

For this commit the serializer is declared as an `opaque` function
matching the behaviour of `crates/gpu/quanta-ir/src/emit_wgsl/*.rs`.
A subsequent commit (B.4) will replace the `opaque` with a Lean
implementation paralleling the Rust emitter, at which point the
shape obligations become decidable.

The load-bearing fact this commit establishes is the **bridge
axiom** in `Quanta.Axioms.Wgsl` — namely, that any structurally
well-formed `Source` serializes to a string that satisfies
`wgsl_string_well_formed` in `Quanta.Axioms.WebGpu`. With the
bridge in place, T410 reduces from "the JIT emitter produces
well-formed WGSL strings" (an operational claim about ~1000 lines
of Rust) to "the JIT emitter's output is structurally well-formed"
(a decidable claim over ~25 KernelOp variants). B.3 will discharge
the latter; B.4 wires the chain.
-/

import Quanta.Wgsl.Grammar

namespace Quanta.Wgsl

/-- The serializer Quanta's runtime uses. Implemented operationally
    in `crates/gpu/quanta-ir/src/emit_wgsl/{kernel,ops,helpers,shader}.rs`;
    here we treat it as an opaque function whose behaviour the
    bridge axiom in `Quanta.Axioms.Wgsl` characterises. -/
opaque Source.serialize : Source → String

end Quanta.Wgsl
