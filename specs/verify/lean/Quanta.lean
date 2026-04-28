-- Quanta formal specifications.
-- Top-level library root; re-exports per-component specs.

import Quanta.Opcodes
import Quanta.WireFormat
import Quanta.VaryingCoord
import Quanta.ComparisonOps
import Quanta.UnaryOps
import Quanta.Scan

-- Machine-model axioms (trusted computing base)
import Quanta.Axioms.Gpu
import Quanta.Axioms.Metal
import Quanta.Axioms.Vulkan
import Quanta.Axioms.Llvm
import Quanta.Axioms.WebGpu
import Quanta.Axioms.MemoryModels

-- Backend instruction semantics
import Quanta.Semantics.SpirV
import Quanta.Semantics.Msl
import Quanta.Semantics.Wgsl
import Quanta.Semantics.Llvm
import Quanta.Semantics.Cpu
import Quanta.Semantics.Agreement

-- WebIDL grammar mirror (B″) and generated spec data
import Quanta.Idl
import Quanta.Idl.WebGpuSpec

-- WGSL grammar mirror (B)
import Quanta.Wgsl.Grammar
import Quanta.Wgsl.Serialize
import Quanta.Wgsl.OpPatterns
import Quanta.Axioms.Wgsl

-- Kernel Rust syntax (E.1a) — top of the source-preservation track
import Quanta.KRust.Syntax

-- Conditional-correctness theorems
import Quanta.Theorems.WebGpu
import Quanta.Theorems.IdlConformance
