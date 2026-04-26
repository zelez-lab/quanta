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

-- Conditional-correctness theorems
import Quanta.Theorems.WebGpu
