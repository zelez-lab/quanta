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
import Quanta.KRust.Semantics

-- KernelOps view + semantics (E.2)
import Quanta.KOps.Syntax
import Quanta.KOps.Semantics

-- KRust → KernelOps translator (E.3)
import Quanta.KRust.Translate

-- Per-rule preservation theorems (E.4)
import Quanta.KRust.Preservation

-- End-to-end source preservation (E.5)
import Quanta.KRust.EndToEnd

-- Indirect Command Buffers (steps 032 + 033)
import Quanta.Icb

-- Bindless resource arrays (steps 034 + 035)
import Quanta.Bindless

-- Tessellation pipelines (steps 022 + 023)
import Quanta.Tessellation

-- Mesh shaders (steps 024 + 025)
import Quanta.MeshShader

-- Ray tracing (steps 026 + 027)
import Quanta.RayTracing

-- Variable rate shading (steps 028 + 029)
import Quanta.Vrs

-- Sparse textures (steps 030 + 031)
import Quanta.SparseTexture

-- Multi-queue (steps 018 + 019)
import Quanta.MultiQueue

-- Conditional-correctness theorems
import Quanta.Theorems.WebGpu
import Quanta.Theorems.IdlConformance
