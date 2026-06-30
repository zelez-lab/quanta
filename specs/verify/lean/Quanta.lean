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

-- KernelOps view + semantics (E.2). The legacy KRust source track
-- (Syntax / Semantics / Translate / Preservation / EndToEnd) was
-- deleted in the WASM-route cutover (2026-05-05) — its production
-- translator is gone and the Lean files no longer correspond to
-- anything that ships. Step 059 reintroduces the source-preservation
-- theorem on top of the WASM operator subset (`Quanta.Wasm.*`).
import Quanta.KOps.Syntax
import Quanta.KOps.Semantics
import Quanta.KOps.Scope

-- WASM source-language view + semantics + translator (step 059).
import Quanta.Wasm.Syntax
import Quanta.Wasm.Structured
import Quanta.Wasm.Semantics
import Quanta.Wasm.Translate
import Quanta.Wasm.TranslatePending
import Quanta.Wasm.Preservation
import Quanta.Wasm.PreservationList
import Quanta.Wasm.PreservationBridge
import Quanta.Wasm.LowerInvariants
import Quanta.Wasm.LowerScopeValid
import Quanta.Wasm.PreservationInduction
import Quanta.Wasm.WellFormed

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

-- Async memory copy (step 044)
import Quanta.AsyncCopy

-- GPU printf (step 049)
import Quanta.Printf

-- Tensor layout algebra (companion crate quanta-tensor)
import Quanta.Tensor.Denotational
import Quanta.Tensor.Layout
import Quanta.Tensor.Bridge

-- Block-cooperative primitives (companion crate quanta-prims)
import Quanta.Prims.Reference

-- Level-1 BLAS forward-error bounds (companion crate quanta-blas)
import Quanta.Blas.Reference
-- Level-3 BLAS GEMM forward-error bound
import Quanta.Blas.Gemm
-- Level-2 BLAS GEMV forward-error bound (reuses the GEMM entry contract)
import Quanta.Blas.Gemv
-- Mixed-precision GEMM forward-error bound (bf16 inputs, f32 accumulate)
import Quanta.Blas.GemmMixed

-- Reverse-mode autodiff VJP correctness (companion crate quanta-autograd)
import Quanta.Autograd.Vjp
import Quanta.Autograd.MatmulVjp
import Quanta.Autograd.ReduceVjp
import Quanta.Autograd.ActivationVjp

-- FFT: Cooley-Tukey radix-2 butterfly identity (companion crate quanta-fft)
import Quanta.Fft.Dft

-- Numeric dtype conversions (step 084.1 — bf16 / fp8 round-trip)
import Quanta.Dtype.Bf16
import Quanta.Dtype.Fp8
import Quanta.Dtype.Quant

-- Conditional-correctness theorems
import Quanta.Theorems.WebGpu
import Quanta.Theorems.IdlConformance
