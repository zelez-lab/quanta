/- GENERATED — DO NOT EDIT. Run `quanta codegen webgpu` to regenerate.

Source: web/webgpu.idl  (sha256: e85178050c10e68381efabbc664cf6c3368055242060abb0627d923ded5c69a7)
Generator: crates/lang/quanta-codegen (B″ track of the FFI TCB shrink).

This file is the Lean mirror of the W3C WebGPU IDL — same data
as `src/webgpu_generated_codes.rs` and `web/src/generated/codes.ts`,
expressed as a `Quanta.Idl.WebGpuSpec` literal so the conformance
theorem `Quanta.Theorems.IdlConformance.quanta_strings_in_spec`
can discharge T1707 (the enum-string component of A11) by
`native_decide` against it.
-/

import Quanta.Idl

namespace Quanta.Idl

/-- WebGPU IDL spec data, generated from `web/webgpu.idl`.
    Only the project-relevant enums are included — the Quanta
    FFI does not ship the full IDL surface yet. Future B″
    passes will widen this with dictionaries and methods. -/
def webGpuSpec : WebGpuSpec :=
  { sourceSha256 :=
      "e85178050c10e68381efabbc664cf6c3368055242060abb0627d923ded5c69a7",
    enums := [
      -- W3C `webgpu.idl` enum `GPUTextureFormat` — 101 values.
      { name := "GPUTextureFormat",
        values := [
          "r8unorm",
          "r8snorm",
          "r8uint",
          "r8sint",
          "r16unorm",
          "r16snorm",
          "r16uint",
          "r16sint",
          "r16float",
          "rg8unorm",
          "rg8snorm",
          "rg8uint",
          "rg8sint",
          "r32uint",
          "r32sint",
          "r32float",
          "rg16unorm",
          "rg16snorm",
          "rg16uint",
          "rg16sint",
          "rg16float",
          "rgba8unorm",
          "rgba8unorm-srgb",
          "rgba8snorm",
          "rgba8uint",
          "rgba8sint",
          "bgra8unorm",
          "bgra8unorm-srgb",
          "rgb9e5ufloat",
          "rgb10a2uint",
          "rgb10a2unorm",
          "rg11b10ufloat",
          "rg32uint",
          "rg32sint",
          "rg32float",
          "rgba16unorm",
          "rgba16snorm",
          "rgba16uint",
          "rgba16sint",
          "rgba16float",
          "rgba32uint",
          "rgba32sint",
          "rgba32float",
          "stencil8",
          "depth16unorm",
          "depth24plus",
          "depth24plus-stencil8",
          "depth32float",
          "depth32float-stencil8",
          "bc1-rgba-unorm",
          "bc1-rgba-unorm-srgb",
          "bc2-rgba-unorm",
          "bc2-rgba-unorm-srgb",
          "bc3-rgba-unorm",
          "bc3-rgba-unorm-srgb",
          "bc4-r-unorm",
          "bc4-r-snorm",
          "bc5-rg-unorm",
          "bc5-rg-snorm",
          "bc6h-rgb-ufloat",
          "bc6h-rgb-float",
          "bc7-rgba-unorm",
          "bc7-rgba-unorm-srgb",
          "etc2-rgb8unorm",
          "etc2-rgb8unorm-srgb",
          "etc2-rgb8a1unorm",
          "etc2-rgb8a1unorm-srgb",
          "etc2-rgba8unorm",
          "etc2-rgba8unorm-srgb",
          "eac-r11unorm",
          "eac-r11snorm",
          "eac-rg11unorm",
          "eac-rg11snorm",
          "astc-4x4-unorm",
          "astc-4x4-unorm-srgb",
          "astc-5x4-unorm",
          "astc-5x4-unorm-srgb",
          "astc-5x5-unorm",
          "astc-5x5-unorm-srgb",
          "astc-6x5-unorm",
          "astc-6x5-unorm-srgb",
          "astc-6x6-unorm",
          "astc-6x6-unorm-srgb",
          "astc-8x5-unorm",
          "astc-8x5-unorm-srgb",
          "astc-8x6-unorm",
          "astc-8x6-unorm-srgb",
          "astc-8x8-unorm",
          "astc-8x8-unorm-srgb",
          "astc-10x5-unorm",
          "astc-10x5-unorm-srgb",
          "astc-10x6-unorm",
          "astc-10x6-unorm-srgb",
          "astc-10x8-unorm",
          "astc-10x8-unorm-srgb",
          "astc-10x10-unorm",
          "astc-10x10-unorm-srgb",
          "astc-12x10-unorm",
          "astc-12x10-unorm-srgb",
          "astc-12x12-unorm",
          "astc-12x12-unorm-srgb"
        ] },
      -- W3C `webgpu.idl` enum `GPUAddressMode` — 3 values.
      { name := "GPUAddressMode",
        values := [
          "clamp-to-edge",
          "repeat",
          "mirror-repeat"
        ] },
      -- W3C `webgpu.idl` enum `GPUFilterMode` — 2 values.
      { name := "GPUFilterMode",
        values := [
          "nearest",
          "linear"
        ] },
      -- W3C `webgpu.idl` enum `GPUMipmapFilterMode` — 2 values.
      { name := "GPUMipmapFilterMode",
        values := [
          "nearest",
          "linear"
        ] },
      -- W3C `webgpu.idl` enum `GPUCompareFunction` — 8 values.
      { name := "GPUCompareFunction",
        values := [
          "never",
          "less",
          "equal",
          "less-equal",
          "greater",
          "not-equal",
          "greater-equal",
          "always"
        ] },
      -- W3C `webgpu.idl` enum `GPUPrimitiveTopology` — 5 values.
      { name := "GPUPrimitiveTopology",
        values := [
          "point-list",
          "line-list",
          "line-strip",
          "triangle-list",
          "triangle-strip"
        ] },
      -- W3C `webgpu.idl` enum `GPUFrontFace` — 2 values.
      { name := "GPUFrontFace",
        values := [
          "ccw",
          "cw"
        ] },
      -- W3C `webgpu.idl` enum `GPUCullMode` — 3 values.
      { name := "GPUCullMode",
        values := [
          "none",
          "front",
          "back"
        ] },
      -- W3C `webgpu.idl` enum `GPUBlendFactor` — 17 values.
      { name := "GPUBlendFactor",
        values := [
          "zero",
          "one",
          "src",
          "one-minus-src",
          "src-alpha",
          "one-minus-src-alpha",
          "dst",
          "one-minus-dst",
          "dst-alpha",
          "one-minus-dst-alpha",
          "src-alpha-saturated",
          "constant",
          "one-minus-constant",
          "src1",
          "one-minus-src1",
          "src1-alpha",
          "one-minus-src1-alpha"
        ] },
      -- W3C `webgpu.idl` enum `GPUBlendOperation` — 5 values.
      { name := "GPUBlendOperation",
        values := [
          "add",
          "subtract",
          "reverse-subtract",
          "min",
          "max"
        ] },
      -- W3C `webgpu.idl` enum `GPUIndexFormat` — 2 values.
      { name := "GPUIndexFormat",
        values := [
          "uint16",
          "uint32"
        ] },
      -- W3C `webgpu.idl` enum `GPUVertexFormat` — 41 values.
      { name := "GPUVertexFormat",
        values := [
          "uint8",
          "uint8x2",
          "uint8x4",
          "sint8",
          "sint8x2",
          "sint8x4",
          "unorm8",
          "unorm8x2",
          "unorm8x4",
          "snorm8",
          "snorm8x2",
          "snorm8x4",
          "uint16",
          "uint16x2",
          "uint16x4",
          "sint16",
          "sint16x2",
          "sint16x4",
          "unorm16",
          "unorm16x2",
          "unorm16x4",
          "snorm16",
          "snorm16x2",
          "snorm16x4",
          "float16",
          "float16x2",
          "float16x4",
          "float32",
          "float32x2",
          "float32x3",
          "float32x4",
          "uint32",
          "uint32x2",
          "uint32x3",
          "uint32x4",
          "sint32",
          "sint32x2",
          "sint32x3",
          "sint32x4",
          "unorm10-10-10-2",
          "unorm8x4-bgra"
        ] },
      -- W3C `webgpu.idl` enum `GPUVertexStepMode` — 2 values.
      { name := "GPUVertexStepMode",
        values := [
          "vertex",
          "instance"
        ] },
      -- W3C `webgpu.idl` enum `GPULoadOp` — 2 values.
      { name := "GPULoadOp",
        values := [
          "load",
          "clear"
        ] },
      -- W3C `webgpu.idl` enum `GPUStoreOp` — 2 values.
      { name := "GPUStoreOp",
        values := [
          "store",
          "discard"
        ] }
    ],
    methods := [
      { interfaceName := "GPU", methodName := "requestAdapter", requiredArity := 0, maxArity := 1, isVariadic := false, params := [{ typeName := "GPURequestAdapterOptions", optional := true }] },
      { interfaceName := "GPU", methodName := "getPreferredCanvasFormat", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUAdapter", methodName := "requestDevice", requiredArity := 0, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUDeviceDescriptor", optional := true }] },
      { interfaceName := "GPUDevice", methodName := "destroy", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUDevice", methodName := "createBuffer", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUBufferDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createTexture", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUTextureDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createSampler", requiredArity := 0, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUSamplerDescriptor", optional := true }] },
      { interfaceName := "GPUDevice", methodName := "importExternalTexture", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUExternalTextureDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createBindGroupLayout", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUBindGroupLayoutDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createPipelineLayout", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUPipelineLayoutDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createBindGroup", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUBindGroupDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createShaderModule", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUShaderModuleDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createComputePipeline", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUComputePipelineDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createRenderPipeline", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPURenderPipelineDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createComputePipelineAsync", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUComputePipelineDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createRenderPipelineAsync", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPURenderPipelineDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createCommandEncoder", requiredArity := 0, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUCommandEncoderDescriptor", optional := true }] },
      { interfaceName := "GPUDevice", methodName := "createRenderBundleEncoder", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPURenderBundleEncoderDescriptor", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "createQuerySet", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUQuerySetDescriptor", optional := false }] },
      { interfaceName := "GPUBuffer", methodName := "mapAsync", requiredArity := 1, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUMapModeFlags", optional := false }, { typeName := "GPUSize64", optional := true }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPUBuffer", methodName := "getMappedRange", requiredArity := 0, maxArity := 2, isVariadic := false, params := [{ typeName := "GPUSize64", optional := true }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPUBuffer", methodName := "unmap", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUBuffer", methodName := "destroy", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUTexture", methodName := "createView", requiredArity := 0, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUTextureViewDescriptor", optional := true }] },
      { interfaceName := "GPUTexture", methodName := "destroy", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUCommandEncoder", methodName := "beginRenderPass", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPURenderPassDescriptor", optional := false }] },
      { interfaceName := "GPUCommandEncoder", methodName := "beginComputePass", requiredArity := 0, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUComputePassDescriptor", optional := true }] },
      { interfaceName := "GPUCommandEncoder", methodName := "copyBufferToBuffer", requiredArity := 2, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPUCommandEncoder", methodName := "copyBufferToBuffer", requiredArity := 4, maxArity := 5, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := false }, { typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := false }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPUCommandEncoder", methodName := "copyBufferToTexture", requiredArity := 3, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUTexelCopyBufferInfo", optional := false }, { typeName := "GPUTexelCopyTextureInfo", optional := false }, { typeName := "GPUExtent3D", optional := false }] },
      { interfaceName := "GPUCommandEncoder", methodName := "copyTextureToBuffer", requiredArity := 3, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUTexelCopyTextureInfo", optional := false }, { typeName := "GPUTexelCopyBufferInfo", optional := false }, { typeName := "GPUExtent3D", optional := false }] },
      { interfaceName := "GPUCommandEncoder", methodName := "copyTextureToTexture", requiredArity := 3, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUTexelCopyTextureInfo", optional := false }, { typeName := "GPUTexelCopyTextureInfo", optional := false }, { typeName := "GPUExtent3D", optional := false }] },
      { interfaceName := "GPUCommandEncoder", methodName := "clearBuffer", requiredArity := 1, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := true }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPUCommandEncoder", methodName := "resolveQuerySet", requiredArity := 5, maxArity := 5, isVariadic := false, params := [{ typeName := "GPUQuerySet", optional := false }, { typeName := "GPUSize32", optional := false }, { typeName := "GPUSize32", optional := false }, { typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := false }] },
      { interfaceName := "GPUCommandEncoder", methodName := "finish", requiredArity := 0, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUCommandBufferDescriptor", optional := true }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "setPipeline", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUComputePipeline", optional := false }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "dispatchWorkgroups", requiredArity := 1, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUSize32", optional := false }, { typeName := "GPUSize32", optional := true }, { typeName := "GPUSize32", optional := true }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "dispatchWorkgroupsIndirect", requiredArity := 2, maxArity := 2, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := false }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "end", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setViewport", requiredArity := 6, maxArity := 6, isVariadic := false, params := [{ typeName := "float", optional := false }, { typeName := "float", optional := false }, { typeName := "float", optional := false }, { typeName := "float", optional := false }, { typeName := "float", optional := false }, { typeName := "float", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setScissorRect", requiredArity := 4, maxArity := 4, isVariadic := false, params := [{ typeName := "GPUIntegerCoordinate", optional := false }, { typeName := "GPUIntegerCoordinate", optional := false }, { typeName := "GPUIntegerCoordinate", optional := false }, { typeName := "GPUIntegerCoordinate", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setBlendConstant", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUColor", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setStencilReference", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUStencilValue", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "beginOcclusionQuery", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUSize32", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "endOcclusionQuery", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPURenderPassEncoder", methodName := "executeBundles", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "sequence<GPURenderBundle>", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "end", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUQueue", methodName := "submit", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "sequence<GPUCommandBuffer>", optional := false }] },
      { interfaceName := "GPUQueue", methodName := "onSubmittedWorkDone", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUQueue", methodName := "writeBuffer", requiredArity := 3, maxArity := 5, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := false }, { typeName := "AllowSharedBufferSource", optional := false }, { typeName := "GPUSize64", optional := true }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPUQueue", methodName := "writeTexture", requiredArity := 4, maxArity := 4, isVariadic := false, params := [{ typeName := "GPUTexelCopyTextureInfo", optional := false }, { typeName := "AllowSharedBufferSource", optional := false }, { typeName := "GPUTexelCopyBufferLayout", optional := false }, { typeName := "GPUExtent3D", optional := false }] },
      { interfaceName := "GPUQueue", methodName := "copyExternalImageToTexture", requiredArity := 3, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUCopyExternalImageSourceInfo", optional := false }, { typeName := "GPUCopyExternalImageDestInfo", optional := false }, { typeName := "GPUExtent3D", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "pushErrorScope", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPUErrorFilter", optional := false }] },
      { interfaceName := "GPUDevice", methodName := "popErrorScope", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUCommandEncoder", methodName := "pushDebugGroup", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "USVString", optional := false }] },
      { interfaceName := "GPUCommandEncoder", methodName := "popDebugGroup", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUCommandEncoder", methodName := "insertDebugMarker", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "USVString", optional := false }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "pushDebugGroup", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "USVString", optional := false }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "popDebugGroup", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPUComputePassEncoder", methodName := "insertDebugMarker", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "USVString", optional := false }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "setBindGroup", requiredArity := 2, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUIndex32", optional := false }, { typeName := "GPUBindGroup?", optional := false }, { typeName := "sequence<GPUBufferDynamicOffset>", optional := true }] },
      { interfaceName := "GPUComputePassEncoder", methodName := "setBindGroup", requiredArity := 5, maxArity := 5, isVariadic := false, params := [{ typeName := "GPUIndex32", optional := false }, { typeName := "GPUBindGroup?", optional := false }, { typeName := "Uint32Array", optional := false }, { typeName := "GPUSize64", optional := false }, { typeName := "GPUSize32", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "pushDebugGroup", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "USVString", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "popDebugGroup", requiredArity := 0, maxArity := 0, isVariadic := false, params := [] },
      { interfaceName := "GPURenderPassEncoder", methodName := "insertDebugMarker", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "USVString", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setBindGroup", requiredArity := 2, maxArity := 3, isVariadic := false, params := [{ typeName := "GPUIndex32", optional := false }, { typeName := "GPUBindGroup?", optional := false }, { typeName := "sequence<GPUBufferDynamicOffset>", optional := true }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setBindGroup", requiredArity := 5, maxArity := 5, isVariadic := false, params := [{ typeName := "GPUIndex32", optional := false }, { typeName := "GPUBindGroup?", optional := false }, { typeName := "Uint32Array", optional := false }, { typeName := "GPUSize64", optional := false }, { typeName := "GPUSize32", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setPipeline", requiredArity := 1, maxArity := 1, isVariadic := false, params := [{ typeName := "GPURenderPipeline", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setIndexBuffer", requiredArity := 2, maxArity := 4, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUIndexFormat", optional := false }, { typeName := "GPUSize64", optional := true }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "setVertexBuffer", requiredArity := 2, maxArity := 4, isVariadic := false, params := [{ typeName := "GPUIndex32", optional := false }, { typeName := "GPUBuffer?", optional := false }, { typeName := "GPUSize64", optional := true }, { typeName := "GPUSize64", optional := true }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "draw", requiredArity := 1, maxArity := 4, isVariadic := false, params := [{ typeName := "GPUSize32", optional := false }, { typeName := "GPUSize32", optional := true }, { typeName := "GPUSize32", optional := true }, { typeName := "GPUSize32", optional := true }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "drawIndexed", requiredArity := 1, maxArity := 5, isVariadic := false, params := [{ typeName := "GPUSize32", optional := false }, { typeName := "GPUSize32", optional := true }, { typeName := "GPUSize32", optional := true }, { typeName := "GPUSignedOffset32", optional := true }, { typeName := "GPUSize32", optional := true }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "drawIndirect", requiredArity := 2, maxArity := 2, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := false }] },
      { interfaceName := "GPURenderPassEncoder", methodName := "drawIndexedIndirect", requiredArity := 2, maxArity := 2, isVariadic := false, params := [{ typeName := "GPUBuffer", optional := false }, { typeName := "GPUSize64", optional := false }] }
    ] }

end Quanta.Idl
