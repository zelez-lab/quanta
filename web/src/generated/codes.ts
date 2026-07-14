// GENERATED — DO NOT EDIT. Run `quanta codegen webgpu` to regenerate.
//
// Source: web/webgpu.idl  (sha256: e85178050c10e68381efabbc664cf6c3368055242060abb0627d923ded5c69a7)
// Generator: crates/lang/quanta-codegen (B′ track of the FFI TCB shrink).
//
// Each `SPEC_*` array holds every value that the W3C `webgpu.idl`
// lists for the corresponding enum, in source order. Quanta's
// hand-written `web/src/codes.ts` (and the Rust mirror in
// `src/driver/webgpu/ffi.rs`) define a *subset* — every string they
// expose is required to appear here. Cross-checked at codegen and at
// runtime via `assertSpecSubset()` below.

export const WEBGPU_IDL_SHA256 = "e85178050c10e68381efabbc664cf6c3368055242060abb0627d923ded5c69a7";

/** W3C `webgpu.idl` enum `GPUTextureFormat` — 101 values. */
export const SPEC_GPUTextureFormat: readonly string[] = [
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
  "astc-12x12-unorm-srgb",
];

/** W3C `webgpu.idl` enum `GPUAddressMode` — 3 values. */
export const SPEC_GPUAddressMode: readonly string[] = [
  "clamp-to-edge",
  "repeat",
  "mirror-repeat",
];

/** W3C `webgpu.idl` enum `GPUFilterMode` — 2 values. */
export const SPEC_GPUFilterMode: readonly string[] = [
  "nearest",
  "linear",
];

/** W3C `webgpu.idl` enum `GPUMipmapFilterMode` — 2 values. */
export const SPEC_GPUMipmapFilterMode: readonly string[] = [
  "nearest",
  "linear",
];

/** W3C `webgpu.idl` enum `GPUCompareFunction` — 8 values. */
export const SPEC_GPUCompareFunction: readonly string[] = [
  "never",
  "less",
  "equal",
  "less-equal",
  "greater",
  "not-equal",
  "greater-equal",
  "always",
];

/** W3C `webgpu.idl` enum `GPUPrimitiveTopology` — 5 values. */
export const SPEC_GPUPrimitiveTopology: readonly string[] = [
  "point-list",
  "line-list",
  "line-strip",
  "triangle-list",
  "triangle-strip",
];

/** W3C `webgpu.idl` enum `GPUFrontFace` — 2 values. */
export const SPEC_GPUFrontFace: readonly string[] = [
  "ccw",
  "cw",
];

/** W3C `webgpu.idl` enum `GPUCullMode` — 3 values. */
export const SPEC_GPUCullMode: readonly string[] = [
  "none",
  "front",
  "back",
];

/** W3C `webgpu.idl` enum `GPUBlendFactor` — 17 values. */
export const SPEC_GPUBlendFactor: readonly string[] = [
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
  "one-minus-src1-alpha",
];

/** W3C `webgpu.idl` enum `GPUBlendOperation` — 5 values. */
export const SPEC_GPUBlendOperation: readonly string[] = [
  "add",
  "subtract",
  "reverse-subtract",
  "min",
  "max",
];

/** W3C `webgpu.idl` enum `GPUIndexFormat` — 2 values. */
export const SPEC_GPUIndexFormat: readonly string[] = [
  "uint16",
  "uint32",
];

/** W3C `webgpu.idl` enum `GPUVertexFormat` — 41 values. */
export const SPEC_GPUVertexFormat: readonly string[] = [
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
  "unorm8x4-bgra",
];

/** W3C `webgpu.idl` enum `GPUVertexStepMode` — 2 values. */
export const SPEC_GPUVertexStepMode: readonly string[] = [
  "vertex",
  "instance",
];

/** W3C `webgpu.idl` enum `GPULoadOp` — 2 values. */
export const SPEC_GPULoadOp: readonly string[] = [
  "load",
  "clear",
];

/** W3C `webgpu.idl` enum `GPUStoreOp` — 2 values. */
export const SPEC_GPUStoreOp: readonly string[] = [
  "store",
  "discard",
];

/**
 * Throw if any of `quantaStrings` is missing from `specTable`.
 * Called from `web/src/codes.ts` once at module-init time so a
 * post-deploy spec drift surfaces as a load error, not a silent
 * mis-dispatch on first use.
 */
export function assertSpecSubset(
  enumName: string,
  specTable: readonly string[],
  quantaStrings: readonly string[],
): void {
  for (const s of quantaStrings) {
    if (!specTable.includes(s)) {
      throw new Error(
        `quanta-codegen: ${enumName} value ${JSON.stringify(s)} is not in the spec ` +
          `(spec sha256: ${WEBGPU_IDL_SHA256.slice(0, 12)}…). ` +
          `Either fix Quanta's enum strings or re-run \`quanta codegen webgpu\`.`,
      );
    }
  }
}
