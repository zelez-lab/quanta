// Integer ↔ WebGPU-string code translations.
//
// The Rust side uses numeric codes for all enum-shaped parameters
// (texture format, blend factor, primitive topology, …). This file is
// the single point where those numbers are turned into the WebGPU IDL
// strings the browser expects. The encoding lives in lockstep with the
// `quanta-codes` mirror in `src/driver/webgpu/ffi.rs`; both sides will
// be derived from the same WebIDL spec under B′ (next track), but are
// hand-aligned here for B⁰.
//
// Every `*Name` function rejects unknown codes with a thrown `Error` —
// silent fallthrough would mask bugs at the FFI boundary.

function lookup(table: readonly string[], code: number, kind: string): string {
  const s = table[code];
  if (s === undefined) {
    throw new Error(`quanta glue: unknown ${kind} code ${code}`);
  }
  return s;
}

// ── Texture format codes ────────────────────────────────────────────────────

export const FORMAT_NAMES: readonly string[] = [
  /* 0  */ "rgba8unorm",
  /* 1  */ "bgra8unorm",
  /* 2  */ "r8unorm",
  /* 3  */ "r16float",
  /* 4  */ "r32float",
  /* 5  */ "rg32float",
  /* 6  */ "rgba16float",
  /* 7  */ "rgba32float",
  /* 8  */ "depth32float",
];

export const formatName = (c: number) => lookup(FORMAT_NAMES, c, "format");

// ── Vertex attribute formats ────────────────────────────────────────────────

export const ATTRIBUTE_FORMAT_NAMES: readonly string[] = [
  /* 0  */ "float32",
  /* 1  */ "float32x2",
  /* 2  */ "float32x3",
  /* 3  */ "float32x4",
  /* 4  */ "sint32",
  /* 5  */ "sint32x2",
  /* 6  */ "sint32x3",
  /* 7  */ "sint32x4",
  /* 8  */ "uint32",
  /* 9  */ "uint32x2",
  /* 10 */ "uint32x3",
  /* 11 */ "uint32x4",
  /* 12 */ "unorm8x4",
];

export const attributeFormatName = (c: number) =>
  lookup(ATTRIBUTE_FORMAT_NAMES, c, "vertex-format");

// ── Primitive topology / cull mode ──────────────────────────────────────────

export const TOPOLOGY_NAMES: readonly string[] = [
  /* 0 */ "point-list",
  /* 1 */ "line-list",
  /* 2 */ "line-strip",
  /* 3 */ "triangle-list",
  /* 4 */ "triangle-strip",
];

export const topologyName = (c: number) => lookup(TOPOLOGY_NAMES, c, "topology");

export const CULL_MODE_NAMES: readonly string[] = [
  /* 0 */ "none",
  /* 1 */ "front",
  /* 2 */ "back",
];

export const cullModeName = (c: number) => lookup(CULL_MODE_NAMES, c, "cull-mode");

// ── Blend factor / op ──────────────────────────────────────────────────────

export const BLEND_FACTOR_NAMES: readonly string[] = [
  /* 0  */ "zero",
  /* 1  */ "one",
  /* 2  */ "src-alpha",
  /* 3  */ "one-minus-src-alpha",
  /* 4  */ "dst-alpha",
  /* 5  */ "one-minus-dst-alpha",
  /* 6  */ "src",
  /* 7  */ "one-minus-src",
  /* 8  */ "dst",
  /* 9  */ "one-minus-dst",
];

export const blendFactorName = (c: number) =>
  lookup(BLEND_FACTOR_NAMES, c, "blend-factor");

export const BLEND_OP_NAMES: readonly string[] = [
  /* 0 */ "add",
  /* 1 */ "subtract",
  /* 2 */ "reverse-subtract",
  /* 3 */ "min",
  /* 4 */ "max",
];

export const blendOpName = (c: number) => lookup(BLEND_OP_NAMES, c, "blend-op");

// ── Sampler filtering / addressing ──────────────────────────────────────────

export const FILTER_NAMES: readonly string[] = [
  /* 0 */ "nearest",
  /* 1 */ "linear",
];

export const filterName = (c: number) => lookup(FILTER_NAMES, c, "filter");

export const ADDRESS_NAMES: readonly string[] = [
  /* 0 */ "clamp-to-edge",
  /* 1 */ "repeat",
  /* 2 */ "mirror-repeat",
];

export const addressName = (c: number) => lookup(ADDRESS_NAMES, c, "address-mode");

// Compare codes — 0 = "no compare configured" sentinel for sampler.
//   1..8 = compare ops (offset by 1).
export const COMPARE_NAMES: readonly string[] = [
  /* 1 */ "never",
  /* 2 */ "less",
  /* 3 */ "equal",
  /* 4 */ "less-equal",
  /* 5 */ "greater",
  /* 6 */ "not-equal",
  /* 7 */ "greater-equal",
  /* 8 */ "always",
];

export function compareName(code: number): string {
  if (code < 1) {
    throw new Error(`quanta glue: compare code ${code} unset (caller should not call)`);
  }
  return lookup(COMPARE_NAMES, code - 1, "compare");
}

// ── Vertex step mode / index format ─────────────────────────────────────────

export const STEP_MODE_NAMES: readonly string[] = [
  /* 0 */ "vertex",
  /* 1 */ "instance",
];

export const stepModeName = (c: number) => lookup(STEP_MODE_NAMES, c, "step-mode");

export const INDEX_FORMAT_NAMES: readonly string[] = [
  /* 0 */ "uint16",
  /* 1 */ "uint32",
];

export const indexFormatName = (c: number) =>
  lookup(INDEX_FORMAT_NAMES, c, "index-format");

// ── Load / store op ────────────────────────────────────────────────────────

export const LOAD_OP_NAMES: readonly string[] = [
  /* 0 */ "load",
  /* 1 */ "clear",
];

export const loadOpName = (c: number) => lookup(LOAD_OP_NAMES, c, "load-op");

export const STORE_OP_NAMES: readonly string[] = [
  /* 0 */ "store",
  /* 1 */ "discard",
];

export const storeOpName = (c: number) => lookup(STORE_OP_NAMES, c, "store-op");
