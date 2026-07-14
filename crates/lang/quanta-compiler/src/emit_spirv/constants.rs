//! SPIR-V opcode and constant definitions.
//!
//! All values match the SPIR-V 1.6 specification, verified against
//! spirv-dis. Kani harnesses in specs/verify/kani/opcodes.rs and
//! opcodes_complete.rs pin each constant to its spec value.

#![allow(dead_code)]

// ── SPIR-V opcodes ──────────────────────────────────────────────────────────

pub const OP_NAME: u16 = 5;
pub const OP_EXTENSION: u16 = 10;
pub const OP_EXT_INST_IMPORT: u16 = 11;
pub const OP_EXT_INST: u16 = 12;
pub const OP_MEMORY_MODEL: u16 = 14;
pub const OP_ENTRY_POINT: u16 = 15;
pub const OP_EXECUTION_MODE: u16 = 16;
pub const OP_CAPABILITY: u16 = 17;
pub const OP_TYPE_VOID: u16 = 19;
pub const OP_TYPE_BOOL: u16 = 20;
pub const OP_TYPE_INT: u16 = 21;
pub const OP_TYPE_FLOAT: u16 = 22;
pub const OP_TYPE_VECTOR: u16 = 23;
pub const OP_TYPE_MATRIX: u16 = 24;
pub const OP_TYPE_IMAGE: u16 = 25;
pub const OP_TYPE_SAMPLER: u16 = 26;
pub const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
pub const OP_TYPE_ARRAY: u16 = 28;
pub const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
pub const OP_TYPE_STRUCT: u16 = 30;
pub const OP_TYPE_POINTER: u16 = 32;
pub const OP_TYPE_FUNCTION: u16 = 33;
pub const OP_CONSTANT: u16 = 43;
pub const OP_CONSTANT_TRUE: u16 = 41;
pub const OP_CONSTANT_FALSE: u16 = 42;
pub const OP_FUNCTION: u16 = 54;
pub const OP_FUNCTION_PARAMETER: u16 = 55;
pub const OP_FUNCTION_END: u16 = 56;
pub const OP_FUNCTION_CALL: u16 = 57;
pub const OP_VARIABLE: u16 = 59;
pub const OP_LOAD: u16 = 61;
pub const OP_STORE: u16 = 62;
pub const OP_ACCESS_CHAIN: u16 = 65;
pub const OP_DECORATE: u16 = 71;
pub const OP_MEMBER_DECORATE: u16 = 72;
pub const OP_VECTOR_SHUFFLE: u16 = 79;
pub const OP_COMPOSITE_CONSTRUCT: u16 = 80;
pub const OP_COMPOSITE_EXTRACT: u16 = 81;
pub const OP_COPY_OBJECT: u16 = 83;
pub const OP_TRANSPOSE: u16 = 84;
pub const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
/// `OpImageSampleExplicitLod` — the only sample form legal under GLCompute
/// (ImplicitLod needs a fragment stage's implicit derivatives). The compute
/// `texture_sample_2d` path emits this with an explicit `Lod` operand; the
/// shader (fragment) `sample()` path keeps ImplicitLod, which is legal there.
pub const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
/// SPIR-V image-operands `Lod` bit (§3.14) — an explicit level-of-detail
/// operand follows the mask in `OpImageSampleExplicitLod`.
pub const IMAGE_OPERANDS_LOD: u32 = 0x2;
pub const OP_IMAGE_FETCH: u16 = 95;
pub const OP_IMAGE_READ: u16 = 98;
pub const OP_IMAGE_WRITE: u16 = 99;
pub const OP_IMAGE: u16 = 100;
// SPIR-V §3.42 numeric conversions — corrected from a prior mis-numbering
// that emitted OpFConvert/OpSConvert (reinterpret) for f↔int casts.
pub const OP_CONVERT_F_TO_U: u16 = 109;
pub const OP_CONVERT_F_TO_S: u16 = 110;
pub const OP_CONVERT_S_TO_F: u16 = 111;
pub const OP_CONVERT_U_TO_F: u16 = 112;
// Width conversions: OpUConvert zero-extends/truncates, OpSConvert
// sign-extends/truncates, OpFConvert changes float width. OpBitcast is
// only valid between types of the SAME total bit width — a 32↔64-bit
// int cast spelled as OpBitcast is invalid SPIR-V.
pub const OP_U_CONVERT: u16 = 113;
pub const OP_S_CONVERT: u16 = 114;
pub const OP_F_CONVERT: u16 = 115;
pub const OP_BITCAST: u16 = 124;
pub const OP_S_NEGATE: u16 = 126;
pub const OP_F_NEGATE: u16 = 127;
pub const OP_IADD: u16 = 128;
pub const OP_FADD: u16 = 129;
pub const OP_ISUB: u16 = 130;
pub const OP_FSUB: u16 = 131;
pub const OP_IMUL: u16 = 132;
pub const OP_FMUL: u16 = 133;
pub const OP_UDIV: u16 = 134;
pub const OP_SDIV: u16 = 135;
pub const OP_FDIV: u16 = 136;
pub const OP_UMOD: u16 = 137;
// 138 is OpSRem (sign follows dividend — matches Rust `%`), not OpSMod (139).
pub const OP_SREM: u16 = 138;
pub const OP_FREM: u16 = 140;
pub const OP_MATRIX_TIMES_VECTOR: u16 = 145;
pub const OP_VECTOR_TIMES_MATRIX: u16 = 144;
pub const OP_MATRIX_TIMES_MATRIX: u16 = 146;
pub const OP_DOT: u16 = 148;
pub const OP_LOGICAL_OR: u16 = 166;
pub const OP_LOGICAL_AND: u16 = 167;
pub const OP_LOGICAL_NOT: u16 = 168;
pub const OP_SELECT: u16 = 169;
pub const OP_IEQUAL: u16 = 170;
pub const OP_INOT_EQUAL: u16 = 171;
// SPIR-V §3.42.18 — signed/unsigned GE and LE pairs were swapped and
// FOrdNotEqual was off by one (181 = OpFUnordEqual). Verified vs spirv-dis.
pub const OP_UGREATER_THAN: u16 = 172;
pub const OP_SGREATER_THAN: u16 = 173;
pub const OP_UGREATER_THAN_EQUAL: u16 = 174;
pub const OP_SGREATER_THAN_EQUAL: u16 = 175;
pub const OP_ULESS_THAN: u16 = 176;
pub const OP_SLESS_THAN: u16 = 177;
pub const OP_ULESS_THAN_EQ: u16 = 178;
pub const OP_SLESS_THAN_EQUAL: u16 = 179;
pub const OP_FORD_EQUAL: u16 = 180;
pub const OP_FORD_NOT_EQUAL: u16 = 182;
pub const OP_FORD_LESS_THAN: u16 = 184;
pub const OP_FORD_GREATER_THAN: u16 = 186;
pub const OP_FORD_LESS_THAN_EQUAL: u16 = 188;
pub const OP_FORD_GREATER_THAN_EQUAL: u16 = 190;
pub const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
pub const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
pub const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
pub const OP_BITWISE_OR: u16 = 197;
pub const OP_BITWISE_XOR: u16 = 198;
pub const OP_BITWISE_AND: u16 = 199;
pub const OP_NOT: u16 = 200;
pub const OP_BIT_COUNT: u16 = 205;
pub const OP_CONTROL_BARRIER: u16 = 224;
pub const OP_ATOMIC_EXCHANGE: u16 = 229;
pub const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
pub const OP_ATOMIC_IADD: u16 = 234;
pub const OP_ATOMIC_ISUB: u16 = 235;
pub const OP_ATOMIC_SMIN: u16 = 236;
pub const OP_ATOMIC_UMIN: u16 = 237;
pub const OP_ATOMIC_SMAX: u16 = 238;
pub const OP_ATOMIC_UMAX: u16 = 239;
pub const OP_ATOMIC_AND: u16 = 240;
pub const OP_ATOMIC_OR: u16 = 241;
pub const OP_ATOMIC_XOR: u16 = 242;
pub const OP_DPDX: u16 = 207;
pub const OP_DPDY: u16 = 208;
pub const OP_FWIDTH: u16 = 209;
pub const OP_PHI: u16 = 245;
pub const OP_LOOP_MERGE: u16 = 246;
pub const OP_SELECTION_MERGE: u16 = 247;
pub const OP_LABEL: u16 = 248;
pub const OP_BRANCH: u16 = 249;
pub const OP_BRANCH_CONDITIONAL: u16 = 250;
pub const OP_RETURN: u16 = 253;
// OpGroupNonUniformAll is 334 (336 is OpGroupNonUniformAllEqual). Any is 335.
pub const OP_GROUP_NON_UNIFORM_ALL: u16 = 334;
pub const OP_GROUP_NON_UNIFORM_ANY: u16 = 335;
pub const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
pub const OP_GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
pub const OP_GROUP_NON_UNIFORM_IADD: u16 = 349;
pub const OP_GROUP_NON_UNIFORM_FADD: u16 = 350;
// SPIR-V §3.42.24 — min/max block was shifted +1. Verified vs spirv-dis.
pub const OP_GROUP_NON_UNIFORM_SMIN: u16 = 353;
pub const OP_GROUP_NON_UNIFORM_UMIN: u16 = 354;
pub const OP_GROUP_NON_UNIFORM_FMIN: u16 = 355;
pub const OP_GROUP_NON_UNIFORM_SMAX: u16 = 356;
pub const OP_GROUP_NON_UNIFORM_UMAX: u16 = 357;
pub const OP_GROUP_NON_UNIFORM_FMAX: u16 = 358;

// ── Storage classes ─────────────────────────────────────────────────────────

pub const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
pub const STORAGE_CLASS_INPUT: u32 = 1;
pub const STORAGE_CLASS_OUTPUT: u32 = 3;
pub const STORAGE_CLASS_WORKGROUP: u32 = 4;
pub const STORAGE_CLASS_FUNCTION: u32 = 7;
pub const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
pub const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// ── Decorations ─────────────────────────────────────────────────────────────

pub const DECORATION_BLOCK: u32 = 2;
pub const DECORATION_ARRAY_STRIDE: u32 = 6;
pub const DECORATION_BUILTIN: u32 = 11;
pub const DECORATION_RESTRICT: u32 = 19;
pub const DECORATION_NON_WRITABLE: u32 = 24;
pub const DECORATION_LOCATION: u32 = 30;
pub const DECORATION_BINDING: u32 = 33;
pub const DECORATION_DESCRIPTOR_SET: u32 = 34;
pub const DECORATION_OFFSET: u32 = 35;
pub const DECORATION_FP_FAST_MATH_MODE: u32 = 40;

// ── FPFastMathMode bits ────────────────────────────────────────────────────
//
// NotNaN (0x1) | NotInf (0x2) | NSZ (0x4) | AllowRecip (0x8) | Fast (0x10)
// Fast implies all of the above.
pub const FP_FAST_MATH_FAST: u32 = 0x10;

// ── Built-in values ─────────────────────────────────────────────────────────

pub const BUILTIN_POSITION: u32 = 0;
pub const BUILTIN_NUM_WORKGROUPS: u32 = 24;
pub const BUILTIN_WORKGROUP_ID: u32 = 26;
pub const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
pub const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// ── Execution model / mode ──────────────────────────────────────────────────

pub const EXECUTION_MODEL_VERTEX: u32 = 0;
pub const EXECUTION_MODEL_FRAGMENT: u32 = 4;
pub const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
pub const EXECUTION_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
pub const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// ── Memory model ────────────────────────────────────────────────────────────

pub const ADDRESSING_MODEL_LOGICAL: u32 = 0;
pub const MEMORY_MODEL_GLSL450: u32 = 1;

// ── Capabilities ────────────────────────────────────────────────────────────

pub const CAPABILITY_SHADER: u32 = 1;
pub const CAPABILITY_FLOAT16: u32 = 9;
pub const CAPABILITY_FLOAT64: u32 = 10;
pub const CAPABILITY_INT64: u32 = 11;
// 16-bit storage-buffer access (SPV_KHR_16bit_storage, core in SPIR-V 1.3).
// Declared for the native bf16 storage element (`ushort` per element).
pub const CAPABILITY_STORAGE_BUFFER_16BIT_ACCESS: u32 = 4433;
// 8-bit storage-buffer access (SPV_KHR_8bit_storage; the capability is only
// core from SPIR-V 1.5, so modules also declare the OpExtension). Declared
// for the native fp8 storage element (`uchar` per element).
pub const CAPABILITY_STORAGE_BUFFER_8BIT_ACCESS: u32 = 4448;
// SPIR-V §3.31 capabilities: NonUniform=61, Vote=62, Arithmetic=63,
// Ballot=64, Shuffle=65. Ballot and Shuffle were each one too low.
pub const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
pub const CAPABILITY_GROUP_NON_UNIFORM_VOTE: u32 = 62;
pub const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
pub const CAPABILITY_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
pub const CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;

// ── Scope / memory semantics ────────────────────────────────────────────────

pub const SCOPE_SUBGROUP: u32 = 3;
pub const SCOPE_WORKGROUP: u32 = 2;
pub const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100;
pub const MEMORY_SEMANTICS_UNIFORM_MEMORY: u32 = 0x40;
pub const MEMORY_SEMANTICS_ACQUIRE: u32 = 0x2;
pub const MEMORY_SEMANTICS_RELEASE: u32 = 0x4;
pub const MEMORY_SEMANTICS_ACQ_REL: u32 = 0x8;
pub const MEMORY_SEMANTICS_SEQ_CST: u32 = 0x10;

pub const OP_MEMORY_BARRIER: u16 = 225;

// ── Group operations ────────────────────────────────────────────────────────

pub const GROUP_OPERATION_REDUCE: u32 = 0;
pub const GROUP_OPERATION_INCLUSIVE_SCAN: u32 = 1;
pub const GROUP_OPERATION_EXCLUSIVE_SCAN: u32 = 2;

// ── Function / selection / loop control ─────────────────────────────────────

pub const FUNCTION_CONTROL_NONE: u32 = 0;
pub const SELECTION_CONTROL_NONE: u32 = 0;
pub const LOOP_CONTROL_NONE: u32 = 0;
pub const LOOP_CONTROL_UNROLL: u32 = 0x1;

// ── SPIR-V header ───────────────────────────────────────────────────────────

pub const SPIRV_MAGIC: u32 = 0x07230203;
pub const SPIRV_VERSION_1_3: u32 = 0x00010300;
pub const SPIRV_GENERATOR: u32 = 0;
pub const SPIRV_SCHEMA: u32 = 0;

// ── GLSL.std.450 extended instruction numbers ───────────────────────────────

pub const GLSL_ROUND: u32 = 1;
pub const GLSL_FABS: u32 = 4;
pub const GLSL_SABS: u32 = 5;
pub const GLSL_FLOOR: u32 = 8;
pub const GLSL_CEIL: u32 = 9;
pub const GLSL_FRACT: u32 = 10;
pub const GLSL_SIN: u32 = 13;
pub const GLSL_COS: u32 = 14;
pub const GLSL_TAN: u32 = 15;
pub const GLSL_ASIN: u32 = 16;
pub const GLSL_ACOS: u32 = 17;
pub const GLSL_ATAN: u32 = 18;
pub const GLSL_ATAN2: u32 = 25;
pub const GLSL_POW: u32 = 26;
pub const GLSL_EXP: u32 = 27;
pub const GLSL_LOG: u32 = 28;
pub const GLSL_EXP2: u32 = 29;
pub const GLSL_LOG2: u32 = 30;
pub const GLSL_SQRT: u32 = 31;
pub const GLSL_INVERSE_SQRT: u32 = 32;
pub const GLSL_FMIN: u32 = 37;
pub const GLSL_UMIN: u32 = 38;
pub const GLSL_SMIN: u32 = 39;
pub const GLSL_FMAX: u32 = 40;
pub const GLSL_UMAX: u32 = 41;
pub const GLSL_SMAX: u32 = 42;
pub const GLSL_FCLAMP: u32 = 43;
pub const GLSL_UCLAMP: u32 = 44;
pub const GLSL_SCLAMP: u32 = 45;
pub const GLSL_FMIX: u32 = 46;
pub const GLSL_STEP: u32 = 48;
pub const GLSL_SMOOTH_STEP: u32 = 49;
pub const GLSL_FMA: u32 = 50;
/// Pack a `vec4<f32>` of unorm channels into one `0xAABBGGRR` u32 (RGBA8-unorm
/// storage-image read boundary — clamps each component to [0,1] × 255).
pub const GLSL_PACK_UNORM_4X8: u32 = 55;
/// Unpack a `0xAABBGGRR` u32 into a `vec4<f32>` of unorm channels (RGBA8-unorm
/// storage-image write boundary — the inverse of `GLSL_PACK_UNORM_4X8`).
pub const GLSL_UNPACK_UNORM_4X8: u32 = 64;
pub const GLSL_LENGTH: u32 = 66;
pub const GLSL_DISTANCE: u32 = 67;
pub const GLSL_CROSS: u32 = 68;
pub const GLSL_NORMALIZE: u32 = 69;
pub const GLSL_FIND_I_LSB: u32 = 73;
pub const GLSL_FIND_U_MSB: u32 = 75;
