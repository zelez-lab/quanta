//! SPIR-V opcode and constant definitions.
//!
//! All values match the SPIR-V 1.3 specification.

#![allow(dead_code)]

// ── SPIR-V opcodes ──────────────────────────────────────────────────────────

pub(crate) const OP_NAME: u16 = 5;
pub(crate) const OP_EXT_INST_IMPORT: u16 = 11;
pub(crate) const OP_EXT_INST: u16 = 12;
pub(crate) const OP_MEMORY_MODEL: u16 = 14;
pub(crate) const OP_ENTRY_POINT: u16 = 15;
pub(crate) const OP_EXECUTION_MODE: u16 = 16;
pub(crate) const OP_CAPABILITY: u16 = 17;
pub(crate) const OP_TYPE_VOID: u16 = 19;
pub(crate) const OP_TYPE_BOOL: u16 = 20;
pub(crate) const OP_TYPE_INT: u16 = 21;
pub(crate) const OP_TYPE_FLOAT: u16 = 22;
pub(crate) const OP_TYPE_VECTOR: u16 = 23;
pub(crate) const OP_TYPE_ARRAY: u16 = 28;
pub(crate) const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
pub(crate) const OP_TYPE_STRUCT: u16 = 30;
pub(crate) const OP_TYPE_POINTER: u16 = 32;
pub(crate) const OP_TYPE_FUNCTION: u16 = 33;
pub(crate) const OP_CONSTANT: u16 = 43;
pub(crate) const OP_CONSTANT_TRUE: u16 = 41;
pub(crate) const OP_CONSTANT_FALSE: u16 = 42;
pub(crate) const OP_FUNCTION: u16 = 54;
pub(crate) const OP_FUNCTION_PARAMETER: u16 = 55;
pub(crate) const OP_FUNCTION_END: u16 = 56;
pub(crate) const OP_FUNCTION_CALL: u16 = 57;
pub(crate) const OP_VARIABLE: u16 = 59;
pub(crate) const OP_LOAD: u16 = 61;
pub(crate) const OP_STORE: u16 = 62;
pub(crate) const OP_ACCESS_CHAIN: u16 = 65;
pub(crate) const OP_DECORATE: u16 = 71;
pub(crate) const OP_MEMBER_DECORATE: u16 = 72;
pub(crate) const OP_COMPOSITE_CONSTRUCT: u16 = 80;
pub(crate) const OP_COMPOSITE_EXTRACT: u16 = 81;
pub(crate) const OP_COPY_OBJECT: u16 = 83;
// SPIR-V §3.42 numeric conversions. These four were previously
// mis-numbered (FToU/FToS/SToF shifted up by 3/5/3), which made the
// emitter spell f→int and int→f conversions as OpFConvert/OpSConvert —
// a float-to-float / int-to-int conversion of mismatched types that
// drivers execute as a reinterpret, returning the source bit pattern.
pub(crate) const OP_CONVERT_F_TO_U: u16 = 109;
pub(crate) const OP_CONVERT_F_TO_S: u16 = 110;
pub(crate) const OP_CONVERT_S_TO_F: u16 = 111;
pub(crate) const OP_CONVERT_U_TO_F: u16 = 112;
pub(crate) const OP_U_CONVERT: u16 = 113; // unsigned int width conversion (u16↔u32)
pub(crate) const OP_BITCAST: u16 = 124;
pub(crate) const OP_S_NEGATE: u16 = 126;
pub(crate) const OP_F_NEGATE: u16 = 127;
pub(crate) const OP_IADD: u16 = 128;
pub(crate) const OP_FADD: u16 = 129;
pub(crate) const OP_ISUB: u16 = 130;
pub(crate) const OP_FSUB: u16 = 131;
pub(crate) const OP_IMUL: u16 = 132;
pub(crate) const OP_FMUL: u16 = 133;
pub(crate) const OP_UDIV: u16 = 134;
pub(crate) const OP_SDIV: u16 = 135;
pub(crate) const OP_FDIV: u16 = 136;
pub(crate) const OP_UMOD: u16 = 137;
// 138 is OpSRem (remainder, sign follows dividend — matches Rust `%`).
// OpSMod (139) differs for mixed-sign operands; not what `BinOp::Rem` wants.
pub(crate) const OP_SREM: u16 = 138;
pub(crate) const OP_FREM: u16 = 140;
pub(crate) const OP_DOT: u16 = 148;
pub(crate) const OP_LOGICAL_OR: u16 = 166;
pub(crate) const OP_LOGICAL_AND: u16 = 167;
pub(crate) const OP_LOGICAL_NOT: u16 = 168;
pub(crate) const OP_SELECT: u16 = 169;
// SPIR-V §3.42.18 comparison opcodes. The signed/unsigned GreaterThanEqual
// and LessThanEqual pairs were swapped, and FOrdNotEqual was off by one
// (181 is OpFUnordEqual — an inverted result). Verified against spirv-dis.
pub(crate) const OP_IEQUAL: u16 = 170;
pub(crate) const OP_INOT_EQUAL: u16 = 171;
pub(crate) const OP_UGREATER_THAN: u16 = 172;
pub(crate) const OP_SGREATER_THAN: u16 = 173;
pub(crate) const OP_UGREATER_THAN_EQUAL: u16 = 174;
pub(crate) const OP_SGREATER_THAN_EQUAL: u16 = 175;
pub(crate) const OP_ULESS_THAN: u16 = 176;
pub(crate) const OP_SLESS_THAN: u16 = 177;
pub(crate) const OP_ULESS_THAN_EQ: u16 = 178;
pub(crate) const OP_SLESS_THAN_EQUAL: u16 = 179;
pub(crate) const OP_FORD_EQUAL: u16 = 180;
pub(crate) const OP_FORD_NOT_EQUAL: u16 = 182;
pub(crate) const OP_FORD_LESS_THAN: u16 = 184;
pub(crate) const OP_FORD_GREATER_THAN: u16 = 186;
pub(crate) const OP_FORD_LESS_THAN_EQUAL: u16 = 188;
pub(crate) const OP_FORD_GREATER_THAN_EQUAL: u16 = 190;
pub(crate) const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
pub(crate) const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
pub(crate) const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
// SPIR-V §3.42.14 — these three were rotated (AND↔Or, OR↔Xor, XOR↔And).
// Verified against spirv-dis. The bug surfaced on i64 rotate (which masks
// with OpBitwiseAnd) and i64 bitxor.
pub(crate) const OP_BITWISE_OR: u16 = 197;
pub(crate) const OP_BITWISE_XOR: u16 = 198;
pub(crate) const OP_BITWISE_AND: u16 = 199;
pub(crate) const OP_NOT: u16 = 200;
pub(crate) const OP_BIT_COUNT: u16 = 205;
pub(crate) const OP_CONTROL_BARRIER: u16 = 224;
pub(crate) const OP_ATOMIC_EXCHANGE: u16 = 229;
pub(crate) const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
pub(crate) const OP_ATOMIC_IADD: u16 = 234;
pub(crate) const OP_ATOMIC_ISUB: u16 = 235;
pub(crate) const OP_ATOMIC_SMIN: u16 = 236;
pub(crate) const OP_ATOMIC_UMIN: u16 = 237;
pub(crate) const OP_ATOMIC_SMAX: u16 = 238;
pub(crate) const OP_ATOMIC_UMAX: u16 = 239;
pub(crate) const OP_ATOMIC_AND: u16 = 240;
pub(crate) const OP_ATOMIC_OR: u16 = 241;
pub(crate) const OP_ATOMIC_XOR: u16 = 242;
pub(crate) const OP_PHI: u16 = 245;
pub(crate) const OP_LOOP_MERGE: u16 = 246;
pub(crate) const OP_SELECTION_MERGE: u16 = 247;
pub(crate) const OP_LABEL: u16 = 248;
pub(crate) const OP_BRANCH: u16 = 249;
pub(crate) const OP_BRANCH_CONDITIONAL: u16 = 250;
pub(crate) const OP_RETURN: u16 = 253;
pub(crate) const OP_GROUP_NON_UNIFORM_SHUFFLE_XOR: u16 = 346;
pub(crate) const OP_GROUP_NON_UNIFORM_IADD: u16 = 349;
pub(crate) const OP_GROUP_NON_UNIFORM_FADD: u16 = 350;
// SPIR-V §3.42.24 — the min/max block was shifted +1 (SMin→354=UMin, …,
// FMax→359=BitwiseAnd), corrupting signed subgroup reductions. Verified
// against spirv-dis.
pub(crate) const OP_GROUP_NON_UNIFORM_SMIN: u16 = 353;
pub(crate) const OP_GROUP_NON_UNIFORM_UMIN: u16 = 354;
pub(crate) const OP_GROUP_NON_UNIFORM_FMIN: u16 = 355;
pub(crate) const OP_GROUP_NON_UNIFORM_SMAX: u16 = 356;
pub(crate) const OP_GROUP_NON_UNIFORM_UMAX: u16 = 357;
pub(crate) const OP_GROUP_NON_UNIFORM_FMAX: u16 = 358;

#[allow(dead_code)]
pub(crate) const OP_IMAGE_FETCH: u16 = 95;

// ── Storage classes ─────────────────────────────────────────────────────────

pub(crate) const STORAGE_CLASS_INPUT: u32 = 1;
pub(crate) const STORAGE_CLASS_OUTPUT: u32 = 3;
pub(crate) const STORAGE_CLASS_WORKGROUP: u32 = 4;
pub(crate) const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
pub(crate) const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// ── Decorations ─────────────────────────────────────────────────────────────

pub(crate) const DECORATION_BLOCK: u32 = 2;
pub(crate) const DECORATION_ARRAY_STRIDE: u32 = 6;
pub(crate) const DECORATION_BUILTIN: u32 = 11;
pub(crate) const DECORATION_RESTRICT: u32 = 19;
pub(crate) const DECORATION_NON_WRITABLE: u32 = 24;
pub(crate) const DECORATION_LOCATION: u32 = 30;
pub(crate) const DECORATION_BINDING: u32 = 33;
pub(crate) const DECORATION_DESCRIPTOR_SET: u32 = 34;
pub(crate) const DECORATION_OFFSET: u32 = 35;
pub(crate) const DECORATION_FP_FAST_MATH_MODE: u32 = 40;

// ── FPFastMathMode bits ────────────────────────────────────────────────────
//
// NotNaN (0x1) | NotInf (0x2) | NSZ (0x4) | AllowRecip (0x8) | Fast (0x10)
// Fast implies all of the above.
pub(crate) const FP_FAST_MATH_FAST: u32 = 0x10;

// ── Built-in values ─────────────────────────────────────────────────────────

pub(crate) const BUILTIN_POSITION: u32 = 0;
pub(crate) const BUILTIN_NUM_WORKGROUPS: u32 = 24;
pub(crate) const BUILTIN_WORKGROUP_ID: u32 = 26;
pub(crate) const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
pub(crate) const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// ── Execution model / mode ──────────────────────────────────────────────────

pub(crate) const EXECUTION_MODEL_VERTEX: u32 = 0;
pub(crate) const EXECUTION_MODEL_FRAGMENT: u32 = 4;
pub(crate) const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
pub(crate) const EXECUTION_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
pub(crate) const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// ── Memory model ────────────────────────────────────────────────────────────

pub(crate) const ADDRESSING_MODEL_LOGICAL: u32 = 0;
pub(crate) const MEMORY_MODEL_GLSL450: u32 = 1;

// ── Capabilities ────────────────────────────────────────────────────────────

pub(crate) const CAPABILITY_SHADER: u32 = 1;
pub(crate) const CAPABILITY_FLOAT16: u32 = 9;
pub(crate) const CAPABILITY_FLOAT64: u32 = 10;
pub(crate) const CAPABILITY_INT64: u32 = 11;
pub(crate) const CAPABILITY_INT16: u32 = 22;
// 16-bit storage-buffer access (SPV_KHR_16bit_storage). Used for native
// bf16 buffers when the device advertises `storageBuffer16BitAccess`.
pub(crate) const CAPABILITY_STORAGE_BUFFER_16BIT_ACCESS: u32 = 4433;
pub(crate) const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
pub(crate) const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
pub(crate) const CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;

// ── Scope / memory semantics ────────────────────────────────────────────────

pub(crate) const SCOPE_SUBGROUP: u32 = 3;
pub(crate) const SCOPE_WORKGROUP: u32 = 2;
pub(crate) const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100;
pub(crate) const MEMORY_SEMANTICS_UNIFORM_MEMORY: u32 = 0x40;
pub(crate) const MEMORY_SEMANTICS_ACQUIRE: u32 = 0x2;
pub(crate) const MEMORY_SEMANTICS_RELEASE: u32 = 0x4;
pub(crate) const MEMORY_SEMANTICS_ACQ_REL: u32 = 0x8;
pub(crate) const MEMORY_SEMANTICS_SEQ_CST: u32 = 0x10;

pub(crate) const OP_MEMORY_BARRIER: u16 = 225;

// ── Group operations ────────────────────────────────────────────────────────

pub(crate) const GROUP_OPERATION_REDUCE: u32 = 0;
pub(crate) const GROUP_OPERATION_INCLUSIVE_SCAN: u32 = 1;
pub(crate) const GROUP_OPERATION_EXCLUSIVE_SCAN: u32 = 2;

// ── Function / selection / loop control ─────────────────────────────────────

pub(crate) const FUNCTION_CONTROL_NONE: u32 = 0;
pub(crate) const SELECTION_CONTROL_NONE: u32 = 0;
pub(crate) const LOOP_CONTROL_NONE: u32 = 0;
pub(crate) const LOOP_CONTROL_UNROLL: u32 = 0x1;

// ── SPIR-V header ───────────────────────────────────────────────────────────

pub(crate) const SPIRV_MAGIC: u32 = 0x07230203;
pub(crate) const SPIRV_VERSION_1_3: u32 = 0x00010300;
pub(crate) const SPIRV_GENERATOR: u32 = 0;
pub(crate) const SPIRV_SCHEMA: u32 = 0;

// ── GLSL.std.450 extended instruction numbers ───────────────────────────────

pub(crate) const GLSL_ROUND: u32 = 1;
/// Round-to-nearest-even (deterministic on .5 — unlike GLSL_ROUND).
pub(crate) const GLSL_ROUND_EVEN: u32 = 2;
pub(crate) const GLSL_FABS: u32 = 4;
pub(crate) const GLSL_SABS: u32 = 5;
pub(crate) const GLSL_FLOOR: u32 = 8;
pub(crate) const GLSL_CEIL: u32 = 9;
pub(crate) const GLSL_SIN: u32 = 13;
pub(crate) const GLSL_COS: u32 = 14;
pub(crate) const GLSL_TAN: u32 = 15;
pub(crate) const GLSL_ASIN: u32 = 16;
pub(crate) const GLSL_ACOS: u32 = 17;
pub(crate) const GLSL_ATAN: u32 = 18;
pub(crate) const GLSL_ATAN2: u32 = 25;
pub(crate) const GLSL_POW: u32 = 26;
pub(crate) const GLSL_EXP: u32 = 27;
pub(crate) const GLSL_LOG: u32 = 28;
pub(crate) const GLSL_EXP2: u32 = 29;
pub(crate) const GLSL_LOG2: u32 = 30;
pub(crate) const GLSL_SQRT: u32 = 31;
pub(crate) const GLSL_INVERSE_SQRT: u32 = 32;
pub(crate) const GLSL_FMIN: u32 = 37;
pub(crate) const GLSL_UMIN: u32 = 38;
pub(crate) const GLSL_SMIN: u32 = 39;
pub(crate) const GLSL_FMAX: u32 = 40;
pub(crate) const GLSL_UMAX: u32 = 41;
pub(crate) const GLSL_SMAX: u32 = 42;
pub(crate) const GLSL_FCLAMP: u32 = 43;
pub(crate) const GLSL_UCLAMP: u32 = 44;
pub(crate) const GLSL_SCLAMP: u32 = 45;
pub(crate) const GLSL_FMA: u32 = 50;
pub(crate) const GLSL_FIND_I_LSB: u32 = 73;
pub(crate) const GLSL_FIND_U_MSB: u32 = 75;
