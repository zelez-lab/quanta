//! KernelDef → Vulkan SPIR-V binary.
//!
//! Walks KernelOps and produces valid Vulkan SPIR-V binary (Shader capability,
//! GLCompute execution model, StorageBuffer storage class). This replaces the
//! LLVM spirv64 backend which emits OpenCL-style SPIR-V that Vulkan rejects.
//!
//! The output is a `Vec<u8>` ready for `vkCreateShaderModule`.

use quanta_ir::*;
use std::collections::HashMap;

// ── SPIR-V opcodes ──────────────────────────────────────────────────────────

const OP_NAME: u16 = 5;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_IMAGE: u16 = 25;
#[allow(dead_code)]
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_CONSTANT_TRUE: u16 = 41;
const OP_CONSTANT_FALSE: u16 = 42;
// const OP_CONSTANT_COMPOSITE: u16 = 44;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_PARAMETER: u16 = 55;
const OP_FUNCTION_END: u16 = 56;
const OP_FUNCTION_CALL: u16 = 57;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_COPY_OBJECT: u16 = 83;
#[allow(dead_code)]
const OP_TRANSPOSE: u16 = 84;
const OP_MATRIX_TIMES_VECTOR: u16 = 145;
#[allow(dead_code)]
const OP_VECTOR_TIMES_MATRIX: u16 = 144;
#[allow(dead_code)]
const OP_MATRIX_TIMES_MATRIX: u16 = 146;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_FETCH: u16 = 95;
#[allow(dead_code)]
const OP_SELECT: u16 = 169;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_CONVERT_F_TO_U: u16 = 113;
const OP_CONVERT_S_TO_F: u16 = 114;
const OP_CONVERT_F_TO_S: u16 = 115;
const OP_BITCAST: u16 = 124;
const OP_S_NEGATE: u16 = 126;
const OP_F_NEGATE: u16 = 127;
const OP_IADD: u16 = 128;
const OP_FADD: u16 = 129;
const OP_ISUB: u16 = 130;
const OP_FSUB: u16 = 131;
const OP_IMUL: u16 = 132;
const OP_FMUL: u16 = 133;
const OP_UDIV: u16 = 134;
const OP_SDIV: u16 = 135;
const OP_FDIV: u16 = 136;
const OP_UMOD: u16 = 137;
const OP_SMOD: u16 = 138;
const OP_FREM: u16 = 140;
const OP_LOGICAL_NOT: u16 = 168;
const OP_IEQUAL: u16 = 170;
const OP_INOT_EQUAL: u16 = 171;
const OP_UGREATER_THAN: u16 = 172;
const OP_SGREATER_THAN: u16 = 174;
const OP_ULESS_THAN: u16 = 176;
const OP_SLESS_THAN: u16 = 178;
const OP_FORD_EQUAL: u16 = 180;
const OP_FORD_NOT_EQUAL: u16 = 181;
const OP_FORD_LESS_THAN: u16 = 184;
const OP_FORD_GREATER_THAN: u16 = 186;
const OP_FORD_LESS_THAN_EQUAL: u16 = 188;
const OP_FORD_GREATER_THAN_EQUAL: u16 = 190;
// OP_ULESS_THAN_EQUAL is 177 — see OP_ULESS_THAN_EQ below
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_AND: u16 = 197;
const OP_BITWISE_OR: u16 = 198;
const OP_BITWISE_XOR: u16 = 199;
const OP_NOT: u16 = 200;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_PHI: u16 = 245;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;
// Additional comparison opcodes
const OP_UGREATER_THAN_EQUAL: u16 = 173;
const OP_SGREATER_THAN_EQUAL: u16 = 175;
const OP_ULESS_THAN_EQ: u16 = 177;
const OP_SLESS_THAN_EQUAL: u16 = 179;

// Extended instruction opcodes (GLSL.std.450)
const OP_EXT_INST: u16 = 12;

// Atomic opcodes
const OP_ATOMIC_EXCHANGE: u16 = 229;
const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
const OP_ATOMIC_IADD: u16 = 234;
const OP_ATOMIC_ISUB: u16 = 235;
const OP_ATOMIC_SMIN: u16 = 236;
const OP_ATOMIC_UMIN: u16 = 237;
const OP_ATOMIC_SMAX: u16 = 238;
const OP_ATOMIC_UMAX: u16 = 239;
const OP_ATOMIC_AND: u16 = 240;
const OP_ATOMIC_OR: u16 = 241;
const OP_ATOMIC_XOR: u16 = 242;

// ── Storage classes ─────────────────────────────────────────────────────────

const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_OUTPUT: u32 = 3;
const STORAGE_CLASS_WORKGROUP: u32 = 4;
// const STORAGE_CLASS_FUNCTION: u32 = 7;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// ── Decorations ─────────────────────────────────────────────────────────────

const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

// ── Built-in values ─────────────────────────────────────────────────────────

const BUILTIN_POSITION: u32 = 0;
const BUILTIN_NUM_WORKGROUPS: u32 = 24;
// const BUILTIN_WORKGROUP_SIZE: u32 = 25;
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// ── Execution model / mode ──────────────────────────────────────────────────

const EXECUTION_MODEL_VERTEX: u32 = 0;
const EXECUTION_MODEL_FRAGMENT: u32 = 4;
const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
const EXECUTION_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// ── Memory model ────────────────────────────────────────────────────────────

const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// ── Capabilities ────────────────────────────────────────────────────────────

const CAPABILITY_SHADER: u32 = 1;

// ── Scope / memory semantics ────────────────────────────────────────────────

const SCOPE_SUBGROUP: u32 = 3;
const SCOPE_WORKGROUP: u32 = 2;
const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100; // WorkgroupMemory
const MEMORY_SEMANTICS_ACQ_REL: u32 = 0x8; // AcquireRelease

// ── GLSL.std.450 extended instruction numbers ───────────────────────────────

const GLSL_ROUND: u32 = 1;
const GLSL_FLOOR: u32 = 8;
const GLSL_CEIL: u32 = 9;
const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;
const GLSL_TAN: u32 = 15;
const GLSL_ASIN: u32 = 16;
const GLSL_ACOS: u32 = 17;
const GLSL_ATAN: u32 = 18;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28;
const GLSL_EXP2: u32 = 29;
const GLSL_LOG2: u32 = 30;
const GLSL_SQRT: u32 = 31;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_FABS: u32 = 4;
const GLSL_SABS: u32 = 5;
const GLSL_FMIN: u32 = 37;
const GLSL_UMIN: u32 = 38;
const GLSL_SMIN: u32 = 39;
const GLSL_FMAX: u32 = 40;
const GLSL_UMAX: u32 = 41;
const GLSL_SMAX: u32 = 42;
const GLSL_FCLAMP: u32 = 43;
const GLSL_UCLAMP: u32 = 44;
const GLSL_SCLAMP: u32 = 45;
const GLSL_FMA: u32 = 50;
const GLSL_POW: u32 = 26;
const GLSL_ATAN2: u32 = 25;
const GLSL_FMIX: u32 = 46;
const GLSL_LENGTH: u32 = 66;
const GLSL_DISTANCE: u32 = 67;
const GLSL_CROSS: u32 = 68;
const GLSL_NORMALIZE: u32 = 69;
const GLSL_FRACT: u32 = 10;
const GLSL_STEP: u32 = 48;
const GLSL_SMOOTH_STEP: u32 = 49;
const GLSL_FIND_I_LSB: u32 = 73;
const GLSL_FIND_U_MSB: u32 = 75;

// ── Subgroup opcodes ───────────────────────────────────────────────────────

const OP_BIT_COUNT: u16 = 205;
const OP_DOT: u16 = 148;
const OP_GROUP_NON_UNIFORM_IADD: u16 = 349;
const OP_GROUP_NON_UNIFORM_FADD: u16 = 350;
const OP_GROUP_NON_UNIFORM_SMIN: u16 = 354;
const OP_GROUP_NON_UNIFORM_UMIN: u16 = 355;
const OP_GROUP_NON_UNIFORM_FMIN: u16 = 356;
const OP_GROUP_NON_UNIFORM_SMAX: u16 = 357;
const OP_GROUP_NON_UNIFORM_UMAX: u16 = 358;
const OP_GROUP_NON_UNIFORM_FMAX: u16 = 359;

// Group operation constants for subgroup ops
const GROUP_OPERATION_REDUCE: u32 = 0;
const GROUP_OPERATION_INCLUSIVE_SCAN: u32 = 1;
const GROUP_OPERATION_EXCLUSIVE_SCAN: u32 = 2;

// Additional capabilities
const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
const CAPABILITY_GROUP_NON_UNIFORM_BALLOT: u32 = 62;
const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
const CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE: u32 = 64;

// Subgroup vote/ballot/shuffle opcodes
const OP_GROUP_NON_UNIFORM_ANY: u16 = 335;
const OP_GROUP_NON_UNIFORM_ALL: u16 = 336;
const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
const OP_GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;

// ── Function control ────────────────────────────────────────────────────────

const FUNCTION_CONTROL_NONE: u32 = 0;

// ── Selection / loop control ────────────────────────────────────────────────

const SELECTION_CONTROL_NONE: u32 = 0;
const LOOP_CONTROL_NONE: u32 = 0;

// ── SPIR-V header ───────────────────────────────────────────────────────────

const SPIRV_MAGIC: u32 = 0x07230203;
const SPIRV_VERSION_1_3: u32 = 0x00010300;
const SPIRV_GENERATOR: u32 = 0; // Unregistered
const SPIRV_SCHEMA: u32 = 0;

/// Emit Vulkan SPIR-V binary from a KernelDef.
///
/// Returns the SPIR-V module as bytes, ready for `vkCreateShaderModule`.
pub fn emit(kernel: &KernelDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    e.emit_kernel(kernel)?;
    Ok(e.finalize())
}

// ── Section buffers ─────────────────────────────────────────────────────────

/// SPIR-V requires instructions in a strict section order. We build each
/// section into its own word buffer, then concatenate at the end.
struct SpvEmitter {
    next_id: u32,

    // Section buffers (in required order)
    sec_capability: Vec<u32>,
    sec_extension: Vec<u32>,
    sec_ext_inst_import: Vec<u32>,
    sec_memory_model: Vec<u32>,
    sec_entry_point: Vec<u32>,
    sec_execution_mode: Vec<u32>,
    sec_debug: Vec<u32>,
    sec_annotation: Vec<u32>,
    sec_type_const: Vec<u32>,
    sec_global_var: Vec<u32>,
    sec_function: Vec<u32>,

    // Type caches to avoid duplicates
    type_void: Option<u32>,
    type_bool: Option<u32>,
    type_u32: Option<u32>,
    type_i32: Option<u32>,
    type_f32: Option<u32>,
    type_f64: Option<u32>,
    type_v3uint: Option<u32>,
    type_cache: HashMap<String, u32>,

    // Constant cache: key = "type_id:bit_pattern"
    const_cache: HashMap<String, u32>,

    // GLSL.std.450 extended instruction set ID
    glsl_ext_id: Option<u32>,

    // Texture sampler variables: slot → (var_id, type_id)
    // Used by both compute kernel texture params and fragment shader sample() calls.
    texture_samplers: HashMap<u32, (u32, u32)>,

    // Stack of loop merge labels for Break support
    loop_merge_stack: Vec<u32>,

    // Register → SPIR-V ID mapping (function-scoped variables)
    reg_ids: HashMap<u32, u32>,
    // Register → type ID (so we know what type a register holds)
    reg_types: HashMap<u32, u32>,

    // Field slot → (variable_id, element_type_id, is_writable)
    field_vars: HashMap<u32, (u32, u32, bool)>,

    // Track total push constant bytes needed
    push_constant_size: u32,
    // Which field slots are push constants (PushConstant storage class)
    push_constant_slots: std::collections::HashSet<u32>,

    // Shared memory: id → (variable_id, element_type_id)
    shared_vars: HashMap<u32, (u32, u32)>,

    // Track which types already have ArrayStride decoration applied
    decorated_stride: std::collections::HashSet<u32>,
    // Track which struct types already have Block decoration
    decorated_block: std::collections::HashSet<u32>,

    // Device function name → (function_id, return_type_id, param_type_ids)
    device_fn_ids: HashMap<String, (u32, u32, Vec<u32>)>,
    // Buffer for device function bodies (emitted after the main function)
    sec_device_fns: Vec<u32>,
}

impl SpvEmitter {
    fn new() -> Self {
        Self {
            next_id: 1,
            sec_capability: Vec::new(),
            sec_extension: Vec::new(),
            sec_ext_inst_import: Vec::new(),
            sec_memory_model: Vec::new(),
            sec_entry_point: Vec::new(),
            sec_execution_mode: Vec::new(),
            sec_debug: Vec::new(),
            sec_annotation: Vec::new(),
            sec_type_const: Vec::new(),
            sec_global_var: Vec::new(),
            sec_function: Vec::new(),
            type_void: None,
            type_bool: None,
            type_u32: None,
            type_i32: None,
            type_f32: None,
            type_f64: None,
            type_v3uint: None,
            type_cache: HashMap::new(),
            const_cache: HashMap::new(),
            glsl_ext_id: None,
            texture_samplers: HashMap::new(),
            loop_merge_stack: Vec::new(),
            reg_ids: HashMap::new(),
            reg_types: HashMap::new(),
            field_vars: HashMap::new(),
            push_constant_size: 0,
            push_constant_slots: std::collections::HashSet::new(),
            shared_vars: HashMap::new(),
            decorated_stride: std::collections::HashSet::new(),
            decorated_block: std::collections::HashSet::new(),
            device_fn_ids: HashMap::new(),
            sec_device_fns: Vec::new(),
        }
    }

    // ── ID allocator ────────────────────────────────────────────────────────

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ── Instruction encoding ────────────────────────────────────────────────

    /// Encode an instruction into a section buffer.
    /// Format: (word_count << 16) | opcode, then operand words.
    fn emit_op(section: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u16;
        section.push(((word_count as u32) << 16) | (opcode as u32));
        section.extend_from_slice(operands);
    }

    /// Encode a string as SPIR-V literal words (null-terminated, padded to
    /// word boundary).
    fn string_words(s: &str) -> Vec<u32> {
        let bytes = s.as_bytes();
        // +1 for null terminator, round up to multiple of 4
        let total_bytes = bytes.len() + 1;
        let word_count = total_bytes.div_ceil(4);
        let mut words = vec![0u32; word_count];
        for (i, &b) in bytes.iter().enumerate() {
            let word_idx = i / 4;
            let byte_idx = i % 4;
            words[word_idx] |= (b as u32) << (byte_idx * 8);
        }
        // Null terminator is already there since we initialized to 0
        words
    }

    // ── Type helpers ────────────────────────────────────────────────────────

    fn ensure_type_void(&mut self) -> u32 {
        if let Some(id) = self.type_void {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_VOID, &[id]);
        self.type_void = Some(id);
        id
    }

    fn ensure_type_bool(&mut self) -> u32 {
        if let Some(id) = self.type_bool {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_BOOL, &[id]);
        self.type_bool = Some(id);
        id
    }

    fn ensure_type_u32(&mut self) -> u32 {
        if let Some(id) = self.type_u32 {
            return id;
        }
        let id = self.alloc_id();
        // OpTypeInt %id 32 0 (unsigned)
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 32, 0]);
        self.type_u32 = Some(id);
        id
    }

    fn ensure_type_i32(&mut self) -> u32 {
        if let Some(id) = self.type_i32 {
            return id;
        }
        let id = self.alloc_id();
        // OpTypeInt %id 32 1 (signed)
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 32, 1]);
        self.type_i32 = Some(id);
        id
    }

    fn ensure_type_f32(&mut self) -> u32 {
        if let Some(id) = self.type_f32 {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_FLOAT, &[id, 32]);
        self.type_f32 = Some(id);
        id
    }

    fn ensure_type_f64(&mut self) -> u32 {
        if let Some(id) = self.type_f64 {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_FLOAT, &[id, 64]);
        self.type_f64 = Some(id);
        id
    }

    fn ensure_type_vector(&mut self, elem_type: u32, count: u32) -> u32 {
        let key = format!("vec_{}_{}", elem_type, count);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_VECTOR,
            &[id, elem_type, count],
        );
        self.type_cache.insert(key, id);
        id
    }

    fn ensure_type_v3uint(&mut self) -> u32 {
        if let Some(id) = self.type_v3uint {
            return id;
        }
        let uint = self.ensure_type_u32();
        let id = self.ensure_type_vector(uint, 3);
        self.type_v3uint = Some(id);
        id
    }

    fn ensure_type_runtime_array(&mut self, elem_type: u32) -> u32 {
        let key = format!("rtarr_{}", elem_type);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_RUNTIME_ARRAY,
            &[id, elem_type],
        );
        self.type_cache.insert(key, id);
        id
    }

    fn ensure_type_array(&mut self, elem_type: u32, length_id: u32) -> u32 {
        let key = format!("arr_{}_{}", elem_type, length_id);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_ARRAY,
            &[id, elem_type, length_id],
        );
        self.type_cache.insert(key, id);
        id
    }

    fn ensure_type_struct(&mut self, members: &[u32]) -> u32 {
        let key = format!(
            "struct_{}",
            members
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let mut ops = vec![id];
        ops.extend_from_slice(members);
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_STRUCT, &ops);
        self.type_cache.insert(key, id);
        id
    }

    fn ensure_type_pointer(&mut self, storage_class: u32, pointee: u32) -> u32 {
        let key = format!("ptr_{}_{}", storage_class, pointee);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_POINTER,
            &[id, storage_class, pointee],
        );
        self.type_cache.insert(key, id);
        id
    }

    fn ensure_type_function(&mut self, return_type: u32, params: &[u32]) -> u32 {
        let key = format!(
            "fn_{}_{}",
            return_type,
            params
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let mut ops = vec![id, return_type];
        ops.extend_from_slice(params);
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_FUNCTION, &ops);
        self.type_cache.insert(key, id);
        id
    }

    /// Map ScalarType to SPIR-V type ID.
    fn scalar_type_id(&mut self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::F32 => self.ensure_type_f32(),
            ScalarType::F64 => self.ensure_type_f64(),
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => self.ensure_type_u32(),
            ScalarType::U64 => {
                // For now, map U64 to U32 (Vulkan compute typically 32-bit)
                self.ensure_type_u32()
            }
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => self.ensure_type_i32(),
            ScalarType::I64 => self.ensure_type_i32(),
            ScalarType::F16 => {
                // Map F16 to F32 for basic support
                self.ensure_type_f32()
            }
            ScalarType::Bool => self.ensure_type_bool(),
        }
    }

    /// Get byte size of a scalar type (for ArrayStride decoration).
    fn scalar_byte_size(ty: ScalarType) -> u32 {
        match ty {
            ScalarType::F16 => 2,
            ScalarType::F32 => 4,
            ScalarType::F64 => 8,
            ScalarType::U8 | ScalarType::I8 => 1,
            ScalarType::U16 | ScalarType::I16 => 2,
            ScalarType::U32 | ScalarType::I32 => 4,
            ScalarType::U64 | ScalarType::I64 => 8,
            ScalarType::Bool => 4,
        }
    }

    // ── Constant helpers ────────────────────────────────────────────────────

    fn emit_constant_u32(&mut self, val: u32) -> u32 {
        let ty = self.ensure_type_u32();
        let key = format!("{}:{}", ty, val);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, val]);
        self.const_cache.insert(key, id);
        id
    }

    fn emit_constant_i32(&mut self, val: i32) -> u32 {
        let ty = self.ensure_type_i32();
        let key = format!("{}:{}", ty, val as u32);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, val as u32]);
        self.const_cache.insert(key, id);
        id
    }

    fn emit_constant_f32(&mut self, val: f32) -> u32 {
        let ty = self.ensure_type_f32();
        let bits = val.to_bits();
        let key = format!("{}:{}", ty, bits);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, bits]);
        self.const_cache.insert(key, id);
        id
    }

    fn emit_constant_f64(&mut self, val: f64) -> u32 {
        let ty = self.ensure_type_f64();
        let bits = val.to_bits();
        let lo = bits as u32;
        let hi = (bits >> 32) as u32;
        let key = format!("{}:{}:{}", ty, lo, hi);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, lo, hi]);
        self.const_cache.insert(key, id);
        id
    }

    fn emit_constant_bool(&mut self, val: bool) -> u32 {
        let ty = self.ensure_type_bool();
        let key = format!("bool:{}", val);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let opcode = if val {
            OP_CONSTANT_TRUE
        } else {
            OP_CONSTANT_FALSE
        };
        Self::emit_op(&mut self.sec_type_const, opcode, &[ty, id]);
        self.const_cache.insert(key, id);
        id
    }

    /// Ensure the GLSL.std.450 extended instruction set is imported.
    fn ensure_glsl_ext(&mut self) -> u32 {
        if let Some(id) = self.glsl_ext_id {
            return id;
        }
        let id = self.alloc_id();
        let name_words = Self::string_words("GLSL.std.450");
        let mut ops = vec![id];
        ops.extend_from_slice(&name_words);
        Self::emit_op(&mut self.sec_ext_inst_import, OP_EXT_INST_IMPORT, &ops);
        self.glsl_ext_id = Some(id);
        id
    }

    // ── Name helper ─────────────────────────────────────────────────────────

    fn emit_name(&mut self, id: u32, name: &str) {
        let name_words = Self::string_words(name);
        let mut ops = vec![id];
        ops.extend_from_slice(&name_words);
        Self::emit_op(&mut self.sec_debug, OP_NAME, &ops);
    }

    // ── Decoration helpers ──────────────────────────────────────────────────

    fn decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(operands);
        Self::emit_op(&mut self.sec_annotation, OP_DECORATE, &ops);
    }

    fn member_decorate(
        &mut self,
        struct_type: u32,
        member: u32,
        decoration: u32,
        operands: &[u32],
    ) {
        let mut ops = vec![struct_type, member, decoration];
        ops.extend_from_slice(operands);
        Self::emit_op(&mut self.sec_annotation, OP_MEMBER_DECORATE, &ops);
    }

    // ── Get or create a function-scoped variable for a register ─────────────

    /// Get the SPIR-V ID for a virtual register. If the register has no ID yet,
    /// allocate a function-scoped variable and record it. Returns the ID of
    /// the value (not the variable pointer).
    fn reg_value_id(&self, reg: Reg) -> Result<u32, String> {
        self.reg_ids
            .get(&reg.0)
            .copied()
            .ok_or_else(|| format!("register r{} used before definition", reg.0))
    }

    fn set_reg(&mut self, reg: Reg, id: u32, type_id: u32) {
        self.reg_ids.insert(reg.0, id);
        self.reg_types.insert(reg.0, type_id);
    }

    #[allow(dead_code)]
    fn reg_type_id(&self, reg: Reg) -> Result<u32, String> {
        self.reg_types
            .get(&reg.0)
            .copied()
            .ok_or_else(|| format!("register r{} type unknown", reg.0))
    }

    // ── Device function emission ─────────────────────────────────────────

    /// Emit device functions as SPIR-V OpFunction definitions.
    /// Must be called before emit_kernel so that function IDs are available
    /// for OpFunctionCall during body emission. The function bodies are
    /// emitted into sec_device_fns which is placed before sec_function
    /// in the final module.
    fn emit_device_functions(
        &mut self,
        kernel: &KernelDef,
        gid_var: u32,
        local_id_var: u32,
        group_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        for device_fn in &kernel.device_functions {
            let ret_ty = self.scalar_type_id(device_fn.return_type);

            // Build parameter type IDs
            let mut param_type_ids = Vec::new();
            for (_name, ty) in &device_fn.params {
                param_type_ids.push(self.scalar_type_id(*ty));
            }

            // Create function type: OpTypeFunction ret_ty param_types...
            let func_type = self.ensure_type_function(ret_ty, &param_type_ids);

            // Allocate function ID
            let fn_id = self.alloc_id();
            self.emit_name(fn_id, &device_fn.name);

            // Store mapping for OpFunctionCall
            self.device_fn_ids.insert(
                device_fn.name.clone(),
                (fn_id, ret_ty, param_type_ids.clone()),
            );

            // OpFunction
            Self::emit_op(
                &mut self.sec_device_fns,
                OP_FUNCTION,
                &[ret_ty, fn_id, FUNCTION_CONTROL_NONE, func_type],
            );

            // OpFunctionParameter for each param — save the register mapping
            let old_reg_ids = self.reg_ids.clone();
            let old_reg_types = self.reg_types.clone();
            self.reg_ids.clear();
            self.reg_types.clear();

            for (i, (pname, ty)) in device_fn.params.iter().enumerate() {
                let param_id = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_device_fns,
                    OP_FUNCTION_PARAMETER,
                    &[param_type_ids[i], param_id],
                );
                self.emit_name(param_id, pname);
                // Map the register that the parser assigned to this param
                // The parser allocates registers 0..N for N params
                let type_id = self.scalar_type_id(*ty);
                self.reg_ids.insert(i as u32, param_id);
                self.reg_types.insert(i as u32, type_id);
            }

            // OpLabel for the function body
            let body_label = self.alloc_id();
            Self::emit_op(&mut self.sec_device_fns, OP_LABEL, &[body_label]);

            // Emit the function body into a temporary buffer, then move to sec_device_fns
            let saved_fn = std::mem::take(&mut self.sec_function);
            self.emit_ops(
                &device_fn.body,
                gid_var,
                local_id_var,
                group_id_var,
                num_wg_var,
            )?;

            // Find the last value produced — that's the return value.
            // The device function body should leave the result in the last
            // register written. We look for it by scanning the reg_ids for
            // the highest register number that was set during body emission.
            let return_val = self.find_return_value(&device_fn.body, device_fn.return_type);

            let body_words = std::mem::replace(&mut self.sec_function, saved_fn);
            self.sec_device_fns.extend_from_slice(&body_words);

            // OpReturnValue with the return value
            if let Some(ret_id) = return_val {
                Self::emit_op(
                    &mut self.sec_device_fns,
                    252, // OpReturnValue
                    &[ret_id],
                );
            } else {
                Self::emit_op(&mut self.sec_device_fns, OP_RETURN, &[]);
            }

            Self::emit_op(&mut self.sec_device_fns, OP_FUNCTION_END, &[]);

            // Restore main function's register context
            self.reg_ids = old_reg_ids;
            self.reg_types = old_reg_types;
        }
        Ok(())
    }

    /// Find the SPIR-V ID of the return value for a device function body.
    /// The last expression in the body determines the return value.
    fn find_return_value(&self, ops: &[KernelOp], _ret_ty: ScalarType) -> Option<u32> {
        // Walk backwards to find the last op that writes to a dst register
        for op in ops.iter().rev() {
            let dst_reg = match op {
                KernelOp::BinOp { dst, .. }
                | KernelOp::UnaryOp { dst, .. }
                | KernelOp::Cmp { dst, .. }
                | KernelOp::Cast { dst, .. }
                | KernelOp::Const { dst, .. }
                | KernelOp::Load { dst, .. }
                | KernelOp::SharedLoad { dst, .. }
                | KernelOp::MathCall { dst, .. }
                | KernelOp::Copy { dst, .. }
                | KernelOp::DeviceCall { dst, .. }
                | KernelOp::Bitcast { dst, .. }
                | KernelOp::CountTrailingZeros { dst, .. }
                | KernelOp::CountLeadingZeros { dst, .. }
                | KernelOp::PopCount { dst, .. }
                | KernelOp::Dot { dst, .. }
                | KernelOp::SubgroupReduceAdd { dst, .. }
                | KernelOp::SubgroupReduceMin { dst, .. }
                | KernelOp::SubgroupReduceMax { dst, .. }
                | KernelOp::SubgroupExclusiveAdd { dst, .. }
                | KernelOp::SubgroupInclusiveAdd { dst, .. }
                | KernelOp::TextureLoad2D { dst, .. }
                | KernelOp::SubgroupSize { dst, .. } => Some(dst.0),
                _ => None,
            };
            if let Some(reg_num) = dst_reg
                && let Some(&id) = self.reg_ids.get(&reg_num)
            {
                return Some(id);
            }
        }
        None
    }

    // ── Main entry: emit a full kernel ──────────────────────────────────────

    /// Check if any ops in the kernel body use subgroup operations.
    fn uses_subgroup_ops(ops: &[KernelOp]) -> bool {
        for op in ops {
            match op {
                KernelOp::SubgroupReduceAdd { .. }
                | KernelOp::SubgroupReduceMin { .. }
                | KernelOp::SubgroupReduceMax { .. }
                | KernelOp::SubgroupExclusiveAdd { .. }
                | KernelOp::SubgroupInclusiveAdd { .. } => return true,
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    if Self::uses_subgroup_ops(then_ops) || Self::uses_subgroup_ops(else_ops) {
                        return true;
                    }
                }
                KernelOp::Loop { body, .. } => {
                    if Self::uses_subgroup_ops(body) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn emit_kernel(&mut self, kernel: &KernelDef) -> Result<(), String> {
        // 1. Capability
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_SHADER],
        );

        // Add subgroup capabilities if needed
        if Self::uses_subgroup_ops(&kernel.body) {
            Self::emit_op(
                &mut self.sec_capability,
                OP_CAPABILITY,
                &[CAPABILITY_GROUP_NON_UNIFORM],
            );
            Self::emit_op(
                &mut self.sec_capability,
                OP_CAPABILITY,
                &[CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC],
            );
        }

        // 2. Memory model
        Self::emit_op(
            &mut self.sec_memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        // 3. Set up built-in: GlobalInvocationId
        let v3uint = self.ensure_type_v3uint();
        let ptr_input_v3uint = self.ensure_type_pointer(STORAGE_CLASS_INPUT, v3uint);
        let gid_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, gid_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(gid_var, "gl_GlobalInvocationId");
        self.decorate(gid_var, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

        // LocalInvocationId
        let local_id_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, local_id_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(local_id_var, "gl_LocalInvocationId");
        self.decorate(
            local_id_var,
            DECORATION_BUILTIN,
            &[BUILTIN_LOCAL_INVOCATION_ID],
        );

        // WorkgroupId
        let group_id_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, group_id_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(group_id_var, "gl_WorkGroupID");
        self.decorate(group_id_var, DECORATION_BUILTIN, &[BUILTIN_WORKGROUP_ID]);

        // NumWorkgroups
        let num_wg_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_input_v3uint, num_wg_var, STORAGE_CLASS_INPUT],
        );
        self.emit_name(num_wg_var, "gl_NumWorkGroups");
        self.decorate(num_wg_var, DECORATION_BUILTIN, &[BUILTIN_NUM_WORKGROUPS]);

        // Collect Input/Output interface variables for the entry point.
        // SPIR-V 1.3 requires only Input/Output variables in the interface list.
        // StorageBuffer, Uniform, and Workgroup variables must NOT be listed.
        let interface_ids = vec![gid_var, local_id_var, group_id_var, num_wg_var];

        // 4. Set up storage buffers for each field parameter
        for param in &kernel.params {
            match param {
                KernelParam::FieldRead {
                    name,
                    slot,
                    scalar_type,
                }
                | KernelParam::FieldWrite {
                    name,
                    slot,
                    scalar_type,
                } => {
                    let is_writable = matches!(param, KernelParam::FieldWrite { .. });
                    let elem_ty = self.scalar_type_id(*scalar_type);
                    let stride = Self::scalar_byte_size(*scalar_type);

                    // RuntimeArray of element type
                    let rt_arr = self.ensure_type_runtime_array(elem_ty);
                    if self.decorated_stride.insert(rt_arr) {
                        self.decorate(rt_arr, DECORATION_ARRAY_STRIDE, &[stride]);
                    }

                    // Struct wrapping the runtime array
                    let struct_ty = self.ensure_type_struct(&[rt_arr]);
                    if self.decorated_block.insert(struct_ty) {
                        self.decorate(struct_ty, DECORATION_BLOCK, &[]);
                        self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
                    }

                    // Pointer to struct in StorageBuffer
                    let ptr_struct =
                        self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, struct_ty);

                    // Variable
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_struct, var_id, STORAGE_CLASS_STORAGE_BUFFER],
                    );
                    self.emit_name(var_id, name);

                    // Decorations
                    self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
                    self.decorate(var_id, DECORATION_BINDING, &[*slot]);
                    if !is_writable {
                        self.decorate(var_id, DECORATION_NON_WRITABLE, &[]);
                    }

                    self.field_vars
                        .insert(*slot, (var_id, elem_ty, is_writable));
                }
                KernelParam::Constant {
                    name,
                    slot,
                    scalar_type,
                } => {
                    let elem_ty = self.scalar_type_id(*scalar_type);
                    // Push constants: wrap in a struct with Block decoration,
                    // use PushConstant storage class (matches vkCmdPushConstants).
                    let struct_ty = self.ensure_type_struct(&[elem_ty]);
                    if self.decorated_block.insert(struct_ty) {
                        self.decorate(struct_ty, DECORATION_BLOCK, &[]);
                        self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[*slot * 16]);
                    }

                    let ptr_struct =
                        self.ensure_type_pointer(STORAGE_CLASS_PUSH_CONSTANT, struct_ty);

                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_struct, var_id, STORAGE_CLASS_PUSH_CONSTANT],
                    );
                    self.emit_name(var_id, name);
                    // PushConstant doesn't use DescriptorSet/Binding — it's accessed
                    // via the push constant range in the pipeline layout.

                    // Store as field_vars — Load with index=MAX will access member 0
                    self.field_vars.insert(*slot, (var_id, elem_ty, false));
                    self.push_constant_slots.insert(*slot);
                    self.push_constant_size += 16;
                }
                KernelParam::Texture2DRead { name, slot, .. } => {
                    // Combined image sampler for texture reads
                    let f32_ty = self.ensure_type_f32();
                    let image_ty = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_type_const,
                        OP_TYPE_IMAGE,
                        &[
                            image_ty, f32_ty, 1, /*Dim2D*/
                            0, 0, 0, 1, /*sampled*/
                            0,
                        ],
                    );
                    let sampled_image_ty = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_type_const,
                        OP_TYPE_SAMPLED_IMAGE,
                        &[sampled_image_ty, image_ty],
                    );
                    let ptr_si =
                        self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, sampled_image_ty);
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_si, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
                    );
                    self.emit_name(var_id, name);
                    self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
                    self.decorate(var_id, DECORATION_BINDING, &[*slot]);
                    self.texture_samplers
                        .insert(*slot, (var_id, sampled_image_ty));
                }
                KernelParam::Texture2DWrite { name, slot, .. } => {
                    // Storage image for texture writes
                    let f32_ty = self.ensure_type_f32();
                    let image_ty = self.alloc_id();
                    // Dim2D, non-depth, non-arrayed, non-MS, sampled=2 (storage), Rgba32f
                    Self::emit_op(
                        &mut self.sec_type_const,
                        OP_TYPE_IMAGE,
                        &[
                            image_ty, f32_ty, 1, 0, 0, 0, 2, /*storage*/
                            3, /*Rgba32f*/
                        ],
                    );
                    let ptr_img =
                        self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, image_ty);
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_img, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
                    );
                    self.emit_name(var_id, name);
                    self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
                    self.decorate(var_id, DECORATION_BINDING, &[*slot]);
                    self.texture_samplers.insert(*slot, (var_id, image_ty));
                }
                KernelParam::Texture3DRead { name, slot, .. } => {
                    let f32_ty = self.ensure_type_f32();
                    let image_ty = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_type_const,
                        OP_TYPE_IMAGE,
                        &[image_ty, f32_ty, 2 /*Dim3D*/, 0, 0, 0, 1, 0],
                    );
                    let sampled_image_ty = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_type_const,
                        OP_TYPE_SAMPLED_IMAGE,
                        &[sampled_image_ty, image_ty],
                    );
                    let ptr_si =
                        self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, sampled_image_ty);
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_si, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
                    );
                    self.emit_name(var_id, name);
                    self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
                    self.decorate(var_id, DECORATION_BINDING, &[*slot]);
                    self.texture_samplers
                        .insert(*slot, (var_id, sampled_image_ty));
                }
            }
        }

        // 5. Scan body for SharedDecl and emit workgroup variables
        self.emit_shared_decls(&kernel.body)?;

        // 5b. Emit device functions as SPIR-V OpFunction definitions.
        // Done before the main function so function IDs are available.
        self.emit_device_functions(kernel, gid_var, local_id_var, group_id_var, num_wg_var)?;

        // 6. Entry point (Input/Output variables only in SPIR-V 1.3)
        let void_ty = self.ensure_type_void();
        let func_ty = self.ensure_type_function(void_ty, &[]);
        let main_id = self.alloc_id();
        self.emit_name(main_id, "main");

        {
            let name_words = Self::string_words("main");
            let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, main_id];
            ops.extend_from_slice(&name_words);
            ops.extend_from_slice(&interface_ids);
            Self::emit_op(&mut self.sec_entry_point, OP_ENTRY_POINT, &ops);
        }

        // 7. Execution mode: LocalSize from kernel workgroup_size
        Self::emit_op(
            &mut self.sec_execution_mode,
            OP_EXECUTION_MODE,
            &[
                main_id,
                EXECUTION_MODE_LOCAL_SIZE,
                kernel.workgroup_size[0],
                kernel.workgroup_size[1],
                kernel.workgroup_size[2],
            ],
        );

        // 8. Function body
        // OpFunction
        Self::emit_op(
            &mut self.sec_function,
            OP_FUNCTION,
            &[void_ty, main_id, FUNCTION_CONTROL_NONE, func_ty],
        );

        let entry_label = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[entry_label]);

        // Emit the body ops
        self.emit_ops(
            &kernel.body,
            gid_var,
            local_id_var,
            group_id_var,
            num_wg_var,
        )?;

        // OpReturn + OpFunctionEnd
        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }

    /// Collect all register numbers written (as dst) in a sequence of ops.
    /// Used to detect loop-carried variables.
    fn collect_dsts(ops: &[KernelOp]) -> Vec<u32> {
        let mut dsts = Vec::new();
        for op in ops {
            match op {
                KernelOp::QuarkId { dst }
                | KernelOp::LocalId { dst }
                | KernelOp::GroupId { dst }
                | KernelOp::QuarkCount { dst }
                | KernelOp::GroupSize { dst }
                | KernelOp::Const { dst, .. }
                | KernelOp::Load { dst, .. }
                | KernelOp::BinOp { dst, .. }
                | KernelOp::UnaryOp { dst, .. }
                | KernelOp::Cmp { dst, .. }
                | KernelOp::Cast { dst, .. }
                | KernelOp::Copy { dst, .. }
                | KernelOp::SharedLoad { dst, .. }
                | KernelOp::MathCall { dst, .. }
                | KernelOp::AtomicOp { dst, .. }
                | KernelOp::AtomicCas { dst, .. }
                | KernelOp::WaveShuffle { dst, .. }
                | KernelOp::WaveBallot { dst, .. }
                | KernelOp::WaveAny { dst, .. }
                | KernelOp::WaveAll { dst, .. }
                | KernelOp::VecConstruct { dst, .. }
                | KernelOp::VecExtract { dst, .. }
                | KernelOp::MatMul { dst, .. }
                | KernelOp::DeviceCall { dst, .. }
                | KernelOp::TextureSample2D { dst, .. }
                | KernelOp::TextureSample3D { dst, .. }
                | KernelOp::Bitcast { dst, .. }
                | KernelOp::CountTrailingZeros { dst, .. }
                | KernelOp::CountLeadingZeros { dst, .. }
                | KernelOp::PopCount { dst, .. }
                | KernelOp::Dot { dst, .. }
                | KernelOp::SubgroupReduceAdd { dst, .. }
                | KernelOp::SubgroupReduceMin { dst, .. }
                | KernelOp::SubgroupReduceMax { dst, .. }
                | KernelOp::SubgroupExclusiveAdd { dst, .. }
                | KernelOp::SubgroupInclusiveAdd { dst, .. }
                | KernelOp::TextureLoad2D { dst, .. }
                | KernelOp::SubgroupSize { dst, .. } => {
                    dsts.push(dst.0);
                }
                KernelOp::TextureSize { dst_w, dst_h, .. } => {
                    dsts.push(dst_w.0);
                    dsts.push(dst_h.0);
                }
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    dsts.extend(Self::collect_dsts(then_ops));
                    dsts.extend(Self::collect_dsts(else_ops));
                }
                KernelOp::Loop { body, iter_reg, .. } => {
                    dsts.push(iter_reg.0);
                    dsts.extend(Self::collect_dsts(body));
                }
                _ => {}
            }
        }
        dsts.sort_unstable();
        dsts.dedup();
        dsts
    }

    /// Scan for SharedDecl ops and create workgroup variables.
    fn emit_shared_decls(&mut self, ops: &[KernelOp]) -> Result<(), String> {
        for op in ops {
            match op {
                KernelOp::SharedDecl { id, ty, count } => {
                    let elem_ty = self.scalar_type_id(*ty);
                    let count_const = self.emit_constant_u32(*count);
                    let arr_ty = self.ensure_type_array(elem_ty, count_const);
                    let stride = Self::scalar_byte_size(*ty);
                    if self.decorated_stride.insert(arr_ty) {
                        self.decorate(arr_ty, DECORATION_ARRAY_STRIDE, &[stride]);
                    }
                    let ptr_arr = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, arr_ty);
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_arr, var_id, STORAGE_CLASS_WORKGROUP],
                    );
                    self.emit_name(var_id, &format!("shared_{}", id));
                    self.shared_vars.insert(*id, (var_id, elem_ty));
                }
                KernelOp::SharedDeclDyn { id, ty } => {
                    // Dynamic shared memory: use a specialization constant for the
                    // array size. Default to 256 elements; overridden at dispatch.
                    let elem_ty = self.scalar_type_id(*ty);
                    let default_count = self.emit_constant_u32(256);
                    let arr_ty = self.ensure_type_array(elem_ty, default_count);
                    let stride = Self::scalar_byte_size(*ty);
                    if self.decorated_stride.insert(arr_ty) {
                        self.decorate(arr_ty, DECORATION_ARRAY_STRIDE, &[stride]);
                    }
                    let ptr_arr = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, arr_ty);
                    let var_id = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_global_var,
                        OP_VARIABLE,
                        &[ptr_arr, var_id, STORAGE_CLASS_WORKGROUP],
                    );
                    self.emit_name(var_id, &format!("shared_dyn_{}", id));
                    self.shared_vars.insert(*id, (var_id, elem_ty));
                }
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    self.emit_shared_decls(then_ops)?;
                    self.emit_shared_decls(else_ops)?;
                }
                KernelOp::Loop { body, .. } => {
                    self.emit_shared_decls(body)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Emit a sequence of KernelOps into the function body.
    fn emit_ops(
        &mut self,
        ops: &[KernelOp],
        gid_var: u32,
        local_id_var: u32,
        group_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        for op in ops {
            self.emit_single_op(op, gid_var, local_id_var, group_id_var, num_wg_var)?;
        }
        Ok(())
    }

    /// Load a built-in vec3<u32> and extract .x component.
    fn load_builtin_x(&mut self, var_id: u32) -> u32 {
        let v3uint = self.ensure_type_v3uint();
        let uint_ty = self.ensure_type_u32();

        let loaded = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[v3uint, loaded, var_id]);

        let x_val = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COMPOSITE_EXTRACT,
            &[uint_ty, x_val, loaded, 0],
        );

        x_val
    }

    fn emit_single_op(
        &mut self,
        op: &KernelOp,
        gid_var: u32,
        local_id_var: u32,
        group_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        match op {
            KernelOp::QuarkId { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.load_builtin_x(gid_var);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::LocalId { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.load_builtin_x(local_id_var);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::GroupId { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.load_builtin_x(group_id_var);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::QuarkCount { dst } => {
                // num_workgroups.x * workgroup_size (64)
                let uint_ty = self.ensure_type_u32();
                let nwg = self.load_builtin_x(num_wg_var);
                let sixty_four = self.emit_constant_u32(64);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_IMUL,
                    &[uint_ty, result, nwg, sixty_four],
                );
                self.set_reg(*dst, result, uint_ty);
            }

            KernelOp::GroupSize { dst } => {
                let uint_ty = self.ensure_type_u32();
                let val = self.emit_constant_u32(64);
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::Const { dst, value } => {
                let (id, ty) = match value {
                    ConstValue::F32(v) => {
                        let ty = self.ensure_type_f32();
                        (self.emit_constant_f32(*v), ty)
                    }
                    ConstValue::F64(v) => {
                        let ty = self.ensure_type_f64();
                        (self.emit_constant_f64(*v), ty)
                    }
                    ConstValue::U32(v) => {
                        let ty = self.ensure_type_u32();
                        (self.emit_constant_u32(*v), ty)
                    }
                    ConstValue::U64(v) => {
                        let ty = self.ensure_type_u32();
                        (self.emit_constant_u32(*v as u32), ty)
                    }
                    ConstValue::I32(v) => {
                        let ty = self.ensure_type_i32();
                        (self.emit_constant_i32(*v), ty)
                    }
                    ConstValue::I64(v) => {
                        let ty = self.ensure_type_i32();
                        (self.emit_constant_i32(*v as i32), ty)
                    }
                    ConstValue::Bool(v) => {
                        let ty = self.ensure_type_bool();
                        (self.emit_constant_bool(*v), ty)
                    }
                    ConstValue::F16(v) => {
                        // Convert F16 to F32
                        let ty = self.ensure_type_f32();
                        let f = f32::from_bits((*v as u32) << 16);
                        (self.emit_constant_f32(f), ty)
                    }
                };
                self.set_reg(*dst, id, ty);
            }

            KernelOp::Load {
                dst,
                field,
                index,
                ty,
            } => {
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;

                let result_ty = self.scalar_type_id(*ty);

                if index.0 == u32::MAX {
                    // Push constant: access member 0 of the struct
                    let zero = self.emit_constant_u32(0);
                    let sc = if self.is_push_constant_field(*field) {
                        STORAGE_CLASS_PUSH_CONSTANT
                    } else {
                        STORAGE_CLASS_STORAGE_BUFFER
                    };
                    let ptr_elem = self.ensure_type_pointer(sc, elem_ty);
                    let chain = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_ACCESS_CHAIN,
                        &[ptr_elem, chain, var_id, zero],
                    );
                    let loaded = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LOAD, &[result_ty, loaded, chain]);
                    self.set_reg(*dst, loaded, result_ty);
                } else {
                    // Array access: struct member 0, then index into runtime array
                    let idx = self.reg_value_id(*index)?;
                    let zero = self.emit_constant_u32(0);
                    let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                    let chain = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_ACCESS_CHAIN,
                        &[ptr_elem, chain, var_id, zero, idx],
                    );
                    let loaded = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LOAD, &[result_ty, loaded, chain]);
                    self.set_reg(*dst, loaded, result_ty);
                }
            }

            KernelOp::Store {
                field,
                index,
                src,
                ty: _,
            } => {
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;

                let idx = self.reg_value_id(*index)?;
                let val = self.reg_value_id(*src)?;
                let zero = self.emit_constant_u32(0);
                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, zero, idx],
                );
                Self::emit_op(&mut self.sec_function, OP_STORE, &[chain, val]);
            }

            KernelOp::BinOp { dst, a, b, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );

                let opcode = match (op, is_float, is_signed) {
                    (BinOp::Add, true, _) => OP_FADD,
                    (BinOp::Add, false, _) => OP_IADD,
                    (BinOp::Sub, true, _) => OP_FSUB,
                    (BinOp::Sub, false, _) => OP_ISUB,
                    (BinOp::Mul, true, _) => OP_FMUL,
                    (BinOp::Mul, false, _) => OP_IMUL,
                    (BinOp::Div, true, _) => OP_FDIV,
                    (BinOp::Div, false, true) => OP_SDIV,
                    (BinOp::Div, false, false) => OP_UDIV,
                    (BinOp::Rem, true, _) => OP_FREM,
                    (BinOp::Rem, false, true) => OP_SMOD,
                    (BinOp::Rem, false, false) => OP_UMOD,
                    (BinOp::BitAnd, _, _) => OP_BITWISE_AND,
                    (BinOp::BitOr, _, _) => OP_BITWISE_OR,
                    (BinOp::BitXor, _, _) => OP_BITWISE_XOR,
                    (BinOp::Shl, _, _) => OP_SHIFT_LEFT_LOGICAL,
                    (BinOp::Shr, _, true) => OP_SHIFT_RIGHT_ARITHMETIC,
                    (BinOp::Shr, _, false) => OP_SHIFT_RIGHT_LOGICAL,
                    // SatAdd/SatSub handled below
                    (BinOp::SatAdd, _, _) | (BinOp::SatSub, _, _) => 0,
                };

                if matches!(op, BinOp::SatAdd | BinOp::SatSub) {
                    if is_float {
                        let base_op = if matches!(op, BinOp::SatAdd) {
                            OP_FADD
                        } else {
                            OP_FSUB
                        };
                        let raw = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            base_op,
                            &[result_ty, raw, a_val, b_val],
                        );
                        self.set_reg(*dst, raw, result_ty);
                    } else if matches!(op, BinOp::SatAdd) {
                        // Unsigned sat add: sum = a + b; overflow = sum < a; result = overflow ? MAX : sum
                        let sum = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_IADD,
                            &[result_ty, sum, a_val, b_val],
                        );
                        let bool_ty = self.ensure_type_bool();
                        let overflow = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_ULESS_THAN,
                            &[bool_ty, overflow, sum, a_val],
                        );
                        let max_val = self.emit_constant_u32(u32::MAX);
                        let result = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_SELECT,
                            &[result_ty, result, overflow, max_val, sum],
                        );
                        self.set_reg(*dst, result, result_ty);
                    } else {
                        // Unsigned sat sub: underflow = a < b; diff = a - b; result = underflow ? 0 : diff
                        let bool_ty = self.ensure_type_bool();
                        let underflow = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_ULESS_THAN,
                            &[bool_ty, underflow, a_val, b_val],
                        );
                        let diff = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_ISUB,
                            &[result_ty, diff, a_val, b_val],
                        );
                        let zero = self.emit_constant_u32(0);
                        let result = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_SELECT,
                            &[result_ty, result, underflow, zero, diff],
                        );
                        self.set_reg(*dst, result, result_ty);
                    }
                } else {
                    let result = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        opcode,
                        &[result_ty, result, a_val, b_val],
                    );
                    self.set_reg(*dst, result, result_ty);
                }
            }

            KernelOp::UnaryOp { dst, a, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);

                let result = self.alloc_id();
                match op {
                    UnaryOp::Neg if is_float => {
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_F_NEGATE,
                            &[result_ty, result, a_val],
                        );
                    }
                    UnaryOp::Neg => {
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_S_NEGATE,
                            &[result_ty, result, a_val],
                        );
                    }
                    UnaryOp::BitNot => {
                        Self::emit_op(&mut self.sec_function, OP_NOT, &[result_ty, result, a_val]);
                    }
                    UnaryOp::LogicalNot => {
                        let bool_ty = self.ensure_type_bool();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_LOGICAL_NOT,
                            &[bool_ty, result, a_val],
                        );
                        self.set_reg(*dst, result, bool_ty);
                        return Ok(());
                    }
                }
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Cmp { dst, a, b, op, ty } => {
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let bool_ty = self.ensure_type_bool();
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );

                let opcode = match (op, is_float, is_signed) {
                    (CmpOp::Eq, true, _) => OP_FORD_EQUAL,
                    (CmpOp::Eq, false, _) => OP_IEQUAL,
                    (CmpOp::Ne, true, _) => OP_FORD_NOT_EQUAL,
                    (CmpOp::Ne, false, _) => OP_INOT_EQUAL,
                    (CmpOp::Lt, true, _) => OP_FORD_LESS_THAN,
                    (CmpOp::Lt, false, true) => OP_SLESS_THAN,
                    (CmpOp::Lt, false, false) => OP_ULESS_THAN,
                    (CmpOp::Le, true, _) => OP_FORD_LESS_THAN_EQUAL,
                    (CmpOp::Le, false, true) => OP_SLESS_THAN_EQUAL,
                    (CmpOp::Le, false, false) => OP_ULESS_THAN_EQ,
                    (CmpOp::Gt, true, _) => OP_FORD_GREATER_THAN,
                    (CmpOp::Gt, false, true) => OP_SGREATER_THAN,
                    (CmpOp::Gt, false, false) => OP_UGREATER_THAN,
                    (CmpOp::Ge, true, _) => OP_FORD_GREATER_THAN_EQUAL,
                    (CmpOp::Ge, false, true) => OP_SGREATER_THAN_EQUAL,
                    (CmpOp::Ge, false, false) => OP_UGREATER_THAN_EQUAL,
                };

                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[bool_ty, result, a_val, b_val],
                );
                self.set_reg(*dst, result, bool_ty);
            }

            KernelOp::Cast { dst, src, from, to } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*to);
                let from_float =
                    matches!(from, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let to_float = matches!(to, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let from_signed = matches!(
                    from,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );
                let to_signed = matches!(
                    to,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );

                let result = self.alloc_id();
                let opcode = match (from_float, to_float, from_signed, to_signed) {
                    (false, true, true, _) => OP_CONVERT_S_TO_F,
                    (false, true, false, _) => OP_CONVERT_U_TO_F,
                    (true, false, _, true) => OP_CONVERT_F_TO_S,
                    (true, false, _, false) => OP_CONVERT_F_TO_U,
                    _ => OP_BITCAST, // int↔int, float↔float of same size
                };
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Copy { dst, src, ty } => {
                // In SSA, Copy is just an alias
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                self.set_reg(*dst, src_val, result_ty);
            }

            KernelOp::Branch {
                cond,
                then_ops,
                else_ops,
            } => {
                let cond_val = self.reg_value_id(*cond)?;
                let then_label = self.alloc_id();
                let else_label = self.alloc_id();
                let merge_label = self.alloc_id();

                // OpSelectionMerge
                Self::emit_op(
                    &mut self.sec_function,
                    OP_SELECTION_MERGE,
                    &[merge_label, SELECTION_CONTROL_NONE],
                );

                if else_ops.is_empty() {
                    // No else branch — branch to then or merge
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_BRANCH_CONDITIONAL,
                        &[cond_val, then_label, merge_label],
                    );
                } else {
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_BRANCH_CONDITIONAL,
                        &[cond_val, then_label, else_label],
                    );
                }

                // Then block
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[then_label]);
                self.emit_ops(then_ops, gid_var, local_id_var, group_id_var, num_wg_var)?;
                Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

                // Else block
                if !else_ops.is_empty() {
                    Self::emit_op(&mut self.sec_function, OP_LABEL, &[else_label]);
                    self.emit_ops(else_ops, gid_var, local_id_var, group_id_var, num_wg_var)?;
                    Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);
                }

                // Merge block
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);
            }

            KernelOp::Loop {
                count,
                iter_reg,
                body,
            } => {
                // Structured loop in SPIR-V with loop-carried variable support.
                //
                //   pre_header:    OpBranch → header
                //   header:        OpPhi (counter + carried vars), OpLoopMerge, OpBranch → cond
                //   cond:          test, OpBranchConditional → body / merge
                //   body:          ... ops ..., OpBranch → continue
                //   continue:      increment counter, OpBranch → header
                //   merge:         (after loop)
                //
                // Loop-carried variables: registers defined before the loop that
                // are also written (dst) inside the loop body. These need OpPhi
                // at the header to merge the pre-loop value with the updated
                // value from the continue block.

                let count_val = self.reg_value_id(*count)?;
                let uint_ty = self.ensure_type_u32();

                // Detect loop-carried registers: defined before the loop AND
                // written inside the body (as dst).
                let written_in_body = Self::collect_dsts(body);
                let mut carried: Vec<(u32, u32, u32)> = Vec::new(); // (reg_num, pre_loop_id, type_id)
                for &reg_num in &written_in_body {
                    if let Some(&pre_id) = self.reg_ids.get(&reg_num)
                        && let Some(&ty_id) = self.reg_types.get(&reg_num)
                    {
                        carried.push((reg_num, pre_id, ty_id));
                    }
                }

                let pre_header_label = self.alloc_id();
                let header_label = self.alloc_id();
                let cond_label = self.alloc_id();
                let body_label = self.alloc_id();
                let continue_label = self.alloc_id();
                let merge_label = self.alloc_id();

                let zero = self.emit_constant_u32(0);
                let one = self.emit_constant_u32(1);

                // Pre-header: branch to header
                Self::emit_op(&mut self.sec_function, OP_BRANCH, &[pre_header_label]);
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[pre_header_label]);
                Self::emit_op(&mut self.sec_function, OP_BRANCH, &[header_label]);

                // Header block
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[header_label]);

                // OpPhi for the loop counter
                let phi_id = self.alloc_id();
                let inc_id = self.alloc_id(); // forward-reference
                Self::emit_op(
                    &mut self.sec_function,
                    OP_PHI,
                    &[
                        uint_ty,
                        phi_id,
                        zero,
                        pre_header_label,
                        inc_id,
                        continue_label,
                    ],
                );
                self.set_reg(*iter_reg, phi_id, uint_ty);

                // OpPhi for each loop-carried variable.
                // Allocate forward-reference IDs for the body's updated values.
                // We use a "copy in continue block" pattern: after the body runs,
                // we copy the current reg value to a fresh ID in the continue block.
                let mut carried_phis: Vec<(u32, u32, u32, u32)> = Vec::new();
                // (reg_num, header_phi_id, continue_copy_id, type_id)
                for &(reg_num, pre_id, ty_id) in &carried {
                    let header_phi = self.alloc_id();
                    let continue_copy = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_PHI,
                        &[
                            ty_id,
                            header_phi,
                            pre_id,
                            pre_header_label,
                            continue_copy,
                            continue_label,
                        ],
                    );
                    // Update register to point to the phi
                    self.set_reg(Reg(reg_num), header_phi, ty_id);
                    carried_phis.push((reg_num, header_phi, continue_copy, ty_id));
                }

                // OpLoopMerge (must be penultimate)
                Self::emit_op(
                    &mut self.sec_function,
                    OP_LOOP_MERGE,
                    &[merge_label, continue_label, LOOP_CONTROL_NONE],
                );

                // OpBranch to condition block (must be last)
                Self::emit_op(&mut self.sec_function, OP_BRANCH, &[cond_label]);

                // Condition block
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[cond_label]);
                let bool_ty = self.ensure_type_bool();
                let cond = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ULESS_THAN,
                    &[bool_ty, cond, phi_id, count_val],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_BRANCH_CONDITIONAL,
                    &[cond, body_label, merge_label],
                );

                // Body block
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[body_label]);
                self.loop_merge_stack.push(merge_label);
                self.emit_ops(body, gid_var, local_id_var, group_id_var, num_wg_var)?;
                self.loop_merge_stack.pop();
                Self::emit_op(&mut self.sec_function, OP_BRANCH, &[continue_label]);

                // Continue block: copy carried values, increment counter
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[continue_label]);
                for &(reg_num, _header_phi, continue_copy, ty_id) in &carried_phis {
                    // The body may have updated reg_num. Read its current value
                    // and emit an OpCopyObject so the continue_copy ID is defined.
                    let current = self.reg_ids[&reg_num];
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COPY_OBJECT,
                        &[ty_id, continue_copy, current],
                    );
                }
                Self::emit_op(
                    &mut self.sec_function,
                    OP_IADD,
                    &[uint_ty, inc_id, phi_id, one],
                );
                Self::emit_op(&mut self.sec_function, OP_BRANCH, &[header_label]);

                // Merge block: carried variables now hold the final value
                // from the last iteration (via the header phi).
                Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);

                // After the loop, registers should point to the header phi values
                // (which are the values from the last iteration when the loop exits).
                for &(reg_num, header_phi, _, ty_id) in &carried_phis {
                    self.set_reg(Reg(reg_num), header_phi, ty_id);
                }
            }

            KernelOp::Barrier => {
                // OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory
                let scope_wg = self.emit_constant_u32(SCOPE_WORKGROUP);
                let semantics =
                    self.emit_constant_u32(MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP);
                Self::emit_op(
                    &mut self.sec_function,
                    OP_CONTROL_BARRIER,
                    &[scope_wg, scope_wg, semantics],
                );
            }

            KernelOp::SharedDecl { .. } => {
                // Already handled in emit_shared_decls
            }

            KernelOp::SharedLoad { dst, id, index, ty } => {
                let (var_id, elem_ty) = *self
                    .shared_vars
                    .get(id)
                    .ok_or_else(|| format!("shared memory {} not declared", id))?;
                let idx = self.reg_value_id(*index)?;
                let result_ty = self.scalar_type_id(*ty);

                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, idx],
                );
                let loaded = self.alloc_id();
                Self::emit_op(&mut self.sec_function, OP_LOAD, &[result_ty, loaded, chain]);
                self.set_reg(*dst, loaded, result_ty);
            }

            KernelOp::SharedStore { id, index, src, .. } => {
                let (var_id, elem_ty) = *self
                    .shared_vars
                    .get(id)
                    .ok_or_else(|| format!("shared memory {} not declared", id))?;
                let idx = self.reg_value_id(*index)?;
                let val = self.reg_value_id(*src)?;

                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, idx],
                );
                Self::emit_op(&mut self.sec_function, OP_STORE, &[chain, val]);
            }

            KernelOp::MathCall {
                dst,
                func,
                args,
                ty,
            } => {
                let ext_id = self.ensure_glsl_ext();
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );

                let glsl_op = match func {
                    MathFn::Sin => GLSL_SIN,
                    MathFn::Cos => GLSL_COS,
                    MathFn::Tan => GLSL_TAN,
                    MathFn::Asin => GLSL_ASIN,
                    MathFn::Acos => GLSL_ACOS,
                    MathFn::Atan => GLSL_ATAN,
                    MathFn::Atan2 => GLSL_ATAN2,
                    MathFn::Sqrt => GLSL_SQRT,
                    MathFn::Rsqrt => GLSL_INVERSE_SQRT,
                    MathFn::Exp => GLSL_EXP,
                    MathFn::Exp2 => GLSL_EXP2,
                    MathFn::Log => GLSL_LOG,
                    MathFn::Log2 => GLSL_LOG2,
                    MathFn::Pow => GLSL_POW,
                    MathFn::Abs if is_float => GLSL_FABS,
                    MathFn::Abs if is_signed => GLSL_SABS,
                    MathFn::Abs => GLSL_FABS,
                    MathFn::Min if is_float => GLSL_FMIN,
                    MathFn::Min if is_signed => GLSL_SMIN,
                    MathFn::Min => GLSL_UMIN,
                    MathFn::Max if is_float => GLSL_FMAX,
                    MathFn::Max if is_signed => GLSL_SMAX,
                    MathFn::Max => GLSL_UMAX,
                    MathFn::Clamp if is_float => GLSL_FCLAMP,
                    MathFn::Clamp if is_signed => GLSL_SCLAMP,
                    MathFn::Clamp => GLSL_UCLAMP,
                    MathFn::Floor => GLSL_FLOOR,
                    MathFn::Ceil => GLSL_CEIL,
                    MathFn::Round => GLSL_ROUND,
                    MathFn::Fma => GLSL_FMA,
                };

                let mut operand_ids = Vec::with_capacity(args.len());
                for arg in args {
                    operand_ids.push(self.reg_value_id(*arg)?);
                }

                let result = self.alloc_id();
                let mut ops = vec![result_ty, result, ext_id, glsl_op];
                ops.extend_from_slice(&operand_ids);
                Self::emit_op(&mut self.sec_function, OP_EXT_INST, &ops);
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Break => {
                // Branch to the current loop's merge block.
                if let Some(&merge_label) = self.loop_merge_stack.last() {
                    Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);
                    // After a break, we need a new label for any following ops
                    // (SPIR-V requires every instruction to be in a block).
                    let dead_label = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LABEL, &[dead_label]);
                } else {
                    return Err("Break outside of loop context".to_string());
                }
            }

            KernelOp::VecConstruct {
                dst,
                components,
                ty,
            } => {
                let elem_ty = self.scalar_type_id(*ty);
                let n = components.len() as u32;
                let vec_ty = self.ensure_type_vector(elem_ty, n);
                let mut ids = Vec::with_capacity(components.len());
                for c in components {
                    ids.push(self.reg_value_id(*c)?);
                }
                let result = self.alloc_id();
                let mut ops = vec![vec_ty, result];
                ops.extend_from_slice(&ids);
                Self::emit_op(&mut self.sec_function, OP_COMPOSITE_CONSTRUCT, &ops);
                self.set_reg(*dst, result, vec_ty);
            }

            KernelOp::VecExtract {
                dst,
                vec,
                component,
                ty,
            } => {
                let vec_val = self.reg_value_id(*vec)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[result_ty, result, vec_val, *component as u32],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::MatMul { dst, a, b, ty, .. } => {
                // For simple cases, matrix multiply is just component-wise
                // or dot product. SPIR-V doesn't have a generic MatMul opcode
                // for scalars. For now, treat as multiply.
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let result_ty = self.scalar_type_id(*ty);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float { OP_FMUL } else { OP_IMUL };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, a_val, b_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::DeviceCall {
                dst,
                func_name,
                args,
                ty,
            } => {
                if let Some((fn_id, ret_ty, _param_tys)) =
                    self.device_fn_ids.get(func_name).cloned()
                {
                    // Emit OpFunctionCall with the function ID and argument values
                    let result = self.alloc_id();
                    let mut operands = vec![ret_ty, result, fn_id];
                    for arg in args {
                        let arg_id = self.reg_value_id(*arg)?;
                        operands.push(arg_id);
                    }
                    Self::emit_op(&mut self.sec_function, OP_FUNCTION_CALL, &operands);
                    self.set_reg(*dst, result, ret_ty);
                } else {
                    // Fallback: emit zero constant if function not found
                    let result_ty = self.scalar_type_id(*ty);
                    let zero = match ty {
                        ScalarType::F32 | ScalarType::F16 => self.emit_constant_f32(0.0),
                        ScalarType::F64 => self.emit_constant_f64(0.0),
                        ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 => {
                            self.emit_constant_i32(0)
                        }
                        _ => self.emit_constant_u32(0),
                    };
                    self.set_reg(*dst, zero, result_ty);
                }
            }

            KernelOp::AtomicOp {
                dst,
                field,
                index,
                val,
                op,
                ty,
            } => {
                // Real SPIR-V atomic instructions (OpAtomicIAdd, etc.)
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;
                let idx = self.reg_value_id(*index)?;
                let val_id = self.reg_value_id(*val)?;
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_u32(0);
                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, zero, idx],
                );

                // Scope: Device (1). Semantics: None (0x0 = relaxed).
                let scope = self.emit_constant_u32(1);
                let semantics = self.emit_constant_u32(0);

                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );
                let atomic_opcode = match op {
                    AtomicOp::Add => OP_ATOMIC_IADD,
                    AtomicOp::Sub => OP_ATOMIC_ISUB,
                    AtomicOp::Min if is_signed => OP_ATOMIC_SMIN,
                    AtomicOp::Min => OP_ATOMIC_UMIN,
                    AtomicOp::Max if is_signed => OP_ATOMIC_SMAX,
                    AtomicOp::Max => OP_ATOMIC_UMAX,
                    AtomicOp::And => OP_ATOMIC_AND,
                    AtomicOp::Or => OP_ATOMIC_OR,
                    AtomicOp::Xor => OP_ATOMIC_XOR,
                    AtomicOp::Exchange => OP_ATOMIC_EXCHANGE,
                    AtomicOp::CompareExchange => OP_ATOMIC_COMPARE_EXCHANGE,
                };

                let result_id = self.alloc_id();
                if matches!(op, AtomicOp::CompareExchange) {
                    // OpAtomicCompareExchange: result_type, result, pointer, scope, equal_sem, unequal_sem, value, comparator
                    Self::emit_op(
                        &mut self.sec_function,
                        atomic_opcode,
                        &[
                            result_ty, result_id, chain, scope, semantics, semantics, val_id,
                            val_id,
                        ],
                    );
                } else {
                    // OpAtomicIAdd etc: result_type, result, pointer, scope, semantics, value
                    Self::emit_op(
                        &mut self.sec_function,
                        atomic_opcode,
                        &[result_ty, result_id, chain, scope, semantics, val_id],
                    );
                }
                self.set_reg(*dst, result_id, result_ty);
            }

            KernelOp::AtomicCas {
                dst,
                field,
                index,
                expected,
                desired,
                ty,
            } => {
                // Non-atomic fallback: load, compare, conditionally store
                let (var_id, elem_ty, _) = *self
                    .field_vars
                    .get(field)
                    .ok_or_else(|| format!("field {} not declared", field))?;
                let idx = self.reg_value_id(*index)?;
                let _exp_val = self.reg_value_id(*expected)?;
                let _des_val = self.reg_value_id(*desired)?;
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_u32(0);
                let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
                let chain = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ACCESS_CHAIN,
                    &[ptr_elem, chain, var_id, zero, idx],
                );
                let old_val = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_LOAD,
                    &[result_ty, old_val, chain],
                );
                self.set_reg(*dst, old_val, result_ty);
            }

            KernelOp::WaveShuffle {
                dst,
                src,
                lane_delta,
                ty,
            } => {
                Self::emit_op(
                    &mut self.sec_capability,
                    OP_CAPABILITY,
                    &[CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE],
                );
                let src_val = self.reg_value_id(*src)?;
                let delta_val = self.reg_value_id(*lane_delta)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_GROUP_NON_UNIFORM_SHUFFLE,
                    &[result_ty, result, scope, src_val, delta_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::WaveBallot { dst, predicate } => {
                Self::emit_op(
                    &mut self.sec_capability,
                    OP_CAPABILITY,
                    &[CAPABILITY_GROUP_NON_UNIFORM_BALLOT],
                );
                let pred_val = self.reg_value_id(*predicate)?;
                let uint_ty = self.ensure_type_u32();
                let vec4_uint = self.ensure_type_vector(uint_ty, 4);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let ballot = self.alloc_id();
                // OpGroupNonUniformBallot returns uvec4
                Self::emit_op(
                    &mut self.sec_function,
                    OP_GROUP_NON_UNIFORM_BALLOT,
                    &[vec4_uint, ballot, scope, pred_val],
                );
                // Extract first component (lanes 0-31)
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[uint_ty, result, ballot, 0],
                );
                self.set_reg(*dst, result, uint_ty);
            }

            KernelOp::WaveAny { dst, predicate } => {
                Self::emit_op(
                    &mut self.sec_capability,
                    OP_CAPABILITY,
                    &[CAPABILITY_GROUP_NON_UNIFORM],
                );
                let pred_val = self.reg_value_id(*predicate)?;
                let bool_ty = self.ensure_type_bool();
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let result_bool = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_GROUP_NON_UNIFORM_ANY,
                    &[bool_ty, result_bool, scope, pred_val],
                );
                // Convert bool to u32 for the register
                let uint_ty = self.ensure_type_u32();
                let one = self.emit_constant_u32(1);
                let zero = self.emit_constant_u32(0);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_SELECT,
                    &[uint_ty, result, result_bool, one, zero],
                );
                self.set_reg(*dst, result, uint_ty);
            }

            KernelOp::WaveAll { dst, predicate } => {
                Self::emit_op(
                    &mut self.sec_capability,
                    OP_CAPABILITY,
                    &[CAPABILITY_GROUP_NON_UNIFORM],
                );
                let pred_val = self.reg_value_id(*predicate)?;
                let bool_ty = self.ensure_type_bool();
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let result_bool = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_GROUP_NON_UNIFORM_ALL,
                    &[bool_ty, result_bool, scope, pred_val],
                );
                let uint_ty = self.ensure_type_u32();
                let one = self.emit_constant_u32(1);
                let zero = self.emit_constant_u32(0);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_SELECT,
                    &[uint_ty, result, result_bool, one, zero],
                );
                self.set_reg(*dst, result, uint_ty);
            }

            KernelOp::TextureSample2D {
                dst,
                texture,
                x,
                y,
                ty,
            } => {
                if let Some(&(var_id, type_id)) = self.texture_samplers.get(texture) {
                    let loaded = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
                    let f32_ty = self.ensure_type_f32();
                    let vec2_ty = self.ensure_type_vector(f32_ty, 2);
                    let x_val = self.reg_value_id(*x)?;
                    let y_val = self.reg_value_id(*y)?;
                    let coord = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_CONSTRUCT,
                        &[vec2_ty, coord, x_val, y_val],
                    );
                    let vec4_ty = self.ensure_type_vector(f32_ty, 4);
                    let sample_result = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_IMAGE_SAMPLE_IMPLICIT_LOD,
                        &[vec4_ty, sample_result, loaded, coord],
                    );
                    // Extract first component (scalar result)
                    let result_ty = self.scalar_type_id(*ty);
                    let result = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_EXTRACT,
                        &[result_ty, result, sample_result, 0],
                    );
                    self.set_reg(*dst, result, result_ty);
                } else {
                    let result_ty = self.scalar_type_id(*ty);
                    let zero = self.emit_constant_f32(0.0);
                    self.set_reg(*dst, zero, result_ty);
                }
            }

            KernelOp::TextureSample3D { dst, ty, .. } => {
                // 3D texture sampling — placeholder until 3D textures are tested
                let result_ty = self.scalar_type_id(*ty);
                let zero = self.emit_constant_f32(0.0);
                self.set_reg(*dst, zero, result_ty);
            }

            KernelOp::TextureWrite2D {
                texture,
                x,
                y,
                value,
                ..
            } => {
                if let Some(&(var_id, type_id)) = self.texture_samplers.get(texture) {
                    let loaded = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
                    let uint_ty = self.ensure_type_u32();
                    let vec2_uint = self.ensure_type_vector(uint_ty, 2);
                    let x_val = self.reg_value_id(*x)?;
                    let y_val = self.reg_value_id(*y)?;
                    let coord = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_CONSTRUCT,
                        &[vec2_uint, coord, x_val, y_val],
                    );
                    let val = self.reg_value_id(*value)?;
                    let f32_ty = self.ensure_type_f32();
                    let vec4_ty = self.ensure_type_vector(f32_ty, 4);
                    // Expand scalar to vec4 for image write
                    let zero = self.emit_constant_f32(0.0);
                    let one = self.emit_constant_f32(1.0);
                    let texel = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_CONSTRUCT,
                        &[vec4_ty, texel, val, zero, zero, one],
                    );
                    // OpImageWrite: void (image, coord, texel)
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_IMAGE_WRITE,
                        &[loaded, coord, texel],
                    );
                }
            }

            KernelOp::TextureSize { dst_w, dst_h, .. } => {
                let uint_ty = self.ensure_type_u32();
                let zero = self.emit_constant_u32(0);
                self.set_reg(*dst_w, zero, uint_ty);
                self.set_reg(*dst_h, zero, uint_ty);
            }

            KernelOp::Bitcast { dst, src, to, .. } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*to);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_BITCAST,
                    &[result_ty, result, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::CountTrailingZeros { dst, src, ty } => {
                // GLSL.std.450 FindILsb (ext inst 73) returns CTZ
                let ext_id = self.ensure_glsl_ext();
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_EXT_INST,
                    &[result_ty, result, ext_id, GLSL_FIND_I_LSB, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::CountLeadingZeros { dst, src, ty } => {
                // GLSL.std.450 FindUMsb (ext inst 75) returns (bits-1 - MSB position).
                // CLZ = 31 - FindUMsb(src) for 32-bit.
                let ext_id = self.ensure_glsl_ext();
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let msb = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_EXT_INST,
                    &[result_ty, msb, ext_id, GLSL_FIND_U_MSB, src_val],
                );
                let thirty_one = self.emit_constant_u32(31);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_ISUB,
                    &[result_ty, result, thirty_one, msb],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::PopCount { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_BIT_COUNT,
                    &[result_ty, result, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::Dot { dst, a, b, ty, .. } => {
                // OpDot requires vector operands. The a and b registers should
                // already hold vectors (from VecConstruct).
                let a_val = self.reg_value_id(*a)?;
                let b_val = self.reg_value_id(*b)?;
                let result_ty = self.scalar_type_id(*ty);
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_DOT,
                    &[result_ty, result, a_val, b_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupReduceAdd { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FADD
                } else {
                    OP_GROUP_NON_UNIFORM_IADD
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, scope, GROUP_OPERATION_REDUCE, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupReduceMin { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FMIN
                } else if is_signed {
                    OP_GROUP_NON_UNIFORM_SMIN
                } else {
                    OP_GROUP_NON_UNIFORM_UMIN
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, scope, GROUP_OPERATION_REDUCE, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupReduceMax { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let is_signed = matches!(
                    ty,
                    ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
                );
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FMAX
                } else if is_signed {
                    OP_GROUP_NON_UNIFORM_SMAX
                } else {
                    OP_GROUP_NON_UNIFORM_UMAX
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[result_ty, result, scope, GROUP_OPERATION_REDUCE, src_val],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupExclusiveAdd { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FADD
                } else {
                    OP_GROUP_NON_UNIFORM_IADD
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[
                        result_ty,
                        result,
                        scope,
                        GROUP_OPERATION_EXCLUSIVE_SCAN,
                        src_val,
                    ],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
                let src_val = self.reg_value_id(*src)?;
                let result_ty = self.scalar_type_id(*ty);
                let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
                let is_float = matches!(ty, ScalarType::F32 | ScalarType::F64 | ScalarType::F16);
                let opcode = if is_float {
                    OP_GROUP_NON_UNIFORM_FADD
                } else {
                    OP_GROUP_NON_UNIFORM_IADD
                };
                let result = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[
                        result_ty,
                        result,
                        scope,
                        GROUP_OPERATION_INCLUSIVE_SCAN,
                        src_val,
                    ],
                );
                self.set_reg(*dst, result, result_ty);
            }

            KernelOp::TextureLoad2D {
                dst,
                texture,
                x,
                y,
                ty,
            } => {
                if let Some(&(var_id, type_id)) = self.texture_samplers.get(texture) {
                    let loaded = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
                    let int_ty = self.ensure_type_i32();
                    let vec2_int = self.ensure_type_vector(int_ty, 2);
                    let x_val = self.reg_value_id(*x)?;
                    let y_val = self.reg_value_id(*y)?;
                    let coord = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_CONSTRUCT,
                        &[vec2_int, coord, x_val, y_val],
                    );
                    let f32_ty = self.ensure_type_f32();
                    let vec4_ty = self.ensure_type_vector(f32_ty, 4);
                    let fetch_result = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_IMAGE_FETCH,
                        &[vec4_ty, fetch_result, loaded, coord],
                    );
                    let result_ty = self.scalar_type_id(*ty);
                    let result = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_EXTRACT,
                        &[result_ty, result, fetch_result, 0],
                    );
                    self.set_reg(*dst, result, result_ty);
                } else {
                    let result_ty = self.scalar_type_id(*ty);
                    let zero = self.emit_constant_f32(0.0);
                    self.set_reg(*dst, zero, result_ty);
                }
            }

            KernelOp::SubgroupSize { dst } => {
                // SubgroupSize built-in: load gl_SubgroupSize
                // For now, emit workgroup size as a constant placeholder.
                // Full implementation requires declaring a SubgroupSize built-in variable.
                let uint_ty = self.ensure_type_u32();
                let val = self.emit_constant_u32(32); // placeholder: common subgroup size
                self.set_reg(*dst, val, uint_ty);
            }

            KernelOp::SharedDeclDyn { .. } => {
                // Dynamic shared memory: handled during shared decl scan phase.
                // The SPIR-V declaration uses a specialization constant for array size.
            }

            KernelOp::DebugPrint { src, ty } => {
                // Debug print: no-op in SPIR-V for now.
                // Real implementation would use NonSemantic.DebugPrintf extension
                // or a debug buffer approach similar to MSL.
                let _ = (src, ty);
            }

            KernelOp::Dispatch { .. } => {
                // Dynamic parallelism not supported in Vulkan compute
            }
        }

        Ok(())
    }

    /// Check if a field slot uses PushConstant storage class.
    fn is_push_constant_field(&self, slot: u32) -> bool {
        self.push_constant_slots.contains(&slot)
    }

    // ── Finalize: concatenate sections and emit header ──────────────────────

    fn finalize(self) -> Vec<u8> {
        let mut words = Vec::new();

        // All sections concatenated (to compute bound = max ID).
        // Device functions are emitted after the main function.
        let all_sections: Vec<&[u32]> = vec![
            &self.sec_capability,
            &self.sec_extension,
            &self.sec_ext_inst_import,
            &self.sec_memory_model,
            &self.sec_entry_point,
            &self.sec_execution_mode,
            &self.sec_debug,
            &self.sec_annotation,
            &self.sec_type_const,
            &self.sec_global_var,
            &self.sec_device_fns,
            &self.sec_function,
        ];

        // Header
        words.push(SPIRV_MAGIC);
        words.push(SPIRV_VERSION_1_3);
        words.push(SPIRV_GENERATOR);
        words.push(self.next_id); // Bound (max ID + 1)
        words.push(SPIRV_SCHEMA);

        // Sections in order
        for section in all_sections {
            words.extend_from_slice(section);
        }

        // Convert to bytes (little-endian)
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    // ── Shader type helpers ────────────────────────────────────────────────

    /// Get the SPIR-V type ID for a ShaderType.
    fn shader_type_id(&mut self, ty: quanta_ir::ShaderType) -> u32 {
        let f32_ty = self.ensure_type_f32();
        match ty {
            quanta_ir::ShaderType::F32 => f32_ty,
            quanta_ir::ShaderType::Vec2 => self.ensure_type_vector(f32_ty, 2),
            quanta_ir::ShaderType::Vec3 => self.ensure_type_vector(f32_ty, 3),
            quanta_ir::ShaderType::Vec4 => self.ensure_type_vector(f32_ty, 4),
            quanta_ir::ShaderType::Mat4 => {
                let vec4_ty = self.ensure_type_vector(f32_ty, 4);
                self.ensure_type_matrix(vec4_ty, 4)
            }
            quanta_ir::ShaderType::Mat3 => {
                let vec3_ty = self.ensure_type_vector(f32_ty, 3);
                self.ensure_type_matrix(vec3_ty, 3)
            }
        }
    }

    /// Ensure an OpTypeMatrix exists: matrix of `count` column vectors of type `col_type`.
    fn ensure_type_matrix(&mut self, col_type: u32, count: u32) -> u32 {
        let key = format!("mat_{}_{}", col_type, count);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_MATRIX,
            &[id, col_type, count],
        );
        self.type_cache.insert(key, id);
        id
    }

    /// Number of f32 components in a ShaderType.
    fn shader_type_components(ty: quanta_ir::ShaderType) -> u32 {
        match ty {
            quanta_ir::ShaderType::F32 => 1,
            quanta_ir::ShaderType::Vec2 => 2,
            quanta_ir::ShaderType::Vec3 => 3,
            quanta_ir::ShaderType::Vec4 => 4,
            quanta_ir::ShaderType::Mat4 => 4,
            quanta_ir::ShaderType::Mat3 => 3,
        }
    }

    // ── Shader body expression parser + SPIR-V emitter ──────────────────
    //
    // Parses tokenized Rust shader body into SPIR-V instructions.
    // Supports: Vec constructors, field access, arithmetic, float literals,
    // let bindings, math functions (GLSL.std.450), matrix-vector multiply,
    // if/else, comparisons, and uniform parameter access via push constants.

    /// Evaluate a shader body_source and emit SPIR-V instructions.
    /// Returns the SPIR-V result ID of the final expression and its type.
    fn eval_shader_body(
        &mut self,
        body_source: &str,
        params: &[(String, u32, u32, quanta_ir::ShaderType)], // (name, var_id, type_id, shader_type)
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let src = body_source.trim();
        let src = if src.starts_with('{') && src.ends_with('}') {
            &src[1..src.len() - 1]
        } else {
            src
        };

        let mut locals: Vec<(String, u32, quanta_ir::ShaderType)> = Vec::new();
        let mut remaining = src.trim();

        // Process let-bindings
        while remaining.starts_with("let ") {
            let semi = remaining.find(';').ok_or("missing ; after let binding")?;
            let binding = &remaining[..semi];
            remaining = remaining[semi + 1..].trim();

            let binding = binding.trim_start_matches("let ").trim();
            let binding = binding.trim_start_matches("mut ").trim();
            let eq_pos = binding.find('=').ok_or("missing = in let binding")?;
            let var_name = binding[..eq_pos].trim().to_string();
            let expr_str = binding[eq_pos + 1..].trim();

            let (val_id, val_ty) = self.eval_expr(expr_str, params, &locals)?;
            locals.push((var_name, val_id, val_ty));
        }

        let remaining = remaining.trim().trim_end_matches(';').trim();
        if remaining.is_empty() {
            return Err("empty shader body".to_string());
        }
        self.eval_expr(remaining, params, &locals)
    }

    fn eval_expr(
        &mut self,
        src: &str,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let tokens = tokenize_shader_expr(src);
        let mut pos = 0;
        self.parse_conditional(&tokens, &mut pos, params, locals)
    }

    // ── Gap 5: if/else ───────────────────────────────────────────────────

    fn parse_conditional(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Ident("if".to_string()) {
            *pos += 1;
            // Parse condition (comparison expression)
            let (cond, _) = self.parse_comparison(tokens, pos, params, locals)?;

            // Skip '{'
            if *pos < tokens.len() && tokens[*pos] == ShaderToken::BraceOpen {
                *pos += 1;
            }
            // Find matching '}' for then-branch by counting braces
            let then_start = *pos;
            let mut depth = 1i32;
            while *pos < tokens.len() && depth > 0 {
                match &tokens[*pos] {
                    ShaderToken::BraceOpen => depth += 1,
                    ShaderToken::BraceClose => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    *pos += 1;
                }
            }
            let then_tokens: Vec<ShaderToken> = tokens[then_start..*pos].to_vec();
            if *pos < tokens.len() {
                *pos += 1; // skip '}'
            }

            // Parse else branch
            let has_else =
                *pos < tokens.len() && tokens[*pos] == ShaderToken::Ident("else".to_string());

            if !has_else {
                return Err("if without else not supported in shader expressions".to_string());
            }
            *pos += 1; // skip 'else'
            if *pos < tokens.len() && tokens[*pos] == ShaderToken::BraceOpen {
                *pos += 1;
            }
            let else_start = *pos;
            depth = 1;
            while *pos < tokens.len() && depth > 0 {
                match &tokens[*pos] {
                    ShaderToken::BraceOpen => depth += 1,
                    ShaderToken::BraceClose => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    *pos += 1;
                }
            }
            let else_tokens: Vec<ShaderToken> = tokens[else_start..*pos].to_vec();
            if *pos < tokens.len() {
                *pos += 1; // skip '}'
            }

            // Emit SPIR-V structured control flow
            let then_label = self.alloc_id();
            let else_label = self.alloc_id();
            let merge_label = self.alloc_id();

            Self::emit_op(
                &mut self.sec_function,
                OP_SELECTION_MERGE,
                &[merge_label, 0], // 0 = None selection control
            );
            Self::emit_op(
                &mut self.sec_function,
                OP_BRANCH_CONDITIONAL,
                &[cond, then_label, else_label],
            );

            // Then block
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[then_label]);
            let mut then_pos = 0;
            let (then_id, then_ty) =
                self.parse_conditional(&then_tokens, &mut then_pos, params, locals)?;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Else block
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[else_label]);
            let mut else_pos = 0;
            let (else_id, _) =
                self.parse_conditional(&else_tokens, &mut else_pos, params, locals)?;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Merge block with OpPhi
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);
            let result = self.alloc_id();
            let ty_id = self.shader_type_id(then_ty);
            Self::emit_op(
                &mut self.sec_function,
                OP_PHI,
                &[ty_id, result, then_id, then_label, else_id, else_label],
            );

            return Ok((result, then_ty));
        }
        self.parse_comparison(tokens, pos, params, locals)
    }

    // ── Comparison operators ─────────────────────────────────────────────

    fn parse_comparison(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (left, ty) = self.parse_additive(tokens, pos, params, locals)?;
        if *pos < tokens.len() {
            let cmp_op = match &tokens[*pos] {
                ShaderToken::Cmp(c) => Some(*c),
                _ => None,
            };
            if let Some(op) = cmp_op {
                *pos += 1;
                let (right, _) = self.parse_additive(tokens, pos, params, locals)?;
                let bool_ty = self.ensure_type_bool();
                let result = self.alloc_id();
                let opcode = match op {
                    ShaderCmpOp::Lt => OP_FORD_LESS_THAN,
                    ShaderCmpOp::Gt => OP_FORD_GREATER_THAN,
                    ShaderCmpOp::Le => OP_FORD_LESS_THAN_EQUAL,
                    ShaderCmpOp::Ge => OP_FORD_GREATER_THAN_EQUAL,
                    ShaderCmpOp::Eq => OP_FORD_EQUAL,
                    ShaderCmpOp::Ne => OP_FORD_NOT_EQUAL,
                };
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[bool_ty, result, left, right],
                );
                return Ok((result, quanta_ir::ShaderType::F32)); // abuse F32 for bool
            }
        }
        Ok((left, ty))
    }

    // ── Additive / multiplicative / unary ────────────────────────────────

    fn parse_additive(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (mut left, ty) = self.parse_multiplicative(tokens, pos, params, locals)?;
        while *pos < tokens.len() {
            match &tokens[*pos] {
                ShaderToken::Op('+') => {
                    *pos += 1;
                    let (right, _) = self.parse_multiplicative(tokens, pos, params, locals)?;
                    let result = self.alloc_id();
                    let ty_id = self.shader_type_id(ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_FADD,
                        &[ty_id, result, left, right],
                    );
                    left = result;
                }
                ShaderToken::Op('-') => {
                    *pos += 1;
                    let (right, _) = self.parse_multiplicative(tokens, pos, params, locals)?;
                    let result = self.alloc_id();
                    let ty_id = self.shader_type_id(ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_FSUB,
                        &[ty_id, result, left, right],
                    );
                    left = result;
                }
                _ => break,
            }
        }
        Ok((left, ty))
    }

    /// Gap 2: Matrix-vector multiplication detection
    fn parse_multiplicative(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (mut left, mut left_ty) = self.parse_unary(tokens, pos, params, locals)?;
        while *pos < tokens.len() {
            match &tokens[*pos] {
                ShaderToken::Op('*') => {
                    *pos += 1;
                    let (right, right_ty) = self.parse_unary(tokens, pos, params, locals)?;
                    let result = self.alloc_id();

                    // Detect matrix × vector
                    let is_left_mat = matches!(
                        left_ty,
                        quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3
                    );
                    let is_right_vec = matches!(
                        right_ty,
                        quanta_ir::ShaderType::Vec4 | quanta_ir::ShaderType::Vec3
                    );

                    if is_left_mat && is_right_vec {
                        // OpMatrixTimesVector: result_type = vector type
                        let result_ty_id = self.shader_type_id(right_ty);
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_MATRIX_TIMES_VECTOR,
                            &[result_ty_id, result, left, right],
                        );
                        left = result;
                        left_ty = right_ty;
                    } else {
                        let ty_id = self.shader_type_id(left_ty);
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_FMUL,
                            &[ty_id, result, left, right],
                        );
                        left = result;
                    }
                }
                ShaderToken::Op('/') => {
                    *pos += 1;
                    let (right, _) = self.parse_unary(tokens, pos, params, locals)?;
                    let result = self.alloc_id();
                    let ty_id = self.shader_type_id(left_ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_FDIV,
                        &[ty_id, result, left, right],
                    );
                    left = result;
                }
                _ => break,
            }
        }
        Ok((left, left_ty))
    }

    fn parse_unary(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Op('-') {
            *pos += 1;
            let (val, ty) = self.parse_unary(tokens, pos, params, locals)?;
            let result = self.alloc_id();
            let ty_id = self.shader_type_id(ty);
            Self::emit_op(&mut self.sec_function, OP_F_NEGATE, &[ty_id, result, val]);
            return Ok((result, ty));
        }
        self.parse_atom(tokens, pos, params, locals)
    }

    // ── Atom: literals, identifiers, constructors, math calls ────────────

    fn parse_atom(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if *pos >= tokens.len() {
            return Err("unexpected end of expression".to_string());
        }

        match &tokens[*pos] {
            ShaderToken::Float(val) => {
                *pos += 1;
                let id = self.emit_constant_f32(*val);
                Ok((id, quanta_ir::ShaderType::F32))
            }
            ShaderToken::Open => {
                *pos += 1;
                let result = self.parse_conditional(tokens, pos, params, locals)?;
                if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                    *pos += 1;
                }
                Ok(result)
            }
            ShaderToken::Ident(name) => {
                let name = name.clone();
                *pos += 1;

                // Vec{2,3,4} :: new ( args )
                if (name == "Vec2" || name == "Vec3" || name == "Vec4")
                    && *pos + 2 <= tokens.len()
                    && tokens.get(*pos) == Some(&ShaderToken::ColonColon)
                    && tokens
                        .get(*pos + 1)
                        .map(|t| matches!(t, ShaderToken::Ident(n) if n == "new"))
                        .unwrap_or(false)
                {
                    *pos += 2; // skip :: new
                    let count = match name.as_str() {
                        "Vec2" => 2u32,
                        "Vec3" => 3,
                        "Vec4" => 4,
                        _ => unreachable!(),
                    };
                    if *pos < tokens.len() && tokens[*pos] == ShaderToken::Open {
                        *pos += 1;
                    }
                    let mut components = Vec::new();
                    for i in 0..count {
                        if i > 0 && *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
                            *pos += 1;
                        }
                        let (c, _) = self.parse_conditional(tokens, pos, params, locals)?;
                        components.push(c);
                    }
                    if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                        *pos += 1;
                    }
                    let f32_ty = self.ensure_type_f32();
                    let vec_ty = self.ensure_type_vector(f32_ty, count);
                    let result = self.alloc_id();
                    let mut ops = vec![vec_ty, result];
                    ops.extend_from_slice(&components);
                    Self::emit_op(&mut self.sec_function, OP_COMPOSITE_CONSTRUCT, &ops);
                    let out_ty = match count {
                        2 => quanta_ir::ShaderType::Vec2,
                        3 => quanta_ir::ShaderType::Vec3,
                        4 => quanta_ir::ShaderType::Vec4,
                        _ => unreachable!(),
                    };
                    return Ok((result, out_ty));
                }

                // Texture sampling: sample(slot, uv) → OpImageSampleImplicitLod
                if name == "sample" && *pos < tokens.len() && tokens[*pos] == ShaderToken::Open {
                    *pos += 1; // skip '('
                    // Parse slot (must be a float literal that we treat as integer)
                    let slot = if let ShaderToken::Float(f) = &tokens[*pos] {
                        let s = *f as u32;
                        *pos += 1;
                        s
                    } else {
                        return Err("sample() first arg must be a literal slot number".to_string());
                    };
                    if *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
                        *pos += 1;
                    }
                    let (uv_id, _) = self.parse_conditional(tokens, pos, params, locals)?;
                    if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                        *pos += 1;
                    }

                    let Some(&(sampler_var, sampled_image_ty)) = self.texture_samplers.get(&slot)
                    else {
                        return Err(format!("texture slot {} not declared", slot));
                    };
                    let loaded_sampler = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_LOAD,
                        &[sampled_image_ty, loaded_sampler, sampler_var],
                    );
                    let f32_ty = self.ensure_type_f32();
                    let vec4_ty = self.ensure_type_vector(f32_ty, 4);
                    let result = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_IMAGE_SAMPLE_IMPLICIT_LOD,
                        &[vec4_ty, result, loaded_sampler, uv_id],
                    );
                    return Ok((result, quanta_ir::ShaderType::Vec4));
                }

                // Gap 3: Math function calls — sin(x), sqrt(x), clamp(x,a,b), etc.
                if *pos < tokens.len() && tokens[*pos] == ShaderToken::Open {
                    if let Some(glsl_op) = glsl_func_id(&name) {
                        *pos += 1; // skip '('
                        let mut args = Vec::new();
                        let mut first_ty = quanta_ir::ShaderType::F32;
                        loop {
                            if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                                break;
                            }
                            if !args.is_empty()
                                && *pos < tokens.len()
                                && tokens[*pos] == ShaderToken::Comma
                            {
                                *pos += 1;
                            }
                            let (a, t) = self.parse_conditional(tokens, pos, params, locals)?;
                            if args.is_empty() {
                                first_ty = t;
                            }
                            args.push(a);
                        }
                        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                            *pos += 1;
                        }

                        // dot() returns f32 regardless of input vector type
                        let result_ty = if name == "dot" || name == "length" || name == "distance" {
                            quanta_ir::ShaderType::F32
                        } else {
                            first_ty
                        };

                        let ext = self.ensure_glsl_ext();
                        let result = self.alloc_id();
                        let ty_id = self.shader_type_id(result_ty);
                        let mut ops = vec![ty_id, result, ext, glsl_op];
                        ops.extend_from_slice(&args);
                        Self::emit_op(&mut self.sec_function, OP_EXT_INST, &ops);
                        return Ok((result, result_ty));
                    }

                    // dot() as SPIR-V OpDot (not GLSL ext)
                    if name == "dot" {
                        *pos += 1;
                        let (a, _) = self.parse_conditional(tokens, pos, params, locals)?;
                        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
                            *pos += 1;
                        }
                        let (b, _) = self.parse_conditional(tokens, pos, params, locals)?;
                        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                            *pos += 1;
                        }
                        let f32_ty = self.ensure_type_f32();
                        let result = self.alloc_id();
                        Self::emit_op(&mut self.sec_function, OP_DOT, &[f32_ty, result, a, b]);
                        return Ok((result, quanta_ir::ShaderType::F32));
                    }
                }

                // param.field (e.g. pos.x)
                if *pos + 1 < tokens.len()
                    && tokens[*pos] == ShaderToken::Dot
                    && let ShaderToken::Ident(field) = &tokens[*pos + 1]
                {
                    let field = field.clone();
                    *pos += 2;

                    let index = match field.as_str() {
                        "x" | "r" => 0u32,
                        "y" | "g" => 1,
                        "z" | "b" => 2,
                        "w" | "a" => 3,
                        _ => return Err(format!("unknown field: {field}")),
                    };

                    if let Some((_, var_id, type_id, _)) =
                        params.iter().find(|(n, _, _, _)| *n == name)
                    {
                        let loaded = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_LOAD,
                            &[*type_id, loaded, *var_id],
                        );
                        let f32_ty = self.ensure_type_f32();
                        let result = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_EXTRACT,
                            &[f32_ty, result, loaded, index],
                        );
                        return Ok((result, quanta_ir::ShaderType::F32));
                    }
                    if let Some((_, val_id, _)) = locals.iter().find(|(n, _, _)| *n == name) {
                        let f32_ty = self.ensure_type_f32();
                        let result = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_EXTRACT,
                            &[f32_ty, result, *val_id, index],
                        );
                        return Ok((result, quanta_ir::ShaderType::F32));
                    }
                    return Err(format!("unknown variable: {name}"));
                }

                // Bare identifier — local, param, or boolean literal
                if name == "true" {
                    let id = self.emit_constant_f32(1.0);
                    return Ok((id, quanta_ir::ShaderType::F32));
                }
                if name == "false" {
                    let id = self.emit_constant_f32(0.0);
                    return Ok((id, quanta_ir::ShaderType::F32));
                }
                if let Some((_, val_id, val_ty)) = locals.iter().find(|(n, _, _)| *n == name) {
                    return Ok((*val_id, *val_ty));
                }
                if let Some((_, var_id, type_id, sty)) =
                    params.iter().find(|(n, _, _, _)| *n == name)
                {
                    let loaded = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_LOAD,
                        &[*type_id, loaded, *var_id],
                    );
                    return Ok((loaded, *sty));
                }
                Err(format!("unknown identifier: {name}"))
            }
            other => Err(format!("unexpected token: {other:?}")),
        }
    }

    // ── Vertex shader ─────────────────────────────────────────────────────

    /// Pass-through fallback: load first input, promote to vec4.
    /// Used when body evaluation fails (unsupported features like uniforms).
    fn passthrough_first_input(
        &mut self,
        attr_params: &[(usize, &quanta_ir::ShaderParam)],
        input_vars: &[(u32, u32)],
        f32_ty: u32,
        vec4_ty: u32,
    ) -> u32 {
        if input_vars.is_empty() {
            let zero = self.emit_constant_f32(0.0);
            let one = self.emit_constant_f32(1.0);
            let r = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_COMPOSITE_CONSTRUCT,
                &[vec4_ty, r, zero, zero, zero, one],
            );
            return r;
        }
        let (var_id, type_id) = input_vars[0];
        let loaded = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[type_id, loaded, var_id]);
        let comps = Self::shader_type_components(attr_params[0].1.ty);
        match comps {
            4 => loaded,
            3 => {
                let x = self.alloc_id();
                let y = self.alloc_id();
                let z = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, x, loaded, 0],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, y, loaded, 1],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, z, loaded, 2],
                );
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, x, y, z, one],
                );
                r
            }
            2 => {
                let x = self.alloc_id();
                let y = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, x, loaded, 0],
                );
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_EXTRACT,
                    &[f32_ty, y, loaded, 1],
                );
                let zero = self.emit_constant_f32(0.0);
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, x, y, zero, one],
                );
                r
            }
            _ => {
                let zero = self.emit_constant_f32(0.0);
                let one = self.emit_constant_f32(1.0);
                let r = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_COMPOSITE_CONSTRUCT,
                    &[vec4_ty, r, loaded, zero, zero, one],
                );
                r
            }
        }
    }

    /// Emit a vertex shader SPIR-V module.
    ///
    /// Evaluates the function body to compute gl_Position. Each value parameter
    /// becomes an Input variable with Location decoration. Uniform parameters
    /// are ignored for V1 (no push constant block yet).
    fn emit_vertex_shader(&mut self, shader: &quanta_ir::ShaderDef) -> Result<(), String> {
        // 1. Capability + memory model
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_SHADER],
        );
        Self::emit_op(
            &mut self.sec_memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        // 2. Declare Input variables for value params
        let attr_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_uniform)
            .collect();

        let mut interface_ids = Vec::new();
        let mut input_vars: Vec<(u32, u32)> = Vec::new(); // (var_id, type_id)

        for (loc, (_, param)) in attr_params.iter().enumerate() {
            let ty_id = self.shader_type_id(param.ty);
            let ptr_ty = self.ensure_type_pointer(STORAGE_CLASS_INPUT, ty_id);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_ty, var_id, STORAGE_CLASS_INPUT],
            );
            self.emit_name(var_id, &param.name);
            self.decorate(var_id, DECORATION_LOCATION, &[loc as u32]);
            interface_ids.push(var_id);
            input_vars.push((var_id, ty_id));
        }

        // 2b. Declare uniform params as push constant struct (Gap 1)
        let uniform_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_uniform)
            .collect();

        // Uniform access: each uniform becomes a member of a push constant struct
        let mut uniform_vars: Vec<(String, u32, u32, quanta_ir::ShaderType)> = Vec::new();
        if !uniform_params.is_empty() {
            let mut member_types = Vec::new();
            let mut member_offsets = Vec::new();
            let mut offset = 0u32;
            for (_, p) in &uniform_params {
                let ty_id = self.shader_type_id(p.ty);
                member_types.push(ty_id);
                member_offsets.push(offset);
                // std430 alignment: mat4=64, vec4=16, vec3=16, vec2=8, f32=4
                let size = match p.ty {
                    quanta_ir::ShaderType::Mat4 => 64u32,
                    quanta_ir::ShaderType::Mat3 => 48,
                    quanta_ir::ShaderType::Vec4 => 16,
                    quanta_ir::ShaderType::Vec3 => 16,
                    quanta_ir::ShaderType::Vec2 => 8,
                    quanta_ir::ShaderType::F32 => 4,
                };
                offset += size;
            }

            // Build struct type
            let struct_ty = self.alloc_id();
            let mut struct_ops = vec![struct_ty];
            struct_ops.extend_from_slice(&member_types);
            Self::emit_op(&mut self.sec_type_const, OP_TYPE_STRUCT, &struct_ops);
            self.decorate(struct_ty, DECORATION_BLOCK, &[]);
            for (i, off) in member_offsets.iter().enumerate() {
                self.member_decorate(struct_ty, i as u32, DECORATION_OFFSET, &[*off]);
                // ColMajor decoration for matrices
                if matches!(
                    uniform_params[i].1.ty,
                    quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3
                ) {
                    self.member_decorate(struct_ty, i as u32, 5 /* ColMajor */, &[]);
                    let stride = match uniform_params[i].1.ty {
                        quanta_ir::ShaderType::Mat4 => 16u32,
                        quanta_ir::ShaderType::Mat3 => 16,
                        _ => 16,
                    };
                    self.member_decorate(struct_ty, i as u32, 7 /* MatrixStride */, &[stride]);
                }
            }

            let ptr_pc = self.ensure_type_pointer(STORAGE_CLASS_PUSH_CONSTANT, struct_ty);
            let pc_var = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_pc, pc_var, STORAGE_CLASS_PUSH_CONSTANT],
            );
            self.emit_name(pc_var, "push_constants");

            // Store uniform info for AccessChain in function body
            for (_, p) in &uniform_params {
                let mty = self.shader_type_id(p.ty);
                uniform_vars.push((p.name.clone(), pc_var, mty, p.ty));
            }
        }

        // 2c. Declare Output variables for varyings
        // Convention: first attr param = position (→ gl_Position, not forwarded).
        // Remaining attr params are forwarded as outputs starting at Location 0,
        // matching the fragment shader's Input locations.
        let mut varying_outputs: Vec<(u32, u32, u32)> = Vec::new(); // (output_var, type_id, input_var)
        for (i, (_, param)) in attr_params.iter().enumerate().skip(1) {
            let varying_loc = (i - 1) as u32;
            let ty_id = self.shader_type_id(param.ty);
            let ptr_ty = self.ensure_type_pointer(STORAGE_CLASS_OUTPUT, ty_id);
            let out_var = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_ty, out_var, STORAGE_CLASS_OUTPUT],
            );
            self.emit_name(out_var, &format!("out_{}", param.name));
            self.decorate(out_var, DECORATION_LOCATION, &[varying_loc]);
            interface_ids.push(out_var);
            varying_outputs.push((out_var, ty_id, input_vars[i].0));
        }

        // 3. Declare Output variable: gl_Position (BuiltIn Position, vec4)
        let f32_ty = self.ensure_type_f32();
        let vec4_ty = self.ensure_type_vector(f32_ty, 4);
        let ptr_output_vec4 = self.ensure_type_pointer(STORAGE_CLASS_OUTPUT, vec4_ty);
        let position_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_output_vec4, position_var, STORAGE_CLASS_OUTPUT],
        );
        self.emit_name(position_var, "gl_Position");
        self.decorate(position_var, DECORATION_BUILTIN, &[BUILTIN_POSITION]);
        interface_ids.push(position_var);

        // 4. Entry point
        let void_ty = self.ensure_type_void();
        let func_ty = self.ensure_type_function(void_ty, &[]);
        let main_id = self.alloc_id();
        self.emit_name(main_id, "main");

        {
            let name_words = Self::string_words("main");
            let mut ops = vec![EXECUTION_MODEL_VERTEX, main_id];
            ops.extend_from_slice(&name_words);
            ops.extend_from_slice(&interface_ids);
            Self::emit_op(&mut self.sec_entry_point, OP_ENTRY_POINT, &ops);
        }

        // 5. Function body
        Self::emit_op(
            &mut self.sec_function,
            OP_FUNCTION,
            &[void_ty, main_id, FUNCTION_CONTROL_NONE, func_ty],
        );
        let entry_label = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[entry_label]);

        // Build param_info: attribute params + uniform params (loaded via AccessChain)
        let mut param_info: Vec<(String, u32, u32, quanta_ir::ShaderType)> = attr_params
            .iter()
            .zip(input_vars.iter())
            .map(|((_, p), (var_id, type_id))| (p.name.clone(), *var_id, *type_id, p.ty))
            .collect();

        // Emit AccessChain + load for each uniform in the function body
        for (member_idx, (name, pc_var, member_ty, sty)) in uniform_vars.iter().enumerate() {
            let ptr_member = self.ensure_type_pointer(STORAGE_CLASS_PUSH_CONSTANT, *member_ty);
            let idx_const = self.emit_constant_u32(member_idx as u32);
            let access = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ACCESS_CHAIN,
                &[ptr_member, access, *pc_var, idx_const],
            );
            // Store access chain pointer as the "var_id" — OpLoad in eval will load from it
            param_info.push((name.clone(), access, *member_ty, *sty));
        }

        // Forward vertex attributes as varying outputs (Gap 4)
        for (out_var, type_id, in_var) in &varying_outputs {
            let loaded = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_LOAD,
                &[*type_id, loaded, *in_var],
            );
            Self::emit_op(&mut self.sec_function, OP_STORE, &[*out_var, loaded]);
        }

        // Save state so we can roll back on failure
        let saved_func = self.sec_function.clone();
        let saved_next = self.next_id;

        let result_id = match self.eval_shader_body(&shader.body_source, &param_info) {
            Ok((id, ty)) => {
                // Promote result to vec4 if needed
                match ty {
                    quanta_ir::ShaderType::Vec4 => id,
                    quanta_ir::ShaderType::Vec3 => {
                        let x = self.alloc_id();
                        let y = self.alloc_id();
                        let z = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_EXTRACT,
                            &[f32_ty, x, id, 0],
                        );
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_EXTRACT,
                            &[f32_ty, y, id, 1],
                        );
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_EXTRACT,
                            &[f32_ty, z, id, 2],
                        );
                        let one = self.emit_constant_f32(1.0);
                        let r = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_CONSTRUCT,
                            &[vec4_ty, r, x, y, z, one],
                        );
                        r
                    }
                    quanta_ir::ShaderType::Vec2 => {
                        let x = self.alloc_id();
                        let y = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_EXTRACT,
                            &[f32_ty, x, id, 0],
                        );
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_EXTRACT,
                            &[f32_ty, y, id, 1],
                        );
                        let zero = self.emit_constant_f32(0.0);
                        let one = self.emit_constant_f32(1.0);
                        let r = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_CONSTRUCT,
                            &[vec4_ty, r, x, y, zero, one],
                        );
                        r
                    }
                    quanta_ir::ShaderType::F32 => {
                        let zero = self.emit_constant_f32(0.0);
                        let one = self.emit_constant_f32(1.0);
                        let r = self.alloc_id();
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_COMPOSITE_CONSTRUCT,
                            &[vec4_ty, r, id, zero, zero, one],
                        );
                        r
                    }
                    _ => {
                        // Unsupported return type — fall back to pass-through
                        self.sec_function = saved_func;
                        self.next_id = saved_next;
                        self.passthrough_first_input(&attr_params, &input_vars, f32_ty, vec4_ty)
                    }
                }
            }
            Err(_) => {
                // Body uses unsupported features — roll back and use pass-through
                self.sec_function = saved_func;
                self.next_id = saved_next;
                self.passthrough_first_input(&attr_params, &input_vars, f32_ty, vec4_ty)
            }
        };

        // Store to gl_Position
        Self::emit_op(&mut self.sec_function, OP_STORE, &[position_var, result_id]);

        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }

    // ── Fragment shader ────────────��───────────────────────────────────────

    /// Emit a fragment shader SPIR-V module.
    ///
    /// Generates a passthrough fragment shader: each value parameter becomes
    /// an Input variable (interpolated varying) with Location decoration.
    /// The first input is passed through to Location(0) Output.
    fn emit_fragment_shader(&mut self, shader: &quanta_ir::ShaderDef) -> Result<(), String> {
        // 1. Capability + memory model
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_SHADER],
        );
        Self::emit_op(
            &mut self.sec_memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        // 2. Declare Input variables for value params
        let stage_in_params: Vec<(usize, &quanta_ir::ShaderParam)> = shader
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_uniform)
            .collect();

        let mut interface_ids = Vec::new();
        let mut input_vars: Vec<(u32, u32)> = Vec::new(); // (var_id, type_id)

        for (loc, (_, param)) in stage_in_params.iter().enumerate() {
            let ty_id = self.shader_type_id(param.ty);
            let ptr_ty = self.ensure_type_pointer(STORAGE_CLASS_INPUT, ty_id);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_global_var,
                OP_VARIABLE,
                &[ptr_ty, var_id, STORAGE_CLASS_INPUT],
            );
            self.emit_name(var_id, &param.name);
            self.decorate(var_id, DECORATION_LOCATION, &[loc as u32]);
            interface_ids.push(var_id);
            input_vars.push((var_id, ty_id));
        }

        // 2b. Declare combined image samplers for texture sampling
        // Scan body_source for sample() calls to determine which slots are used
        let f32_ty = self.ensure_type_f32();
        let vec4_ty = self.ensure_type_vector(f32_ty, 4);

        let max_tex_slot = (0..8u32)
            .filter(|slot| shader.body_source.contains(&format!("sample({}", slot)))
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        self.texture_samplers.clear();
        if max_tex_slot > 0 {
            // OpTypeImage: float, 2D, non-depth, non-arrayed, non-MS, sampled=1, Unknown format
            let image_ty = self.alloc_id();
            Self::emit_op(
                &mut self.sec_type_const,
                OP_TYPE_IMAGE,
                &[
                    image_ty, f32_ty, 1, /*Dim2D*/
                    0, /*depth*/
                    0, /*arrayed*/
                    0, /*MS*/
                    1, /*sampled*/
                    0, /*Unknown*/
                ],
            );
            let sampled_image_ty = self.alloc_id();
            Self::emit_op(
                &mut self.sec_type_const,
                OP_TYPE_SAMPLED_IMAGE,
                &[sampled_image_ty, image_ty],
            );
            let ptr_uniform_si =
                self.ensure_type_pointer(STORAGE_CLASS_UNIFORM_CONSTANT, sampled_image_ty);

            for slot in 0..max_tex_slot {
                let var_id = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_global_var,
                    OP_VARIABLE,
                    &[ptr_uniform_si, var_id, STORAGE_CLASS_UNIFORM_CONSTANT],
                );
                self.emit_name(var_id, &format!("tex_{}", slot));
                self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
                self.decorate(var_id, DECORATION_BINDING, &[slot + 8]); // bindings 8+ for textures (matching Vulkan layout)
                self.texture_samplers
                    .insert(slot, (var_id, sampled_image_ty));
            }
        }

        // 3. Declare Output variable: fragment color at Location(0)
        let ptr_output_vec4 = self.ensure_type_pointer(STORAGE_CLASS_OUTPUT, vec4_ty);
        let color_var = self.alloc_id();
        Self::emit_op(
            &mut self.sec_global_var,
            OP_VARIABLE,
            &[ptr_output_vec4, color_var, STORAGE_CLASS_OUTPUT],
        );
        self.emit_name(color_var, "out_color");
        self.decorate(color_var, DECORATION_LOCATION, &[0]);
        interface_ids.push(color_var);

        // 4. Entry point
        let void_ty = self.ensure_type_void();
        let func_ty = self.ensure_type_function(void_ty, &[]);
        let main_id = self.alloc_id();
        self.emit_name(main_id, "main");

        {
            let name_words = Self::string_words("main");
            let mut ops = vec![EXECUTION_MODEL_FRAGMENT, main_id];
            ops.extend_from_slice(&name_words);
            ops.extend_from_slice(&interface_ids);
            Self::emit_op(&mut self.sec_entry_point, OP_ENTRY_POINT, &ops);
        }

        // 5. Execution mode: OriginUpperLeft (required for Fragment)
        Self::emit_op(
            &mut self.sec_execution_mode,
            OP_EXECUTION_MODE,
            &[main_id, EXECUTION_MODE_ORIGIN_UPPER_LEFT],
        );

        // 6. Function body
        Self::emit_op(
            &mut self.sec_function,
            OP_FUNCTION,
            &[void_ty, main_id, FUNCTION_CONTROL_NONE, func_ty],
        );
        let entry_label = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[entry_label]);

        // Evaluate function body to compute output color.
        // Falls back to pass-through on failure.
        let param_info: Vec<(String, u32, u32, quanta_ir::ShaderType)> = stage_in_params
            .iter()
            .zip(input_vars.iter())
            .map(|((_, p), (var_id, type_id))| (p.name.clone(), *var_id, *type_id, p.ty))
            .collect();

        let saved_func = self.sec_function.clone();
        let saved_next = self.next_id;

        let result_id = match self.eval_shader_body(&shader.body_source, &param_info) {
            Ok((id, ty)) => match ty {
                quanta_ir::ShaderType::Vec4 => id,
                quanta_ir::ShaderType::Vec3 => {
                    let x = self.alloc_id();
                    let y = self.alloc_id();
                    let z = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_EXTRACT,
                        &[f32_ty, x, id, 0],
                    );
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_EXTRACT,
                        &[f32_ty, y, id, 1],
                    );
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_EXTRACT,
                        &[f32_ty, z, id, 2],
                    );
                    let one = self.emit_constant_f32(1.0);
                    let r = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_CONSTRUCT,
                        &[vec4_ty, r, x, y, z, one],
                    );
                    r
                }
                quanta_ir::ShaderType::Vec2 => {
                    let x = self.alloc_id();
                    let y = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_EXTRACT,
                        &[f32_ty, x, id, 0],
                    );
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_EXTRACT,
                        &[f32_ty, y, id, 1],
                    );
                    let zero = self.emit_constant_f32(0.0);
                    let one = self.emit_constant_f32(1.0);
                    let r = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_CONSTRUCT,
                        &[vec4_ty, r, x, y, zero, one],
                    );
                    r
                }
                quanta_ir::ShaderType::F32 => {
                    let zero = self.emit_constant_f32(0.0);
                    let one = self.emit_constant_f32(1.0);
                    let r = self.alloc_id();
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_COMPOSITE_CONSTRUCT,
                        &[vec4_ty, r, id, zero, zero, one],
                    );
                    r
                }
                _ => {
                    self.sec_function = saved_func;
                    self.next_id = saved_next;
                    self.passthrough_first_input(&stage_in_params, &input_vars, f32_ty, vec4_ty)
                }
            },
            Err(_) => {
                self.sec_function = saved_func;
                self.next_id = saved_next;
                self.passthrough_first_input(&stage_in_params, &input_vars, f32_ty, vec4_ty)
            }
        };

        // Store to output color
        Self::emit_op(&mut self.sec_function, OP_STORE, &[color_var, result_id]);

        Self::emit_op(&mut self.sec_function, OP_RETURN, &[]);
        Self::emit_op(&mut self.sec_function, OP_FUNCTION_END, &[]);

        Ok(())
    }
}

// ── Public API for vertex/fragment shader SPIR-V emission ──────────────────

/// Emit SPIR-V for a vertex shader from a [`ShaderDef`].
pub fn emit_vertex(shader: &quanta_ir::ShaderDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    e.emit_vertex_shader(shader)?;
    Ok(e.finalize())
}

/// Emit SPIR-V for a fragment shader from a [`ShaderDef`].
pub fn emit_fragment(shader: &quanta_ir::ShaderDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    e.emit_fragment_shader(shader)?;
    Ok(e.finalize())
}

// ── Shader expression tokenizer ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum ShaderCmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq)]
enum ShaderToken {
    Float(f32),
    Ident(String),
    Op(char),         // + - * /
    Cmp(ShaderCmpOp), // < > <= >= == !=
    Dot,              // .
    ColonColon,       // ::
    Comma,            // ,
    Open,             // (
    Close,            // )
    BraceOpen,        // {
    BraceClose,       // }
}

fn tokenize_shader_expr(src: &str) -> Vec<ShaderToken> {
    let mut tokens = Vec::new();
    for w in src.split_whitespace() {
        tokenize_word(w, &mut tokens);
    }
    tokens
}

fn tokenize_word(w: &str, tokens: &mut Vec<ShaderToken>) {
    match w {
        "::" => tokens.push(ShaderToken::ColonColon),
        "." => tokens.push(ShaderToken::Dot),
        "," => tokens.push(ShaderToken::Comma),
        "(" => tokens.push(ShaderToken::Open),
        ")" => tokens.push(ShaderToken::Close),
        "{" => tokens.push(ShaderToken::BraceOpen),
        "}" => tokens.push(ShaderToken::BraceClose),
        "+" => tokens.push(ShaderToken::Op('+')),
        "-" => tokens.push(ShaderToken::Op('-')),
        "*" => tokens.push(ShaderToken::Op('*')),
        "/" => tokens.push(ShaderToken::Op('/')),
        "<" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Lt)),
        ">" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Gt)),
        "<=" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Le)),
        ">=" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Ge)),
        "==" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Eq)),
        "!=" => tokens.push(ShaderToken::Cmp(ShaderCmpOp::Ne)),
        ";" => {} // skip semicolons
        _ => {
            // Split on embedded punctuation
            if let Some(split_pos) = w.find(['(', ')', ',', '{', '}']) {
                let (before, rest) = w.split_at(split_pos);
                if !before.is_empty() {
                    tokenize_word(before, tokens);
                }
                tokenize_word(&rest[..1], tokens);
                if rest.len() > 1 {
                    tokenize_word(&rest[1..], tokens);
                }
            } else if w.contains('.') && !w.starts_with(|c: char| c.is_ascii_digit()) {
                if let Some(dot_pos) = w.find('.') {
                    let (before, rest) = w.split_at(dot_pos);
                    if !before.is_empty() {
                        tokenize_word(before, tokens);
                    }
                    tokens.push(ShaderToken::Dot);
                    if rest.len() > 1 {
                        tokenize_word(&rest[1..], tokens);
                    }
                }
            } else if let Ok(f) = w.parse::<f32>() {
                tokens.push(ShaderToken::Float(f));
            } else {
                tokens.push(ShaderToken::Ident(w.to_string()));
            }
        }
    }
}

/// Map function name to GLSL.std.450 extended instruction opcode.
fn glsl_func_id(name: &str) -> Option<u32> {
    match name {
        "sin" => Some(GLSL_SIN),
        "cos" => Some(GLSL_COS),
        "tan" => Some(GLSL_TAN),
        "asin" => Some(GLSL_ASIN),
        "acos" => Some(GLSL_ACOS),
        "atan" => Some(GLSL_ATAN),
        "sqrt" => Some(GLSL_SQRT),
        "inverseSqrt" | "inverse_sqrt" => Some(GLSL_INVERSE_SQRT),
        "abs" => Some(GLSL_FABS),
        "floor" => Some(GLSL_FLOOR),
        "ceil" => Some(GLSL_CEIL),
        "round" => Some(GLSL_ROUND),
        "fract" => Some(GLSL_FRACT),
        "min" => Some(GLSL_FMIN),
        "max" => Some(GLSL_FMAX),
        "clamp" => Some(GLSL_FCLAMP),
        "mix" => Some(GLSL_FMIX),
        "step" => Some(GLSL_STEP),
        "smoothstep" | "smooth_step" => Some(GLSL_SMOOTH_STEP),
        "pow" => Some(GLSL_POW),
        "exp" => Some(GLSL_EXP),
        "log" => Some(GLSL_LOG),
        "exp2" => Some(GLSL_EXP2),
        "log2" => Some(GLSL_LOG2),
        "normalize" => Some(GLSL_NORMALIZE),
        "length" => Some(GLSL_LENGTH),
        "distance" => Some(GLSL_DISTANCE),
        "cross" => Some(GLSL_CROSS),
        "fma" => Some(GLSL_FMA),
        "atan2" => Some(GLSL_ATAN2),
        _ => None,
    }
}
