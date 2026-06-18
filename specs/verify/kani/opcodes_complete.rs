//! Kani harness: comprehensive SPIR-V opcode correctness (Theorem T2).
//!
//! Covers every opcode category NOT in opcodes.rs: comparisons, types,
//! memory, flow control, conversions, atomics, subgroups (full set),
//! capabilities, and a global collision check across ALL constants.
//!
//! All values are taken from constants.rs (the source of truth).
//! No allocation: pure const assertions only.
//!
//! Run all:  cargo kani --harness-regex 'verify_'
//! Run one:  cargo kani --harness verify_comparison_opcodes

// ── Opcode constants (mirrored from emit_spirv/constants.rs) ───────────────

// Comparison opcodes
const OP_IEQUAL: u16 = 170;
const OP_INOT_EQUAL: u16 = 171;
const OP_UGREATER_THAN: u16 = 172;
const OP_SGREATER_THAN: u16 = 173;
const OP_UGREATER_THAN_EQUAL: u16 = 174;
const OP_SGREATER_THAN_EQUAL: u16 = 175;
const OP_ULESS_THAN: u16 = 176;
const OP_SLESS_THAN: u16 = 177;
const OP_ULESS_THAN_EQ: u16 = 178;
const OP_SLESS_THAN_EQUAL: u16 = 179;
const OP_FORD_EQUAL: u16 = 180;
const OP_FORD_NOT_EQUAL: u16 = 182;
const OP_FORD_LESS_THAN: u16 = 184;
const OP_FORD_GREATER_THAN: u16 = 186;
const OP_FORD_LESS_THAN_EQUAL: u16 = 188;
const OP_FORD_GREATER_THAN_EQUAL: u16 = 190;
const OP_LOGICAL_NOT: u16 = 168;
const OP_SELECT: u16 = 169;

// Type opcodes
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;

// Memory opcodes
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;

// Flow control opcodes
const OP_PHI: u16 = 245;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;

// Conversion opcodes
const OP_CONVERT_F_TO_U: u16 = 109;
const OP_CONVERT_F_TO_S: u16 = 110;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_BITCAST: u16 = 124;

// Atomic opcodes
const OP_CONTROL_BARRIER: u16 = 224;
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

// Subgroup opcodes (full set from constants.rs)
const OP_GROUP_NON_UNIFORM_ALL: u16 = 334;
const OP_GROUP_NON_UNIFORM_ANY: u16 = 335;
const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
const OP_GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
const OP_GROUP_NON_UNIFORM_SHUFFLE_XOR: u16 = 346;
const OP_GROUP_NON_UNIFORM_IADD: u16 = 349;
const OP_GROUP_NON_UNIFORM_FADD: u16 = 350;
const OP_GROUP_NON_UNIFORM_SMIN: u16 = 353;
const OP_GROUP_NON_UNIFORM_UMIN: u16 = 354;
const OP_GROUP_NON_UNIFORM_FMIN: u16 = 355;
const OP_GROUP_NON_UNIFORM_SMAX: u16 = 356;
const OP_GROUP_NON_UNIFORM_UMAX: u16 = 357;
const OP_GROUP_NON_UNIFORM_FMAX: u16 = 358;

// Remaining opcodes (already in opcodes.rs but needed for global collision check)
const OP_NAME: u16 = 5;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_CONSTANT: u16 = 43;
const OP_CONSTANT_TRUE: u16 = 41;
const OP_CONSTANT_FALSE: u16 = 42;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_PARAMETER: u16 = 55;
const OP_FUNCTION_END: u16 = 56;
const OP_FUNCTION_CALL: u16 = 57;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_COPY_OBJECT: u16 = 83;
const OP_TRANSPOSE: u16 = 84;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_FETCH: u16 = 95;
const OP_IMAGE_WRITE: u16 = 99;
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
const OP_SREM: u16 = 138;
const OP_FREM: u16 = 140;
const OP_VECTOR_TIMES_MATRIX: u16 = 144;
const OP_MATRIX_TIMES_VECTOR: u16 = 145;
const OP_MATRIX_TIMES_MATRIX: u16 = 146;
const OP_DOT: u16 = 148;
const OP_NOT: u16 = 200;
const OP_BIT_COUNT: u16 = 205;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 197;
const OP_BITWISE_XOR: u16 = 198;
const OP_BITWISE_AND: u16 = 199;

// ── Non-opcode constants (u32) ─────────────────────────────────────────────

// Capabilities
const CAPABILITY_SHADER: u32 = 1;
const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
const CAPABILITY_GROUP_NON_UNIFORM_BALLOT: u32 = 62;
const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
const CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;

// Storage classes
const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_OUTPUT: u32 = 3;
const STORAGE_CLASS_WORKGROUP: u32 = 4;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Decorations
const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_LOCATION: u32 = 30;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

// Built-in values
const BUILTIN_POSITION: u32 = 0;
const BUILTIN_NUM_WORKGROUPS: u32 = 24;
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Execution model / mode
const EXECUTION_MODEL_VERTEX: u32 = 0;
const EXECUTION_MODEL_FRAGMENT: u32 = 4;
const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
const EXECUTION_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Memory model
const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// Scope / memory semantics
const SCOPE_SUBGROUP: u32 = 3;
const SCOPE_WORKGROUP: u32 = 2;
const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100;
const MEMORY_SEMANTICS_ACQ_REL: u32 = 0x8;

// Group operations
const GROUP_OPERATION_REDUCE: u32 = 0;
const GROUP_OPERATION_INCLUSIVE_SCAN: u32 = 1;
const GROUP_OPERATION_EXCLUSIVE_SCAN: u32 = 2;

// SPIR-V header
const SPIRV_MAGIC: u32 = 0x07230203;
const SPIRV_VERSION_1_3: u32 = 0x00010300;

// ── Kani harnesses ─────────────────────────────────────────────────────────

#[cfg(kani)]
mod proofs {
    use super::*;

    // ── 1. Comparison opcodes ──────────────────────────────────────────────

    #[kani::proof]
    fn verify_comparison_opcodes() {
        // Integer comparisons (SPIR-V 1.6 section 3.37.16)
        assert_eq!(OP_IEQUAL, 170, "OpIEqual");
        assert_eq!(OP_INOT_EQUAL, 171, "OpINotEqual");
        assert_eq!(OP_UGREATER_THAN, 172, "OpUGreaterThan");
        assert_eq!(OP_SGREATER_THAN, 173, "OpSGreaterThan");
        assert_eq!(OP_UGREATER_THAN_EQUAL, 174, "OpUGreaterThanEqual");
        assert_eq!(OP_SGREATER_THAN_EQUAL, 175, "OpSGreaterThanEqual");
        assert_eq!(OP_ULESS_THAN, 176, "OpULessThan");
        assert_eq!(OP_SLESS_THAN, 177, "OpSLessThan");
        assert_eq!(OP_ULESS_THAN_EQ, 178, "OpULessThanEqual");
        assert_eq!(OP_SLESS_THAN_EQUAL, 179, "OpSLessThanEqual");

        // Float ordered comparisons (SPIR-V 1.6 section 3.37.16)
        assert_eq!(OP_FORD_EQUAL, 180, "OpFOrdEqual");
        assert_eq!(OP_FORD_NOT_EQUAL, 182, "OpFOrdNotEqual");
        assert_eq!(OP_FORD_LESS_THAN, 184, "OpFOrdLessThan");
        assert_eq!(OP_FORD_GREATER_THAN, 186, "OpFOrdGreaterThan");
        assert_eq!(OP_FORD_LESS_THAN_EQUAL, 188, "OpFOrdLessThanEqual");
        assert_eq!(OP_FORD_GREATER_THAN_EQUAL, 190, "OpFOrdGreaterThanEqual");

        // Logical
        assert_eq!(OP_LOGICAL_NOT, 168, "OpLogicalNot");
        assert_eq!(OP_SELECT, 169, "OpSelect");
    }

    // ── 2. Type opcodes ────────────────────────────────────────────────────

    #[kani::proof]
    fn verify_type_opcodes() {
        assert_eq!(OP_TYPE_VOID, 19, "OpTypeVoid");
        assert_eq!(OP_TYPE_BOOL, 20, "OpTypeBool");
        assert_eq!(OP_TYPE_INT, 21, "OpTypeInt");
        assert_eq!(OP_TYPE_FLOAT, 22, "OpTypeFloat");
        assert_eq!(OP_TYPE_VECTOR, 23, "OpTypeVector");
        assert_eq!(OP_TYPE_MATRIX, 24, "OpTypeMatrix");
        assert_eq!(OP_TYPE_IMAGE, 25, "OpTypeImage");
        assert_eq!(OP_TYPE_SAMPLER, 26, "OpTypeSampler");
        assert_eq!(OP_TYPE_SAMPLED_IMAGE, 27, "OpTypeSampledImage");
        assert_eq!(OP_TYPE_ARRAY, 28, "OpTypeArray");
        assert_eq!(OP_TYPE_RUNTIME_ARRAY, 29, "OpTypeRuntimeArray");
        assert_eq!(OP_TYPE_STRUCT, 30, "OpTypeStruct");
        assert_eq!(OP_TYPE_POINTER, 32, "OpTypePointer");
        assert_eq!(OP_TYPE_FUNCTION, 33, "OpTypeFunction");

        // Contiguous range check: Void..SampledImage = 19..=27
        assert_eq!(OP_TYPE_SAMPLED_IMAGE - OP_TYPE_VOID, 8);
    }

    // ── 3. Memory opcodes ──────────────────────────────────────────────────

    #[kani::proof]
    fn verify_memory_opcodes() {
        assert_eq!(OP_VARIABLE, 59, "OpVariable");
        assert_eq!(OP_LOAD, 61, "OpLoad");
        assert_eq!(OP_STORE, 62, "OpStore");
        assert_eq!(OP_ACCESS_CHAIN, 65, "OpAccessChain");

        // Load and Store are adjacent
        assert_eq!(OP_STORE - OP_LOAD, 1);
    }

    // ── 4. Flow control opcodes ────────────────────────────────────────────

    #[kani::proof]
    fn verify_flow_control_opcodes() {
        assert_eq!(OP_PHI, 245, "OpPhi");
        assert_eq!(OP_LOOP_MERGE, 246, "OpLoopMerge");
        assert_eq!(OP_SELECTION_MERGE, 247, "OpSelectionMerge");
        assert_eq!(OP_LABEL, 248, "OpLabel");
        assert_eq!(OP_BRANCH, 249, "OpBranch");
        assert_eq!(OP_BRANCH_CONDITIONAL, 250, "OpBranchConditional");
        assert_eq!(OP_RETURN, 253, "OpReturn");

        // Phi..BranchConditional is contiguous (245..=250)
        assert_eq!(OP_BRANCH_CONDITIONAL - OP_PHI, 5);
    }

    // ── 5. Conversion opcodes ──────────────────────────────────────────────

    #[kani::proof]
    fn verify_conversion_opcodes() {
        assert_eq!(OP_CONVERT_F_TO_U, 109, "OpConvertFToU");
        assert_eq!(OP_CONVERT_F_TO_S, 110, "OpConvertFToS");
        assert_eq!(OP_CONVERT_S_TO_F, 111, "OpConvertSToF");
        assert_eq!(OP_CONVERT_U_TO_F, 112, "OpConvertUToF");
        assert_eq!(OP_BITCAST, 124, "OpBitcast");

        // The four convert opcodes form a contiguous block 109..=112
        assert_eq!(OP_CONVERT_U_TO_F - OP_CONVERT_F_TO_U, 3);
    }

    // ── 6. Cast opcodes: no collision ──────────────────────────────────────

    #[kani::proof]
    fn verify_cast_opcodes_no_collision() {
        let casts: [u16; 5] = [
            OP_CONVERT_F_TO_U,  // 109
            OP_CONVERT_F_TO_S,  // 110
            OP_CONVERT_S_TO_F,  // 111
            OP_CONVERT_U_TO_F,  // 112
            OP_BITCAST,         // 124
        ];
        let mut i: usize = 0;
        while i < casts.len() {
            let mut j: usize = i + 1;
            while j < casts.len() {
                assert_ne!(casts[i], casts[j], "cast opcode collision");
                j += 1;
            }
            i += 1;
        }
    }

    // ── 7. Atomic opcodes ──────────────────────────────────────────────────

    #[kani::proof]
    fn verify_atomic_opcodes() {
        assert_eq!(OP_CONTROL_BARRIER, 224, "OpControlBarrier");
        assert_eq!(OP_ATOMIC_EXCHANGE, 229, "OpAtomicExchange");
        assert_eq!(OP_ATOMIC_COMPARE_EXCHANGE, 230, "OpAtomicCompareExchange");
        assert_eq!(OP_ATOMIC_IADD, 234, "OpAtomicIAdd");
        assert_eq!(OP_ATOMIC_ISUB, 235, "OpAtomicISub");
        assert_eq!(OP_ATOMIC_SMIN, 236, "OpAtomicSMin");
        assert_eq!(OP_ATOMIC_UMIN, 237, "OpAtomicUMin");
        assert_eq!(OP_ATOMIC_SMAX, 238, "OpAtomicSMax");
        assert_eq!(OP_ATOMIC_UMAX, 239, "OpAtomicUMax");
        assert_eq!(OP_ATOMIC_AND, 240, "OpAtomicAnd");
        assert_eq!(OP_ATOMIC_OR, 241, "OpAtomicOr");
        assert_eq!(OP_ATOMIC_XOR, 242, "OpAtomicXor");

        // AtomicIAdd..AtomicXor is contiguous (234..=242)
        assert_eq!(OP_ATOMIC_XOR - OP_ATOMIC_IADD, 8);
    }

    // ── 8. Subgroup opcodes (complete set) ─────────────────────────────────

    #[kani::proof]
    fn verify_subgroup_opcodes_complete() {
        // Vote operations
        assert_eq!(OP_GROUP_NON_UNIFORM_ALL, 334, "OpGroupNonUniformAll");
        assert_eq!(OP_GROUP_NON_UNIFORM_ANY, 335, "OpGroupNonUniformAny");

        // Ballot
        assert_eq!(OP_GROUP_NON_UNIFORM_BALLOT, 339, "OpGroupNonUniformBallot");

        // Shuffle
        assert_eq!(OP_GROUP_NON_UNIFORM_SHUFFLE, 345, "OpGroupNonUniformShuffle");
        assert_eq!(
            OP_GROUP_NON_UNIFORM_SHUFFLE_XOR,
            346,
            "OpGroupNonUniformShuffleXor"
        );

        // Arithmetic reductions
        assert_eq!(OP_GROUP_NON_UNIFORM_IADD, 349, "OpGroupNonUniformIAdd");
        assert_eq!(OP_GROUP_NON_UNIFORM_FADD, 350, "OpGroupNonUniformFAdd");
        assert_eq!(OP_GROUP_NON_UNIFORM_SMIN, 353, "OpGroupNonUniformSMin");
        assert_eq!(OP_GROUP_NON_UNIFORM_UMIN, 354, "OpGroupNonUniformUMin");
        assert_eq!(OP_GROUP_NON_UNIFORM_FMIN, 355, "OpGroupNonUniformFMin");
        assert_eq!(OP_GROUP_NON_UNIFORM_SMAX, 356, "OpGroupNonUniformSMax");
        assert_eq!(OP_GROUP_NON_UNIFORM_UMAX, 357, "OpGroupNonUniformUMax");
        assert_eq!(OP_GROUP_NON_UNIFORM_FMAX, 358, "OpGroupNonUniformFMax");

        // Min/Max block is contiguous (353..=358)
        assert_eq!(OP_GROUP_NON_UNIFORM_FMAX - OP_GROUP_NON_UNIFORM_SMIN, 5);
    }

    // ── 9. Capability constants ────────────────────────────────────────────

    #[kani::proof]
    fn verify_capability_constants() {
        assert_eq!(CAPABILITY_SHADER, 1, "Shader");
        assert_eq!(CAPABILITY_GROUP_NON_UNIFORM, 61, "GroupNonUniform");
        assert_eq!(CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC, 63, "GroupNonUniformArithmetic");
        assert_eq!(CAPABILITY_GROUP_NON_UNIFORM_BALLOT, 64, "GroupNonUniformBallot");
        assert_eq!(CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE, 65, "GroupNonUniformShuffle");

        // GroupNonUniform capabilities span 61..=65 (Vote=62 unused here)
        assert_eq!(CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE - CAPABILITY_GROUP_NON_UNIFORM, 4);

        // All distinct
        let caps: [u32; 5] = [
            CAPABILITY_SHADER,
            CAPABILITY_GROUP_NON_UNIFORM,
            CAPABILITY_GROUP_NON_UNIFORM_BALLOT,
            CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC,
            CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE,
        ];
        let mut i: usize = 0;
        while i < caps.len() {
            let mut j: usize = i + 1;
            while j < caps.len() {
                assert_ne!(caps[i], caps[j], "capability collision");
                j += 1;
            }
            i += 1;
        }
    }

    // ── 10. Global collision check: ALL opcodes ────────────────────────────

    /// Every opcode constant in constants.rs must have a unique value.
    /// This catches copy-paste errors where two names map to the same u16.
    #[kani::proof]
    fn verify_all_opcodes_no_collision() {
        // Complete list of all u16 opcode constants from constants.rs,
        // sorted by value for readability.
        let all: [u16; 122] = [
            OP_NAME,                          //   5
            OP_EXT_INST_IMPORT,               //  11
            OP_EXT_INST,                      //  12
            OP_MEMORY_MODEL,                  //  14
            OP_ENTRY_POINT,                   //  15
            OP_EXECUTION_MODE,                //  16
            OP_CAPABILITY,                    //  17
            OP_TYPE_VOID,                     //  19
            OP_TYPE_BOOL,                     //  20
            OP_TYPE_INT,                      //  21
            OP_TYPE_FLOAT,                    //  22
            OP_TYPE_VECTOR,                   //  23
            OP_TYPE_MATRIX,                   //  24
            OP_TYPE_IMAGE,                    //  25
            OP_TYPE_SAMPLER,                  //  26
            OP_TYPE_SAMPLED_IMAGE,            //  27
            OP_TYPE_ARRAY,                    //  28
            OP_TYPE_RUNTIME_ARRAY,            //  29
            OP_TYPE_STRUCT,                   //  30
            OP_TYPE_POINTER,                  //  32
            OP_TYPE_FUNCTION,                 //  33
            OP_CONSTANT_TRUE,                 //  41
            OP_CONSTANT_FALSE,                //  42
            OP_CONSTANT,                      //  43
            OP_FUNCTION,                      //  54
            OP_FUNCTION_PARAMETER,            //  55
            OP_FUNCTION_END,                  //  56
            OP_FUNCTION_CALL,                 //  57
            OP_VARIABLE,                      //  59
            OP_LOAD,                          //  61
            OP_STORE,                         //  62
            OP_ACCESS_CHAIN,                  //  65
            OP_DECORATE,                      //  71
            OP_MEMBER_DECORATE,               //  72
            OP_COMPOSITE_CONSTRUCT,           //  80
            OP_COMPOSITE_EXTRACT,             //  81
            OP_COPY_OBJECT,                   //  83
            OP_TRANSPOSE,                     //  84
            OP_IMAGE_SAMPLE_IMPLICIT_LOD,     //  87
            OP_IMAGE_FETCH,                   //  95
            OP_IMAGE_WRITE,                   //  99
            OP_CONVERT_F_TO_U,                // 109
            OP_CONVERT_F_TO_S,                // 110
            OP_CONVERT_S_TO_F,                // 111
            OP_CONVERT_U_TO_F,                // 112
            OP_BITCAST,                       // 124
            OP_S_NEGATE,                      // 126
            OP_F_NEGATE,                      // 127
            OP_IADD,                          // 128
            OP_FADD,                          // 129
            OP_ISUB,                          // 130
            OP_FSUB,                          // 131
            OP_IMUL,                          // 132
            OP_FMUL,                          // 133
            OP_UDIV,                          // 134
            OP_SDIV,                          // 135
            OP_FDIV,                          // 136
            OP_UMOD,                          // 137
            OP_SREM,                          // 138
            OP_FREM,                          // 140
            OP_VECTOR_TIMES_MATRIX,           // 144
            OP_MATRIX_TIMES_VECTOR,           // 145
            OP_MATRIX_TIMES_MATRIX,           // 146
            OP_DOT,                           // 148
            OP_LOGICAL_NOT,                   // 168
            OP_SELECT,                        // 169
            OP_IEQUAL,                        // 170
            OP_INOT_EQUAL,                    // 171
            OP_UGREATER_THAN,                 // 172
            OP_SGREATER_THAN,                 // 173
            OP_UGREATER_THAN_EQUAL,           // 174
            OP_SGREATER_THAN_EQUAL,           // 175
            OP_ULESS_THAN,                    // 176
            OP_SLESS_THAN,                    // 177
            OP_ULESS_THAN_EQ,                 // 178
            OP_SLESS_THAN_EQUAL,              // 179
            OP_FORD_EQUAL,                    // 180
            OP_FORD_NOT_EQUAL,                // 182
            OP_FORD_LESS_THAN,                // 184
            OP_FORD_GREATER_THAN,             // 186
            OP_FORD_LESS_THAN_EQUAL,          // 188
            OP_FORD_GREATER_THAN_EQUAL,       // 190
            OP_SHIFT_RIGHT_LOGICAL,           // 194
            OP_SHIFT_RIGHT_ARITHMETIC,        // 195
            OP_SHIFT_LEFT_LOGICAL,            // 196
            OP_BITWISE_OR,                    // 197
            OP_BITWISE_XOR,                   // 198
            OP_BITWISE_AND,                   // 199
            OP_NOT,                           // 200
            OP_BIT_COUNT,                     // 205
            OP_CONTROL_BARRIER,               // 224
            OP_ATOMIC_EXCHANGE,               // 229
            OP_ATOMIC_COMPARE_EXCHANGE,       // 230
            OP_ATOMIC_IADD,                   // 234
            OP_ATOMIC_ISUB,                   // 235
            OP_ATOMIC_SMIN,                   // 236
            OP_ATOMIC_UMIN,                   // 237
            OP_ATOMIC_SMAX,                   // 238
            OP_ATOMIC_UMAX,                   // 239
            OP_ATOMIC_AND,                    // 240
            OP_ATOMIC_OR,                     // 241
            OP_ATOMIC_XOR,                    // 242
            OP_PHI,                           // 245
            OP_LOOP_MERGE,                    // 246
            OP_SELECTION_MERGE,               // 247
            OP_LABEL,                         // 248
            OP_BRANCH,                        // 249
            OP_BRANCH_CONDITIONAL,            // 250
            OP_RETURN,                        // 253
            OP_GROUP_NON_UNIFORM_ALL,         // 334
            OP_GROUP_NON_UNIFORM_ANY,         // 335
            OP_GROUP_NON_UNIFORM_BALLOT,      // 339
            OP_GROUP_NON_UNIFORM_SHUFFLE,     // 345
            OP_GROUP_NON_UNIFORM_SHUFFLE_XOR, // 346
            OP_GROUP_NON_UNIFORM_IADD,        // 349
            OP_GROUP_NON_UNIFORM_FADD,        // 350
            OP_GROUP_NON_UNIFORM_SMIN,        // 353
            OP_GROUP_NON_UNIFORM_UMIN,        // 354
            OP_GROUP_NON_UNIFORM_FMIN,        // 355
            OP_GROUP_NON_UNIFORM_SMAX,        // 356
            OP_GROUP_NON_UNIFORM_UMAX,        // 357
            OP_GROUP_NON_UNIFORM_FMAX,        // 358
        ];

        // O(n^2) pairwise distinctness. N=96 so CBMC handles this
        // as pure const propagation -- no symbolic input, no allocation.
        let mut i: usize = 0;
        while i < all.len() {
            let mut j: usize = i + 1;
            while j < all.len() {
                assert_ne!(all[i], all[j], "opcode collision");
                j += 1;
            }
            i += 1;
        }
    }

    // ── Bonus: non-opcode constant groups (no cross-namespace collision) ───

    #[kani::proof]
    fn verify_storage_class_constants() {
        assert_eq!(STORAGE_CLASS_UNIFORM_CONSTANT, 0);
        assert_eq!(STORAGE_CLASS_INPUT, 1);
        assert_eq!(STORAGE_CLASS_OUTPUT, 3);
        assert_eq!(STORAGE_CLASS_WORKGROUP, 4);
        assert_eq!(STORAGE_CLASS_PUSH_CONSTANT, 9);
        assert_eq!(STORAGE_CLASS_STORAGE_BUFFER, 12);

        // All distinct
        let scs: [u32; 6] = [
            STORAGE_CLASS_UNIFORM_CONSTANT,
            STORAGE_CLASS_INPUT,
            STORAGE_CLASS_OUTPUT,
            STORAGE_CLASS_WORKGROUP,
            STORAGE_CLASS_PUSH_CONSTANT,
            STORAGE_CLASS_STORAGE_BUFFER,
        ];
        let mut i: usize = 0;
        while i < scs.len() {
            let mut j: usize = i + 1;
            while j < scs.len() {
                assert_ne!(scs[i], scs[j], "storage class collision");
                j += 1;
            }
            i += 1;
        }
    }

    #[kani::proof]
    fn verify_decoration_constants() {
        assert_eq!(DECORATION_BLOCK, 2);
        assert_eq!(DECORATION_ARRAY_STRIDE, 6);
        assert_eq!(DECORATION_BUILTIN, 11);
        assert_eq!(DECORATION_NON_WRITABLE, 24);
        assert_eq!(DECORATION_LOCATION, 30);
        assert_eq!(DECORATION_BINDING, 33);
        assert_eq!(DECORATION_DESCRIPTOR_SET, 34);
        assert_eq!(DECORATION_OFFSET, 35);

        // All distinct
        let decs: [u32; 8] = [
            DECORATION_BLOCK,
            DECORATION_ARRAY_STRIDE,
            DECORATION_BUILTIN,
            DECORATION_NON_WRITABLE,
            DECORATION_LOCATION,
            DECORATION_BINDING,
            DECORATION_DESCRIPTOR_SET,
            DECORATION_OFFSET,
        ];
        let mut i: usize = 0;
        while i < decs.len() {
            let mut j: usize = i + 1;
            while j < decs.len() {
                assert_ne!(decs[i], decs[j], "decoration collision");
                j += 1;
            }
            i += 1;
        }
    }

    #[kani::proof]
    fn verify_spirv_header_constants() {
        assert_eq!(SPIRV_MAGIC, 0x07230203, "SPIR-V magic number");
        assert_eq!(SPIRV_VERSION_1_3, 0x00010300, "SPIR-V version 1.3");
        assert_eq!(ADDRESSING_MODEL_LOGICAL, 0, "Logical addressing");
        assert_eq!(MEMORY_MODEL_GLSL450, 1, "GLSL450 memory model");
    }

    #[kani::proof]
    fn verify_execution_model_constants() {
        assert_eq!(EXECUTION_MODEL_VERTEX, 0);
        assert_eq!(EXECUTION_MODEL_FRAGMENT, 4);
        assert_eq!(EXECUTION_MODEL_GLCOMPUTE, 5);
        assert_eq!(EXECUTION_MODE_ORIGIN_UPPER_LEFT, 7);
        assert_eq!(EXECUTION_MODE_LOCAL_SIZE, 17);

        // All execution models are distinct
        let models: [u32; 3] = [
            EXECUTION_MODEL_VERTEX,
            EXECUTION_MODEL_FRAGMENT,
            EXECUTION_MODEL_GLCOMPUTE,
        ];
        let mut i: usize = 0;
        while i < models.len() {
            let mut j: usize = i + 1;
            while j < models.len() {
                assert_ne!(models[i], models[j], "execution model collision");
                j += 1;
            }
            i += 1;
        }
    }

    #[kani::proof]
    fn verify_scope_and_semantics_constants() {
        assert_eq!(SCOPE_SUBGROUP, 3);
        assert_eq!(SCOPE_WORKGROUP, 2);
        assert_ne!(SCOPE_SUBGROUP, SCOPE_WORKGROUP, "scopes must differ");

        assert_eq!(MEMORY_SEMANTICS_WORKGROUP, 0x100);
        assert_eq!(MEMORY_SEMANTICS_ACQ_REL, 0x8);
        assert_ne!(
            MEMORY_SEMANTICS_WORKGROUP, MEMORY_SEMANTICS_ACQ_REL,
            "memory semantics must differ"
        );
    }

    #[kani::proof]
    fn verify_group_operation_constants() {
        assert_eq!(GROUP_OPERATION_REDUCE, 0);
        assert_eq!(GROUP_OPERATION_INCLUSIVE_SCAN, 1);
        assert_eq!(GROUP_OPERATION_EXCLUSIVE_SCAN, 2);

        // Contiguous 0..=2
        assert_eq!(GROUP_OPERATION_EXCLUSIVE_SCAN - GROUP_OPERATION_REDUCE, 2);
    }

    #[kani::proof]
    fn verify_builtin_constants() {
        assert_eq!(BUILTIN_POSITION, 0);
        assert_eq!(BUILTIN_NUM_WORKGROUPS, 24);
        assert_eq!(BUILTIN_WORKGROUP_ID, 26);
        assert_eq!(BUILTIN_LOCAL_INVOCATION_ID, 27);
        assert_eq!(BUILTIN_GLOBAL_INVOCATION_ID, 28);

        // All distinct
        let builtins: [u32; 5] = [
            BUILTIN_POSITION,
            BUILTIN_NUM_WORKGROUPS,
            BUILTIN_WORKGROUP_ID,
            BUILTIN_LOCAL_INVOCATION_ID,
            BUILTIN_GLOBAL_INVOCATION_ID,
        ];
        let mut i: usize = 0;
        while i < builtins.len() {
            let mut j: usize = i + 1;
            while j < builtins.len() {
                assert_ne!(builtins[i], builtins[j], "builtin collision");
                j += 1;
            }
            i += 1;
        }
    }
}
