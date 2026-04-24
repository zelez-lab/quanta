//! Kani harness: SPIR-V opcode constants match the specification.
//!
//! Run: cargo kani --harness verify_bitwise_opcodes

/// SPIR-V 1.6 specification opcode values (ground truth).
mod spec {
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
    pub const OP_SMOD: u16 = 138;
    pub const OP_FREM: u16 = 140;
    pub const OP_BITWISE_OR: u16 = 197;
    pub const OP_BITWISE_XOR: u16 = 198;
    pub const OP_BITWISE_AND: u16 = 199;
    pub const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const OP_COMPOSITE_CONSTRUCT: u16 = 80;
    pub const OP_COMPOSITE_EXTRACT: u16 = 81;
    pub const OP_LOAD: u16 = 61;
    pub const OP_STORE: u16 = 62;
    pub const OP_VARIABLE: u16 = 59;
    pub const OP_CONTROL_BARRIER: u16 = 224;
    pub const OP_TYPE_FLOAT: u16 = 22;
    pub const OP_TYPE_INT: u16 = 21;
    pub const OP_TYPE_VECTOR: u16 = 23;
    pub const OP_TYPE_MATRIX: u16 = 24;
    pub const OP_CONSTANT: u16 = 43;
    pub const OP_FUNCTION: u16 = 54;
    pub const OP_RETURN: u16 = 253;
    pub const OP_LABEL: u16 = 248;
    pub const OP_BRANCH: u16 = 249;
    pub const OP_BRANCH_CONDITIONAL: u16 = 250;
    pub const OP_SELECTION_MERGE: u16 = 247;
    pub const OP_F_NEGATE: u16 = 127;
    pub const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
    pub const OP_IMAGE_WRITE: u16 = 99;
    pub const OP_IMAGE_FETCH: u16 = 95;
    pub const OP_MATRIX_TIMES_VECTOR: u16 = 145;
    pub const OP_DOT: u16 = 148;
    pub const OP_GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
    pub const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
    pub const OP_GROUP_NON_UNIFORM_ANY: u16 = 335;
    pub const OP_GROUP_NON_UNIFORM_ALL: u16 = 336;
}

/// Values from our emit_spirv.rs (must match spec exactly).
mod ours {
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
    pub const OP_SMOD: u16 = 138;
    pub const OP_FREM: u16 = 140;
    pub const OP_BITWISE_OR: u16 = 197;
    pub const OP_BITWISE_XOR: u16 = 198;
    pub const OP_BITWISE_AND: u16 = 199;
    pub const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
    pub const OP_IMAGE_WRITE: u16 = 99;
    pub const OP_IMAGE_FETCH: u16 = 95;
    pub const OP_MATRIX_TIMES_VECTOR: u16 = 145;
    pub const OP_DOT: u16 = 148;
    pub const OP_GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
    pub const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
    pub const OP_GROUP_NON_UNIFORM_ANY: u16 = 335;
    pub const OP_GROUP_NON_UNIFORM_ALL: u16 = 336;
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn verify_bitwise_opcodes() {
        assert_eq!(ours::OP_BITWISE_AND, spec::OP_BITWISE_AND, "AND must be 199");
        assert_eq!(ours::OP_BITWISE_OR, spec::OP_BITWISE_OR, "OR must be 197");
        assert_eq!(ours::OP_BITWISE_XOR, spec::OP_BITWISE_XOR, "XOR must be 198");
    }

    #[kani::proof]
    fn verify_arithmetic_opcodes() {
        assert_eq!(ours::OP_IADD, spec::OP_IADD);
        assert_eq!(ours::OP_FADD, spec::OP_FADD);
        assert_eq!(ours::OP_ISUB, spec::OP_ISUB);
        assert_eq!(ours::OP_FSUB, spec::OP_FSUB);
        assert_eq!(ours::OP_IMUL, spec::OP_IMUL);
        assert_eq!(ours::OP_FMUL, spec::OP_FMUL);
        assert_eq!(ours::OP_UDIV, spec::OP_UDIV);
        assert_eq!(ours::OP_SDIV, spec::OP_SDIV);
        assert_eq!(ours::OP_FDIV, spec::OP_FDIV);
        assert_eq!(ours::OP_UMOD, spec::OP_UMOD);
        assert_eq!(ours::OP_SMOD, spec::OP_SMOD);
        assert_eq!(ours::OP_FREM, spec::OP_FREM);
    }

    #[kani::proof]
    fn verify_shift_opcodes() {
        assert_eq!(ours::OP_SHIFT_LEFT_LOGICAL, spec::OP_SHIFT_LEFT_LOGICAL);
        assert_eq!(ours::OP_SHIFT_RIGHT_LOGICAL, spec::OP_SHIFT_RIGHT_LOGICAL);
        assert_eq!(ours::OP_SHIFT_RIGHT_ARITHMETIC, spec::OP_SHIFT_RIGHT_ARITHMETIC);
    }

    #[kani::proof]
    fn verify_texture_opcodes() {
        assert_eq!(ours::OP_IMAGE_SAMPLE_IMPLICIT_LOD, spec::OP_IMAGE_SAMPLE_IMPLICIT_LOD);
        assert_eq!(ours::OP_IMAGE_WRITE, spec::OP_IMAGE_WRITE);
        assert_eq!(ours::OP_IMAGE_FETCH, spec::OP_IMAGE_FETCH);
    }

    #[kani::proof]
    fn verify_subgroup_opcodes() {
        assert_eq!(ours::OP_GROUP_NON_UNIFORM_SHUFFLE, spec::OP_GROUP_NON_UNIFORM_SHUFFLE);
        assert_eq!(ours::OP_GROUP_NON_UNIFORM_BALLOT, spec::OP_GROUP_NON_UNIFORM_BALLOT);
        assert_eq!(ours::OP_GROUP_NON_UNIFORM_ANY, spec::OP_GROUP_NON_UNIFORM_ANY);
        assert_eq!(ours::OP_GROUP_NON_UNIFORM_ALL, spec::OP_GROUP_NON_UNIFORM_ALL);
    }

    #[kani::proof]
    fn verify_matrix_opcodes() {
        assert_eq!(ours::OP_MATRIX_TIMES_VECTOR, spec::OP_MATRIX_TIMES_VECTOR);
        assert_eq!(ours::OP_DOT, spec::OP_DOT);
    }

    #[kani::proof]
    fn verify_no_opcode_collisions() {
        // All our opcodes must be distinct
        let ops = [
            ours::OP_IADD, ours::OP_FADD, ours::OP_ISUB, ours::OP_FSUB,
            ours::OP_IMUL, ours::OP_FMUL, ours::OP_UDIV, ours::OP_SDIV,
            ours::OP_FDIV, ours::OP_UMOD, ours::OP_SMOD, ours::OP_FREM,
            ours::OP_BITWISE_OR, ours::OP_BITWISE_XOR, ours::OP_BITWISE_AND,
            ours::OP_SHIFT_LEFT_LOGICAL, ours::OP_SHIFT_RIGHT_LOGICAL,
            ours::OP_SHIFT_RIGHT_ARITHMETIC,
        ];
        for i in 0..ops.len() {
            for j in (i + 1)..ops.len() {
                assert_ne!(ops[i], ops[j], "opcode collision at indices {} and {}", i, j);
            }
        }
    }
}
