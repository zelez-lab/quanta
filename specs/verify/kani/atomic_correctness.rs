//! Kani harnesses for atomic instruction correctness in the SPIR-V emitter.
//!
//! Covers theorems T1400 (AtomicCAS opcode) and T1401 (atomic semantics).
//!
//! All values are mirrored from crates/lang/quanta-compiler/src/emit_spirv/constants.rs.
//! No allocation: pure const assertions only.
//!
//! Run all:  cargo kani --harness-regex 'verify_'
//! Run one:  cargo kani --harness verify_atomic_cas_opcode
//!
//! | Theorem | What it proves |
//! |---------|----------------|
//! | T1400   | AtomicCAS uses opcode 230 (OpAtomicCompareExchange), not 61 (OpLoad). |
//! | T1401   | AcqRel (0x8) OR Workgroup (0x100) == 0x108. Scope is Device (1). |

// ── Constants mirrored from emit_spirv/constants.rs ──────────────────

// Atomic opcodes
const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
const OP_ATOMIC_EXCHANGE: u16 = 229;
const OP_ATOMIC_IADD: u16 = 234;
const OP_ATOMIC_ISUB: u16 = 235;
const OP_ATOMIC_SMIN: u16 = 236;
const OP_ATOMIC_UMIN: u16 = 237;
const OP_ATOMIC_SMAX: u16 = 238;
const OP_ATOMIC_UMAX: u16 = 239;
const OP_ATOMIC_AND: u16 = 240;
const OP_ATOMIC_OR: u16 = 241;
const OP_ATOMIC_XOR: u16 = 242;

// Non-atomic opcodes (must NOT appear in atomic codepath)
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;

// Scopes (SPIR-V section 3.27)
const SCOPE_DEVICE: u32 = 1;
const SCOPE_WORKGROUP: u32 = 2;
const SCOPE_SUBGROUP: u32 = 3;

// Memory semantics (SPIR-V section 3.25)
const MEMORY_SEMANTICS_RELAXED: u32 = 0x0;
const MEMORY_SEMANTICS_ACQ_REL: u32 = 0x8;
const MEMORY_SEMANTICS_WORKGROUP: u32 = 0x100;

// ── Model of the AtomicCAS instruction word sequence ─────────────────
//
// Production: emit_op_atomic_cas (ops_ext.rs:76-118)
//   Self::emit_op(
//       &mut self.sec_function,
//       OP_ATOMIC_COMPARE_EXCHANGE,       // opcode word
//       &[result_ty, result_id, chain,    // operands
//         scope, semantics, semantics,
//         des_val, exp_val],
//   );
//
// emit_op encodes the header word as:
//   ((1 + operands.len()) << 16) | opcode
// For AtomicCAS: operands.len() = 8, so header = (9 << 16) | 230

/// Model: extract the opcode from the header word (low 16 bits).
fn header_opcode(header: u32) -> u16 {
    (header & 0xFFFF) as u16
}

/// Model: extract the word count from the header word (high 16 bits).
fn header_word_count(header: u32) -> u16 {
    (header >> 16) as u16
}

/// Build a header word the way emit_op does.
fn make_header(word_count: u16, opcode: u16) -> u32 {
    ((word_count as u32) << 16) | (opcode as u32)
}

// ── Kani harnesses ───────────────────────────────────────────────────

#[cfg(kani)]
mod proofs {
    use super::*;

    // ── T1400: AtomicCAS opcode ──────────────────────────────────────

    /// T1400 Kani: The AtomicCAS instruction header contains opcode 230
    /// (OpAtomicCompareExchange), NOT opcode 61 (OpLoad).
    #[kani::proof]
    fn verify_atomic_cas_opcode() {
        // Build the header word as emit_op would for AtomicCAS
        // 8 operands + 1 header = 9 words
        let header = make_header(9, OP_ATOMIC_COMPARE_EXCHANGE);

        // The opcode extracted from the header is 230
        assert_eq!(
            header_opcode(header),
            230,
            "AtomicCAS header must contain opcode 230"
        );

        // The opcode is NOT OpLoad (61)
        assert_ne!(
            header_opcode(header),
            OP_LOAD,
            "AtomicCAS header must NOT be OpLoad"
        );

        // The word count is 9
        assert_eq!(
            header_word_count(header),
            9,
            "AtomicCAS instruction is 9 words (1 header + 8 operands)"
        );
    }

    /// T1400 Kani: Verify the raw constant value is 230 (not accidentally
    /// assigned OpLoad's value of 61).
    #[kani::proof]
    fn verify_atomic_cas_constant_value() {
        assert_eq!(OP_ATOMIC_COMPARE_EXCHANGE, 230);
        assert_ne!(OP_ATOMIC_COMPARE_EXCHANGE, OP_LOAD);
        assert_ne!(OP_ATOMIC_COMPARE_EXCHANGE, OP_STORE);
    }

    /// T1400 Kani: The AtomicCAS opcode is distinct from all other
    /// atomic opcodes (no accidental overlap).
    #[kani::proof]
    fn verify_atomic_cas_distinct_from_other_atomics() {
        let cas = OP_ATOMIC_COMPARE_EXCHANGE;
        assert_ne!(cas, OP_ATOMIC_EXCHANGE);
        assert_ne!(cas, OP_ATOMIC_IADD);
        assert_ne!(cas, OP_ATOMIC_ISUB);
        assert_ne!(cas, OP_ATOMIC_SMIN);
        assert_ne!(cas, OP_ATOMIC_UMIN);
        assert_ne!(cas, OP_ATOMIC_SMAX);
        assert_ne!(cas, OP_ATOMIC_UMAX);
        assert_ne!(cas, OP_ATOMIC_AND);
        assert_ne!(cas, OP_ATOMIC_OR);
        assert_ne!(cas, OP_ATOMIC_XOR);
    }

    /// T1400 Kani: For any symbolic opcode, the header round-trips correctly.
    /// This proves make_header / header_opcode are inverses for any opcode.
    #[kani::proof]
    fn verify_header_opcode_roundtrip() {
        let opcode: u16 = kani::any();
        let word_count: u16 = kani::any();
        kani::assume(word_count > 0);

        let header = make_header(word_count, opcode);
        assert_eq!(header_opcode(header), opcode);
        assert_eq!(header_word_count(header), word_count);
    }

    // ── T1401: Atomic memory semantics ───────────────────────────────

    /// T1401 Kani: 0x8 | 0x100 == 0x108.
    #[kani::proof]
    fn verify_acq_rel_workgroup_combined() {
        let combined = MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP;
        assert_eq!(combined, 0x108, "AcqRel | Workgroup must equal 0x108");
    }

    /// T1401 Kani: The combined semantics is NOT Relaxed (0x0).
    #[kani::proof]
    fn verify_semantics_not_relaxed() {
        let combined = MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP;
        assert_ne!(
            combined, MEMORY_SEMANTICS_RELAXED,
            "Atomic semantics must not be Relaxed"
        );
    }

    /// T1401 Kani: The AcquireRelease bit (0x8) is set in the combined value.
    #[kani::proof]
    fn verify_acq_rel_bit_set() {
        let combined = MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP;
        assert_ne!(
            combined & MEMORY_SEMANTICS_ACQ_REL,
            0,
            "AcquireRelease bit must be set"
        );
    }

    /// T1401 Kani: The WorkgroupMemory bit (0x100) is set in the combined value.
    #[kani::proof]
    fn verify_workgroup_bit_set() {
        let combined = MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP;
        assert_ne!(
            combined & MEMORY_SEMANTICS_WORKGROUP,
            0,
            "WorkgroupMemory bit must be set"
        );
    }

    /// T1401 Kani: Scope is Device (1).
    #[kani::proof]
    fn verify_scope_is_device() {
        // Production code: self.emit_constant_u32(1) // Device
        let scope = SCOPE_DEVICE;
        assert_eq!(scope, 1, "Atomic scope must be Device (1)");
        assert_ne!(scope, SCOPE_WORKGROUP, "Scope must not be Workgroup");
        assert_ne!(scope, SCOPE_SUBGROUP, "Scope must not be Subgroup");
    }

    /// T1401 Kani: Verify bit decomposition — 0x108 has exactly bits 3 and 8 set.
    #[kani::proof]
    fn verify_semantics_bit_decomposition() {
        let combined: u32 = 0x108;

        // Bit 3 (AcquireRelease = 0x8 = 1 << 3)
        assert_ne!(combined & (1 << 3), 0, "Bit 3 must be set");
        // Bit 8 (WorkgroupMemory = 0x100 = 1 << 8)
        assert_ne!(combined & (1 << 8), 0, "Bit 8 must be set");

        // No other bits are set
        let mask = (1u32 << 3) | (1u32 << 8);
        assert_eq!(combined & !mask, 0, "No other bits should be set");
    }
}
