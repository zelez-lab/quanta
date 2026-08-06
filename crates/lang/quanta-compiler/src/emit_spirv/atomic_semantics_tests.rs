//! Memory-semantics words across the AOT emitter's atomic/fence sites.
//!
//! Mirrors `quanta-ir/tests/emit_spirv_atomic_semantics.rs` for the JIT
//! emitter. Two Vulkan-environment rules are pinned:
//!
//! 1. No emitted semantics word carries SequentiallyConsistent (0x10) —
//!    Vulkan forbids it (VUID-StandaloneSpirv-None-04732), so `SeqCst`
//!    clamps to AcquireRelease.
//! 2. `OpAtomicCompareExchange` gets DISTINCT Equal/Unequal operands: the
//!    Unequal (failure) semantics cannot carry a release component, so
//!    Release → None and AcqRel/SeqCst → Acquire — the same clamp
//!    emit_llvm applies to `cmpxchg`'s failure ordering.
//!
//! Modules are pushed through `spirv-val --target-env vulkan1.3` where the
//! validator is installed (plain `spirv-val` does NOT arm VUID-04732).

use std::collections::HashMap;

use quanta_ir::{
    AtomicOp, ConstValue, KernelDef, KernelOp, KernelParam, MemoryOrder, Reg, ScalarType,
};

// Semantics bits (SPIR-V section 3.25).
const ACQUIRE: u32 = 0x2;
const RELEASE: u32 = 0x4;
const ACQ_REL: u32 = 0x8;
const SEQ_CST: u32 = 0x10;
const UNIFORM_MEM: u32 = 0x40;
const WORKGROUP_MEM: u32 = 0x100;

const OP_CONSTANT_WORD: u16 = 43;
const OP_ATOMIC_CAS_WORD: u16 = 230;
const OP_ATOMIC_IADD_WORD: u16 = 234;
const OP_MEMORY_BARRIER_WORD: u16 = 225;

fn kernel(name: &str, body: Vec<KernelOp>, next_reg: u32) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![KernelParam::FieldWrite {
            name: "out".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body,
        body_source: None,
        next_reg,
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

fn cas_kernel(success: MemoryOrder, failure: MemoryOrder) -> KernelDef {
    let body = vec![
        KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(0),
        },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(1),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(2),
        },
        KernelOp::AtomicCas {
            dst: Reg(3),
            field: 0,
            index: Reg(0),
            expected: Reg(1),
            desired: Reg(2),
            ty: ScalarType::U32,
            success_order: success,
            failure_order: failure,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(3),
            ty: ScalarType::U32,
        },
    ];
    kernel("cas_semantics", body, 4)
}

fn rmw_kernel(order: MemoryOrder) -> KernelDef {
    let body = vec![
        KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(0),
        },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(1),
        },
        KernelOp::AtomicOp {
            dst: Reg(2),
            field: 0,
            index: Reg(0),
            val: Reg(1),
            op: AtomicOp::Add,
            ty: ScalarType::U32,
            order,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::U32,
        },
    ];
    kernel("rmw_semantics", body, 3)
}

fn fence_kernel(order: MemoryOrder) -> KernelDef {
    let body = vec![
        KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(0),
        },
        KernelOp::Fence { order },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(0),
            ty: ScalarType::U32,
        },
    ];
    kernel("fence_semantics", body, 1)
}

fn shared_cas_kernel(order: MemoryOrder) -> KernelDef {
    let body = vec![
        KernelOp::SharedDecl {
            id: 0,
            ty: ScalarType::U32,
            count: 64,
        },
        KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(0),
        },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(1),
        },
        KernelOp::SharedAtomicOp {
            dst: Reg(2),
            slot: 0,
            index: Reg(0),
            val: Reg(1),
            op: AtomicOp::CompareExchange,
            ty: ScalarType::U32,
            order,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::U32,
        },
    ];
    kernel("shared_cas_semantics", body, 3)
}

// ── Word-stream decoding ────────────────────────────────────────────────

fn to_words(spv: &[u8]) -> Vec<u32> {
    spv.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `(opcode, operands)` for every instruction after the 5-word header.
fn instructions(words: &[u32]) -> Vec<(u16, Vec<u32>)> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let opcode = (words[i] & 0xFFFF) as u16;
        let len = (words[i] >> 16) as usize;
        assert!(len > 0, "zero-length instruction at word {i}");
        out.push((opcode, words[i + 1..i + len].to_vec()));
        i += len;
    }
    out
}

/// result-id → value for every 32-bit OpConstant.
fn const_values(instrs: &[(u16, Vec<u32>)]) -> HashMap<u32, u32> {
    instrs
        .iter()
        .filter(|(op, operands)| *op == OP_CONSTANT_WORD && operands.len() == 3)
        .map(|(_, operands)| (operands[1], operands[2]))
        .collect()
}

/// Operand indices holding a memory-semantics id, per opcode.
fn semantics_positions(opcode: u16) -> &'static [usize] {
    match opcode {
        224 => &[2],             // OpControlBarrier
        225 => &[1],             // OpMemoryBarrier
        227 => &[4],             // OpAtomicLoad
        228 => &[2],             // OpAtomicStore
        230 | 231 => &[4, 5],    // OpAtomicCompareExchange[Weak]: Equal, Unequal
        229 | 232..=242 => &[4], // OpAtomicExchange / IIncrement / … / Xor
        _ => &[],
    }
}

/// Every memory-semantics value in the module, resolved through the
/// constant table, in instruction order.
fn all_semantics_values(spv: &[u8]) -> Vec<u32> {
    let words = to_words(spv);
    let instrs = instructions(&words);
    let consts = const_values(&instrs);
    let mut out = Vec::new();
    for (opcode, operands) in &instrs {
        for &pos in semantics_positions(*opcode) {
            let id = operands[pos];
            let value = *consts
                .get(&id)
                .unwrap_or_else(|| panic!("semantics id %{id} of opcode {opcode} not a constant"));
            out.push(value);
        }
    }
    out
}

/// The `(Equal, Unequal)` semantics pair of the module's single
/// OpAtomicCompareExchange.
fn cas_semantics_pair(spv: &[u8]) -> (u32, u32) {
    let words = to_words(spv);
    let instrs = instructions(&words);
    let consts = const_values(&instrs);
    let cas: Vec<_> = instrs
        .iter()
        .filter(|(op, _)| *op == OP_ATOMIC_CAS_WORD)
        .collect();
    assert_eq!(cas.len(), 1, "expected exactly one OpAtomicCompareExchange");
    let operands = &cas[0].1;
    (consts[&operands[4]], consts[&operands[5]])
}

/// Run `spirv-val --target-env vulkan1.3` if it is on PATH.
fn validate_vulkan(name: &str, spv: &[u8]) {
    let dir = std::env::temp_dir().join("quanta_compiler_atomic_semantics_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.spv"));
    std::fs::write(&path, spv).unwrap();
    match std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", path.to_str().unwrap()])
        .output()
    {
        Ok(out) => assert!(
            out.status.success(),
            "spirv-val rejected {name}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(_) => eprintln!("spirv-val not on PATH — skipping external validation"),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

/// Buffer CAS at every order: Equal keeps the (clamped) success order,
/// Unequal is the failure order under the failure clamp.
#[test]
fn cas_semantics_split_and_clamped() {
    let cases = [
        (MemoryOrder::Relaxed, MemoryOrder::Relaxed, 0, 0),
        (
            MemoryOrder::Acquire,
            MemoryOrder::Acquire,
            ACQUIRE | WORKGROUP_MEM,
            ACQUIRE | WORKGROUP_MEM,
        ),
        // Failure Release → None: a failed CAS has no store to release.
        (
            MemoryOrder::Release,
            MemoryOrder::Release,
            RELEASE | WORKGROUP_MEM,
            0,
        ),
        // Failure AcqRel → Acquire: the release half is dropped.
        (
            MemoryOrder::AcqRel,
            MemoryOrder::AcqRel,
            ACQ_REL | WORKGROUP_MEM,
            ACQUIRE | WORKGROUP_MEM,
        ),
        // SeqCst → AcqRel (VUID-04732), then failure AcqRel → Acquire.
        (
            MemoryOrder::SeqCst,
            MemoryOrder::SeqCst,
            ACQ_REL | WORKGROUP_MEM,
            ACQUIRE | WORKGROUP_MEM,
        ),
        // Split orders: Unequal must follow failure_order, not success.
        (
            MemoryOrder::Release,
            MemoryOrder::Relaxed,
            RELEASE | WORKGROUP_MEM,
            0,
        ),
        (
            MemoryOrder::AcqRel,
            MemoryOrder::Relaxed,
            ACQ_REL | WORKGROUP_MEM,
            0,
        ),
    ];
    for (success, failure, want_eq, want_uneq) in cases {
        let spv = crate::emit_spirv::emit(&cas_kernel(success, failure)).expect("emit");
        let (got_eq, got_uneq) = cas_semantics_pair(&spv);
        assert_eq!(
            (got_eq, got_uneq),
            (want_eq, want_uneq),
            "CAS ({success:?}, {failure:?}) semantics pair"
        );
        validate_vulkan(&format!("cas_{success:?}_{failure:?}"), &spv);
    }
}

/// Atomic RMW at every order: SeqCst emits AcqRel, never the 0x10 bit.
#[test]
fn rmw_semantics_never_seq_cst() {
    let cases = [
        (MemoryOrder::Relaxed, 0),
        (MemoryOrder::Acquire, ACQUIRE | WORKGROUP_MEM),
        (MemoryOrder::Release, RELEASE | WORKGROUP_MEM),
        (MemoryOrder::AcqRel, ACQ_REL | WORKGROUP_MEM),
        (MemoryOrder::SeqCst, ACQ_REL | WORKGROUP_MEM),
    ];
    for (order, want) in cases {
        let spv = crate::emit_spirv::emit(&rmw_kernel(order)).expect("emit");
        let words = to_words(&spv);
        let instrs = instructions(&words);
        let consts = const_values(&instrs);
        let adds: Vec<_> = instrs
            .iter()
            .filter(|(op, _)| *op == OP_ATOMIC_IADD_WORD)
            .collect();
        assert_eq!(adds.len(), 1, "expected exactly one OpAtomicIAdd");
        assert_eq!(
            consts[&adds[0].1[4]], want,
            "OpAtomicIAdd {order:?} semantics"
        );
        for value in all_semantics_values(&spv) {
            assert_eq!(value & SEQ_CST, 0, "SeqCst bit in {order:?} module");
        }
        validate_vulkan(&format!("rmw_{order:?}"), &spv);
    }
}

/// Fence at every order: SeqCst emits AcqRel, never 0x10 — and a Relaxed
/// fence emits NO barrier at all (a relaxed fence orders nothing, and an
/// OpMemoryBarrier without an ordering bit is itself invalid under Vulkan
/// — VUID-StandaloneSpirv-None-04733).
#[test]
fn fence_semantics_never_seq_cst() {
    let cases = [
        (MemoryOrder::Acquire, ACQUIRE),
        (MemoryOrder::Release, RELEASE),
        (MemoryOrder::AcqRel, ACQ_REL),
        (MemoryOrder::SeqCst, ACQ_REL),
    ];
    for (order, want_bits) in cases {
        let spv = crate::emit_spirv::emit(&fence_kernel(order)).expect("emit");
        let words = to_words(&spv);
        let instrs = instructions(&words);
        let consts = const_values(&instrs);
        let fences: Vec<_> = instrs
            .iter()
            .filter(|(op, _)| *op == OP_MEMORY_BARRIER_WORD)
            .collect();
        assert_eq!(fences.len(), 1, "expected exactly one OpMemoryBarrier");
        assert_eq!(
            consts[&fences[0].1[1]],
            want_bits | UNIFORM_MEM | WORKGROUP_MEM,
            "OpMemoryBarrier {order:?} semantics"
        );
        for value in all_semantics_values(&spv) {
            assert_eq!(value & SEQ_CST, 0, "SeqCst bit in {order:?} module");
        }
        validate_vulkan(&format!("fence_{order:?}"), &spv);
    }

    // The Relaxed fence lowers to nothing.
    let spv = crate::emit_spirv::emit(&fence_kernel(MemoryOrder::Relaxed)).expect("emit");
    let instrs = instructions(&to_words(&spv));
    assert!(
        instrs.iter().all(|(op, _)| *op != OP_MEMORY_BARRIER_WORD),
        "a Relaxed fence must emit no OpMemoryBarrier"
    );
    validate_vulkan("fence_Relaxed", &spv);
}

/// The shared-memory CompareExchange arm splits its operands the same way.
#[test]
fn shared_cas_failure_clamped() {
    let spv = crate::emit_spirv::emit(&shared_cas_kernel(MemoryOrder::SeqCst)).expect("emit");
    let (got_eq, got_uneq) = cas_semantics_pair(&spv);
    assert_eq!(
        (got_eq, got_uneq),
        (ACQ_REL | WORKGROUP_MEM, ACQUIRE | WORKGROUP_MEM),
        "shared CAS SeqCst semantics pair"
    );
    for value in all_semantics_values(&spv) {
        assert_eq!(value & SEQ_CST, 0, "SeqCst bit in shared CAS module");
    }
    validate_vulkan("shared_cas_SeqCst", &spv);
}
