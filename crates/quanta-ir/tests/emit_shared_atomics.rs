//! Shared-memory atomics across the JIT emitters.
//!
//! Until 2026-06-12 only the MSL emitter lowered
//! `KernelOp::SharedAtomicOp`; SPIR-V refused and WGSL emitted an
//! `/* unsupported */` marker. These tests pin the new lowerings on
//! a histogram-shaped kernel: zero-init the shared bucket array,
//! barrier, atomically bump a bucket, barrier, read the counts back.
//!
//! Where the host has the official validators (`spirv-val` from
//! SPIRV-Tools, `naga` for WGSL), the emitted module is also pushed
//! through them — emission bugs in atomic operand order or pointer
//! storage classes show up there, not in string asserts.

#![cfg(feature = "jit")]

use quanta_ir::{
    AtomicOp, ConstValue, KernelDef, KernelOp, KernelParam, MemoryOrder, Reg, ScalarType,
};

/// Histogram-shaped kernel: shared_0 is the per-block bucket array,
/// touched by a zero-init store, an atomic add, and a readback load.
fn shared_atomic_kernel() -> KernelDef {
    let body = vec![
        KernelOp::SharedDecl {
            id: 0,
            ty: ScalarType::U32,
            count: 256,
        },
        KernelOp::ProtonId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(0),
        },
        // Zero-init own slot, then sync.
        KernelOp::SharedStore {
            id: 0,
            index: Reg(0),
            src: Reg(1),
            ty: ScalarType::U32,
        },
        KernelOp::Barrier,
        // Atomically bump bucket[lane & 15].
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(15),
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(0),
            b: Reg(2),
            op: quanta_ir::BinOp::BitAnd,
            ty: ScalarType::U32,
        },
        KernelOp::Const {
            dst: Reg(4),
            value: ConstValue::U32(1),
        },
        KernelOp::SharedAtomicOp {
            dst: Reg(5),
            slot: 0,
            index: Reg(3),
            val: Reg(4),
            op: AtomicOp::Add,
            ty: ScalarType::U32,
            order: MemoryOrder::Relaxed,
        },
        KernelOp::Barrier,
        // Read own slot back out to the field.
        KernelOp::SharedLoad {
            dst: Reg(6),
            id: 0,
            index: Reg(0),
            ty: ScalarType::U32,
        },
        KernelOp::QuarkId { dst: Reg(7) },
        KernelOp::Store {
            field: 0,
            index: Reg(7),
            src: Reg(6),
            ty: ScalarType::U32,
        },
    ];
    KernelDef {
        name: "shared_atomic_hist".into(),
        params: vec![KernelParam::FieldWrite {
            name: "out".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body,
        body_source: None,
        next_reg: 8,
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [256, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Run an external validator if it exists on PATH; `None` = tool not
/// installed (test still passes — emission-level asserts carry it).
fn run_validator(tool: &str, args: &[&str]) -> Option<std::process::Output> {
    match std::process::Command::new(tool).args(args).output() {
        Ok(out) => Some(out),
        Err(_) => {
            eprintln!("{tool} not on PATH — skipping external validation");
            None
        }
    }
}

#[test]
fn spirv_emits_and_validates() {
    let kernel = shared_atomic_kernel();
    let spv = quanta_ir::emit_spirv::emit(&kernel).expect("SPIR-V emission must succeed");

    // OpAtomicIAdd = 234. Scan the word stream for it.
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut found_atomic = false;
    let mut i = 5; // skip header
    while i < words.len() {
        let opcode = (words[i] & 0xFFFF) as u16;
        let len = (words[i] >> 16) as usize;
        if opcode == 234 {
            found_atomic = true;
            // OpAtomicIAdd: result_type, result, pointer, scope, semantics, value
            // Scope operand is an id of a constant — we can't decode
            // it positionally without a full parse; presence is the
            // emission-level assert, spirv-val checks the rest.
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    assert!(found_atomic, "emitted SPIR-V must contain OpAtomicIAdd");

    let dir = std::env::temp_dir().join("quanta_shared_atomics_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shared_atomic_hist.spv");
    std::fs::write(&path, &spv).unwrap();
    if let Some(out) = run_validator("spirv-val", &[path.to_str().unwrap()]) {
        assert!(
            out.status.success(),
            "spirv-val rejected the module:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn wgsl_emits_atomic_decoration_and_validates() {
    let kernel = shared_atomic_kernel();
    let wgsl = quanta_ir::emit_wgsl::emit(&kernel).expect("WGSL emission must succeed");

    // The slot touched by SharedAtomicOp must be atomic<u32>, and
    // every access must go through the atomic builtins.
    assert!(
        wgsl.contains("var<workgroup> shared_0: array<atomic<u32>, 256>;"),
        "shared slot must be declared atomic<u32>:\n{wgsl}"
    );
    assert!(wgsl.contains("atomicAdd(&shared_0["), "missing atomicAdd:\n{wgsl}");
    assert!(
        wgsl.contains("atomicStore(&shared_0["),
        "plain store on an atomic slot must become atomicStore:\n{wgsl}"
    );
    assert!(
        wgsl.contains("atomicLoad(&shared_0["),
        "plain load on an atomic slot must become atomicLoad:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("unsupported"),
        "no unsupported markers expected:\n{wgsl}"
    );

    let dir = std::env::temp_dir().join("quanta_shared_atomics_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shared_atomic_hist.wgsl");
    std::fs::write(&path, &wgsl).unwrap();
    if let Some(out) = run_validator("naga", &[path.to_str().unwrap()]) {
        assert!(
            out.status.success(),
            "naga rejected the module:\n{}\n--- wgsl ---\n{wgsl}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn wgsl_non_atomic_slots_stay_plain() {
    // A kernel with a shared array but NO SharedAtomicOp must keep
    // the plain declaration and plain accesses — the atomic tagging
    // must not leak onto untouched slots.
    let mut kernel = shared_atomic_kernel();
    kernel.body.retain(|op| !matches!(op, KernelOp::SharedAtomicOp { .. }));
    let wgsl = quanta_ir::emit_wgsl::emit(&kernel).expect("WGSL emission must succeed");
    assert!(
        wgsl.contains("var<workgroup> shared_0: array<u32, 256>;"),
        "untouched slot must stay plain:\n{wgsl}"
    );
    assert!(!wgsl.contains("atomicLoad"), "no atomicLoad expected:\n{wgsl}");
}
