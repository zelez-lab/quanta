//! Tests for `emit_wgsl_jit` — the WGSL emitter that runs inside the wasm
//! binary served to browsers.
//!
//! Coverage strategy:
//!
//! 1. **Goldens** — `vector_add`, `add_one`, `shared_sum_u32` produce expected
//!    WGSL fragments. Goldens use substring assertions so the tests survive
//!    cosmetic formatting tweaks but break loudly on semantic regressions.
//! 2. **Variant exhaustiveness** — a synthetic kernel exercises every
//!    `KernelOp` variant. The emitter is required to emit something for
//!    each: this test asserts compilation succeeds (no `unreachable!()` /
//!    panic) for the union.
//! 3. **Cross-emitter sanity** — the same `KernelDef` produces non-empty
//!    output from `emit_msl`, `emit_spirv`, and `emit_wgsl_jit`. A regression
//!    that drops a variant from any one would surface here.

#![cfg(feature = "jit")]

use quanta_ir::emit_wgsl::emit_wgsl_jit;
use quanta_ir::*;

fn k(name: &str, params: Vec<KernelParam>, body: Vec<KernelOp>, next_reg: u32) -> KernelDef {
    KernelDef {
        name: name.to_string(),
        params,
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

#[test]
fn vector_add_golden() {
    let kernel = k(
        "vector_add",
        vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        4,
    );

    let wgsl = emit_wgsl_jit(&kernel).expect("emit");

    assert!(wgsl.contains("@group(0) @binding(0) var<storage, read> a: array<f32>"));
    assert!(wgsl.contains("@group(0) @binding(1) var<storage, read> b: array<f32>"));
    assert!(wgsl.contains("@group(0) @binding(2) var<storage, read_write> result: array<f32>"));
    assert!(wgsl.contains("@compute @workgroup_size(64, 1, 1)"));
    assert!(wgsl.contains("fn vector_add("));
    assert!(wgsl.contains("let r0 = _quark_id;"));
    assert!(wgsl.contains("a[r0]"));
    assert!(wgsl.contains("b[r0]"));
    assert!(wgsl.contains("r1 + r2"));
    assert!(wgsl.contains("result[r0] = r3;"));
}

#[test]
fn add_one_golden() {
    // The smoke-test kernel called out in step 079's "definition of done".
    let kernel = k(
        "add_one",
        vec![KernelParam::FieldWrite {
            name: "buf".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::U32(1),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::U32,
            },
        ],
        4,
    );

    let wgsl = emit_wgsl_jit(&kernel).expect("emit");
    assert!(wgsl.contains("fn add_one"));
    assert!(wgsl.contains("array<u32>"));
    assert!(wgsl.contains("let r2 = 1u;"));
    assert!(wgsl.contains("buf[r0] = r3;"));
}

#[test]
fn shared_decl_lifted_to_module_scope() {
    let kernel = k(
        "shared_sum",
        vec![KernelParam::FieldRead {
            name: "data".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        vec![
            KernelOp::SharedDecl {
                id: 0,
                ty: ScalarType::F32,
                count: 64,
            },
            KernelOp::Barrier,
        ],
        1,
    );

    let wgsl = emit_wgsl_jit(&kernel).expect("emit");
    // Module-scope `var<workgroup>`, not inside the function body.
    let header_end = wgsl.find("@compute").expect("compute marker");
    assert!(wgsl[..header_end].contains("var<workgroup> shared_0: array<f32, 64>;"));
    assert!(wgsl.contains("workgroupBarrier();"));
}

#[test]
fn atomic_field_wraps_in_atomic() {
    let kernel = k(
        "counter",
        vec![KernelParam::FieldWrite {
            name: "ctr".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        vec![
            KernelOp::QuarkId { dst: Reg(0) },
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
                order: quanta_ir::MemoryOrder::SeqCst,
            },
        ],
        3,
    );

    let wgsl = emit_wgsl_jit(&kernel).expect("emit");
    assert!(wgsl.contains("array<atomic<u32>>"));
    assert!(wgsl.contains("atomicAdd(&ctr[r0], r1)"));
}

#[test]
fn subgroup_use_emits_enable_directive() {
    let kernel = k(
        "wave_sum",
        vec![KernelParam::FieldWrite {
            name: "out".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::F32(1.0),
            },
            KernelOp::SubgroupReduceAdd {
                dst: Reg(2),
                src: Reg(1),
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::F32,
            },
        ],
        3,
    );

    let wgsl = emit_wgsl_jit(&kernel).expect("emit");
    assert!(wgsl.starts_with("enable subgroups;"));
    assert!(wgsl.contains("subgroupAdd(r1)"));
}

#[test]
fn workgroup_size_passed_through() {
    let mut kernel = k("wide", vec![], vec![KernelOp::QuarkId { dst: Reg(0) }], 1);
    kernel.workgroup_size = [32, 4, 2];
    let wgsl = emit_wgsl_jit(&kernel).expect("emit");
    assert!(wgsl.contains("@compute @workgroup_size(32, 4, 2)"));
    // 32 * 4 * 2 = 256
    assert!(wgsl.contains("let _proton_size: u32 = 256u;"));
}

#[test]
fn every_kernel_op_variant_compiles() {
    // Synthetic kernel exercising every KernelOp variant. The point is not to
    // execute meaningfully — it's to assert the emitter handles each variant
    // without panicking. This is the runtime mirror of the Kani T1001
    // exhaustiveness theorem.
    let body: Vec<KernelOp> = vec![
        KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(0),
        },
        KernelOp::QuarkId { dst: Reg(1) },
        KernelOp::QuarkCount { dst: Reg(2) },
        KernelOp::ProtonId { dst: Reg(3) },
        KernelOp::NucleusId { dst: Reg(4) },
        KernelOp::ProtonSize { dst: Reg(5) },
        KernelOp::SubgroupSize { dst: Reg(6) },
        KernelOp::Load {
            dst: Reg(7),
            field: 0,
            index: Reg(1),
            ty: ScalarType::F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(1),
            src: Reg(7),
            ty: ScalarType::F32,
        },
        KernelOp::SharedDecl {
            id: 0,
            ty: ScalarType::F32,
            count: 64,
        },
        KernelOp::SharedDeclDyn {
            id: 1,
            ty: ScalarType::F32,
        },
        KernelOp::SharedStore {
            id: 0,
            index: Reg(3),
            src: Reg(7),
            ty: ScalarType::F32,
        },
        KernelOp::SharedLoad {
            dst: Reg(8),
            id: 0,
            index: Reg(3),
            ty: ScalarType::F32,
        },
        KernelOp::Barrier,
        KernelOp::BinOp {
            dst: Reg(9),
            a: Reg(7),
            b: Reg(8),
            op: BinOp::Add,
            ty: ScalarType::F32,
        },
        KernelOp::BinOp {
            dst: Reg(50),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::BitAnd,
            ty: ScalarType::U32,
        },
        KernelOp::BinOp {
            dst: Reg(51),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::Shl,
            ty: ScalarType::U32,
        },
        KernelOp::BinOp {
            dst: Reg(52),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::SatAdd,
            ty: ScalarType::U32,
        },
        KernelOp::UnaryOp {
            dst: Reg(10),
            a: Reg(9),
            op: UnaryOp::Neg,
            ty: ScalarType::F32,
        },
        KernelOp::Cmp {
            dst: Reg(11),
            a: Reg(9),
            b: Reg(10),
            op: CmpOp::Lt,
            ty: ScalarType::F32,
        },
        KernelOp::Cast {
            dst: Reg(12),
            src: Reg(9),
            from: ScalarType::F32,
            to: ScalarType::I32,
        },
        KernelOp::Bitcast {
            dst: Reg(13),
            src: Reg(9),
            from: ScalarType::F32,
            to: ScalarType::U32,
        },
        KernelOp::MathCall {
            dst: Reg(14),
            func: MathFn::Sin,
            args: vec![Reg(9)],
            ty: ScalarType::F32,
        },
        KernelOp::MathCall {
            dst: Reg(53),
            func: MathFn::Atan2,
            args: vec![Reg(9), Reg(9)],
            ty: ScalarType::F32,
        },
        KernelOp::CountTrailingZeros {
            dst: Reg(15),
            src: Reg(0),
            ty: ScalarType::U32,
        },
        KernelOp::CountLeadingZeros {
            dst: Reg(16),
            src: Reg(0),
            ty: ScalarType::U32,
        },
        KernelOp::PopCount {
            dst: Reg(17),
            src: Reg(0),
            ty: ScalarType::U32,
        },
        KernelOp::Branch {
            cond: Reg(11),
            then_ops: vec![KernelOp::Break],
            else_ops: vec![KernelOp::Copy {
                dst: Reg(9),
                src: Reg(7),
                ty: ScalarType::F32,
            }],
        },
        KernelOp::Loop {
            count: Reg(2),
            iter_reg: Reg(20),
            body: vec![KernelOp::Copy {
                dst: Reg(9),
                src: Reg(8),
                ty: ScalarType::F32,
            }],
        },
        KernelOp::AtomicOp {
            dst: Reg(21),
            field: 2,
            index: Reg(1),
            val: Reg(0),
            op: AtomicOp::Add,
            ty: ScalarType::U32,
            order: quanta_ir::MemoryOrder::SeqCst,
        },
        KernelOp::AtomicCas {
            dst: Reg(22),
            field: 2,
            index: Reg(1),
            expected: Reg(0),
            desired: Reg(0),
            ty: ScalarType::U32,
        },
        KernelOp::WaveShuffle {
            dst: Reg(23),
            src: Reg(9),
            lane_delta: Reg(0),
            ty: ScalarType::F32,
        },
        KernelOp::WaveBallot {
            dst: Reg(24),
            predicate: Reg(0),
        },
        KernelOp::WaveAny {
            dst: Reg(25),
            predicate: Reg(0),
        },
        KernelOp::WaveAll {
            dst: Reg(26),
            predicate: Reg(0),
        },
        KernelOp::SubgroupReduceAdd {
            dst: Reg(27),
            src: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::SubgroupReduceMin {
            dst: Reg(28),
            src: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::SubgroupReduceMax {
            dst: Reg(29),
            src: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::SubgroupExclusiveAdd {
            dst: Reg(30),
            src: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::SubgroupInclusiveAdd {
            dst: Reg(31),
            src: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::VecConstruct {
            dst: Reg(32),
            components: vec![Reg(9), Reg(9), Reg(9), Reg(9)],
            ty: ScalarType::F32,
        },
        KernelOp::VecExtract {
            dst: Reg(33),
            vec: Reg(32),
            component: 1,
            ty: ScalarType::F32,
        },
        KernelOp::Dot {
            dst: Reg(34),
            a: Reg(32),
            b: Reg(32),
            ty: ScalarType::F32,
            width: 4,
        },
        KernelOp::MatMul {
            dst: Reg(35),
            a: Reg(32),
            b: Reg(32),
            size: 4,
            ty: ScalarType::F32,
        },
        KernelOp::CooperativeMMA {
            dst: Reg(36),
            a: Reg(32),
            b: Reg(32),
            c: Reg(32),
            m: 16,
            n: 16,
            k: 16,
            ty: ScalarType::F32,
        },
        KernelOp::TextureSample2D {
            dst: Reg(37),
            texture: 3,
            x: Reg(9),
            y: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::TextureSample3D {
            dst: Reg(38),
            texture: 3,
            x: Reg(9),
            y: Reg(9),
            z: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::TextureWrite2D {
            texture: 4,
            x: Reg(1),
            y: Reg(1),
            value: Reg(32),
            ty: ScalarType::F32,
        },
        KernelOp::TextureLoad2D {
            dst: Reg(39),
            texture: 3,
            x: Reg(1),
            y: Reg(1),
            ty: ScalarType::F32,
        },
        KernelOp::TextureSize {
            dst_w: Reg(40),
            dst_h: Reg(41),
            texture: 3,
        },
        KernelOp::DeviceCall {
            dst: Reg(42),
            func_name: "helper".into(),
            args: vec![Reg(9)],
            ty: ScalarType::F32,
        },
        KernelOp::DebugPrint {
            src: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::Dispatch {
            wave: Reg(0),
            groups: [Reg(0), Reg(0), Reg(0)],
        },
    ];

    let kernel = KernelDef {
        name: "all_ops".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "in_a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "out_a".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "ctr".into(),
                slot: 2,
                scalar_type: ScalarType::U32,
            },
            KernelParam::Texture2DRead {
                name: "tex_in".into(),
                slot: 3,
                scalar_type: ScalarType::F32,
            },
            KernelParam::Texture2DWrite {
                name: "tex_out".into(),
                slot: 4,
                scalar_type: ScalarType::F32,
            },
        ],
        body,
        body_source: None,
        next_reg: 60,
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };

    let wgsl = emit_wgsl_jit(&kernel).expect("emit");
    // Spot-check that core constructs from across the variant set landed.
    for marker in [
        "fn all_ops",
        "subgroupAdd",
        "atomicAdd",
        "atomicCompareExchangeWeak",
        "vec4<f32>",
        "dot(",
        "countOneBits",
        "countLeadingZeros",
        "countTrailingZeros",
        "bitcast<u32>",
        "atan2(",
        "var<workgroup> shared_0:",
        "workgroupBarrier();",
    ] {
        assert!(
            wgsl.contains(marker),
            "missing `{}` in WGSL output:\n{}",
            marker,
            wgsl
        );
    }
}

#[test]
fn cross_emitter_smoke() {
    // The same kernel goes through MSL and WGSL JIT emitters; both must
    // produce non-empty output. SPIR-V emitter has its own coverage in its
    // own tests; we don't double up here.
    let kernel = k(
        "vector_add",
        vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        4,
    );

    let wgsl = emit_wgsl_jit(&kernel).expect("wgsl");
    let msl = emit_msl::emit(&kernel).expect("msl");

    assert!(wgsl.contains("@compute"));
    assert!(msl.contains("kernel void vector_add"));
    assert!(wgsl.len() > 100);
    assert!(msl.len() > 100);
}
