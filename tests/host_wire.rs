//! Tier 1 (host, no GPU) conformance tests — IR wire format.
//!
//! Roundtrip all KernelOp variants, KernelDef with device_sources,
//! CompilerOutput with all fields, edge cases, and invalid inputs.
//!
//! Run: cargo test --test host_wire

use quanta_ir::*;

// ---------------------------------------------------------------------------
// Helper: create a minimal KernelDef wrapping a body
// ---------------------------------------------------------------------------

fn wrap_ops(ops: Vec<KernelOp>) -> KernelDef {
    KernelDef {
        name: String::from("test"),
        params: Vec::new(),
        body: ops,
        body_source: None,
        next_reg: 100,
        opt_level: 3,
        device_sources: Vec::new(),
    }
}

fn roundtrip_ops(ops: Vec<KernelOp>) {
    let k = wrap_ops(ops.clone());
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), ops.len());
    assert_eq!(k2.name, "test");
    assert_eq!(k2.next_reg, 100);
}

// ===========================================================================
// Test: roundtrip ALL 36 KernelOp variants
// ===========================================================================

#[test]
fn roundtrip_all_kernel_op_variants() {
    let ops = vec![
        // Memory (5)
        KernelOp::Load {
            dst: Reg(0),
            field: 0,
            index: Reg(1),
            ty: ScalarType::F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::U32,
        },
        KernelOp::SharedDecl {
            id: 0,
            ty: ScalarType::F32,
            count: 256,
        },
        KernelOp::SharedLoad {
            dst: Reg(3),
            id: 0,
            index: Reg(4),
            ty: ScalarType::F32,
        },
        KernelOp::SharedStore {
            id: 0,
            index: Reg(5),
            src: Reg(6),
            ty: ScalarType::F32,
        },
        // Arithmetic (3)
        KernelOp::BinOp {
            dst: Reg(7),
            a: Reg(8),
            b: Reg(9),
            op: BinOp::Add,
            ty: ScalarType::F32,
        },
        KernelOp::UnaryOp {
            dst: Reg(10),
            a: Reg(11),
            op: UnaryOp::Neg,
            ty: ScalarType::I32,
        },
        KernelOp::Cmp {
            dst: Reg(12),
            a: Reg(13),
            b: Reg(14),
            op: CmpOp::Lt,
            ty: ScalarType::F64,
        },
        // Control flow (4)
        KernelOp::Branch {
            cond: Reg(15),
            then_ops: vec![KernelOp::Barrier],
            else_ops: vec![KernelOp::Break],
        },
        KernelOp::Loop {
            count: Reg(16),
            iter_reg: Reg(17),
            body: vec![KernelOp::Barrier],
        },
        KernelOp::Break,
        KernelOp::Dispatch {
            wave: Reg(18),
            groups: [Reg(19), Reg(20), Reg(21)],
        },
        // Math (1)
        KernelOp::MathCall {
            dst: Reg(22),
            func: MathFn::Sin,
            args: vec![Reg(23)],
            ty: ScalarType::F32,
        },
        // Thread indexing (5)
        KernelOp::QuarkId { dst: Reg(24) },
        KernelOp::QuarkCount { dst: Reg(25) },
        KernelOp::LocalId { dst: Reg(26) },
        KernelOp::GroupId { dst: Reg(27) },
        KernelOp::GroupSize { dst: Reg(28) },
        // Synchronization (3)
        KernelOp::Barrier,
        KernelOp::AtomicOp {
            dst: Reg(29),
            field: 0,
            index: Reg(30),
            val: Reg(31),
            op: AtomicOp::Add,
            ty: ScalarType::U32,
        },
        KernelOp::AtomicCas {
            dst: Reg(32),
            field: 0,
            index: Reg(33),
            expected: Reg(34),
            desired: Reg(35),
            ty: ScalarType::U32,
        },
        // Warp/wave (4)
        KernelOp::WaveShuffle {
            dst: Reg(36),
            src: Reg(37),
            lane_delta: Reg(38),
            ty: ScalarType::F32,
        },
        KernelOp::WaveBallot {
            dst: Reg(39),
            predicate: Reg(40),
        },
        KernelOp::WaveAny {
            dst: Reg(41),
            predicate: Reg(42),
        },
        KernelOp::WaveAll {
            dst: Reg(43),
            predicate: Reg(44),
        },
        // Type conversion (2)
        KernelOp::Cast {
            dst: Reg(45),
            src: Reg(46),
            from: ScalarType::F32,
            to: ScalarType::I32,
        },
        KernelOp::Const {
            dst: Reg(47),
            value: ConstValue::F32(42.0),
        },
        // Vector (3)
        KernelOp::VecConstruct {
            dst: Reg(48),
            components: vec![Reg(49), Reg(50), Reg(51)],
            ty: ScalarType::F32,
        },
        KernelOp::VecExtract {
            dst: Reg(52),
            vec: Reg(53),
            component: 2,
            ty: ScalarType::F32,
        },
        KernelOp::MatMul {
            dst: Reg(54),
            a: Reg(55),
            b: Reg(56),
            size: 4,
            ty: ScalarType::F32,
        },
        // Texture (4)
        KernelOp::TextureSample2D {
            dst: Reg(57),
            texture: 0,
            x: Reg(58),
            y: Reg(59),
            ty: ScalarType::F32,
        },
        KernelOp::TextureSample3D {
            dst: Reg(60),
            texture: 1,
            x: Reg(61),
            y: Reg(62),
            z: Reg(63),
            ty: ScalarType::F32,
        },
        KernelOp::TextureWrite2D {
            texture: 2,
            x: Reg(64),
            y: Reg(65),
            value: Reg(66),
            ty: ScalarType::F32,
        },
        KernelOp::TextureSize {
            dst_w: Reg(67),
            dst_h: Reg(68),
            texture: 0,
        },
        // Register copy (1)
        KernelOp::Copy {
            dst: Reg(69),
            src: Reg(70),
            ty: ScalarType::F32,
        },
        // Device function call (1)
        KernelOp::DeviceCall {
            dst: Reg(71),
            func_name: String::from("helper"),
            args: vec![Reg(72), Reg(73)],
            ty: ScalarType::F32,
        },
    ];

    // Verify we have all 36 variant types
    assert_eq!(ops.len(), 36, "Must test exactly 36 KernelOp variants");
    roundtrip_ops(ops);
}

// ===========================================================================
// Test: KernelDef with device_sources
// ===========================================================================

#[test]
fn roundtrip_kernel_def_with_device_sources() {
    let k = KernelDef {
        name: String::from("kernel_with_helpers"),
        params: vec![
            KernelParam::FieldRead {
                name: String::from("input"),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: String::from("output"),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::Constant {
                name: String::from("scale"),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::DeviceCall {
                dst: Reg(2),
                func_name: String::from("activate"),
                args: vec![Reg(1)],
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::F32,
            },
        ],
        body_source: Some(String::from(
            "let i = quark_id(); output[i] = activate(input[i]);",
        )),
        next_reg: 3,
        opt_level: 2,
        device_sources: vec![
            String::from("fn activate(x: f32) -> f32 { if x > 0.0 { x } else { x * 0.01 } }"),
            String::from("fn clamp_val(x: f32, lo: f32, hi: f32) -> f32 { x.max(lo).min(hi) }"),
            String::from("fn identity(x: f32) -> f32 { x }"),
        ],
    };

    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();

    assert_eq!(k2.name, "kernel_with_helpers");
    assert_eq!(k2.params.len(), 3);
    assert_eq!(k2.body.len(), 4);
    assert_eq!(
        k2.body_source,
        Some(String::from(
            "let i = quark_id(); output[i] = activate(input[i]);"
        ))
    );
    assert_eq!(k2.next_reg, 3);
    assert_eq!(k2.opt_level, 2);
    assert_eq!(k2.device_sources.len(), 3);
    assert!(k2.device_sources[0].contains("activate"));
    assert!(k2.device_sources[1].contains("clamp_val"));
    assert!(k2.device_sources[2].contains("identity"));
}

// ===========================================================================
// Test: CompilerOutput with all fields populated
// ===========================================================================

#[test]
fn roundtrip_compiler_output_all_fields() {
    let o = CompilerOutput {
        amd: Some(vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03]),
        nvidia: Some(vec![0x7F, 0x45, 0x4C, 0x46]),
        spirv: Some(vec![0x03, 0x02, 0x23, 0x07, 0x00, 0x01, 0x00, 0x00]),
        metallib: Some(vec![0x4D, 0x54, 0x4C, 0x42, 0x01, 0x00]),
    };

    let bytes = serialize_output(&o);
    let o2 = deserialize_output(&bytes).unwrap();

    assert_eq!(o2.amd, o.amd);
    assert_eq!(o2.nvidia, o.nvidia);
    assert_eq!(o2.spirv, o.spirv);
    assert_eq!(o2.metallib, o.metallib);
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn roundtrip_empty_body() {
    let k = KernelDef {
        name: String::from("empty"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 0,
        device_sources: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert!(k2.body.is_empty());
    assert!(k2.params.is_empty());
}

#[test]
fn roundtrip_huge_body_1000_ops() {
    let ops: Vec<KernelOp> = (0..1000)
        .map(|i| KernelOp::Const {
            dst: Reg(i),
            value: ConstValue::F32(i as f32),
        })
        .collect();

    let k = wrap_ops(ops);
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 1000);
}

#[test]
fn roundtrip_deeply_nested_branch_loop_5_levels() {
    fn make_nested(depth: u32) -> KernelOp {
        if depth == 0 {
            KernelOp::Barrier
        } else {
            KernelOp::Branch {
                cond: Reg(depth),
                then_ops: vec![KernelOp::Loop {
                    count: Reg(depth + 100),
                    iter_reg: Reg(depth + 200),
                    body: vec![make_nested(depth - 1)],
                }],
                else_ops: vec![make_nested(depth - 1)],
            }
        }
    }

    let op = make_nested(5);
    let k = wrap_ops(vec![op]);
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 1);

    // Verify depth by walking the structure
    fn verify_depth(op: &KernelOp, expected: u32) {
        match op {
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                assert!(expected > 0, "unexpected nesting beyond expected depth");
                if let KernelOp::Loop { body, .. } = &then_ops[0] {
                    verify_depth(&body[0], expected - 1);
                }
                verify_depth(&else_ops[0], expected - 1);
            }
            KernelOp::Barrier => {
                assert_eq!(expected, 0, "expected leaf at depth 0");
            }
            _ => panic!("unexpected op type"),
        }
    }
    verify_depth(&k2.body[0], 5);
}

// ===========================================================================
// Invalid input rejection
// ===========================================================================

#[test]
fn invalid_truncated_bytes() {
    // Single byte cannot be a valid KernelDef
    assert!(deserialize_kernel(&[0x01]).is_err());
    assert!(deserialize_kernel(&[]).is_err());
    assert!(deserialize_kernel(&[0x00, 0x00]).is_err());
}

#[test]
fn invalid_random_bytes() {
    // Random-looking data should not parse cleanly
    let garbage: Vec<u8> = (0..64).map(|i| (i * 37 + 13) as u8).collect();
    assert!(deserialize_kernel(&garbage).is_err());
}

#[test]
fn invalid_truncated_compiler_output() {
    assert!(deserialize_output(&[]).is_err());
    assert!(deserialize_output(&[0x01, 0x02, 0x03]).is_err());
}

#[test]
fn invalid_trailing_bytes_kernel() {
    let k = KernelDef {
        name: String::from("x"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 0,
        device_sources: Vec::new(),
    };
    let mut bytes = serialize_kernel(&k);
    bytes.push(0xFF);
    assert_eq!(
        deserialize_kernel(&bytes).unwrap_err(),
        "trailing bytes after KernelDef"
    );
}

#[test]
fn invalid_trailing_bytes_output() {
    let o = CompilerOutput {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
    };
    let mut bytes = serialize_output(&o);
    bytes.push(0xAB);
    assert_eq!(
        deserialize_output(&bytes).unwrap_err(),
        "trailing bytes after CompilerOutput"
    );
}

// ===========================================================================
// All BinOp variants roundtrip
// ===========================================================================

#[test]
fn roundtrip_all_binop_variants() {
    let bin_ops = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
    ];
    let ops: Vec<KernelOp> = bin_ops
        .iter()
        .enumerate()
        .map(|(i, op)| KernelOp::BinOp {
            dst: Reg(i as u32 * 3),
            a: Reg(i as u32 * 3 + 1),
            b: Reg(i as u32 * 3 + 2),
            op: *op,
            ty: ScalarType::U32,
        })
        .collect();
    roundtrip_ops(ops);
}

// ===========================================================================
// All AtomicOp variants roundtrip
// ===========================================================================

#[test]
fn roundtrip_all_atomic_op_variants() {
    let atomic_ops = [
        AtomicOp::Add,
        AtomicOp::Sub,
        AtomicOp::Min,
        AtomicOp::Max,
        AtomicOp::And,
        AtomicOp::Or,
        AtomicOp::Xor,
        AtomicOp::Exchange,
        AtomicOp::CompareExchange,
    ];
    let ops: Vec<KernelOp> = atomic_ops
        .iter()
        .enumerate()
        .map(|(i, op)| KernelOp::AtomicOp {
            dst: Reg(i as u32),
            field: 0,
            index: Reg(100),
            val: Reg(101),
            op: *op,
            ty: ScalarType::U32,
        })
        .collect();
    roundtrip_ops(ops);
}

// ===========================================================================
// All MathFn variants roundtrip
// ===========================================================================

#[test]
fn roundtrip_all_math_fn_variants() {
    let math_fns = [
        MathFn::Sin,
        MathFn::Cos,
        MathFn::Tan,
        MathFn::Asin,
        MathFn::Acos,
        MathFn::Atan,
        MathFn::Atan2,
        MathFn::Sqrt,
        MathFn::Rsqrt,
        MathFn::Exp,
        MathFn::Exp2,
        MathFn::Log,
        MathFn::Log2,
        MathFn::Pow,
        MathFn::Abs,
        MathFn::Min,
        MathFn::Max,
        MathFn::Clamp,
        MathFn::Floor,
        MathFn::Ceil,
        MathFn::Round,
        MathFn::Fma,
    ];
    let ops: Vec<KernelOp> = math_fns
        .iter()
        .enumerate()
        .map(|(i, func)| KernelOp::MathCall {
            dst: Reg(i as u32),
            func: *func,
            args: vec![Reg(200), Reg(201)],
            ty: ScalarType::F32,
        })
        .collect();
    assert_eq!(ops.len(), 22);
    roundtrip_ops(ops);
}

// ===========================================================================
// All UnaryOp variants roundtrip
// ===========================================================================

#[test]
fn roundtrip_all_unary_op_variants() {
    let unary_ops = [UnaryOp::Neg, UnaryOp::BitNot, UnaryOp::LogicalNot];
    let ops: Vec<KernelOp> = unary_ops
        .iter()
        .enumerate()
        .map(|(i, op)| KernelOp::UnaryOp {
            dst: Reg(i as u32),
            a: Reg(100),
            op: *op,
            ty: ScalarType::I32,
        })
        .collect();
    roundtrip_ops(ops);
}

// ===========================================================================
// All CmpOp variants roundtrip
// ===========================================================================

#[test]
fn roundtrip_all_cmp_op_variants() {
    let cmp_ops = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ];
    let ops: Vec<KernelOp> = cmp_ops
        .iter()
        .enumerate()
        .map(|(i, op)| KernelOp::Cmp {
            dst: Reg(i as u32),
            a: Reg(100),
            b: Reg(101),
            op: *op,
            ty: ScalarType::F32,
        })
        .collect();
    roundtrip_ops(ops);
}

// ===========================================================================
// All KernelParam variants roundtrip
// ===========================================================================

#[test]
fn roundtrip_all_kernel_param_variants() {
    let k = KernelDef {
        name: String::from("all_params"),
        params: vec![
            KernelParam::FieldRead {
                name: String::from("read_field"),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: String::from("write_field"),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
            KernelParam::Constant {
                name: String::from("constant"),
                slot: 2,
                scalar_type: ScalarType::I32,
            },
            KernelParam::Texture2DRead {
                name: String::from("tex2d_r"),
                slot: 3,
                scalar_type: ScalarType::F32,
            },
            KernelParam::Texture2DWrite {
                name: String::from("tex2d_w"),
                slot: 4,
                scalar_type: ScalarType::F32,
            },
            KernelParam::Texture3DRead {
                name: String::from("tex3d_r"),
                slot: 5,
                scalar_type: ScalarType::F16,
            },
        ],
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 1,
        device_sources: Vec::new(),
    };

    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.params.len(), 6);
}

// ===========================================================================
// All ScalarType variants roundtrip
// ===========================================================================

#[test]
fn roundtrip_all_scalar_types_in_load() {
    let types = [
        ScalarType::F16,
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::U8,
        ScalarType::U16,
        ScalarType::U32,
        ScalarType::U64,
        ScalarType::I8,
        ScalarType::I16,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::Bool,
    ];
    let ops: Vec<KernelOp> = types
        .iter()
        .enumerate()
        .map(|(i, ty)| KernelOp::Load {
            dst: Reg(i as u32),
            field: 0,
            index: Reg(100),
            ty: *ty,
        })
        .collect();
    assert_eq!(ops.len(), 12);
    roundtrip_ops(ops);
}
