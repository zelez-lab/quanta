//! Property-based wire format fuzzing for the Quanta crate.
//!
//! Tests that the public `quanta_ir` serialization layer handles all valid
//! enum tag ranges correctly and never panics on arbitrary byte input.

use proptest::prelude::*;
use quanta_ir::*;

// ── ScalarType tag roundtrip ────────────────────────────────────────────────

proptest! {
    #[test]
    fn scalar_type_roundtrip(tag in 0u8..12) {
        let ty = scalar_type_from_tag(tag);
        let k = KernelDef {
            name: String::from("s"),
            params: vec![KernelParam::FieldRead {
                name: String::from("x"),
                slot: 0,
                scalar_type: ty,
            }],
            body: Vec::new(),
            body_source: None,
            next_reg: 0,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        let rt_ty = param_scalar_type(&k2.params[0]);
        assert_eq!(rt_ty, ty);
    }
}

// ── BinOp tag roundtrip ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn binop_roundtrip(tag in 0u8..12) {
        let op = binop_from_tag(tag);
        let k = KernelDef {
            name: String::from("b"),
            params: Vec::new(),
            body: vec![KernelOp::BinOp {
                dst: Reg(0),
                a: Reg(1),
                b: Reg(2),
                op,
                ty: ScalarType::U32,
            }],
            body_source: None,
            next_reg: 3,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::BinOp { op: rt_op, .. } = &k2.body[0] {
            assert_eq!(*rt_op, op);
        } else {
            panic!("expected BinOp");
        }
    }
}

// ── CmpOp tag roundtrip ────────────────────────────────────────────────────

proptest! {
    #[test]
    fn cmpop_roundtrip(tag in 0u8..6) {
        let op = cmpop_from_tag(tag);
        let k = KernelDef {
            name: String::from("c"),
            params: Vec::new(),
            body: vec![KernelOp::Cmp {
                dst: Reg(0),
                a: Reg(1),
                b: Reg(2),
                op,
                ty: ScalarType::I32,
            }],
            body_source: None,
            next_reg: 3,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::Cmp { op: rt_op, .. } = &k2.body[0] {
            assert_eq!(*rt_op, op);
        } else {
            panic!("expected Cmp");
        }
    }
}

// ── UnaryOp tag roundtrip ───────────────────────────────────────────────────

proptest! {
    #[test]
    fn unaryop_roundtrip(tag in 0u8..3) {
        let op = unaryop_from_tag(tag);
        let k = KernelDef {
            name: String::from("u"),
            params: Vec::new(),
            body: vec![KernelOp::UnaryOp {
                dst: Reg(0),
                a: Reg(1),
                op,
                ty: ScalarType::F32,
            }],
            body_source: None,
            next_reg: 2,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::UnaryOp { op: rt_op, .. } = &k2.body[0] {
            assert_eq!(*rt_op, op);
        } else {
            panic!("expected UnaryOp");
        }
    }
}

// ── AtomicOp tag roundtrip ──────────────────────────────────────────────────

proptest! {
    #[test]
    fn atomicop_roundtrip(tag in 0u8..9) {
        let op = atomicop_from_tag(tag);
        let k = KernelDef {
            name: String::from("a"),
            params: Vec::new(),
            body: vec![KernelOp::AtomicOp {
                dst: Reg(0),
                field: 0,
                index: Reg(1),
                val: Reg(2),
                op,
                ty: ScalarType::U32,
                order: quanta_ir::MemoryOrder::SeqCst,
            }],
            body_source: None,
            next_reg: 3,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::AtomicOp { op: rt_op, .. } = &k2.body[0] {
            assert_eq!(*rt_op, op);
        } else {
            panic!("expected AtomicOp");
        }
    }
}

// ── MathFn tag roundtrip ────────────────────────────────────────────────────

proptest! {
    #[test]
    fn mathfn_roundtrip(tag in 0u8..22) {
        let func = mathfn_from_tag(tag);
        let k = KernelDef {
            name: String::from("m"),
            params: Vec::new(),
            body: vec![KernelOp::MathCall {
                dst: Reg(0),
                func,
                args: vec![Reg(1), Reg(2)],
                ty: ScalarType::F32,
            }],
            body_source: None,
            next_reg: 3,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::MathCall { func: rt_func, .. } = &k2.body[0] {
            assert_eq!(*rt_func, func);
        } else {
            panic!("expected MathCall");
        }
    }
}

// ── KernelDef with random fields ────────────────────────────────────────────

proptest! {
    #[test]
    fn kernel_def_roundtrip(
        name in "[a-z]{1,8}",
        opt_level in 0u8..4,
        wg_x in 1u32..1024,
        subgroup in proptest::option::of(prop_oneof![Just(8u32), Just(16), Just(32), Just(64)]),
        dyn_shared in 0u32..65536,
    ) {
        let k = KernelDef {
            name: name.clone(),
            params: Vec::new(),
            body: Vec::new(),
            body_source: None,
            next_reg: 0,
            opt_level,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [wg_x, 1, 1],
            subgroup_size: subgroup,
            dynamic_shared_bytes: dyn_shared,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.name, name);
        assert_eq!(k2.opt_level, opt_level);
        assert_eq!(k2.workgroup_size[0], wg_x);
        assert_eq!(k2.subgroup_size, subgroup);
        assert_eq!(k2.dynamic_shared_bytes, dyn_shared);
    }
}

// ── ConstValue::F32 roundtrip (NaN, inf, denormal) ──────────────────────────

proptest! {
    #[test]
    fn const_value_f32_roundtrip(bits in any::<u32>()) {
        let val = f32::from_bits(bits);
        let k = KernelDef {
            name: String::from("c"),
            params: Vec::new(),
            body: vec![KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::F32(val),
            }],
            body_source: None,
            next_reg: 1,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::Const { value: ConstValue::F32(v), .. } = &k2.body[0] {
            // Compare bits, not float equality (NaN != NaN)
            assert_eq!(v.to_bits(), bits);
        } else {
            panic!("expected Const F32");
        }
    }
}

// ── Arbitrary bytes must not panic ──────────────────────────────────────────

proptest! {
    #[test]
    fn arbitrary_bytes_dont_panic(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = quanta_ir::deserialize_kernel(&data);
    }

    #[test]
    fn arbitrary_bytes_dont_panic_output(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = quanta_ir::deserialize_output(&data);
    }

    #[test]
    fn arbitrary_bytes_dont_panic_shader(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = quanta_ir::deserialize_shader(&data);
    }

    #[test]
    fn arbitrary_bytes_dont_panic_shader_output(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = quanta_ir::deserialize_shader_output(&data);
    }
}

// ── ShaderOutput roundtrip ──────────────────────────────────────────────────

proptest! {
    #[test]
    fn shader_output_roundtrip(
        has_spirv in any::<bool>(),
        has_metallib in any::<bool>(),
        has_wgsl in any::<bool>(),
    ) {
        let o = ShaderOutput {
            spirv: if has_spirv { Some(vec![0x03, 0x02]) } else { None },
            metallib: if has_metallib { Some(vec![0x4D]) } else { None },
            wgsl: if has_wgsl { Some(String::from("fn main() {}")) } else { None },
        };
        let bytes = serialize_shader_output(&o);
        let o2 = deserialize_shader_output(&bytes).unwrap();
        assert_eq!(o2.spirv.is_some(), has_spirv);
        assert_eq!(o2.metallib.is_some(), has_metallib);
        assert_eq!(o2.wgsl.is_some(), has_wgsl);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn scalar_type_from_tag(tag: u8) -> ScalarType {
    match tag {
        0 => ScalarType::F16,
        1 => ScalarType::F32,
        2 => ScalarType::F64,
        3 => ScalarType::U8,
        4 => ScalarType::U16,
        5 => ScalarType::U32,
        6 => ScalarType::U64,
        7 => ScalarType::I8,
        8 => ScalarType::I16,
        9 => ScalarType::I32,
        10 => ScalarType::I64,
        11 => ScalarType::Bool,
        _ => unreachable!(),
    }
}

fn binop_from_tag(tag: u8) -> BinOp {
    match tag {
        0 => BinOp::Add,
        1 => BinOp::Sub,
        2 => BinOp::Mul,
        3 => BinOp::Div,
        4 => BinOp::Rem,
        5 => BinOp::BitAnd,
        6 => BinOp::BitOr,
        7 => BinOp::BitXor,
        8 => BinOp::Shl,
        9 => BinOp::Shr,
        10 => BinOp::SatAdd,
        11 => BinOp::SatSub,
        _ => unreachable!(),
    }
}

fn cmpop_from_tag(tag: u8) -> CmpOp {
    match tag {
        0 => CmpOp::Eq,
        1 => CmpOp::Ne,
        2 => CmpOp::Lt,
        3 => CmpOp::Le,
        4 => CmpOp::Gt,
        5 => CmpOp::Ge,
        _ => unreachable!(),
    }
}

fn unaryop_from_tag(tag: u8) -> UnaryOp {
    match tag {
        0 => UnaryOp::Neg,
        1 => UnaryOp::BitNot,
        2 => UnaryOp::LogicalNot,
        _ => unreachable!(),
    }
}

fn atomicop_from_tag(tag: u8) -> AtomicOp {
    match tag {
        0 => AtomicOp::Add,
        1 => AtomicOp::Sub,
        2 => AtomicOp::Min,
        3 => AtomicOp::Max,
        4 => AtomicOp::And,
        5 => AtomicOp::Or,
        6 => AtomicOp::Xor,
        7 => AtomicOp::Exchange,
        8 => AtomicOp::CompareExchange,
        _ => unreachable!(),
    }
}

fn mathfn_from_tag(tag: u8) -> MathFn {
    match tag {
        0 => MathFn::Sin,
        1 => MathFn::Cos,
        2 => MathFn::Tan,
        3 => MathFn::Asin,
        4 => MathFn::Acos,
        5 => MathFn::Atan,
        6 => MathFn::Atan2,
        7 => MathFn::Sqrt,
        8 => MathFn::Rsqrt,
        9 => MathFn::Exp,
        10 => MathFn::Exp2,
        11 => MathFn::Log,
        12 => MathFn::Log2,
        13 => MathFn::Pow,
        14 => MathFn::Abs,
        15 => MathFn::Min,
        16 => MathFn::Max,
        17 => MathFn::Clamp,
        18 => MathFn::Floor,
        19 => MathFn::Ceil,
        20 => MathFn::Round,
        21 => MathFn::Fma,
        _ => unreachable!(),
    }
}

fn param_scalar_type(p: &KernelParam) -> ScalarType {
    match p {
        KernelParam::FieldRead { scalar_type, .. }
        | KernelParam::FieldWrite { scalar_type, .. }
        | KernelParam::Constant { scalar_type, .. }
        | KernelParam::Texture2DRead { scalar_type, .. }
        | KernelParam::Texture2DWrite { scalar_type, .. }
        | KernelParam::Texture3DRead { scalar_type, .. } => *scalar_type,
    }
}
