//! Property-based tests for quanta-ir types using proptest.
//!
//! Tests wire-format roundtrips for all IR enum types, KernelDef serialization
//! with random fields, and robustness against arbitrary byte inputs.

use proptest::prelude::*;
use quanta_ir::*;

// ── Scalar type tag roundtrip ───────────────────────────────────────────────

proptest! {
    /// Every valid ScalarType tag (0..12) should roundtrip through
    /// serialize_kernel / deserialize_kernel when embedded in a KernelParam.
    #[test]
    fn scalar_type_roundtrip(tag in 0u8..12) {
        let ty = scalar_type_from_tag(tag);
        let k = minimal_kernel_with_param(ty);
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.params.len(), 1);
        // The scalar type survives the roundtrip.
        let rt_ty = param_scalar_type(&k2.params[0]);
        assert_eq!(rt_ty, ty);
    }
}

// ── BinOp tag roundtrip ─────────────────────────────────────────────────────

proptest! {
    /// Every valid BinOp tag (0..14) should roundtrip when embedded in a
    /// KernelOp::BinOp inside a KernelDef.
    #[test]
    fn binop_roundtrip(tag in 0u8..14) {
        let op = binop_from_tag(tag);
        let k = KernelDef {
            name: String::from("binop_rt"),
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
        assert_eq!(k2.body.len(), 1);
        if let KernelOp::BinOp { op: rt_op, .. } = &k2.body[0] {
            assert_eq!(*rt_op, op);
        } else {
            panic!("expected BinOp");
        }
    }
}

// ── KernelDef roundtrip with random fields ──────────────────────────────────

proptest! {
    /// A KernelDef with random name, opt_level, and workgroup size should
    /// survive a serialize/deserialize roundtrip with all fields intact.
    #[test]
    fn kernel_def_roundtrip(
        name in "[a-z]{1,8}",
        opt_level in 0u8..4,
        wg_x in 1u32..1024,
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
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.name, name);
        assert_eq!(k2.opt_level, opt_level);
        assert_eq!(k2.workgroup_size[0], wg_x);
    }
}

// ── ConstValue::F32 roundtrip (covers NaN, inf, denormal) ───────────────────

proptest! {
    /// Any f32 bit pattern should survive a ConstValue roundtrip through
    /// the wire format. We compare bits, not float equality, to handle NaN.
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
            assert_eq!(v.to_bits(), bits);
        } else {
            panic!("expected Const F32");
        }
    }
}

// ── ConstValue::F64 roundtrip ───────────────────────────────────────────────

proptest! {
    #[test]
    fn const_value_f64_roundtrip(bits in any::<u64>()) {
        let val = f64::from_bits(bits);
        let k = KernelDef {
            name: String::from("c"),
            params: Vec::new(),
            body: vec![KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::F64(val),
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
        if let KernelOp::Const { value: ConstValue::F64(v), .. } = &k2.body[0] {
            assert_eq!(v.to_bits(), bits);
        } else {
            panic!("expected Const F64");
        }
    }
}

// ── ConstValue integer roundtrips ───────────────────────────────────────────

proptest! {
    #[test]
    fn const_value_u32_roundtrip(val in any::<u32>()) {
        let k = kernel_with_const(ConstValue::U32(val));
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::Const { value: ConstValue::U32(v), .. } = &k2.body[0] {
            assert_eq!(*v, val);
        } else {
            panic!("expected Const U32");
        }
    }

    #[test]
    fn const_value_i32_roundtrip(val in any::<i32>()) {
        let k = kernel_with_const(ConstValue::I32(val));
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::Const { value: ConstValue::I32(v), .. } = &k2.body[0] {
            assert_eq!(*v, val);
        } else {
            panic!("expected Const I32");
        }
    }

    #[test]
    fn const_value_u64_roundtrip(val in any::<u64>()) {
        let k = kernel_with_const(ConstValue::U64(val));
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::Const { value: ConstValue::U64(v), .. } = &k2.body[0] {
            assert_eq!(*v, val);
        } else {
            panic!("expected Const U64");
        }
    }

    #[test]
    fn const_value_i64_roundtrip(val in any::<i64>()) {
        let k = kernel_with_const(ConstValue::I64(val));
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        if let KernelOp::Const { value: ConstValue::I64(v), .. } = &k2.body[0] {
            assert_eq!(*v, val);
        } else {
            panic!("expected Const I64");
        }
    }
}

// ── Arbitrary bytes must not panic ──────────────────────────────────────────

proptest! {
    /// Feeding random bytes to the deserializer must never panic.
    /// Returning Err is fine — crashing is not.
    #[test]
    fn arbitrary_bytes_dont_panic_kernel(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = deserialize_kernel(&data);
    }

    #[test]
    fn arbitrary_bytes_dont_panic_output(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = deserialize_output(&data);
    }

    #[test]
    fn arbitrary_bytes_dont_panic_shader(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = deserialize_shader(&data);
    }

    #[test]
    fn arbitrary_bytes_dont_panic_shader_output(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = deserialize_shader_output(&data);
    }
}

// ── CompilerOutput roundtrip ────────────────────────────────────────────────

proptest! {
    #[test]
    fn compiler_output_roundtrip(
        has_spirv in any::<bool>(),
        has_metallib in any::<bool>(),
        has_metallib_ios in any::<bool>(),
        has_metallib_ios_sim in any::<bool>(),
        has_wgsl in any::<bool>(),
    ) {
        let output = CompilerOutput {
            amd: None,
            nvidia: None,
            spirv: if has_spirv { Some(vec![0x03, 0x02, 0x23, 0x07]) } else { None },
            metallib: if has_metallib { Some(vec![0x4D, 0x54]) } else { None },
            metallib_ios: if has_metallib_ios { Some(vec![0x4D, 0x54, 0x01]) } else { None },
            metallib_ios_sim: if has_metallib_ios_sim { Some(vec![0x4D, 0x54, 0x02]) } else { None },
            wgsl: if has_wgsl { Some(String::from("@compute fn main() {}")) } else { None },
        };
        let bytes = serialize_output(&output);
        let o2 = deserialize_output(&bytes).unwrap();
        assert_eq!(o2.spirv.is_some(), has_spirv);
        assert_eq!(o2.metallib.is_some(), has_metallib);
        assert_eq!(o2.metallib_ios.is_some(), has_metallib_ios);
        assert_eq!(o2.metallib_ios_sim.is_some(), has_metallib_ios_sim);
        assert_eq!(o2.wgsl.is_some(), has_wgsl);
    }
}

// ── ShaderDef roundtrip ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn shader_def_roundtrip(
        name in "[a-z]{1,8}",
        stage_tag in 0u8..2,
        ret_tag in 0u8..7,
        // The appended varyings interface: absent, or a struct with 0..4
        // varying fields (each any ShaderType tag) and an optional receiver.
        varyings in proptest::option::of((
            "[A-Z][a-z]{0,6}",
            "[a-z]{1,6}",
            proptest::collection::vec(("[a-z]{1,6}", 0u8..7), 0..4),
            proptest::option::of("[a-z]{1,4}"),
        )),
    ) {
        let stage = if stage_tag == 0 { ShaderStage::Vertex } else { ShaderStage::Fragment };
        let return_type = shader_type_from_tag(ret_tag);
        let varyings = varyings.map(|(struct_name, position, fields, binding)| ShaderVaryings {
            struct_name,
            position,
            fields: fields
                .into_iter()
                .map(|(name, tag)| VaryingField { name, ty: shader_type_from_tag(tag) })
                .collect(),
            binding,
        });
        let shader = ShaderDef {
            name: name.clone(),
            stage,
            params: Vec::new(),
            return_type,
            body_source: String::from("return x;"),
            varyings: varyings.clone(),
        };
        let bytes = serialize_shader(&shader);
        let s2 = deserialize_shader(&bytes).unwrap();
        assert_eq!(s2.name, name);
        assert_eq!(s2.stage, stage);
        assert_eq!(s2.return_type, return_type);
        assert_eq!(s2.varyings, varyings);
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
        12 => BinOp::Rotl,
        13 => BinOp::Rotr,
        _ => unreachable!(),
    }
}

fn shader_type_from_tag(tag: u8) -> ShaderType {
    match tag {
        0 => ShaderType::F32,
        1 => ShaderType::Vec2,
        2 => ShaderType::Vec3,
        3 => ShaderType::Vec4,
        4 => ShaderType::Mat4,
        5 => ShaderType::Mat3,
        6 => ShaderType::U32,
        _ => unreachable!(),
    }
}

fn minimal_kernel_with_param(ty: ScalarType) -> KernelDef {
    KernelDef {
        name: String::from("p"),
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

fn kernel_with_const(cv: ConstValue) -> KernelDef {
    KernelDef {
        name: String::from("c"),
        params: Vec::new(),
        body: vec![KernelOp::Const {
            dst: Reg(0),
            value: cv,
        }],
        body_source: None,
        next_reg: 1,
        opt_level: 0,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}
