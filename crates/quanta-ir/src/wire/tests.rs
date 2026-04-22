use super::*;
use crate::{BinOp, CompilerOutput, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

#[test]
fn roundtrip_empty_kernel() {
    let k = KernelDef {
        name: String::from("test_kernel"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k.name, k2.name);
    assert!(k2.params.is_empty());
    assert!(k2.body.is_empty());
    assert!(k2.body_source.is_none());
    assert_eq!(k2.next_reg, 0);
    assert_eq!(k2.opt_level, 3);
    assert!(k2.device_sources.is_empty());
}

#[test]
fn roundtrip_kernel_with_body_source() {
    let k = KernelDef {
        name: String::from("k"),
        params: vec![KernelParam::FieldRead {
            name: String::from("input"),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: Vec::new(),
        body_source: Some(String::from("let x = input[gid];")),
        next_reg: 5,
        opt_level: 2,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body_source, Some(String::from("let x = input[gid];")));
    assert_eq!(k2.opt_level, 2);
}

#[test]
fn roundtrip_kernel_ops() {
    let ops = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::F32(3.14),
        },
        KernelOp::Load {
            dst: Reg(2),
            field: 0,
            index: Reg(0),
            ty: ScalarType::F32,
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(2),
            b: Reg(1),
            op: BinOp::Mul,
            ty: ScalarType::F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(3),
            ty: ScalarType::F32,
        },
        KernelOp::Barrier,
        KernelOp::Break,
    ];
    let k = KernelDef {
        name: String::from("mul_pi"),
        params: vec![
            KernelParam::FieldRead {
                name: String::from("in"),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: String::from("out"),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: ops,
        body_source: None,
        next_reg: 4,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 7);
    assert_eq!(k2.next_reg, 4);
}

#[test]
fn roundtrip_branch_and_loop() {
    let k = KernelDef {
        name: String::from("branchy"),
        params: Vec::new(),
        body: vec![
            KernelOp::Branch {
                cond: Reg(0),
                then_ops: vec![KernelOp::Barrier],
                else_ops: vec![KernelOp::Break],
            },
            KernelOp::Loop {
                count: Reg(1),
                iter_reg: Reg(2),
                body: vec![KernelOp::Const {
                    dst: Reg(3),
                    value: ConstValue::Bool(true),
                }],
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 2);
}

#[test]
fn roundtrip_compiler_output_empty() {
    let o = CompilerOutput {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
    };
    let bytes = serialize_output(&o);
    let o2 = deserialize_output(&bytes).unwrap();
    assert!(o2.amd.is_none());
    assert!(o2.nvidia.is_none());
    assert!(o2.spirv.is_none());
    assert!(o2.metallib.is_none());
}

#[test]
fn roundtrip_compiler_output_full() {
    let o = CompilerOutput {
        amd: Some(vec![0xDE, 0xAD]),
        nvidia: Some(vec![0xBE, 0xEF]),
        spirv: Some(vec![0x03, 0x02, 0x23, 0x07]),
        metallib: Some(vec![0x4D, 0x54]),
    };
    let bytes = serialize_output(&o);
    let o2 = deserialize_output(&bytes).unwrap();
    assert_eq!(o2.amd, Some(vec![0xDE, 0xAD]));
    assert_eq!(o2.nvidia, Some(vec![0xBE, 0xEF]));
    assert_eq!(o2.spirv, Some(vec![0x03, 0x02, 0x23, 0x07]));
    assert_eq!(o2.metallib, Some(vec![0x4D, 0x54]));
}

#[test]
fn trailing_bytes_rejected() {
    let k = KernelDef {
        name: String::from("x"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 0,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let mut bytes = serialize_kernel(&k);
    bytes.push(0xFF);
    assert_eq!(
        deserialize_kernel(&bytes).unwrap_err(),
        "trailing bytes after KernelDef"
    );
}

#[test]
fn truncated_input_rejected() {
    let bytes = [0x01]; // too short for any KernelDef
    assert!(deserialize_kernel(&bytes).is_err());
}

#[test]
fn all_scalar_types_roundtrip() {
    use ScalarType::*;
    let types = [F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, Bool];
    for ty in &types {
        let mut w = encode::Writer::new();
        encode::write_scalar_type(&mut w, ty);
        let buf = w.finish();
        let mut r = decode::Reader::new(&buf);
        let ty2 = decode::read_scalar_type(&mut r).unwrap();
        assert_eq!(*ty, ty2);
    }
}

#[test]
fn all_const_values_roundtrip() {
    let values = [
        ConstValue::F16(0x3C00),
        ConstValue::F32(1.0),
        ConstValue::F64(2.0),
        ConstValue::U32(42),
        ConstValue::U64(1_000_000),
        ConstValue::I32(-1),
        ConstValue::I64(-999),
        ConstValue::Bool(true),
    ];
    for cv in &values {
        let mut w = encode::Writer::new();
        encode::write_const_value(&mut w, cv);
        let buf = w.finish();
        let mut r = decode::Reader::new(&buf);
        let _ = decode::read_const_value(&mut r).unwrap();
    }
}

#[test]
fn dispatch_roundtrip() {
    let op = KernelOp::Dispatch {
        wave: Reg(10),
        groups: [Reg(1), Reg(2), Reg(3)],
    };
    let k = KernelDef {
        name: String::from("d"),
        params: Vec::new(),
        body: vec![op],
        body_source: None,
        next_reg: 11,
        opt_level: 0,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 1);
}

#[test]
fn all_kernel_params_roundtrip() {
    let params = vec![
        KernelParam::FieldRead {
            name: String::from("a"),
            slot: 0,
            scalar_type: ScalarType::F32,
        },
        KernelParam::FieldWrite {
            name: String::from("b"),
            slot: 1,
            scalar_type: ScalarType::U32,
        },
        KernelParam::Constant {
            name: String::from("c"),
            slot: 2,
            scalar_type: ScalarType::I32,
        },
        KernelParam::Texture2DRead {
            name: String::from("t0"),
            slot: 3,
            scalar_type: ScalarType::F32,
        },
        KernelParam::Texture2DWrite {
            name: String::from("t1"),
            slot: 4,
            scalar_type: ScalarType::F32,
        },
        KernelParam::Texture3DRead {
            name: String::from("t2"),
            slot: 5,
            scalar_type: ScalarType::F16,
        },
    ];
    let k = KernelDef {
        name: String::from("all_params"),
        params,
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 1,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.params.len(), 6);
}

#[test]
fn texture_ops_roundtrip() {
    let ops = vec![
        KernelOp::TextureSample2D {
            dst: Reg(0),
            texture: 0,
            x: Reg(1),
            y: Reg(2),
            ty: ScalarType::F32,
        },
        KernelOp::TextureSample3D {
            dst: Reg(3),
            texture: 1,
            x: Reg(4),
            y: Reg(5),
            z: Reg(6),
            ty: ScalarType::F16,
        },
        KernelOp::TextureWrite2D {
            texture: 2,
            x: Reg(7),
            y: Reg(8),
            value: Reg(9),
            ty: ScalarType::F32,
        },
        KernelOp::TextureSize {
            dst_w: Reg(10),
            dst_h: Reg(11),
            texture: 0,
        },
    ];
    let k = KernelDef {
        name: String::from("tex"),
        params: Vec::new(),
        body: ops,
        body_source: None,
        next_reg: 12,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 4);
}

#[test]
fn wave_ops_roundtrip() {
    let ops = vec![
        KernelOp::WaveShuffle {
            dst: Reg(0),
            src: Reg(1),
            lane_delta: Reg(2),
            ty: ScalarType::F32,
        },
        KernelOp::WaveBallot {
            dst: Reg(3),
            predicate: Reg(4),
        },
        KernelOp::WaveAny {
            dst: Reg(5),
            predicate: Reg(6),
        },
        KernelOp::WaveAll {
            dst: Reg(7),
            predicate: Reg(8),
        },
    ];
    let k = KernelDef {
        name: String::from("wave"),
        params: Vec::new(),
        body: ops,
        body_source: None,
        next_reg: 9,
        opt_level: 0,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 4);
}

#[test]
fn roundtrip_device_sources() {
    let k = KernelDef {
        name: String::from("with_device"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 3,
        device_sources: vec![
            String::from("fn activate(x: f32, t: f32) -> f32 { if x > t { x } else { x * 0.99 } }"),
            String::from("fn helper(a: f32) -> f32 { a + 1.0 }"),
        ],
        device_functions: Vec::new(),
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.device_sources.len(), 2);
    assert!(k2.device_sources[0].contains("activate"));
    assert!(k2.device_sources[1].contains("helper"));
}

#[test]
fn roundtrip_device_functions() {
    use crate::DeviceFnDef;
    let k = KernelDef {
        name: String::from("with_device_fns"),
        params: Vec::new(),
        body: vec![KernelOp::DeviceCall {
            dst: Reg(2),
            func_name: String::from("add_one"),
            args: vec![Reg(0)],
            ty: ScalarType::F32,
        }],
        body_source: None,
        next_reg: 3,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: vec![DeviceFnDef {
            name: String::from("add_one"),
            params: vec![(String::from("x"), ScalarType::F32)],
            return_type: ScalarType::F32,
            body: vec![
                KernelOp::Const {
                    dst: Reg(1),
                    value: ConstValue::F32(1.0),
                },
                KernelOp::BinOp {
                    dst: Reg(2),
                    a: Reg(0),
                    b: Reg(1),
                    op: BinOp::Add,
                    ty: ScalarType::F32,
                },
            ],
            next_reg: 3,
        }],
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.device_functions.len(), 1);
    assert_eq!(k2.device_functions[0].name, "add_one");
    assert_eq!(k2.device_functions[0].params.len(), 1);
    assert_eq!(k2.device_functions[0].body.len(), 2);
    assert_eq!(k2.device_functions[0].next_reg, 3);
}

// === ShaderDef / ShaderOutput roundtrip tests ===

#[test]
fn roundtrip_shader_def_vertex() {
    use crate::*;
    let s = ShaderDef {
        name: "transform".to_string(),
        stage: ShaderStage::Vertex,
        params: vec![
            ShaderParam {
                name: "pos".to_string(),
                ty: ShaderType::Vec3,
                is_uniform: false,
            },
            ShaderParam {
                name: "mvp".to_string(),
                ty: ShaderType::Mat4,
                is_uniform: true,
            },
        ],
        return_type: ShaderType::Vec4,
        body_source: "mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)".to_string(),
    };
    let bytes = serialize_shader(&s);
    let s2 = deserialize_shader(&bytes).unwrap();
    assert_eq!(s2.name, "transform");
    assert_eq!(s2.stage, ShaderStage::Vertex);
    assert_eq!(s2.params.len(), 2);
    assert_eq!(s2.params[0].name, "pos");
    assert_eq!(s2.params[0].ty, ShaderType::Vec3);
    assert!(!s2.params[0].is_uniform);
    assert_eq!(s2.params[1].name, "mvp");
    assert_eq!(s2.params[1].ty, ShaderType::Mat4);
    assert!(s2.params[1].is_uniform);
    assert_eq!(s2.return_type, ShaderType::Vec4);
    assert!(s2.body_source.contains("Vec4::new"));
}

#[test]
fn roundtrip_shader_def_fragment() {
    use crate::*;
    let s = ShaderDef {
        name: "shade".to_string(),
        stage: ShaderStage::Fragment,
        params: vec![ShaderParam {
            name: "uv".to_string(),
            ty: ShaderType::Vec2,
            is_uniform: false,
        }],
        return_type: ShaderType::Vec4,
        body_source: "Vec4::new(uv.x, uv.y, 0.0, 1.0)".to_string(),
    };
    let bytes = serialize_shader(&s);
    let s2 = deserialize_shader(&bytes).unwrap();
    assert_eq!(s2.name, "shade");
    assert_eq!(s2.stage, ShaderStage::Fragment);
    assert_eq!(s2.params.len(), 1);
    assert_eq!(s2.return_type, ShaderType::Vec4);
}

#[test]
fn roundtrip_shader_output_both() {
    use crate::*;
    let o = ShaderOutput {
        spirv: Some(vec![0x03, 0x02, 0x23, 0x07, 0x00, 0x01, 0x03, 0x00]),
        metallib: Some(vec![b'M', b'T', b'L', b'B', 0x01, 0x02]),
    };
    let bytes = serialize_shader_output(&o);
    let o2 = deserialize_shader_output(&bytes).unwrap();
    assert_eq!(o2.spirv.as_ref().unwrap().len(), 8);
    assert_eq!(o2.metallib.as_ref().unwrap().len(), 6);
}

#[test]
fn roundtrip_shader_output_none() {
    use crate::*;
    let o = ShaderOutput {
        spirv: None,
        metallib: None,
    };
    let bytes = serialize_shader_output(&o);
    let o2 = deserialize_shader_output(&bytes).unwrap();
    assert!(o2.spirv.is_none());
    assert!(o2.metallib.is_none());
}

#[test]
fn roundtrip_shader_def_all_types() {
    use crate::*;
    let s = ShaderDef {
        name: "all_types".to_string(),
        stage: ShaderStage::Fragment,
        params: vec![
            ShaderParam {
                name: "a".into(),
                ty: ShaderType::F32,
                is_uniform: false,
            },
            ShaderParam {
                name: "b".into(),
                ty: ShaderType::Vec2,
                is_uniform: false,
            },
            ShaderParam {
                name: "c".into(),
                ty: ShaderType::Vec3,
                is_uniform: false,
            },
            ShaderParam {
                name: "d".into(),
                ty: ShaderType::Vec4,
                is_uniform: false,
            },
            ShaderParam {
                name: "e".into(),
                ty: ShaderType::Mat3,
                is_uniform: true,
            },
            ShaderParam {
                name: "f".into(),
                ty: ShaderType::Mat4,
                is_uniform: true,
            },
        ],
        return_type: ShaderType::Vec4,
        body_source: "return d;".to_string(),
    };
    let bytes = serialize_shader(&s);
    let s2 = deserialize_shader(&bytes).unwrap();
    assert_eq!(s2.params.len(), 6);
    assert_eq!(s2.params[0].ty, ShaderType::F32);
    assert_eq!(s2.params[1].ty, ShaderType::Vec2);
    assert_eq!(s2.params[2].ty, ShaderType::Vec3);
    assert_eq!(s2.params[3].ty, ShaderType::Vec4);
    assert_eq!(s2.params[4].ty, ShaderType::Mat3);
    assert_eq!(s2.params[5].ty, ShaderType::Mat4);
}
