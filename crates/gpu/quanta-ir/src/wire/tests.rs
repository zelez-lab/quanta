use super::*;
use crate::{
    BinOp, CompilerOutput, ConstValue, KernelDef, KernelOp, KernelParam, MemoryOrder, Reg,
    ScalarType,
};

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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
    assert_eq!(k2.workgroup_size, [64, 1, 1]);
    assert_eq!(k2.subgroup_size, None);
    assert_eq!(k2.dynamic_shared_bytes, 0);
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
            value: ConstValue::F32(3.25),
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };
    let bytes = serialize_output(&o);
    let o2 = deserialize_output(&bytes).unwrap();
    assert!(o2.amd.is_none());
    assert!(o2.nvidia.is_none());
    assert!(o2.spirv.is_none());
    assert!(o2.metallib.is_none());
    assert!(o2.metallib_ios.is_none());
    assert!(o2.metallib_ios_sim.is_none());
    assert!(o2.wgsl.is_none());
}

#[test]
fn roundtrip_compiler_output_full() {
    let o = CompilerOutput {
        amd: Some(vec![0xDE, 0xAD]),
        nvidia: Some(vec![0xBE, 0xEF]),
        spirv: Some(vec![0x03, 0x02, 0x23, 0x07]),
        metallib: Some(vec![0x4D, 0x54]),
        metallib_ios: Some(vec![b'M', b'T', b'L', b'B', 0x10]),
        metallib_ios_sim: Some(vec![b'M', b'T', b'L', b'B', 0x20]),
        wgsl: Some(String::from("@compute fn main() {}")),
    };
    let bytes = serialize_output(&o);
    let o2 = deserialize_output(&bytes).unwrap();
    assert_eq!(o2.amd, Some(vec![0xDE, 0xAD]));
    assert_eq!(o2.nvidia, Some(vec![0xBE, 0xEF]));
    assert_eq!(o2.spirv, Some(vec![0x03, 0x02, 0x23, 0x07]));
    assert_eq!(o2.metallib, Some(vec![0x4D, 0x54]));
    assert_eq!(o2.metallib_ios, Some(vec![b'M', b'T', b'L', b'B', 0x10]));
    assert_eq!(
        o2.metallib_ios_sim,
        Some(vec![b'M', b'T', b'L', b'B', 0x20])
    );
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
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
                is_slice: false,
            },
            ShaderParam {
                name: "mvp".to_string(),
                ty: ShaderType::Mat4,
                is_uniform: true,
                is_slice: false,
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
            is_slice: false,
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
        metallib_ios: Some(vec![b'M', b'T', b'L', b'B', 0x11, 0x22, 0x33]),
        metallib_ios_sim: Some(vec![b'M', b'T', b'L', b'B', 0x44]),
        wgsl: Some(String::from(
            "fn vertex_main() -> @builtin(position) vec4f {}",
        )),
    };
    let bytes = serialize_shader_output(&o);
    let o2 = deserialize_shader_output(&bytes).unwrap();
    assert_eq!(o2.spirv.as_ref().unwrap().len(), 8);
    assert_eq!(o2.metallib.as_ref().unwrap().len(), 6);
    assert_eq!(o2.metallib_ios.as_ref().unwrap().len(), 7);
    assert_eq!(o2.metallib_ios_sim.as_ref().unwrap().len(), 5);
    assert!(o2.wgsl.is_some());
}

#[test]
fn roundtrip_shader_output_none() {
    use crate::*;
    let o = ShaderOutput {
        spirv: None,
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };
    let bytes = serialize_shader_output(&o);
    let o2 = deserialize_shader_output(&bytes).unwrap();
    assert!(o2.spirv.is_none());
    assert!(o2.metallib.is_none());
    assert!(o2.metallib_ios.is_none());
    assert!(o2.metallib_ios_sim.is_none());
    assert!(o2.wgsl.is_none());
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
                is_slice: false,
            },
            ShaderParam {
                name: "b".into(),
                ty: ShaderType::Vec2,
                is_uniform: false,
                is_slice: false,
            },
            ShaderParam {
                name: "c".into(),
                ty: ShaderType::Vec3,
                is_uniform: false,
                is_slice: false,
            },
            ShaderParam {
                name: "d".into(),
                ty: ShaderType::Vec4,
                is_uniform: false,
                is_slice: false,
            },
            ShaderParam {
                name: "e".into(),
                ty: ShaderType::Mat3,
                is_uniform: true,
                is_slice: false,
            },
            ShaderParam {
                name: "f".into(),
                ty: ShaderType::Mat4,
                is_uniform: true,
                is_slice: false,
            },
            ShaderParam {
                name: "g".into(),
                ty: ShaderType::U32,
                is_uniform: false,
                is_slice: false,
            },
        ],
        return_type: ShaderType::Vec4,
        body_source: "return d;".to_string(),
    };
    let bytes = serialize_shader(&s);
    let s2 = deserialize_shader(&bytes).unwrap();
    assert_eq!(s2.params.len(), 7);
    assert_eq!(s2.params[0].ty, ShaderType::F32);
    assert_eq!(s2.params[1].ty, ShaderType::Vec2);
    assert_eq!(s2.params[2].ty, ShaderType::Vec3);
    assert_eq!(s2.params[3].ty, ShaderType::Vec4);
    assert_eq!(s2.params[4].ty, ShaderType::Mat3);
    assert_eq!(s2.params[5].ty, ShaderType::Mat4);
    assert_eq!(s2.params[6].ty, ShaderType::U32);
}

#[test]
fn roundtrip_workgroup_size() {
    let k = KernelDef {
        name: String::from("custom_wg"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [16, 16, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.workgroup_size, [16, 16, 1]);
}

#[test]
fn roundtrip_workgroup_size_1d() {
    let k = KernelDef {
        name: String::from("wg_1d"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [256, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.workgroup_size, [256, 1, 1]);
}

#[test]
fn roundtrip_new_ops() {
    let ops = vec![
        KernelOp::Bitcast {
            dst: Reg(0),
            src: Reg(1),
            from: ScalarType::U32,
            to: ScalarType::F32,
        },
        KernelOp::CountTrailingZeros {
            dst: Reg(2),
            src: Reg(3),
            ty: ScalarType::U32,
        },
        KernelOp::CountLeadingZeros {
            dst: Reg(4),
            src: Reg(5),
            ty: ScalarType::U32,
        },
        KernelOp::PopCount {
            dst: Reg(6),
            src: Reg(7),
            ty: ScalarType::U32,
        },
        KernelOp::Dot {
            dst: Reg(8),
            a: Reg(9),
            b: Reg(10),
            ty: ScalarType::F32,
            width: 4,
        },
        KernelOp::SubgroupReduceAdd {
            dst: Reg(11),
            src: Reg(12),
            ty: ScalarType::F32,
        },
        KernelOp::SubgroupReduceMin {
            dst: Reg(13),
            src: Reg(14),
            ty: ScalarType::I32,
        },
        KernelOp::SubgroupReduceMax {
            dst: Reg(15),
            src: Reg(16),
            ty: ScalarType::U32,
        },
        KernelOp::SubgroupExclusiveAdd {
            dst: Reg(17),
            src: Reg(18),
            ty: ScalarType::F32,
        },
        KernelOp::SubgroupInclusiveAdd {
            dst: Reg(19),
            src: Reg(20),
            ty: ScalarType::F32,
        },
        KernelOp::TextureLoad2D {
            dst: Reg(21),
            texture: 0,
            x: Reg(22),
            y: Reg(23),
            ty: ScalarType::F32,
        },
    ];
    let k = KernelDef {
        name: String::from("new_ops"),
        params: Vec::new(),
        body: ops,
        body_source: None,
        next_reg: 24,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 11);
}

#[test]
fn roundtrip_subgroup_size_and_dynamic_shared() {
    let k = KernelDef {
        name: String::from("subgroup_test"),
        params: Vec::new(),
        body: vec![
            KernelOp::SubgroupSize { dst: Reg(0) },
            KernelOp::SharedDeclDyn {
                id: 0,
                ty: ScalarType::F32,
            },
            KernelOp::DebugPrint {
                src: Reg(0),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 1,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: Some(32),
        dynamic_shared_bytes: 4096,
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.body.len(), 3);
    assert_eq!(k2.subgroup_size, Some(32));
    assert_eq!(k2.dynamic_shared_bytes, 4096);
}

#[test]
fn roundtrip_subgroup_size_none() {
    let k = KernelDef {
        name: String::from("no_subgroup"),
        params: Vec::new(),
        body: Vec::new(),
        body_source: None,
        next_reg: 0,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let bytes = serialize_kernel(&k);
    let k2 = deserialize_kernel(&bytes).unwrap();
    assert_eq!(k2.subgroup_size, None);
    assert_eq!(k2.dynamic_shared_bytes, 0);
}

#[test]
fn roundtrip_fence_all_orderings() {
    // D-ext.3a: every MemoryOrder variant must survive a wire-format
    // roundtrip in a Fence opcode. The decoder is the only place where
    // the read_memory_order helper is exercised in a real kernel, so a
    // bug there would surface here as a tag mismatch.
    let orderings = [
        MemoryOrder::Relaxed,
        MemoryOrder::Acquire,
        MemoryOrder::Release,
        MemoryOrder::AcqRel,
        MemoryOrder::SeqCst,
    ];
    for &order in &orderings {
        let k = KernelDef {
            name: String::from("fence_test"),
            params: Vec::new(),
            body: vec![KernelOp::Fence { order }],
            body_source: None,
            next_reg: 0,
            opt_level: 0,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.body.len(), 1);
        match k2.body[0] {
            KernelOp::Fence { order: o2 } => assert_eq!(o2, order, "ordering survived roundtrip"),
            _ => panic!("expected Fence, got {:?}", k2.body[0]),
        }
    }
}

#[test]
fn roundtrip_shader_def_slice() {
    use crate::*;
    let s = ShaderDef {
        name: "gradient".to_string(),
        stage: ShaderStage::Fragment,
        params: vec![
            ShaderParam {
                name: "tint".to_string(),
                ty: ShaderType::Vec4,
                is_uniform: true,
                is_slice: false,
            },
            ShaderParam {
                name: "stops".to_string(),
                ty: ShaderType::Vec4,
                is_uniform: false,
                is_slice: true,
            },
        ],
        return_type: ShaderType::Vec4,
        body_source: "tint * stops[0]".to_string(),
    };
    let bytes = serialize_shader(&s);
    let s2 = deserialize_shader(&bytes).unwrap();
    assert!(!s2.params[0].is_slice);
    assert!(s2.params[1].is_slice);
    assert!(s2.params[0].is_uniform);
    assert!(!s2.params[1].is_uniform);
}
