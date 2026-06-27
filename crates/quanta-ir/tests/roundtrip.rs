//! Test: serialize + deserialize KernelDef round-trips correctly.

use quanta_ir::*;

#[test]
fn roundtrip_vector_add() {
    let kernel = KernelDef {
        name: "vector_add".to_string(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".to_string(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".to_string(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".to_string(),
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
        body_source: None,
        next_reg: 4,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };

    let bytes = serialize_kernel(&kernel);
    let restored = deserialize_kernel(&bytes).unwrap();

    assert_eq!(restored.name, "vector_add");
    assert_eq!(restored.params.len(), 3);
    assert_eq!(restored.body.len(), 5);
    assert_eq!(restored.next_reg, 4);
    assert_eq!(restored.opt_level, 3);
}

#[test]
fn roundtrip_compiler_output() {
    let output = CompilerOutput {
        amd: Some(vec![0x7f, 0x45, 0x4c, 0x46]),
        nvidia: Some(b".visible .entry test()".to_vec()),
        spirv: None,
        metallib: None,
        wgsl: None,
    };

    let bytes = serialize_output(&output);
    let restored = deserialize_output(&bytes).unwrap();

    assert_eq!(
        restored.amd.as_ref().unwrap()[0..4],
        [0x7f, 0x45, 0x4c, 0x46]
    );
    assert!(restored.nvidia.is_some());
    assert!(restored.spirv.is_none());
    assert!(restored.metallib.is_none());
}

#[test]
fn roundtrip_all_scalar_types() {
    for ty in [
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
    ] {
        let kernel = KernelDef {
            name: format!("test_{:?}", ty),
            params: vec![KernelParam::FieldRead {
                name: "x".to_string(),
                slot: 0,
                scalar_type: ty,
            }],
            body: vec![],
            body_source: None,
            next_reg: 0,
            opt_level: 3,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let bytes = serialize_kernel(&kernel);
        let restored = deserialize_kernel(&bytes).unwrap();
        assert_eq!(restored.params.len(), 1);
    }
}

#[test]
fn roundtrip_all_ops() {
    let kernel = KernelDef {
        name: "all_ops".to_string(),
        params: vec![],
        body: vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::F32(3.14),
            },
            KernelOp::QuarkId { dst: Reg(1) },
            KernelOp::ProtonId { dst: Reg(2) },
            KernelOp::NucleusId { dst: Reg(3) },
            KernelOp::ProtonSize { dst: Reg(4) },
            KernelOp::Barrier,
            KernelOp::Branch {
                cond: Reg(0),
                then_ops: vec![],
                else_ops: vec![],
            },
            KernelOp::Loop {
                count: Reg(0),
                iter_reg: Reg(5),
                body: vec![],
            },
        ],
        body_source: None,
        next_reg: 6,
        opt_level: 2,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };

    let bytes = serialize_kernel(&kernel);
    let restored = deserialize_kernel(&bytes).unwrap();
    assert_eq!(restored.body.len(), 8);
    assert_eq!(restored.opt_level, 2);
}

#[test]
fn roundtrip_cooperative_matrix_ops() {
    // The three cooperative-matrix ops (load A/B fragments, MMA, store the
    // accumulator) round-trip through the wire format. KernelOp has no
    // PartialEq (it carries Vecs), so we check stability by re-encoding the
    // decoded kernel and comparing bytes — a faithful round-trip witness.
    let kernel = KernelDef {
        name: "coop_matrix".to_string(),
        params: vec![],
        body: vec![
            KernelOp::CooperativeMatrixLoad {
                dst: Reg(0),
                field: 0,
                index: Reg(10),
                stride: Reg(11),
                frag: MatrixFrag::A,
                from_shared: false,
                m: 8,
                n: 8,
                k: 8,
                ty: ScalarType::F32,
            },
            KernelOp::CooperativeMatrixLoad {
                dst: Reg(1),
                field: 1,
                index: Reg(12),
                stride: Reg(13),
                frag: MatrixFrag::B,
                from_shared: true,
                m: 8,
                n: 8,
                k: 8,
                ty: ScalarType::F32,
            },
            KernelOp::CooperativeMMA {
                dst: Reg(2),
                a: Reg(0),
                b: Reg(1),
                c: Reg(2),
                m: 8,
                n: 8,
                k: 8,
                ty: ScalarType::F32,
            },
            KernelOp::CooperativeMatrixStore {
                field: 2,
                index: Reg(14),
                stride: Reg(15),
                src: Reg(2),
                m: 8,
                n: 8,
                k: 8,
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 16,
        opt_level: 0,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [32, 1, 1],
        subgroup_size: Some(32),
        dynamic_shared_bytes: 0,
    };

    let bytes = serialize_kernel(&kernel);
    let restored = deserialize_kernel(&bytes).unwrap();
    assert_eq!(restored.body.len(), 4);
    // Re-encode and compare bytes: the decode reproduced every field exactly.
    assert_eq!(serialize_kernel(&restored), bytes);
}
