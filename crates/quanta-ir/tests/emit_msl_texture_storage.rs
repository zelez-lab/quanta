//! JIT MSL emitter: storage-image (`&mut Texture2D<f32>`) kernel params.
//!
//! The JIT MSL path (wave_jit, used by the Metal driver when no AOT metallib
//! is present) previously dropped all texture params (`_ => {}`) while its
//! ops emitter still referenced `tex_N` — producing undeclared-identifier MSL.
//! These pin the ported declarations: a write-declared slot becomes an
//! `access::read_write` storage texture (so `texture_load_2d` against it is a
//! valid `.read()`), a read-declared slot keeps `access::sample` + its
//! sampler, and sampling a write slot is rejected before emission.

#![cfg(feature = "jit")]

use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_msl};

fn write_kernel() -> KernelDef {
    KernelDef {
        name: "write_tex".into(),
        params: vec![
            KernelParam::Texture2DWrite {
                name: "tex".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "values".into(),
                slot: 0,
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
            KernelOp::TextureWrite2D {
                texture: 1,
                x: Reg(0),
                y: Reg(0),
                value: Reg(1),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 2,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[test]
fn write_slot_is_read_write_storage_texture() {
    let msl = emit_msl::emit(&write_kernel()).expect("emit write kernel");
    assert!(
        msl.contains("texture2d<float, access::read_write> tex_1 [[texture(1)]]"),
        "write slot must be declared read_write storage; MSL:\n{msl}"
    );
    // A write slot must NOT declare a sampler (storage images have none).
    assert!(
        !msl.contains("samp_1"),
        "write slot must not declare a sampler; MSL:\n{msl}"
    );
}

#[test]
fn read_slot_keeps_sampler() {
    let mut def = write_kernel();
    def.params = vec![KernelParam::Texture2DRead {
        name: "tex".into(),
        slot: 0,
        scalar_type: ScalarType::F32,
    }];
    def.body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::TextureLoad2D {
            dst: Reg(1),
            texture: 0,
            x: Reg(0),
            y: Reg(0),
            ty: ScalarType::F32,
        },
    ];
    def.next_reg = 2;
    let msl = emit_msl::emit(&def).expect("emit read kernel");
    assert!(
        msl.contains("texture2d<float, access::sample> tex_0 [[texture(0)]]"),
        "read slot must be a sampled texture; MSL:\n{msl}"
    );
    assert!(
        msl.contains("sampler samp_0 [[sampler(0)]]"),
        "read slot must declare a sampler; MSL:\n{msl}"
    );
}

#[test]
fn sample_on_write_slot_is_rejected() {
    let mut def = write_kernel();
    def.body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::TextureSample2D {
            dst: Reg(1),
            texture: 1,
            x: Reg(0),
            y: Reg(0),
            ty: ScalarType::F32,
        },
    ];
    let err = emit_msl::emit(&def).expect_err("sampling a storage image must fail");
    assert!(
        err.contains("storage") && err.contains("sampled"),
        "error should explain the storage/sample mismatch; got: {err}"
    );
}
