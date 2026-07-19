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
            KernelParam::Texture2DReadWrite {
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
    def.params = vec![KernelParam::Sampled2D {
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

/// A packed-RGBA8 (`&mut Texture2D<u32>`) slot: the declaration stays a `float`
/// read_write texture (pixel format is host-side), but the read/write ops go
/// through MSL's pack_float_to_unorm4x8 / unpack_unorm4x8_to_float.
#[test]
fn rgba8_slot_packs_and_unpacks_at_the_op() {
    let mut def = write_kernel();
    def.params = vec![
        KernelParam::Texture2DReadWrite {
            name: "tex".into(),
            slot: 1,
            scalar_type: ScalarType::U32,
        },
        KernelParam::FieldRead {
            name: "values".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        },
    ];
    def.body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Load {
            dst: Reg(1),
            field: 0,
            index: Reg(0),
            ty: ScalarType::U32,
        },
        // read-modify-write so both the pack (load) and unpack (write) appear.
        KernelOp::TextureLoad2D {
            dst: Reg(2),
            texture: 1,
            x: Reg(0),
            y: Reg(0),
            ty: ScalarType::U32,
        },
        KernelOp::TextureWrite2D {
            texture: 1,
            x: Reg(0),
            y: Reg(0),
            value: Reg(1),
            ty: ScalarType::U32,
        },
    ];
    def.next_reg = 3;
    let msl = emit_msl::emit(&def).expect("emit rgba8 kernel");
    // Declaration is still a float read_write texture — the format is host-side.
    assert!(
        msl.contains("texture2d<float, access::read_write> tex_1 [[texture(1)]]"),
        "packed-RGBA8 slot must still be a float read_write texture; MSL:\n{msl}"
    );
    assert!(
        msl.contains("unpack_unorm4x8_to_float(r1)"),
        "packed-RGBA8 write must unpack the u32 into the float4 texel; MSL:\n{msl}"
    );
    assert!(
        msl.contains("pack_float_to_unorm4x8(tex_1.read("),
        "packed-RGBA8 load must pack the float4 texel into a u32; MSL:\n{msl}"
    );
}

/// A sampled `&Texture2D<u32>` is rejected before emission.
#[test]
fn sampled_u32_texture_is_rejected() {
    let mut def = write_kernel();
    def.params = vec![KernelParam::Sampled2D {
        name: "tex".into(),
        slot: 0,
        scalar_type: ScalarType::U32,
    }];
    def.body = vec![KernelOp::QuarkId { dst: Reg(0) }];
    def.next_reg = 1;
    let err = emit_msl::emit(&def).expect_err("sampled &Texture2D<u32> must be rejected");
    assert!(
        err.contains("u32") && err.contains("sampled"),
        "error should explain the sampled-u32 restriction; got: {err}"
    );
}

// ── Read-only texel slots (`&Texture2D`) ─────────────────────────────────────

/// `&Texture2D` is `access::read` — no sampler, no read_write (and therefore
/// no MTLReadWriteTextureTier gate, which is the reason the form exists).
#[test]
fn read_only_slot_uses_access_read_without_sampler() {
    let def = KernelDef {
        name: "ro_msl".into(),
        params: vec![
            KernelParam::Texture2DRead {
                name: "tex".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::TextureLoad2D {
                dst: Reg(1),
                texture: 0,
                x: Reg(0),
                y: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(1),
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
    };
    let msl = emit_msl::emit(&def).expect("emit ro msl");
    assert!(
        msl.contains("texture2d<float, access::read> tex_0 [[texture(0)]]"),
        "read-only texel slot must be access::read; got:\n{msl}"
    );
    assert!(
        !msl.contains("samp_0"),
        "read-only texel slot must not bind a sampler; got:\n{msl}"
    );
    assert!(
        msl.contains("tex_0.read("),
        "load must lower to .read(); got:\n{msl}"
    );
}

/// Writing a read-only slot fails in the MSL emitter with the same error as
/// SPIR-V (`reject_write_on_read_only` is shared).
#[test]
fn msl_write_on_read_only_slot_is_rejected() {
    let def = KernelDef {
        name: "ro_msl_w".into(),
        params: vec![KernelParam::Texture2DRead {
            name: "tex".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::TextureWrite2D {
                texture: 0,
                x: Reg(0),
                y: Reg(0),
                value: Reg(0),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 1,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let err = emit_msl::emit(&def).expect_err("writing a read-only texel slot must fail");
    assert!(
        err.contains("read-only") && err.contains("&mut Texture2D"),
        "error should name the read-only slot and the fix; got: {err}"
    );
}
