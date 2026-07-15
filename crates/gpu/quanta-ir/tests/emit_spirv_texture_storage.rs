//! JIT SPIR-V emitter: storage-image (`&mut Texture2D<f32>`) kernel params.
//!
//! The JIT path (wave_jit, used by the Vulkan/software drivers when there is
//! no AOT metallib/SPIR-V) previously stubbed texture params and ops — a
//! texture write was a no-op and a load returned zero. These pin the ported
//! emission: a write emits `OpImageWrite`, a load against a write-declared
//! slot emits `OpImageRead` (not `OpImageFetch`, which is sampled-only), and
//! the storage image carries a scalar-driven `R32f` format (ImageFormat = 3),
//! `sampled = 2`.
//!
//! Opcode / enum values verified against the SPIR-V spec (§3.42 / §3.14).

#![cfg(feature = "jit")]

use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

const OP_TYPE_IMAGE: u16 = 25;
const OP_EXT_INST: u16 = 12;
const OP_IMAGE_READ: u16 = 98;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_FETCH: u16 = 95;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const IMAGE_FORMAT_R32F: u32 = 3;
const IMAGE_FORMAT_RGBA8: u32 = 4;
// GLSL.std.450 extended instruction numbers for the packed-RGBA8 boundary.
const GLSL_PACK_UNORM_4X8: u32 = 55;
const GLSL_UNPACK_UNORM_4X8: u32 = 64;

/// `texture_write_2d(tex, x, y, values[i])` — a pure write-only storage kernel.
fn write_kernel() -> KernelDef {
    KernelDef {
        name: "write_tex".into(),
        params: vec![
            KernelParam::Texture2DWrite {
                name: "tex".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "values".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 1,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::TextureWrite2D {
                texture: 0,
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

/// `out[i] = texture_load_2d(tex, x, y)` where `tex` is `&mut Texture2D` — the
/// DSL admits a read against a write-declared slot; it must lower to a storage
/// read (OpImageRead), not a sampled fetch.
fn load_from_storage_kernel() -> KernelDef {
    KernelDef {
        name: "load_storage".into(),
        params: vec![
            KernelParam::Texture2DWrite {
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
    }
}

/// Packed-RGBA8 twin of `write_kernel`: `texture_write_2d_u32(tex, x, y, v)`
/// where `tex` is `&mut Texture2D<u32>`. The u32 value must UnpackUnorm4x8 into
/// the vec4<f32> texel and the storage image must carry ImageFormat Rgba8 (4).
fn write_rgba8_kernel() -> KernelDef {
    KernelDef {
        name: "write_tex_rgba8".into(),
        params: vec![
            KernelParam::Texture2DWrite {
                name: "tex".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldRead {
                name: "values".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 1,
                index: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::TextureWrite2D {
                texture: 0,
                x: Reg(0),
                y: Reg(0),
                value: Reg(1),
                ty: ScalarType::U32,
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

/// Packed-RGBA8 twin of `load_from_storage_kernel`: read a `&mut
/// Texture2D<u32>` slot. The vec4<f32> from OpImageRead must PackUnorm4x8 into
/// the u32 result.
fn load_rgba8_kernel() -> KernelDef {
    KernelDef {
        name: "load_storage_rgba8".into(),
        params: vec![
            KernelParam::Texture2DWrite {
                name: "tex".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::TextureLoad2D {
                dst: Reg(1),
                texture: 0,
                x: Reg(0),
                y: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(1),
                ty: ScalarType::U32,
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

/// The src+dst ping-pong shape: TWO `&mut Texture2D<u32>` storage-image params.
/// Both storage images share one identical `OpTypeImage` (`%float 2D 0 0 0 2
/// Rgba8`). SPIR-V forbids duplicate non-aggregate type declarations, so the
/// emitter must emit that image type once and reuse it for the second param —
/// emitting it twice makes spirv-val reject the module (the downstream dija
/// bug). The kernel reads slot 0 and writes the texel to slot 1.
fn ping_pong_rgba8_kernel() -> KernelDef {
    KernelDef {
        name: "ping_pong_rgba8".into(),
        params: vec![
            KernelParam::Texture2DWrite {
                name: "src".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::Texture2DWrite {
                name: "dst".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::TextureLoad2D {
                dst: Reg(1),
                texture: 0,
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

/// f32 twin of the ping-pong: two `&mut Texture2D<f32>` (R32Float) storage
/// images. Same dedup requirement — one shared `%float 2D 0 0 0 2 R32f`.
fn ping_pong_f32_kernel() -> KernelDef {
    let mut def = ping_pong_rgba8_kernel();
    def.name = "ping_pong_f32".into();
    def.params = vec![
        KernelParam::Texture2DWrite {
            name: "src".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        },
        KernelParam::Texture2DWrite {
            name: "dst".into(),
            slot: 1,
            scalar_type: ScalarType::F32,
        },
    ];
    def.body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::TextureLoad2D {
            dst: Reg(1),
            texture: 0,
            x: Reg(0),
            y: Reg(0),
            ty: ScalarType::F32,
        },
        KernelOp::TextureWrite2D {
            texture: 1,
            x: Reg(0),
            y: Reg(0),
            value: Reg(1),
            ty: ScalarType::F32,
        },
    ];
    def
}

/// A sampled `&Texture2D<u32>` must be rejected at emit — sampled u32 is a
/// distinct, unwired meaning (storage-position u32 is the packed-RGBA8 image).
fn sampled_u32_kernel() -> KernelDef {
    KernelDef {
        name: "sampled_u32".into(),
        params: vec![
            KernelParam::Texture2DRead {
                name: "tex".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
        ],
        body: vec![KernelOp::QuarkId { dst: Reg(0) }],
        body_source: None,
        next_reg: 1,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// `out[i] = texture_sample_2d(tex, x, y)` where `tex` is `&Texture2D<f32>` — a
/// sampled read. Under GLCompute the sample must be `OpImageSampleExplicitLod`
/// with an explicit Lod (ImplicitLod needs a fragment stage's derivatives and
/// spirv-val rejects it in a compute module).
fn sample_f32_kernel() -> KernelDef {
    KernelDef {
        name: "sample_f32".into(),
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
            KernelOp::TextureSample2D {
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
    }
}

fn words(spirv: &[u8]) -> Vec<u32> {
    spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut ops = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let word = words[i];
        ops.push((word & 0xFFFF) as u16);
        let wc = (word >> 16) as usize;
        assert!(wc >= 1);
        i += wc;
    }
    ops
}

/// Count how many times `opcode` appears in the module (past the 5-word header).
fn count_opcode(words: &[u32], opcode: u16) -> usize {
    let mut i = 5;
    let mut n = 0;
    while i < words.len() {
        let word = words[i];
        let wc = (word >> 16) as usize;
        assert!(wc >= 1);
        if (word & 0xFFFF) as u16 == opcode {
            n += 1;
        }
        i += wc;
    }
    n
}

/// Find the operands of the first `OpTypeImage` in the module.
fn type_image_operands(words: &[u32]) -> Vec<u32> {
    let mut i = 5;
    while i < words.len() {
        let word = words[i];
        let opcode = (word & 0xFFFF) as u16;
        let wc = (word >> 16) as usize;
        assert!(wc >= 1);
        if opcode == OP_TYPE_IMAGE {
            return words[i..i + wc].to_vec();
        }
        i += wc;
    }
    panic!("no OpTypeImage in module");
}

/// True if the module contains an `OpExtInst` whose extended-instruction
/// number is `glsl_instr` (operand index 4 of the instruction, after
/// result-type / result-id / ext-set-id).
fn has_ext_inst(words: &[u32], glsl_instr: u32) -> bool {
    let mut i = 5;
    while i < words.len() {
        let word = words[i];
        let opcode = (word & 0xFFFF) as u16;
        let wc = (word >> 16) as usize;
        assert!(wc >= 1);
        if opcode == OP_EXT_INST && wc >= 5 && words[i + 4] == glsl_instr {
            return true;
        }
        i += wc;
    }
    false
}

#[test]
fn write_emits_op_image_write() {
    let spirv = emit_spirv::emit(&write_kernel()).expect("emit write kernel");
    let ops = opcodes(&words(&spirv));
    assert!(
        ops.contains(&OP_IMAGE_WRITE),
        "storage write must emit OpImageWrite; opcodes: {ops:?}"
    );
}

#[test]
fn storage_image_is_r32f_sampled2() {
    let spirv = emit_spirv::emit(&write_kernel()).expect("emit write kernel");
    // OpTypeImage %result %SampledType Dim Depth Arrayed MS Sampled Format
    // words[0]=header, [1]=result, [2]=sampled_type, [3]=Dim, [4]=Depth,
    // [5]=Arrayed, [6]=MS, [7]=Sampled, [8]=Format.
    let ops = type_image_operands(&words(&spirv));
    assert_eq!(ops[7], 2, "storage image must be sampled=2; got {ops:?}");
    assert_eq!(
        ops[8], IMAGE_FORMAT_R32F,
        "Texture2D<f32> storage image must be R32f (3), not Rgba32f (1); got {ops:?}"
    );
}

#[test]
fn load_from_write_slot_emits_op_image_read() {
    let spirv = emit_spirv::emit(&load_from_storage_kernel()).expect("emit load kernel");
    let ops = opcodes(&words(&spirv));
    assert!(
        ops.contains(&OP_IMAGE_READ),
        "load against a `&mut Texture2D` slot must emit OpImageRead; opcodes: {ops:?}"
    );
    assert!(
        !ops.contains(&OP_IMAGE_FETCH),
        "a storage image must NOT be OpImageFetch'd; opcodes: {ops:?}"
    );
}

/// Sampling a write-declared slot must be rejected at emit time (a storage
/// image has no sampler). Both emitters agree via `reject_sample_on_write`.
#[test]
fn sample_on_write_slot_is_rejected() {
    let mut def = write_kernel();
    def.body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::TextureSample2D {
            dst: Reg(1),
            texture: 0,
            x: Reg(0),
            y: Reg(0),
            ty: ScalarType::F32,
        },
    ];
    let err = emit_spirv::emit(&def).expect_err("sampling a storage image must fail");
    assert!(
        err.contains("storage") && err.contains("sampled"),
        "error should explain the storage/sample mismatch; got: {err}"
    );
}

/// If `spirv-val` is on PATH, the emitted storage-image module must validate.
#[test]
fn storage_module_validates() {
    let spirv = emit_spirv::emit(&write_kernel()).expect("emit write kernel");
    assert_spirv_val_clean("write_tex", &spirv);
    let spirv = emit_spirv::emit(&load_from_storage_kernel()).expect("emit load kernel");
    assert_spirv_val_clean("load_storage", &spirv);
}

// ── Sampled reads (`&Texture2D<f32>` + texture_sample_2d) ────────────────────

/// A compute sample must lower to `OpImageSampleExplicitLod`, never the
/// `OpImageSampleImplicitLod` that is illegal under GLCompute.
#[test]
fn sample_emits_explicit_lod_not_implicit() {
    let spirv = emit_spirv::emit(&sample_f32_kernel()).expect("emit sample kernel");
    let ops = opcodes(&words(&spirv));
    assert!(
        ops.contains(&OP_IMAGE_SAMPLE_EXPLICIT_LOD),
        "compute texture_sample_2d must emit OpImageSampleExplicitLod (88); opcodes: {ops:?}"
    );
    assert!(
        !ops.contains(&OP_IMAGE_SAMPLE_IMPLICIT_LOD),
        "compute sample must NOT emit OpImageSampleImplicitLod (87), illegal under \
         GLCompute; opcodes: {ops:?}"
    );
}

/// The emitted sampled-read module must validate under `spirv-val` — the guard
/// that would have caught an ImplicitLod-in-compute regression (nothing else
/// spirv-vals a sample module).
#[test]
fn sample_module_validates() {
    let spirv = emit_spirv::emit(&sample_f32_kernel()).expect("emit sample kernel");
    assert_spirv_val_clean("sample_f32", &spirv);
}

// ── Packed-RGBA8 (`&mut Texture2D<u32>`) storage images ─────────────────────

#[test]
fn rgba8_write_unpacks_to_vec4_and_is_rgba8_format() {
    let spirv = emit_spirv::emit(&write_rgba8_kernel()).expect("emit rgba8 write kernel");
    let w = words(&spirv);
    // The u32 value is UnpackUnorm4x8'd into the vec4<f32> texel.
    assert!(
        has_ext_inst(&w, GLSL_UNPACK_UNORM_4X8),
        "packed-RGBA8 write must OpExtInst UnpackUnorm4x8; opcodes: {:?}",
        opcodes(&w)
    );
    assert!(
        opcodes(&w).contains(&OP_IMAGE_WRITE),
        "packed-RGBA8 write must still emit OpImageWrite"
    );
    // The storage image's SPIR-V sampled type stays f32 (component of the vec4);
    // only the format word is Rgba8 (4), not R32Uint or R32f.
    let ops = type_image_operands(&w);
    assert_eq!(ops[7], 2, "storage image must be sampled=2; got {ops:?}");
    assert_eq!(
        ops[8], IMAGE_FORMAT_RGBA8,
        "Texture2D<u32> storage image must be Rgba8 (4); got {ops:?}"
    );
}

#[test]
fn rgba8_load_packs_from_vec4() {
    let spirv = emit_spirv::emit(&load_rgba8_kernel()).expect("emit rgba8 load kernel");
    let w = words(&spirv);
    assert!(
        opcodes(&w).contains(&OP_IMAGE_READ),
        "packed-RGBA8 load must emit OpImageRead"
    );
    assert!(
        has_ext_inst(&w, GLSL_PACK_UNORM_4X8),
        "packed-RGBA8 load must OpExtInst PackUnorm4x8 the vec4 into a u32; opcodes: {:?}",
        opcodes(&w)
    );
}

/// A sampled `&Texture2D<u32>` is rejected at emit (both emitters agree via
/// `reject_sampled_u32_texture`).
#[test]
fn sampled_u32_texture_is_rejected() {
    let err = emit_spirv::emit(&sampled_u32_kernel())
        .expect_err("sampled &Texture2D<u32> must be rejected at emit");
    assert!(
        err.contains("u32") && err.contains("sampled"),
        "error should explain the sampled-u32 restriction; got: {err}"
    );
}

/// The emitted packed-RGBA8 modules must validate under `spirv-val`.
#[test]
fn rgba8_storage_module_validates() {
    let spirv = emit_spirv::emit(&write_rgba8_kernel()).expect("emit rgba8 write kernel");
    assert_spirv_val_clean("write_tex_rgba8", &spirv);
    let spirv = emit_spirv::emit(&load_rgba8_kernel()).expect("emit rgba8 load kernel");
    assert_spirv_val_clean("load_storage_rgba8", &spirv);
}

// ── Two storage images share one OpTypeImage (the ping-pong dedup) ──────────

/// The regression tripwire for the dija-reported bug: two same-shaped
/// `&mut Texture2D<u32>` storage-image params must emit exactly ONE
/// `OpTypeImage`, reused for both. Emitting it per-param produced a duplicate
/// non-aggregate type declaration that spirv-val rejects.
#[test]
fn two_storage_images_share_one_op_type_image() {
    let spirv = emit_spirv::emit(&ping_pong_rgba8_kernel()).expect("emit ping-pong rgba8");
    let n = count_opcode(&words(&spirv), OP_TYPE_IMAGE);
    assert_eq!(
        n, 1,
        "two &mut Texture2D<u32> params must share ONE OpTypeImage (SPIR-V \
         forbids duplicate non-aggregate types); got {n}"
    );

    // The f32 twin must dedupe just the same.
    let spirv = emit_spirv::emit(&ping_pong_f32_kernel()).expect("emit ping-pong f32");
    let n = count_opcode(&words(&spirv), OP_TYPE_IMAGE);
    assert_eq!(
        n, 1,
        "two &mut Texture2D<f32> params must share ONE OpTypeImage; got {n}"
    );
}

/// End-to-end validation of the fix: both ping-pong modules must pass
/// `spirv-val` (before the fix the duplicate `OpTypeImage` failed with
/// "Duplicate non-aggregate type declarations are not allowed").
#[test]
fn ping_pong_storage_module_validates() {
    let spirv = emit_spirv::emit(&ping_pong_rgba8_kernel()).expect("emit ping-pong rgba8");
    assert_spirv_val_clean("ping_pong_rgba8", &spirv);
    let spirv = emit_spirv::emit(&ping_pong_f32_kernel()).expect("emit ping-pong f32");
    assert_spirv_val_clean("ping_pong_f32", &spirv);
}

fn assert_spirv_val_clean(name: &str, spirv: &[u8]) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(_) => return, // spirv-val not installed
    };
    child.stdin.as_mut().unwrap().write_all(spirv).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{name}: emitted SPIR-V is invalid (spirv-val):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
