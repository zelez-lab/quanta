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
const OP_IMAGE_READ: u16 = 98;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_FETCH: u16 = 95;
const IMAGE_FORMAT_R32F: u32 = 3;

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
