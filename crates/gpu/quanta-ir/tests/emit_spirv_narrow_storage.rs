//! Narrow-scalar (bf16 / fp8) storage contract on the SPIR-V emitter.
//!
//! bf16 buffers are 16-bit elements (ArrayStride 2) and fp8 buffers are
//! 8-bit elements (ArrayStride 1) — native stride, matching the host's
//! tight `Field<u16>` / `Field<u8>` upload and the CPU executor. The
//! module must declare the matching storage-access capability (and the
//! `SPV_KHR_8bit_storage` extension for fp8, which is only core from
//! SPIR-V 1.5). When `spirv-val` is on PATH the emitted module is also
//! run through the reference validator.

#![cfg(feature = "jit")]

use quanta_ir::{BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

const CAP_STORAGE_BUFFER_16BIT: u32 = 4433;
const CAP_STORAGE_BUFFER_8BIT: u32 = 4448;
const OP_CAPABILITY: u32 = 17;
const OP_DECORATE: u32 = 71;
const DECORATION_ARRAY_STRIDE: u32 = 6;

/// A minimal load → f32 math → store round-trip over `ty` storage.
fn narrow_kernel(ty: ScalarType, name: &str) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: ty,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(2.0),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Mul,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(3),
                ty,
            },
        ],
        body_source: None,
        next_reg: 4,
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

/// All `(opcode, operands)` instructions of the module.
fn instructions(w: &[u32]) -> Vec<(u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut i = 5; // skip header
    while i < w.len() {
        let wc = (w[i] >> 16) as usize;
        let op = w[i] & 0xFFFF;
        out.push((op, w[i + 1..i + wc].to_vec()));
        i += wc;
    }
    out
}

fn has_capability(w: &[u32], cap: u32) -> bool {
    instructions(w)
        .iter()
        .any(|(op, args)| *op == OP_CAPABILITY && args == &[cap])
}

fn array_strides(w: &[u32]) -> Vec<u32> {
    instructions(w)
        .iter()
        .filter(|(op, args)| *op == OP_DECORATE && args.get(1) == Some(&DECORATION_ARRAY_STRIDE))
        .map(|(_, args)| args[2])
        .collect()
}

/// Run `spirv-val --target-env vulkan1.3` when available; skip silently
/// (like the build-time gate) when it isn't installed.
fn spirv_val(name: &str, spirv: &[u8]) {
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
        Err(_) => return,
    };
    child.stdin.as_mut().unwrap().write_all(spirv).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{name}: spirv-val rejected the module:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn bf16_buffers_use_native_16bit_storage() {
    let spirv = emit_spirv::emit(&narrow_kernel(ScalarType::BF16, "bf16_roundtrip")).unwrap();
    let w = words(&spirv);
    assert!(
        has_capability(&w, CAP_STORAGE_BUFFER_16BIT),
        "bf16 module must declare StorageBuffer16BitAccess"
    );
    let strides = array_strides(&w);
    assert!(
        strides.contains(&2),
        "bf16 buffer must have ArrayStride 2, got {strides:?}"
    );
    spirv_val("bf16_roundtrip", &spirv);
}

#[test]
fn fp8_buffers_use_native_8bit_storage() {
    for (ty, name) in [
        (ScalarType::FP8E5M2, "fp8_e5m2_roundtrip"),
        (ScalarType::FP8E4M3, "fp8_e4m3_roundtrip"),
    ] {
        let spirv = emit_spirv::emit(&narrow_kernel(ty, name)).unwrap();
        let w = words(&spirv);
        assert!(
            has_capability(&w, CAP_STORAGE_BUFFER_8BIT),
            "{name}: module must declare StorageBuffer8BitAccess"
        );
        let strides = array_strides(&w);
        assert!(
            strides.contains(&1),
            "{name}: fp8 buffer must have ArrayStride 1, got {strides:?}"
        );
        spirv_val(name, &spirv);
    }
}

/// The mixed-GEMM kernel shape (`quanta-blas`'s `mixed_kernel::build_def`):
/// a loop-carried f32 accumulator over narrow A/B loads with f32 casts and
/// an f32 C read-modify-write. This is the exact kernel the bf16/fp8 GEMM
/// dispatches on Vulkan, so validating it here covers the production module.
fn gemm_shaped_kernel(in_ty: ScalarType, name: &str) -> KernelDef {
    use quanta_ir::ConstValue::{F32, U32};
    use quanta_ir::ScalarType::{F32 as SF32, U32 as SU32};
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: U32(4),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: U32(4),
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::Div,
            ty: SU32,
        },
        KernelOp::BinOp {
            dst: Reg(4),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::Rem,
            ty: SU32,
        },
        KernelOp::Const {
            dst: Reg(5),
            value: F32(0.0),
        },
        KernelOp::Loop {
            count: Reg(2),
            iter_reg: Reg(6),
            body: vec![
                KernelOp::BinOp {
                    dst: Reg(7),
                    a: Reg(3),
                    b: Reg(2),
                    op: BinOp::Mul,
                    ty: SU32,
                },
                KernelOp::BinOp {
                    dst: Reg(7),
                    a: Reg(7),
                    b: Reg(6),
                    op: BinOp::Add,
                    ty: SU32,
                },
                KernelOp::BinOp {
                    dst: Reg(8),
                    a: Reg(6),
                    b: Reg(1),
                    op: BinOp::Mul,
                    ty: SU32,
                },
                KernelOp::BinOp {
                    dst: Reg(8),
                    a: Reg(8),
                    b: Reg(4),
                    op: BinOp::Add,
                    ty: SU32,
                },
                KernelOp::Load {
                    dst: Reg(9),
                    field: 0,
                    index: Reg(7),
                    ty: in_ty,
                },
                KernelOp::Load {
                    dst: Reg(10),
                    field: 1,
                    index: Reg(8),
                    ty: in_ty,
                },
                KernelOp::Cast {
                    dst: Reg(18),
                    src: Reg(9),
                    from: in_ty,
                    to: SF32,
                },
                KernelOp::Cast {
                    dst: Reg(19),
                    src: Reg(10),
                    from: in_ty,
                    to: SF32,
                },
                KernelOp::BinOp {
                    dst: Reg(11),
                    a: Reg(18),
                    b: Reg(19),
                    op: BinOp::Mul,
                    ty: SF32,
                },
                KernelOp::BinOp {
                    dst: Reg(5),
                    a: Reg(5),
                    b: Reg(11),
                    op: BinOp::Add,
                    ty: SF32,
                },
            ],
        },
        KernelOp::Load {
            dst: Reg(14),
            field: 2,
            index: Reg(0),
            ty: SF32,
        },
        KernelOp::BinOp {
            dst: Reg(17),
            a: Reg(5),
            b: Reg(14),
            op: BinOp::Add,
            ty: SF32,
        },
        KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: Reg(17),
            ty: SF32,
        },
    ];
    KernelDef {
        name: name.into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: in_ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: in_ty,
            },
            KernelParam::FieldWrite {
                name: "c".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        body,
        body_source: None,
        next_reg: 20,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[test]
fn gemm_shaped_narrow_kernels_validate() {
    for (ty, name) in [
        (ScalarType::BF16, "blas_gemm_mixed_bf16"),
        (ScalarType::FP8E5M2, "blas_gemm_mixed_fp8e5m2"),
        (ScalarType::FP8E4M3, "blas_gemm_mixed_fp8e4m3"),
    ] {
        let spirv = emit_spirv::emit(&gemm_shaped_kernel(ty, name)).unwrap();
        let w = words(&spirv);
        let expect_stride = if matches!(ty, ScalarType::BF16) { 2 } else { 1 };
        assert!(
            array_strides(&w).contains(&expect_stride),
            "{name}: expected ArrayStride {expect_stride}"
        );
        spirv_val(name, &spirv);
    }
}

#[test]
fn f32_buffers_declare_no_narrow_capabilities() {
    let spirv = emit_spirv::emit(&narrow_kernel(ScalarType::F32, "f32_roundtrip")).unwrap();
    let w = words(&spirv);
    assert!(!has_capability(&w, CAP_STORAGE_BUFFER_16BIT));
    assert!(!has_capability(&w, CAP_STORAGE_BUFFER_8BIT));
    spirv_val("f32_roundtrip", &spirv);
}
