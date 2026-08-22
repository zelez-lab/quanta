//! Cooperative matrices on whatever device `quanta::init()` finds — the
//! probe and the execution check for the `SPV_KHR_cooperative_matrix`
//! path (Vulkan) and the `simdgroup_matrix` path (Metal).
//!
//! Three things, in order:
//! 1. PROBE — print every shape the device enumerates. Always runs; on
//!    lavapipe this is the line that says whether the CI runner's Mesa
//!    exposes `VK_KHR_cooperative_matrix` at all.
//! 2. REFUSAL — a kernel built on a shape no device has (3×3×3 f32) must
//!    be refused with `NotSupported` at wave creation, never dispatched.
//!    Asserted on every device, including the software lane.
//! 3. EXECUTION — if the device enumerates a uniform-type shape (one
//!    element type for A, B, C and D — the only form Quanta's ops can
//!    express today), run one `C += A·B` tile of that shape on integer-
//!    valued inputs and compare bit-exactly with a host oracle. Skips
//!    loudly when no such shape exists (NVIDIA/AMD enumerate f16-in /
//!    f32-out first; that mixed form is the next op-shape increment).

use quanta::{CoopMatrixShape, ScalarType};
use quanta_ir::{ConstValue, KernelDef, KernelOp, KernelParam, MatrixFrag, Reg};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// One tile: `C[m×n] += A[m×k] · B[k×n]`, all three dense row-major with
/// their natural strides, one subgroup (every subgroup in the workgroup
/// computes the same tile — idempotent, so the subgroup width does not
/// matter).
fn tile_kernel(m: u8, n: u8, k: u8, ty: ScalarType) -> KernelDef {
    let field = |name: &str, slot: u32, write: bool| {
        if write {
            KernelParam::FieldWrite {
                name: name.into(),
                slot,
                scalar_type: ty,
            }
        } else {
            KernelParam::FieldRead {
                name: name.into(),
                slot,
                scalar_type: ty,
            }
        }
    };
    let konst = |dst: u32, v: u32| KernelOp::Const {
        dst: Reg(dst),
        value: ConstValue::U32(v),
    };
    let body = vec![
        konst(0, 0),
        konst(1, u32::from(k)), // A row stride
        konst(2, u32::from(n)), // B and C row stride
        KernelOp::CooperativeMatrixLoad {
            dst: Reg(3),
            field: 0,
            index: Reg(0),
            stride: Reg(1),
            frag: MatrixFrag::A,
            from_shared: false,
            m,
            n,
            k,
            ty,
        },
        KernelOp::CooperativeMatrixLoad {
            dst: Reg(4),
            field: 1,
            index: Reg(0),
            stride: Reg(2),
            frag: MatrixFrag::B,
            from_shared: false,
            m,
            n,
            k,
            ty,
        },
        KernelOp::CooperativeMatrixLoad {
            dst: Reg(5),
            field: 2,
            index: Reg(0),
            stride: Reg(2),
            frag: MatrixFrag::Accumulator,
            from_shared: false,
            m,
            n,
            k,
            ty,
        },
        KernelOp::CooperativeMMA {
            dst: Reg(6),
            a: Reg(3),
            b: Reg(4),
            c: Reg(5),
            m,
            n,
            k,
            ty,
        },
        KernelOp::CooperativeMatrixStore {
            field: 2,
            index: Reg(0),
            stride: Reg(2),
            src: Reg(6),
            m,
            n,
            k,
            ty,
        },
    ];
    KernelDef {
        name: format!("coopmat_tile_{m}x{n}x{k}_{ty:?}"),
        params: vec![
            field("a", 0, false),
            field("b", 1, false),
            field("c", 2, true),
        ],
        body,
        body_source: None,
        next_reg: 7,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [32, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

fn uniform_shape(s: &CoopMatrixShape) -> bool {
    s.ab_ty == s.c_ty && s.c_ty == s.result_ty
}

#[test]
fn probe_prints_enumerated_shapes() {
    let Some(gpu) = try_gpu() else {
        eprintln!("SKIP: no device");
        return;
    };
    let shapes = gpu.cooperative_matrix_shapes();
    eprintln!(
        "[coopmat probe] device `{}`: supports_cooperative_matrix = {}, {} shape(s)",
        gpu.name(),
        gpu.supports_cooperative_matrix(),
        shapes.len()
    );
    for s in &shapes {
        eprintln!(
            "[coopmat probe]   {}x{}x{}  A/B {:?}  C {:?}  D {:?}{}",
            s.m,
            s.n,
            s.k,
            s.ab_ty,
            s.c_ty,
            s.result_ty,
            if uniform_shape(s) {
                "  (uniform — runnable by the IR today)"
            } else {
                ""
            }
        );
    }
    // The capability answer must be exactly "some shape exists".
    assert_eq!(gpu.supports_cooperative_matrix(), !shapes.is_empty());
}

#[test]
fn unenumerated_shape_is_refused_at_wave_creation() {
    let Some(gpu) = try_gpu() else {
        eprintln!("SKIP: no device");
        return;
    };
    // 3×3×3 is no hardware's shape; every device must refuse it before
    // anything reaches a compiler or a driver.
    let bytes = quanta_ir::serialize_kernel(&tile_kernel(3, 3, 3, ScalarType::F32));
    match gpu.wave_jit(&bytes) {
        Ok(_) => panic!(
            "a 3x3x3 cooperative-matrix kernel was accepted by `{}`",
            gpu.name()
        ),
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("cooperative") || msg.contains("Cooperative"),
                "refusal must name cooperative matrices, got: {msg}"
            );
        }
    }
}

#[test]
fn uniform_shape_tile_matches_host_oracle() {
    let Some(gpu) = try_gpu() else {
        eprintln!("SKIP: no device");
        return;
    };
    let shapes = gpu.cooperative_matrix_shapes();
    // Uniform f32 only: f16 fields have no host type yet (storage is u16
    // bits), so an f16-only device is reported, not exercised.
    let Some(shape) = shapes
        .iter()
        .copied()
        .find(|s| uniform_shape(s) && s.ab_ty == ScalarType::F32)
    else {
        eprintln!(
            "SKIP: `{}` enumerates no uniform-f32 cooperative-matrix shape ({} shape(s) total{})",
            gpu.name(),
            shapes.len(),
            if shapes
                .iter()
                .any(|s| uniform_shape(s) && s.ab_ty == ScalarType::F16)
            {
                "; a uniform f16 shape exists — needs f16 fields"
            } else {
                ""
            }
        );
        return;
    };
    let (m, n, k) = (shape.m as usize, shape.n as usize, shape.k as usize);
    eprintln!(
        "[coopmat exec] `{}`: {}x{}x{} {:?}",
        gpu.name(),
        m,
        n,
        k,
        shape.ab_ty
    );
    // Small integer-valued inputs (|values| ≤ 3): every product and
    // partial sum is an exact f32, so the comparison is bitwise.
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 + 1) % 7) as f32 - 3.0).collect();
    let c0: Vec<f32> = (0..m * n).map(|i| ((i * 3 + 2) % 7) as f32 - 3.0).collect();
    let mut expected = c0.clone();
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[r * k + i] * b[i * n + col];
            }
            expected[r * n + col] += acc;
        }
    }
    let bytes = quanta_ir::serialize_kernel(&tile_kernel(shape.m, shape.n, shape.k, shape.ab_ty));
    let mut wave = gpu
        .wave_jit(&bytes)
        .expect("enumerated shape must create a wave");
    let af = gpu.field::<f32>(m * k).unwrap();
    let bf = gpu.field::<f32>(k * n).unwrap();
    let cf = gpu.field::<f32>(m * n).unwrap();
    af.write(&a).unwrap();
    bf.write(&b).unwrap();
    cf.write(&c0).unwrap();
    wave.bind(0, &af);
    wave.bind(1, &bf);
    wave.bind(2, &cf);
    gpu.dispatch(&wave, 32).unwrap().wait().unwrap();
    let got: Vec<f32> = cf.read().unwrap();
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "C[{}] (row {}, col {}): got {g}, expected {e}",
            i,
            i / n,
            i % n
        );
    }
}
