//! SAXPY: out[i] = a * x[i] + y[i]
//!
//! Float kernel. Differential tolerance is ≤ 1 ULP versus the
//! pure-Rust reference, which uses the same `mul-then-add`
//! sequence as the IR (no FMA contraction).

use super::super::lane::Lane;
use super::super::output::{RawOutput, RawValues};

pub const NAME: &str = "saxpy";
pub const N: usize = 1024;
pub const A: f32 = 2.5;

pub fn inputs() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..N).map(|i| (i as f32) * 0.125).collect();
    let y: Vec<f32> = (0..N).map(|i| 1.0 - (i as f32) * 0.0625).collect();
    (x, y)
}

pub fn run_reference() -> RawOutput {
    let (x, y) = inputs();
    let mut out = vec![0.0f32; N];
    for i in 0..N {
        out[i] = A * x[i] + y[i];
    }
    RawOutput {
        lane: Lane::Reference,
        kernel: NAME,
        values: RawValues::F32(out),
    }
}

#[cfg(feature = "software")]
pub fn run_software() -> RawOutput {
    use quanta::kernel::*;

    let (x, y) = inputs();

    let def = KernelDef {
        name: "saxpy".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "x".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "y".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
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
            KernelOp::Const {
                dst: Reg(3),
                value: ConstValue::F32(A),
            },
            KernelOp::BinOp {
                dst: Reg(4),
                a: Reg(3),
                b: Reg(1),
                op: BinOp::Mul,
                ty: ScalarType::F32,
            },
            KernelOp::BinOp {
                dst: Reg(5),
                a: Reg(4),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(5),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 6,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };

    let gpu = quanta::init_cpu();
    let fx = gpu.field::<f32>(N).unwrap();
    let fy = gpu.field::<f32>(N).unwrap();
    let fout = gpu.field::<f32>(N).unwrap();
    fx.write(&x).unwrap();
    fy.write(&y).unwrap();

    let bytes = quanta_ir::serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &fx);
    wave.bind(1, &fy);
    wave.bind(2, &fout);
    let mut pulse = gpu.dispatch(&wave, N as u32).unwrap();
    pulse.wait().unwrap();

    RawOutput {
        lane: Lane::Software,
        kernel: NAME,
        values: RawValues::F32(fout.read().unwrap()),
    }
}
