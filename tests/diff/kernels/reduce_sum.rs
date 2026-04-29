//! Reduce-sum: out[0] = Σ data[i]   for i in 0..N
//!
//! Integer kernel — comparator tolerance is bit-exact across lanes.
//! Single-workgroup reduction:
//!  1. each quark loads `data[quark_id]` into shared memory at `proton_id`
//!  2. workgroup barrier
//!  3. quark 0 accumulates into `out[0]` by reading-modifying-writing
//!     it once per iteration
//!
//! Why `out[0]` is the accumulator instead of a register: the WGSL
//! emitter lowers `BinOp { dst, .. }` to `let r{dst} = ...`, which is
//! immutable. Mutating a register inside a `Loop` body would shadow
//! rather than update. Routing the accumulator through a `read_write`
//! storage binding sidesteps this — every Load+Store in the loop
//! body lowers to plain assignment which works on every backend.

use super::super::lane::Lane;
use super::super::output::{RawOutput, RawValues};

pub const NAME: &str = "reduce_sum";
pub const N: u32 = 64;

pub fn inputs() -> Vec<u32> {
    (1..=N).collect()
}

pub fn run_reference() -> RawOutput {
    let sum: u32 = inputs().iter().sum();
    RawOutput {
        lane: Lane::Reference,
        kernel: NAME,
        values: RawValues::U32(vec![sum]),
    }
}

#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn build_def() -> quanta::kernel::KernelDef {
    use quanta::kernel::*;
    KernelDef {
        name: "reduce_sum".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "data".into(),
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
            KernelOp::ProtonId { dst: Reg(1) },
            KernelOp::ProtonSize { dst: Reg(2) },
            KernelOp::SharedDecl {
                id: 0,
                ty: ScalarType::U32,
                count: N,
            },
            KernelOp::Load {
                dst: Reg(3),
                field: 0,
                index: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::SharedStore {
                id: 0,
                index: Reg(1),
                src: Reg(3),
                ty: ScalarType::U32,
            },
            KernelOp::Barrier,
            KernelOp::Const {
                dst: Reg(4),
                value: ConstValue::U32(0),
            },
            KernelOp::Cmp {
                dst: Reg(5),
                a: Reg(1),
                b: Reg(4),
                op: CmpOp::Eq,
                ty: ScalarType::U32,
            },
            KernelOp::Branch {
                cond: Reg(5),
                then_ops: vec![KernelOp::Loop {
                    count: Reg(2),
                    iter_reg: Reg(7),
                    body: vec![
                        KernelOp::Load {
                            dst: Reg(6),
                            field: 1,
                            index: Reg(4),
                            ty: ScalarType::U32,
                        },
                        KernelOp::SharedLoad {
                            dst: Reg(8),
                            id: 0,
                            index: Reg(7),
                            ty: ScalarType::U32,
                        },
                        KernelOp::BinOp {
                            dst: Reg(9),
                            a: Reg(6),
                            b: Reg(8),
                            op: BinOp::Add,
                            ty: ScalarType::U32,
                        },
                        KernelOp::Store {
                            field: 1,
                            index: Reg(4),
                            src: Reg(9),
                            ty: ScalarType::U32,
                        },
                    ],
                }],
                else_ops: vec![],
            },
        ],
        body_source: None,
        next_reg: 10,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [N, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_on(gpu: &quanta::Gpu, lane: Lane) -> RawOutput {
    let data = inputs();
    let def = build_def();

    let fdata = gpu.field::<u32>(N as usize).unwrap();
    let fout = gpu.field::<u32>(1).unwrap();
    fdata.write(&data).unwrap();
    fout.write(&[0u32]).unwrap();

    let bytes = quanta_ir::serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &fdata);
    wave.bind(1, &fout);
    let mut pulse = gpu.dispatch(&wave, N).unwrap();
    pulse.wait().unwrap();

    RawOutput {
        lane,
        kernel: NAME,
        values: RawValues::U32(fout.read().unwrap()),
    }
}

#[cfg(feature = "software")]
pub fn run_software() -> RawOutput {
    dispatch_on(&quanta::init_cpu(), Lane::Software)
}

#[cfg(feature = "metal")]
pub fn run_metal() -> RawOutput {
    let gpu = quanta::init().expect("metal lane requires a metal-capable device");
    dispatch_on(&gpu, Lane::Metal)
}

#[cfg(feature = "vulkan")]
pub fn run_vulkan() -> RawOutput {
    let gpu = quanta::init().expect("vulkan lane requires a vulkan-capable device");
    dispatch_on(&gpu, Lane::Vulkan)
}
