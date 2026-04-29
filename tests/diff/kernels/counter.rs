//! Counter: N quarks each `atomic_add(&ctr, 1)`.
//!
//! Integer kernel. Bit-exact tolerance: every backend must produce
//! `ctr[0] == N` regardless of dispatch order. The atomicity of
//! `AtomicOp::Add` is the load-bearing property — a non-atomic
//! increment would race and produce a value < N. Differential
//! agreement here is the empirical complement to the 055 / 056
//! memory-model axioms (every observed final value must lie in the
//! permitted set, which for `atomic_add` collapses to {N}).

use super::super::lane::Lane;
use super::super::output::{RawOutput, RawValues};

pub const NAME: &str = "counter";
pub const N: u32 = 128;

pub fn run_reference() -> RawOutput {
    RawOutput {
        lane: Lane::Reference,
        kernel: NAME,
        values: RawValues::U32(vec![N]),
    }
}

#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn build_def() -> quanta::kernel::KernelDef {
    use quanta::kernel::*;
    KernelDef {
        name: "counter".into(),
        params: vec![KernelParam::FieldWrite {
            // Field name must differ from the kernel name — WGSL
            // forbids redeclaration at module scope, so a storage
            // binding and the @compute function can't share an
            // identifier. Saxpy / reduce_sum don't trip this because
            // their field names happen to differ from the kernel name.
            name: "ctr".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body: vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::U32(0),
            },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(1),
            },
            KernelOp::AtomicOp {
                dst: Reg(2),
                field: 0,
                index: Reg(0),
                val: Reg(1),
                op: AtomicOp::Add,
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 3,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_on(gpu: &quanta::Gpu, lane: Lane) -> RawOutput {
    let def = build_def();
    let fctr = gpu.field::<u32>(1).unwrap();
    fctr.write(&[0u32]).unwrap();

    let bytes = quanta_ir::serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &fctr);
    let mut pulse = gpu.dispatch(&wave, N).unwrap();
    pulse.wait().unwrap();

    RawOutput {
        lane,
        kernel: NAME,
        values: RawValues::U32(fctr.read().unwrap()),
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
