//! Race litmus: N quarks each `atomic_exchange(&shared[0], quark_id)`
//! and write the prior value they observed to `out[quark_id]`.
//!
//! D-ext.3b.2: the first kernel exercising the trace-membership
//! comparator. Final state is non-deterministic across backends:
//!  - `shared[0]` holds whichever quark_id was the LAST exchanger.
//!    Permitted set = {0, 1, ..., N-1}.
//!  - `out[quark_id]` holds the value `shared[0]` had when this quark
//!    exchanged. Permitted set = each `out` value must be a quark_id
//!    OR the initial value (0).
//!
//! For N = 2 the full permitted candidate space is small enough to
//! enumerate in the test. We use this lane as the comparator
//! cross-check: every backend (software / metal / WGSL) lane output
//! must equal at least one of the enumerated outcomes.

use super::super::lane::Lane;
use super::super::output::{RawOutput, RawValues};

pub const NAME: &str = "race";
pub const N: u32 = 2;

/// Returns the full permitted-outcome set for the N=2 race.
///
/// Output layout: `[shared_final, out_0, out_1]` (length 3).
///
/// Two quarks (0, 1) both atomic_exchange `shared` from initial 0
/// with their own quark_id. Each captures the prior value into its
/// `out[i]` slot. The four possible interleavings:
///
///   q0 first, q1 second:  shared = 1; out_0 = 0 (saw initial); out_1 = 0 (saw q0's value).
///   q1 first, q0 second:  shared = 0; out_1 = 0 (saw initial); out_0 = 1 (saw q1's value).
///
/// Plus the "stale read" / "atomic exchange not actually atomic"
/// pathologies where both quarks see the initial 0 — these would
/// indicate a non-atomic backend and SHOULD fail.
pub fn permitted() -> Vec<Vec<u32>> {
    vec![
        // q0 first
        vec![1, 0, 0],
        // q1 first
        vec![0, 1, 0],
    ]
}

#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn build_def() -> quanta::kernel::KernelDef {
    use quanta::kernel::*;
    KernelDef {
        name: "race".into(),
        params: vec![
            // The contended cell. Field name avoids `shared` because that's
            // likely-reserved in MSL contexts and also because — like in
            // the `counter` kernel — we need it distinct from the kernel
            // name so WGSL doesn't reject it as a module-scope re-decl.
            KernelParam::FieldWrite {
                name: "cell".into(),
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
            // r0 = quark_id
            KernelOp::QuarkId { dst: Reg(0) },
            // r1 = 0 (index into shared)
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(0),
            },
            // r2 = atomic_exchange(&shared[0], quark_id)
            //   AtomicOp's val parameter is the value to swap in;
            //   dst captures the prior value.
            KernelOp::AtomicOp {
                dst: Reg(2),
                field: 0,
                index: Reg(1),
                val: Reg(0),
                op: AtomicOp::Exchange,
                ty: ScalarType::U32,
                // Relaxed: the litmus is testing exchange atomicity, not
                // ordering. The 055/056 model permits Relaxed for this
                // kernel. The trace-membership comparator accepts every
                // model-permitted outcome.
                order: MemoryOrder::Relaxed,
            },
            // out[quark_id] = prior_value
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 3,
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
    let def = build_def();

    let fshared = gpu.field::<u32>(1).unwrap();
    let fout = gpu.field::<u32>(N as usize).unwrap();
    fshared.write(&[0u32]).unwrap();
    fout.write(&vec![0u32; N as usize]).unwrap();

    let bytes = quanta_ir::serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &fshared);
    wave.bind(1, &fout);
    // Dispatch one workgroup of `workgroup_size` threads. Using
    // `wave_dispatch([1, 1, 1])` (groups, not total quarks) avoids
    // platform-specific `dispatchThreads`-with-mismatched-grid quirks
    // — Metal's dispatchThreads with grid=[2,1,1] threadgroup=[2,1,1]
    // was observed to silently drop the second thread on macOS-14.
    let mut pulse = gpu.wave_dispatch(&wave, [1, 1, 1]).unwrap();
    pulse.wait().unwrap();

    let mut combined = vec![fshared.read().unwrap()[0]];
    combined.extend(fout.read().unwrap());
    RawOutput {
        lane,
        kernel: NAME,
        values: RawValues::U32(combined),
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
