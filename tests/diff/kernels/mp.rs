//! Message-Passing litmus kernel (race-freedom L2, Phase 1).
//!
//! Mirrors `specs/verify/herd7/message_passing.litmus`. Many independent
//! MP instances are packed into one dispatch (see `diff::histogram` for
//! the instance layout). Ordering is carried by the pair of atomic
//! accesses plus a `Fence { order }`.
//!
//! Per instance `i` (two quarks, global id `g`, `i = g % INSTANCES`):
//!   role 0 (producer):  atomic data[i] = 42; Fence(Release);
//!                       atomic flag[i] = 1
//!   role 1 (consumer):  f = atomic flag[i]; Fence(Acquire);
//!                       d = atomic data[i];
//!                       obs_flag[i] = f; obs_data[i] = d
//!
//! Observation vector per instance = `[flag_seen, data_seen]`.
//!
//! The herd7-forbidden outcome is `[1, 0]` (saw the flag set but not the
//! data). Good outcomes: `[0, 0]`, `[0, 42]`, `[1, 42]`.
//!
//! Why the accesses are ATOMIC, not plain `Store`/`Load` — an empirical
//! finding. On real Metal, an acquire/release device thread-fence between
//! *plain* (non-atomic) buffer accesses does NOT forbid the MP anomaly:
//! `[1, 0]` was observed reproducibly (tens to ~1500 per 131072). The
//! fence orders memory operations, but a plain store and a plain load to
//! different addresses are not pulled into the synchronizes-with edge
//! that MP relies on. Making both `data` and `flag` atomic (which enters
//! them into coherence order) plus the fence forbids `[1, 0]` reliably
//! (0 / 131072 across many runs). We therefore emit atomic accesses:
//! atomic store = an `Exchange` (old value discarded), atomic load = an
//! `Or` with 0 (value unchanged, returns current). NOTE: Quanta's IR has
//! no dedicated ordered atomic load/store today, and `AtomicOp` is
//! `memory_order_relaxed` on Metal, so the fence carries all ordering —
//! this is the honest current shape of the surface, tracked as the L2
//! follow-up (ordered atomic load/store ops).

use super::super::histogram::Histogram;

pub const NAME: &str = "mp";

/// Instances per dispatch. `2 * INSTANCES` quarks run per dispatch.
pub const INSTANCES: u32 = 131_072; // 2^17 -> 262_144 quarks in one dispatch

pub const DATA_VALUE: u32 = 42;

/// The four model-permitted observation vectors `[flag_seen, data_seen]`.
pub fn allowed() -> Vec<Vec<u32>> {
    vec![vec![0, 0], vec![0, DATA_VALUE], vec![1, DATA_VALUE]]
}

/// The herd7-forbidden observation vector.
pub fn forbidden() -> Vec<Vec<u32>> {
    vec![vec![1, 0]]
}

#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn build_def() -> quanta::kernel::KernelDef {
    use quanta::kernel::*;

    // Instance layout (split, to spread each pair across workgroups):
    // global quark id `g` in [0, 2*INSTANCES). Producers occupy the low
    // half (g < INSTANCES), consumers the high half. Instance index is
    // `i = g % INSTANCES` (== g for producers, g - INSTANCES for
    // consumers). Producer i and consumer i therefore live INSTANCES
    // quarks apart in the grid — different workgroups — so they are NOT
    // co-scheduled in lockstep, which is what a weak-memory window needs.
    //
    // Registers:
    //   r0 = quark_id (g)
    //   r1 = INSTANCES
    //   r2 = i   = g % INSTANCES
    //   r3 = producer? = (g < INSTANCES)
    //   r4 = 42  (data value, exchanged in)
    //   r5 = 1   (flag value, exchanged in)
    //   r6 = 0   (OR operand for the atomic loads: value unchanged)
    //   r7 = discarded RMW result on the producer side
    //   r8 = loaded flag
    //   r9 = loaded data
    let producer = vec![
        // atomic data[i] = 42 (exchange, old value discarded)
        KernelOp::AtomicOp {
            dst: Reg(7),
            field: 0,
            index: Reg(2),
            val: Reg(4),
            op: AtomicOp::Exchange,
            ty: ScalarType::U32,
            order: MemoryOrder::Relaxed,
        },
        // release fence: prior data store becomes visible to any acquire
        KernelOp::Fence {
            order: MemoryOrder::Release,
        },
        // atomic flag[i] = 1 (exchange, old value discarded)
        KernelOp::AtomicOp {
            dst: Reg(7),
            field: 1,
            index: Reg(2),
            val: Reg(5),
            op: AtomicOp::Exchange,
            ty: ScalarType::U32,
            order: MemoryOrder::Relaxed,
        },
    ];

    let consumer = vec![
        // f = atomic flag[i] (OR 0: value unchanged, returns current)
        KernelOp::AtomicOp {
            dst: Reg(8),
            field: 1,
            index: Reg(2),
            val: Reg(6),
            op: AtomicOp::Or,
            ty: ScalarType::U32,
            order: MemoryOrder::Relaxed,
        },
        // acquire fence: if we saw the flag, the paired data store is visible
        KernelOp::Fence {
            order: MemoryOrder::Acquire,
        },
        // d = atomic data[i] (OR 0)
        KernelOp::AtomicOp {
            dst: Reg(9),
            field: 0,
            index: Reg(2),
            val: Reg(6),
            op: AtomicOp::Or,
            ty: ScalarType::U32,
            order: MemoryOrder::Relaxed,
        },
        // obs_flag[i] = f
        KernelOp::Store {
            field: 2,
            index: Reg(2),
            src: Reg(8),
            ty: ScalarType::U32,
        },
        // obs_data[i] = d
        KernelOp::Store {
            field: 3,
            index: Reg(2),
            src: Reg(9),
            ty: ScalarType::U32,
        },
    ];

    KernelDef {
        name: "mp".into(),
        params: vec![
            KernelParam::FieldWrite {
                name: "mp_data".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "mp_flag".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "mp_obs_flag".into(),
                slot: 2,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "mp_obs_data".into(),
                slot: 3,
                scalar_type: ScalarType::U32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(INSTANCES),
            },
            // i = g % INSTANCES
            KernelOp::BinOp {
                dst: Reg(2),
                a: Reg(0),
                b: Reg(1),
                op: BinOp::Rem,
                ty: ScalarType::U32,
            },
            // producer? = (g < INSTANCES)
            KernelOp::Cmp {
                dst: Reg(3),
                a: Reg(0),
                b: Reg(1),
                op: CmpOp::Lt,
                ty: ScalarType::U32,
            },
            KernelOp::Const {
                dst: Reg(4),
                value: ConstValue::U32(DATA_VALUE),
            },
            KernelOp::Const {
                dst: Reg(5),
                value: ConstValue::U32(1),
            },
            KernelOp::Const {
                dst: Reg(6),
                value: ConstValue::U32(0),
            },
            KernelOp::Branch {
                cond: Reg(3),
                then_ops: producer,
                else_ops: consumer,
            },
        ],
        body_source: None,
        next_reg: 10,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [256, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Run the MP litmus batch on `gpu`, returning the observation
/// histogram over `INSTANCES` instances.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
pub fn run_on(gpu: &quanta::Gpu) -> Histogram {
    let n = INSTANCES as usize;
    let def = build_def();

    let data = gpu.field::<u32>(n).unwrap();
    let flag = gpu.field::<u32>(n).unwrap();
    let obs_flag = gpu.field::<u32>(n).unwrap();
    let obs_data = gpu.field::<u32>(n).unwrap();
    data.write(&vec![0u32; n]).unwrap();
    flag.write(&vec![0u32; n]).unwrap();
    // Seed observers with a sentinel that is NOT a legal outcome so a
    // quark that never ran would surface as an out-of-allowed-set value.
    obs_flag.write(&vec![0xFFFF_FFFFu32; n]).unwrap();
    obs_data.write(&vec![0xFFFF_FFFFu32; n]).unwrap();

    let bytes = quanta_ir::serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &data);
    wave.bind(1, &flag);
    wave.bind(2, &obs_flag);
    wave.bind(3, &obs_data);

    let mut pulse = gpu.dispatch(&wave, 2 * INSTANCES).unwrap();
    pulse.wait().unwrap();

    let of = obs_flag.read().unwrap();
    let od = obs_data.read().unwrap();

    let mut hist = Histogram::new();
    for i in 0..n {
        hist.record(vec![of[i], od[i]]);
    }
    hist
}

#[cfg(feature = "software")]
pub fn run_software() -> Histogram {
    run_on(&quanta::init_cpu())
}

#[cfg(feature = "metal")]
pub fn run_metal(gpu: &quanta::Gpu) -> Histogram {
    run_on(gpu)
}

#[cfg(feature = "vulkan")]
pub fn run_vulkan(gpu: &quanta::Gpu) -> Histogram {
    run_on(gpu)
}
