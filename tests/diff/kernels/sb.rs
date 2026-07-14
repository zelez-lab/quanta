//! Store-Buffer litmus kernel (race-freedom L2, Phase 1).
//!
//! Mirrors `specs/verify/herd7/store_buffer.litmus` and
//! `store_buffer_sc.litmus`. Two variants exercise the boundary between
//! release/acquire (which ALLOWS the SB anomaly) and sequential
//! consistency (which FORBIDS it):
//!
//!   - `Variant::RelAcq` — the shared cells are accessed with plain
//!     `Store` / `Load` and an `AcqRel` `Fence` between each quark's
//!     store and its load. Under pure release/acquire the SB anomaly
//!     (both readers see 0) is ALLOWED. We assert it is in the allowed
//!     set but never hard-require it (an in-order device never shows it).
//!
//!   - `Variant::SeqCst` — the shared cells are accessed with **atomic**
//!     operations (an atomic exchange for the store, an atomic OR-0 for
//!     the load) with a `SeqCst` `Fence` between. The anomaly is
//!     FORBIDDEN; we assert its count is 0.
//!
//! Why the SeqCst variant switches to atomics — an empirical finding.
//! On real Metal, a `SeqCst` device thread-fence between *plain*
//! (non-atomic) buffer accesses does NOT forbid SB: the fence orders
//! memory operations but the plain store and load to different addresses
//! are not drawn into a single sequentially-consistent total order, so
//! `[0,0]` still appears (~6k / 131072 observed). Making the shared
//! accesses atomic pulls them into coherence order; with the SeqCst
//! fence the anomaly then vanishes (0 / 131072 observed on Metal). This
//! matches the abstract model only when the SB accesses are atomic —
//! a fence alone is not enough. The rel/acq variant deliberately keeps
//! plain accesses so the anomaly remains reachable.
//!
//! Split instance layout (two quarks per instance, see mp.rs):
//!   role 0:  x[i] = 1; Fence(order); r0 = y[i]; obs0[i] = r0
//!   role 1:  y[i] = 1; Fence(order); r1 = x[i]; obs1[i] = r1
//!
//! Observation vector per instance = `[r0, r1]`. The SB anomaly is
//! `[0, 0]`.

use super::super::histogram::Histogram;

pub const NAME: &str = "sb";

/// Instances per dispatch (`2 * INSTANCES` quarks).
pub const INSTANCES: u32 = 131_072; // 2^17

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    RelAcq,
    SeqCst,
}

impl Variant {
    pub fn tag(self) -> &'static str {
        match self {
            Variant::RelAcq => "sb-relacq",
            Variant::SeqCst => "sb-seqcst",
        }
    }
}

/// Outcomes permitted by the model for a variant:
///  - `[1,1]` both saw the other, `[0,1]`/`[1,0]` one saw the other:
///    always allowed.
///  - `[0,0]` the SB anomaly: allowed only under RelAcq.
pub fn allowed(variant: Variant) -> Vec<Vec<u32>> {
    let mut v = vec![vec![0, 1], vec![1, 0], vec![1, 1]];
    if variant == Variant::RelAcq {
        v.push(vec![0, 0]);
    }
    v
}

/// The SB anomaly.
pub fn anomaly() -> Vec<u32> {
    vec![0, 0]
}

#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn build_def(variant: Variant) -> quanta::kernel::KernelDef {
    use quanta::kernel::*;

    let order = match variant {
        Variant::RelAcq => MemoryOrder::AcqRel,
        Variant::SeqCst => MemoryOrder::SeqCst,
    };
    let atomic = variant == Variant::SeqCst;

    // Split instance layout (see mp.rs): role-0 quarks occupy the low
    // half of the grid (g < INSTANCES), role-1 the high half. Instance
    // index `i = g % INSTANCES`. The two sides of each instance live in
    // different workgroups so they are not co-scheduled in lockstep.
    //
    // Registers:
    //   r0 = quark_id (g); r1 = INSTANCES; r2 = i = g % INSTANCES;
    //   r6 = role0? = (g < INSTANCES); r4 = 1 (store value);
    //   r5 = 0 (atomic-load OR operand); r7 = loaded value;
    //   r8 = discarded RMW result

    // A "store to `field`" — atomic exchange (discard old) when `atomic`,
    // else a plain Store. `val_reg` holds the value to write.
    let store = |field: u32, val_reg: u32| -> Vec<KernelOp> {
        if atomic {
            vec![KernelOp::AtomicOp {
                dst: Reg(8),
                field,
                index: Reg(2),
                val: Reg(val_reg),
                op: AtomicOp::Exchange,
                ty: ScalarType::U32,
                order: MemoryOrder::Relaxed,
            }]
        } else {
            vec![KernelOp::Store {
                field,
                index: Reg(2),
                src: Reg(val_reg),
                ty: ScalarType::U32,
            }]
        }
    };

    // A "load from `field` into r7" — atomic OR-0 (no-op write, returns
    // current value) when `atomic`, else a plain Load.
    let load = |field: u32| -> Vec<KernelOp> {
        if atomic {
            vec![KernelOp::AtomicOp {
                dst: Reg(7),
                field,
                index: Reg(2),
                val: Reg(5), // OR 0 => value unchanged, returns current
                op: AtomicOp::Or,
                ty: ScalarType::U32,
                order: MemoryOrder::Relaxed,
            }]
        } else {
            vec![KernelOp::Load {
                dst: Reg(7),
                field,
                index: Reg(2),
                ty: ScalarType::U32,
            }]
        }
    };

    let mut role0 = store(0, 4); // x[i] = 1
    role0.push(KernelOp::Fence { order });
    role0.extend(load(1)); // r0 = y[i]
    role0.push(KernelOp::Store {
        field: 2,
        index: Reg(2),
        src: Reg(7),
        ty: ScalarType::U32,
    });

    let mut role1 = store(1, 4); // y[i] = 1
    role1.push(KernelOp::Fence { order });
    role1.extend(load(0)); // r1 = x[i]
    role1.push(KernelOp::Store {
        field: 3,
        index: Reg(2),
        src: Reg(7),
        ty: ScalarType::U32,
    });

    KernelDef {
        name: "sb".into(),
        params: vec![
            KernelParam::FieldWrite {
                name: "sb_x".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "sb_y".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "sb_obs0".into(),
                slot: 2,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "sb_obs1".into(),
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
            KernelOp::Const {
                dst: Reg(4),
                value: ConstValue::U32(1),
            },
            KernelOp::Const {
                dst: Reg(5),
                value: ConstValue::U32(0),
            },
            // role0? = (g < INSTANCES)
            KernelOp::Cmp {
                dst: Reg(6),
                a: Reg(0),
                b: Reg(1),
                op: CmpOp::Lt,
                ty: ScalarType::U32,
            },
            KernelOp::Branch {
                cond: Reg(6),
                then_ops: role0,
                else_ops: role1,
            },
        ],
        body_source: None,
        next_reg: 9,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [256, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Run one SB variant on `gpu`, returning the `[r0, r1]` histogram over
/// `INSTANCES` instances.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
pub fn run_on(gpu: &quanta::Gpu, variant: Variant) -> Histogram {
    let n = INSTANCES as usize;
    let def = build_def(variant);

    let x = gpu.field::<u32>(n).unwrap();
    let y = gpu.field::<u32>(n).unwrap();
    let obs0 = gpu.field::<u32>(n).unwrap();
    let obs1 = gpu.field::<u32>(n).unwrap();
    x.write(&vec![0u32; n]).unwrap();
    y.write(&vec![0u32; n]).unwrap();
    obs0.write(&vec![0xFFFF_FFFFu32; n]).unwrap();
    obs1.write(&vec![0xFFFF_FFFFu32; n]).unwrap();

    let bytes = quanta_ir::serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &x);
    wave.bind(1, &y);
    wave.bind(2, &obs0);
    wave.bind(3, &obs1);

    let mut pulse = gpu.dispatch(&wave, 2 * INSTANCES).unwrap();
    pulse.wait().unwrap();

    let r0 = obs0.read().unwrap();
    let r1 = obs1.read().unwrap();

    let mut hist = Histogram::new();
    for i in 0..n {
        hist.record(vec![r0[i], r1[i]]);
    }
    hist
}

#[cfg(feature = "software")]
pub fn run_software(variant: Variant) -> Histogram {
    run_on(&quanta::init_cpu(), variant)
}
