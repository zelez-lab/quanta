//! Integration tests for the CPU software device.
//!
//! Verifies that `quanta::init_cpu()` returns a working device that can:
//! - Allocate and read/write fields
//! - JIT-compile a KernelDef and dispatch it
//! - Produce correct computation results
//!
//! Run: cargo test --test cpu_device --features software

#![cfg(feature = "software")]

use quanta::kernel::*;

// ═══════════════════════════════════════════════════════════════════════════
// Basic device operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn init_cpu_returns_device() {
    let gpu = quanta::init_cpu();
    assert_eq!(gpu.caps().vendor, quanta::Vendor::Software);
    assert_eq!(gpu.name(), "Quanta CPU (software)");
}

#[test]
fn cpu_field_roundtrip() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let data = field.read().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn cpu_field_u32_roundtrip() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<u32>(3).unwrap();
    field.write(&[10, 20, 30]).unwrap();
    let data = field.read().unwrap();
    assert_eq!(data, vec![10, 20, 30]);
}

#[test]
fn cpu_wave_rejects_binary() {
    let gpu = quanta::init_cpu();
    let result = gpu.wave(&[0, 1, 2, 3]);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// JIT kernel dispatch — add_one
// ═══════════════════════════════════════════════════════════════════════════

fn build_add_one_kernel() -> Vec<u8> {
    // Build a KernelDef that does: data[quark_id()] += 1.0
    let def = KernelDef {
        name: "add_one".into(),
        params: vec![KernelParam::FieldWrite {
            name: "data".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(1.0),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_add_one() {
    let gpu = quanta::init_cpu();

    // Create field with data [0, 10, 20, 30]
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[0.0, 10.0, 20.0, 30.0]).unwrap();

    // Compile and dispatch
    let kernel_bytes = build_add_one_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();

    // Verify: each element should be incremented by 1
    let result = field.read().unwrap();
    assert_eq!(result, vec![1.0, 11.0, 21.0, 31.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// JIT kernel dispatch — vector_add (two inputs, one output)
// ═══════════════════════════════════════════════════════════════════════════

fn build_vector_add_kernel() -> Vec<u8> {
    // out[i] = a[i] + b[i]
    let def = KernelDef {
        name: "vector_add".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
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
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_vector_add() {
    let gpu = quanta::init_cpu();

    let a = gpu.field::<f32>(4).unwrap();
    let b = gpu.field::<f32>(4).unwrap();
    let out = gpu.field::<f32>(4).unwrap();

    a.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    b.write(&[10.0, 20.0, 30.0, 40.0]).unwrap();

    let kernel_bytes = build_vector_add_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &out);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();

    let result = out.read().unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Branching kernel — conditional write
// ═══════════════════════════════════════════════════════════════════════════

fn build_threshold_kernel() -> Vec<u8> {
    // if data[i] > 5.0 { data[i] = 1.0 } else { data[i] = 0.0 }
    let def = KernelDef {
        name: "threshold".into(),
        params: vec![KernelParam::FieldWrite {
            name: "data".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(5.0),
            },
            KernelOp::Cmp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: CmpOp::Gt,
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(4),
                value: ConstValue::F32(1.0),
            },
            KernelOp::Const {
                dst: Reg(5),
                value: ConstValue::F32(0.0),
            },
            KernelOp::Branch {
                cond: Reg(3),
                then_ops: vec![KernelOp::Store {
                    field: 0,
                    index: Reg(0),
                    src: Reg(4),
                    ty: ScalarType::F32,
                }],
                else_ops: vec![KernelOp::Store {
                    field: 0,
                    index: Reg(0),
                    src: Reg(5),
                    ty: ScalarType::F32,
                }],
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
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_branch() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(6).unwrap();
    field.write(&[1.0, 10.0, 3.0, 7.0, 5.0, 6.0]).unwrap();

    let kernel_bytes = build_threshold_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);
    let mut pulse = gpu.dispatch(&wave, 6).unwrap();
    pulse.wait().unwrap();

    let result = field.read().unwrap();
    // >5: 10, 7, 6 -> 1.0; <=5: 1, 3, 5 -> 0.0
    assert_eq!(result, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Loop kernel — sum over a loop
// ═══════════════════════════════════════════════════════════════════════════

fn build_loop_sum_kernel() -> Vec<u8> {
    // data[quark_id()] = sum of 0..10 = 45
    let def = KernelDef {
        name: "loop_sum".into(),
        params: vec![KernelParam::FieldWrite {
            name: "data".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(0), // accumulator
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::U32(10), // loop count
            },
            KernelOp::Loop {
                count: Reg(2),
                iter_reg: Reg(3),
                body: vec![KernelOp::BinOp {
                    dst: Reg(1),
                    a: Reg(1),
                    b: Reg(3),
                    op: BinOp::Add,
                    ty: ScalarType::U32,
                }],
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(1),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_loop() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<u32>(3).unwrap();

    let kernel_bytes = build_loop_sum_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);
    let mut pulse = gpu.dispatch(&wave, 3).unwrap();
    pulse.wait().unwrap();

    let result = field.read().unwrap();
    // sum(0..10) = 0+1+2+...+9 = 45
    assert_eq!(result, vec![45, 45, 45]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch reuse — same wave dispatched multiple times
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cpu_dispatch_reuse_wave() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[0.0, 0.0, 0.0, 0.0]).unwrap();

    let kernel_bytes = build_add_one_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);

    // Dispatch 3 times: each adds 1
    for _ in 0..3 {
        let mut pulse = gpu.dispatch(&wave, 4).unwrap();
        pulse.wait().unwrap();
    }

    let result = field.read().unwrap();
    assert_eq!(result, vec![3.0, 3.0, 3.0, 3.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Larger dispatch — verify correct thread indexing
// ═══════════════════════════════════════════════════════════════════════════

fn build_identity_kernel() -> Vec<u8> {
    // data[i] = i (write quark_id as f32)
    let def = KernelDef {
        name: "identity".into(),
        params: vec![KernelParam::FieldWrite {
            name: "data".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Cast {
                dst: Reg(1),
                src: Reg(0),
                from: ScalarType::U32,
                to: ScalarType::F32,
            },
            KernelOp::Store {
                field: 0,
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
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_256_threads() {
    let gpu = quanta::init_cpu();
    let n = 256;
    let field = gpu.field::<f32>(n).unwrap();

    let kernel_bytes = build_identity_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();

    let result = field.read().unwrap();
    let expected: Vec<f32> = (0..n).map(|i| i as f32).collect();
    assert_eq!(result, expected);
}

// ═══════════════════════════════════════════════════════════════════════════
// Math operations
// ═══════════════════════════════════════════════════════════════════════════

fn build_sqrt_kernel() -> Vec<u8> {
    // data[i] = sqrt(data[i])
    let def = KernelDef {
        name: "sqrt".into(),
        params: vec![KernelParam::FieldWrite {
            name: "data".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::MathCall {
                dst: Reg(2),
                func: MathFn::Sqrt,
                args: vec![Reg(1)],
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::F32,
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
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_math_sqrt() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[4.0, 9.0, 16.0, 25.0]).unwrap();

    let kernel_bytes = build_sqrt_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();

    let result = field.read().unwrap();
    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Atomic operations
// ═══════════════════════════════════════════════════════════════════════════

fn build_atomic_add_kernel() -> Vec<u8> {
    // atomic_add(&counter[0], 1) for each thread
    let def = KernelDef {
        name: "atomic_add".into(),
        params: vec![KernelParam::FieldWrite {
            name: "counter".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body: vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::U32(0), // index 0
            },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(1), // add 1
            },
            KernelOp::AtomicOp {
                dst: Reg(2),
                field: 0,
                index: Reg(0),
                val: Reg(1),
                op: AtomicOp::Add,
                ty: ScalarType::U32,
                order: MemoryOrder::SeqCst,
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
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_atomic_add() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<u32>(1).unwrap();
    field.write(&[0u32]).unwrap();

    let kernel_bytes = build_atomic_add_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);
    // Dispatch 100 threads — each atomically adds 1 to counter[0]
    let mut pulse = gpu.wave_dispatch(&wave, [2, 1, 1]).unwrap();
    pulse.wait().unwrap();

    let result = field.read().unwrap();
    // 2 groups * 64 threads = 128
    assert_eq!(result, vec![128]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared memory
// ═══════════════════════════════════════════════════════════════════════════

fn build_shared_sum_kernel() -> Vec<u8> {
    // Each thread loads data[quark_id] into shared[proton_id],
    // then thread 0 sums all shared values and writes to output[0].
    let def = KernelDef {
        name: "shared_sum".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "data".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "output".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::ProtonId { dst: Reg(1) },
            KernelOp::ProtonSize { dst: Reg(2) },
            // Declare shared memory
            KernelOp::SharedDecl {
                id: 0,
                ty: ScalarType::F32,
                count: 64,
            },
            // Load from global to shared
            KernelOp::Load {
                dst: Reg(3),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::SharedStore {
                id: 0,
                index: Reg(1),
                src: Reg(3),
                ty: ScalarType::F32,
            },
            KernelOp::Barrier,
            // Thread 0 sums all values
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
                then_ops: vec![
                    KernelOp::Const {
                        dst: Reg(6),
                        value: ConstValue::F32(0.0),
                    },
                    KernelOp::Loop {
                        count: Reg(2),
                        iter_reg: Reg(7),
                        body: vec![
                            KernelOp::SharedLoad {
                                dst: Reg(8),
                                id: 0,
                                index: Reg(7),
                                ty: ScalarType::F32,
                            },
                            KernelOp::BinOp {
                                dst: Reg(6),
                                a: Reg(6),
                                b: Reg(8),
                                op: BinOp::Add,
                                ty: ScalarType::F32,
                            },
                        ],
                    },
                    KernelOp::Store {
                        field: 1,
                        index: Reg(4),
                        src: Reg(6),
                        ty: ScalarType::F32,
                    },
                ],
                else_ops: vec![],
            },
        ],
        body_source: None,
        next_reg: 9,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [4, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_dispatch_shared_memory() {
    let gpu = quanta::init_cpu();
    let data = gpu.field::<f32>(4).unwrap();
    let output = gpu.field::<f32>(1).unwrap();

    data.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();

    let kernel_bytes = build_shared_sum_kernel();
    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &data);
    wave.bind(1, &output);
    let mut pulse = gpu.wave_dispatch(&wave, [1, 1, 1]).unwrap();
    pulse.wait().unwrap();

    let result = output.read().unwrap();
    assert!(
        (result[0] - 10.0).abs() < 1e-6,
        "sum should be 10.0, got {}",
        result[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Scan library integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn scan_kernel_def_roundtrip_via_cpu() {
    let gpu = quanta::init_cpu();

    // Just verify the scan kernel can be JIT-compiled
    let bytes = quanta::scan::exclusive_scan_f32_bytes();
    let wave = gpu.wave_jit(&bytes);
    assert!(wave.is_ok(), "scan kernel should JIT-compile on CPU");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cooperative subgroup (warp) reductions on the CPU lane
//
// Regression guard: the software executor must reduce SubgroupReduceAdd
// cooperatively across a warp (32 lanes), not return each lane's own value.
// Before the fix, a 32-lane SubgroupReduceAdd returned the lane's input
// (and SubgroupSize returned 1); now every lane in a warp gets the warp sum.
// ═══════════════════════════════════════════════════════════════════════════

fn build_subgroup_reduce_add_kernel() -> Vec<u8> {
    // out[i] = subgroup_reduce_add(in[i]) — every lane in the warp gets the
    // warp-wide sum.
    let def = KernelDef {
        name: "sg_reduce_add".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
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
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::SubgroupReduceAdd {
                dst: Reg(2),
                src: Reg(1),
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 3,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [32, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    quanta_ir::serialize_kernel(&def)
}

#[test]
fn cpu_subgroup_reduce_add_is_cooperative() {
    let gpu = quanta::init_cpu();
    // One warp of 32 lanes: inputs 1.0..=32.0, sum = 528.
    let input: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let a = gpu.field::<f32>(32).unwrap();
    let out = gpu.field::<f32>(32).unwrap();
    a.write(&input).unwrap();

    let bytes = build_subgroup_reduce_add_kernel();
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 32).unwrap().wait().unwrap();

    let got = out.read().unwrap();
    // Every lane in the warp sees the full warp sum (not its own value).
    for (i, &v) in got.iter().enumerate() {
        assert!((v - 528.0).abs() <= 1e-3, "lane {i}: {v} (want 528)");
    }
}
