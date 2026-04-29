//! Browser-side SAXPY runner for step D.2 (differential CI, WGSL lane).
//!
//! Mirrors `tests/diff/kernels/saxpy.rs` byte-for-byte: same `A`,
//! same `inputs()`, same KernelDef shape — but dispatched through
//! `WebgpuDevice` instead of `init_cpu()`. The output bytes are
//! returned to JS, which compares them ≤ 1 ULP against a reference
//! oracle computed in JS using the *same* constants.
//!
//! ## Build
//! ```sh
//! quanta build web web_diff
//! ```
//!
//! ## What the test asserts
//! - Inputs:  x[i] = i * 0.125,   y[i] = 1 - i * 0.0625,   A = 2.5,   N = 1024
//! - Output:  out[i] = A * x[i] + y[i]   (mul-then-add, no FMA contraction)
//! - Verdict: every element within 1 ULP of the JS-side reference.

#![cfg(target_arch = "wasm32")]

use quanta::GpuDevice;
use quanta::webgpu::spawn_local;
use quanta_ir::{
    AtomicOp, BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType,
    serialize_kernel,
};

unsafe extern "C" {
    fn quanta_complete_bytes(task: u32, ptr: *const u8, len: usize);
    fn quanta_complete_err(task: u32, ptr: *const u8, len: usize);
}

const A: f32 = 2.5;
const N: usize = 1024;

fn build_saxpy_kernel() -> KernelDef {
    KernelDef {
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
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

async fn run() -> Result<Vec<u8>, String> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| format!("new_async: {:?}", e))?;

    let bytes_per_field = N * core::mem::size_of::<f32>();
    let usage = quanta::FieldUsage::default_compute();

    let fx = dev
        .field_alloc(bytes_per_field, usage)
        .map_err(|e| format!("field_alloc x: {:?}", e))?;
    let fy = dev
        .field_alloc(bytes_per_field, usage)
        .map_err(|e| format!("field_alloc y: {:?}", e))?;
    let fout = dev
        .field_alloc(bytes_per_field, usage)
        .map_err(|e| format!("field_alloc out: {:?}", e))?;

    let mut x_bytes = Vec::with_capacity(bytes_per_field);
    let mut y_bytes = Vec::with_capacity(bytes_per_field);
    for i in 0..N {
        let xi = (i as f32) * 0.125;
        let yi = 1.0f32 - (i as f32) * 0.0625;
        x_bytes.extend_from_slice(&xi.to_le_bytes());
        y_bytes.extend_from_slice(&yi.to_le_bytes());
    }
    dev.field_write_bytes(fx, &x_bytes)
        .map_err(|e| format!("field_write x: {:?}", e))?;
    dev.field_write_bytes(fy, &y_bytes)
        .map_err(|e| format!("field_write y: {:?}", e))?;

    let kernel = build_saxpy_kernel();
    let kernel_bytes = serialize_kernel(&kernel);
    let mut wave = dev
        .wave_jit(&kernel_bytes)
        .map_err(|e| format!("wave_jit: {:?}", e))?;
    wave.bind_handle(0, fx);
    wave.bind_handle(1, fy);
    wave.bind_handle(2, fout);

    // 1024 quarks at workgroup_size 64 → 16 workgroups along x.
    let _pulse = dev
        .wave_dispatch(&wave, [(N as u32) / 64, 1, 1])
        .map_err(|e| format!("wave_dispatch: {:?}", e))?;

    dev.field_read_bytes_async(fout, bytes_per_field)
        .await
        .map_err(|e| format!("read_back: {:?}", e))
}

#[unsafe(no_mangle)]
pub extern "C" fn web_diff_saxpy_run(task: u32) {
    spawn_local(async move {
        match run().await {
            Ok(bytes) => unsafe {
                quanta_complete_bytes(task, bytes.as_ptr(), bytes.len());
            },
            Err(msg) => unsafe {
                quanta_complete_err(task, msg.as_ptr(), msg.len());
            },
        }
    });
}

// ───────────────────────── reduce_sum (D.3a) ─────────────────────────
//
// Mirrors `tests/diff/kernels/reduce_sum.rs`: shared-memory + barrier
// + thread-0 sums the 64 cells linearly. Output is one u32. Tolerance
// is bit-exact; the JS verdict checks `out[0] === Σ inputs`.

const REDUCE_N: u32 = 64;

fn build_reduce_sum_kernel() -> KernelDef {
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
                count: REDUCE_N,
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
                        // Accumulator lives in `out[0]` rather than a
                        // register: the WGSL emitter lowers BinOp to
                        // an immutable `let`, which would shadow not
                        // mutate inside the loop. Reading + writing
                        // the storage binding works on every backend.
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
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [REDUCE_N, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

async fn run_reduce_sum() -> Result<Vec<u8>, String> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| format!("new_async: {:?}", e))?;

    let n = REDUCE_N as usize;
    let in_bytes = n * core::mem::size_of::<u32>();
    let out_bytes = core::mem::size_of::<u32>();
    let usage = quanta::FieldUsage::default_compute();

    let fdata = dev
        .field_alloc(in_bytes, usage)
        .map_err(|e| format!("field_alloc data: {:?}", e))?;
    let fout = dev
        .field_alloc(out_bytes, usage)
        .map_err(|e| format!("field_alloc out: {:?}", e))?;

    let mut data_bytes = Vec::with_capacity(in_bytes);
    for i in 1u32..=REDUCE_N {
        data_bytes.extend_from_slice(&i.to_le_bytes());
    }
    dev.field_write_bytes(fdata, &data_bytes)
        .map_err(|e| format!("field_write data: {:?}", e))?;
    dev.field_write_bytes(fout, &0u32.to_le_bytes())
        .map_err(|e| format!("field_write out: {:?}", e))?;

    let kernel = build_reduce_sum_kernel();
    let kernel_bytes = serialize_kernel(&kernel);
    let mut wave = dev
        .wave_jit(&kernel_bytes)
        .map_err(|e| format!("wave_jit: {:?}", e))?;
    wave.bind_handle(0, fdata);
    wave.bind_handle(1, fout);

    let _pulse = dev
        .wave_dispatch(&wave, [1, 1, 1])
        .map_err(|e| format!("wave_dispatch: {:?}", e))?;

    dev.field_read_bytes_async(fout, out_bytes)
        .await
        .map_err(|e| format!("read_back: {:?}", e))
}

#[unsafe(no_mangle)]
pub extern "C" fn web_diff_reduce_sum_run(task: u32) {
    spawn_local(async move {
        match run_reduce_sum().await {
            Ok(bytes) => unsafe {
                quanta_complete_bytes(task, bytes.as_ptr(), bytes.len());
            },
            Err(msg) => unsafe {
                quanta_complete_err(task, msg.as_ptr(), msg.len());
            },
        }
    });
}

// ───────────────────────── counter (D.3b) ────────────────────────────
//
// N quarks each `atomic_add(&counter, 1)`. The atomic field is
// detected by the WGSL emitter (collect_atomic_fields) and emitted
// as `atomic<u32>` storage with `atomicAdd` in the kernel body. The
// final value must equal exactly N — anything less indicates a lost
// update from a non-atomic backend implementation.

const COUNTER_N: u32 = 128;

fn build_counter_kernel() -> KernelDef {
    KernelDef {
        name: "counter".into(),
        params: vec![KernelParam::FieldWrite {
            // Field name must differ from the kernel name — WGSL
            // forbids module-scope redeclaration, so the storage
            // binding and the @compute function can't share an
            // identifier.
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
                order: quanta_ir::MemoryOrder::SeqCst,
            },
        ],
        body_source: None,
        next_reg: 3,
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

async fn run_counter() -> Result<Vec<u8>, String> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| format!("new_async: {:?}", e))?;

    let out_bytes = core::mem::size_of::<u32>();
    let usage = quanta::FieldUsage::default_compute();

    let fcounter = dev
        .field_alloc(out_bytes, usage)
        .map_err(|e| format!("field_alloc counter: {:?}", e))?;
    dev.field_write_bytes(fcounter, &0u32.to_le_bytes())
        .map_err(|e| format!("field_write counter: {:?}", e))?;

    let kernel = build_counter_kernel();
    let kernel_bytes = serialize_kernel(&kernel);
    let mut wave = dev
        .wave_jit(&kernel_bytes)
        .map_err(|e| format!("wave_jit: {:?}", e))?;
    wave.bind_handle(0, fcounter);

    // 128 quarks at workgroup_size 64 → 2 workgroups along x.
    let _pulse = dev
        .wave_dispatch(&wave, [COUNTER_N / 64, 1, 1])
        .map_err(|e| format!("wave_dispatch: {:?}", e))?;

    dev.field_read_bytes_async(fcounter, out_bytes)
        .await
        .map_err(|e| format!("read_back: {:?}", e))
}

#[unsafe(no_mangle)]
pub extern "C" fn web_diff_counter_run(task: u32) {
    spawn_local(async move {
        match run_counter().await {
            Ok(bytes) => unsafe {
                quanta_complete_bytes(task, bytes.as_ptr(), bytes.len());
            },
            Err(msg) => unsafe {
                quanta_complete_err(task, msg.as_ptr(), msg.len());
            },
        }
    });
}
