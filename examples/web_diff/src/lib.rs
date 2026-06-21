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

#[link(wasm_import_module = "env")]
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

// ───────────────────────── race (D-ext.3b.2) ─────────────────────────
//
// 2 quarks each `atomic_exchange(&cell, quark_id)`. Final layout:
// [cell_final, out_0, out_1] — non-deterministic across runs but
// always a member of the model-permitted set the page enumerates.

const RACE_N: u32 = 2;

fn build_race_kernel() -> KernelDef {
    KernelDef {
        name: "race".into(),
        params: vec![
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
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(0),
            },
            KernelOp::AtomicOp {
                dst: Reg(2),
                field: 0,
                index: Reg(1),
                val: Reg(0),
                op: AtomicOp::Exchange,
                ty: ScalarType::U32,
                order: quanta_ir::MemoryOrder::Relaxed,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 3,
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [RACE_N, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

async fn run_race() -> Result<Vec<u8>, String> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| format!("new_async: {:?}", e))?;

    let cell_bytes = core::mem::size_of::<u32>();
    let out_bytes_total = (RACE_N as usize) * core::mem::size_of::<u32>();
    let usage = quanta::FieldUsage::default_compute();

    let fcell = dev
        .field_alloc(cell_bytes, usage)
        .map_err(|e| format!("field_alloc cell: {:?}", e))?;
    let fout = dev
        .field_alloc(out_bytes_total, usage)
        .map_err(|e| format!("field_alloc out: {:?}", e))?;

    let zeros: Vec<u8> = vec![0u8; out_bytes_total];
    dev.field_write_bytes(fcell, &0u32.to_le_bytes())
        .map_err(|e| format!("field_write cell: {:?}", e))?;
    dev.field_write_bytes(fout, &zeros)
        .map_err(|e| format!("field_write out: {:?}", e))?;

    let kernel = build_race_kernel();
    let kernel_bytes = serialize_kernel(&kernel);
    let mut wave = dev
        .wave_jit(&kernel_bytes)
        .map_err(|e| format!("wave_jit: {:?}", e))?;
    wave.bind_handle(0, fcell);
    wave.bind_handle(1, fout);

    let _pulse = dev
        .wave_dispatch(&wave, [1, 1, 1])
        .map_err(|e| format!("wave_dispatch: {:?}", e))?;

    let mut combined = dev
        .field_read_bytes_async(fcell, cell_bytes)
        .await
        .map_err(|e| format!("read_back cell: {:?}", e))?;
    let out_part = dev
        .field_read_bytes_async(fout, out_bytes_total)
        .await
        .map_err(|e| format!("read_back out: {:?}", e))?;
    combined.extend(out_part);
    Ok(combined)
}

#[unsafe(no_mangle)]
pub extern "C" fn web_diff_race_run(task: u32) {
    spawn_local(async move {
        match run_race().await {
            Ok(bytes) => unsafe {
                quanta_complete_bytes(task, bytes.as_ptr(), bytes.len());
            },
            Err(msg) => unsafe {
                quanta_complete_err(task, msg.as_ptr(), msg.len());
            },
        }
    });
}

// ───────────────────── op-matrix (WGSL audit) ─────────────────────
//
// Runs the shared per-op differential matrix through real WebGPU and
// compares each case against its CPU-computed `expected`. This is the
// WGSL counterpart to the software / Metal / Vulkan op_matrix lanes —
// the lane that exposed the SPIR-V emitter opcode bugs. WGSL has no
// 64-bit scalar types, so u64/i64/f64 cases are skipped (the native
// op_matrix lanes gate them the same way).

use quanta_ir::op_matrix_cases::{OpCase, RawValues, cases};

/// Little-endian byte image of a typed buffer.
fn raw_to_bytes(v: &RawValues) -> Vec<u8> {
    let mut out = Vec::new();
    match v {
        RawValues::F32(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        RawValues::U32(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        RawValues::I32(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        RawValues::F64(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        RawValues::U64(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        RawValues::I64(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        // Narrow floats use the portable u32-slot storage (one value per
        // 32-bit word), carrying the raw pattern zero-extended. These cases
        // are skipped in the run loop (see `case_is_narrow_float`), but the
        // byte image stays well-defined.
        RawValues::BF16(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&(*x as u32).to_le_bytes())),
        RawValues::FP8E5M2(xs) | RawValues::FP8E4M3(xs) => xs
            .iter()
            .for_each(|x| out.extend_from_slice(&(*x as u32).to_le_bytes())),
    }
    out
}

fn raw_elem_size(v: &RawValues) -> usize {
    match v {
        RawValues::F32(_) | RawValues::U32(_) | RawValues::I32(_) => 4,
        RawValues::F64(_) | RawValues::U64(_) | RawValues::I64(_) => 8,
        // u32-slot storage for the narrow floats.
        RawValues::BF16(_) | RawValues::FP8E5M2(_) | RawValues::FP8E4M3(_) => 4,
    }
}

/// Narrow-float cases (bf16, fp8) run through the portable u32-slot path
/// with pack/unpack helpers in the emitted shader. The browser differential
/// lane doesn't exercise them yet (the native software / Metal / Vulkan
/// op_matrix lanes already prove them bit-exact), so skip them here.
fn case_is_narrow_float(c: &OpCase) -> bool {
    let narrow = |v: &RawValues| {
        matches!(
            v,
            RawValues::BF16(_) | RawValues::FP8E5M2(_) | RawValues::FP8E4M3(_)
        )
    };
    narrow(&c.input_a) || narrow(&c.input_b) || narrow(&c.expected)
}

/// WGSL (and the WebGPU shaders Quanta emits) has no 64-bit scalar type;
/// skip any case whose inputs or output are 64-bit, matching how the
/// native op_matrix lanes gate u64/i64/f64.
fn case_is_64bit(c: &OpCase) -> bool {
    let is64 =
        |v: &RawValues| matches!(v, RawValues::F64(_) | RawValues::U64(_) | RawValues::I64(_));
    is64(&c.input_a) || is64(&c.input_b) || is64(&c.expected)
}

/// f32 ULP distance (sign-magnitude ordering), mirroring the host
/// comparator's tolerance handling.
fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    let order = |i: i32| if i < 0 { i32::MIN.wrapping_sub(i) } else { i };
    (order(ai).wrapping_sub(order(bi))).unsigned_abs()
}

/// Compare a readback against the expected buffer with the case's
/// tolerance. Returns Ok(()) or a short mismatch description.
fn compare_case(case: &OpCase, got: &[u8]) -> Result<(), String> {
    let want = raw_to_bytes(&case.expected);
    if got.len() < want.len() {
        return Err(format!("short readback ({} < {})", got.len(), want.len()));
    }
    // Float output with a ULP tolerance: compare element-wise.
    if let RawValues::F32(exp) = &case.expected {
        if case.max_ulps > 0 {
            for (i, &e) in exp.iter().enumerate() {
                let g = f32::from_le_bytes([
                    got[i * 4],
                    got[i * 4 + 1],
                    got[i * 4 + 2],
                    got[i * 4 + 3],
                ]);
                let d = ulp_distance_f32(e, g);
                if d > case.max_ulps {
                    return Err(format!("got {g}, want {e} ({d} ULP > {})", case.max_ulps));
                }
            }
            return Ok(());
        }
    }
    // Everything else: bit-exact.
    if got[..want.len()] == want[..] {
        Ok(())
    } else {
        Err("bit-exact mismatch".into())
    }
}

async fn dispatch_case(
    dev: &quanta::webgpu::WebgpuDevice,
    case: &OpCase,
) -> Result<Vec<u8>, String> {
    let usage = quanta::FieldUsage::default_compute();
    let in_sz = raw_elem_size(&case.input_a);
    let out_sz = raw_elem_size(&case.expected);

    let fa = dev
        .field_alloc(in_sz, usage)
        .map_err(|e| format!("alloc a: {e:?}"))?;
    let fb = dev
        .field_alloc(in_sz, usage)
        .map_err(|e| format!("alloc b: {e:?}"))?;
    let fout = dev
        .field_alloc(out_sz, usage)
        .map_err(|e| format!("alloc out: {e:?}"))?;

    dev.field_write_bytes(fa, &raw_to_bytes(&case.input_a))
        .map_err(|e| format!("write a: {e:?}"))?;
    dev.field_write_bytes(fb, &raw_to_bytes(&case.input_b))
        .map_err(|e| format!("write b: {e:?}"))?;

    let kernel_bytes = serialize_kernel(&case.def);
    let mut wave = dev
        .wave_jit(&kernel_bytes)
        .map_err(|e| format!("wave_jit: {e:?}"))?;
    wave.bind_handle(0, fa);
    wave.bind_handle(1, fb);
    wave.bind_handle(2, fout);
    let _pulse = dev
        .wave_dispatch(&wave, [1, 1, 1])
        .map_err(|e| format!("dispatch: {e:?}"))?;

    dev.field_read_bytes_async(fout, out_sz)
        .await
        .map_err(|e| format!("readback: {e:?}"))
}

async fn run_op_matrix() -> Result<Vec<u8>, String> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| format!("new_async: {e:?}"))?;

    let mut passed = 0u32;
    let mut skipped = 0u32;
    let mut total = 0u32;
    let mut first_fail = String::new();

    for case in cases() {
        if case_is_64bit(&case) || case_is_narrow_float(&case) {
            skipped += 1;
            continue;
        }
        total += 1;
        match dispatch_case(&dev, &case).await {
            Ok(bytes) => match compare_case(&case, &bytes) {
                Ok(()) => passed += 1,
                Err(why) => {
                    if first_fail.is_empty() {
                        first_fail = format!("{}: {}", case.name, why);
                    }
                }
            },
            Err(why) => {
                if first_fail.is_empty() {
                    first_fail = format!("{}: dispatch {}", case.name, why);
                }
            }
        }
    }

    // Verdict line consumed by the page: "passed total skipped | detail".
    let verdict = format!("{passed} {total} {skipped} | {first_fail}");
    Ok(verdict.into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn web_diff_op_matrix_run(task: u32) {
    spawn_local(async move {
        match run_op_matrix().await {
            Ok(bytes) => unsafe {
                quanta_complete_bytes(task, bytes.as_ptr(), bytes.len());
            },
            Err(msg) => unsafe {
                quanta_complete_err(task, msg.as_ptr(), msg.len());
            },
        }
    });
}
