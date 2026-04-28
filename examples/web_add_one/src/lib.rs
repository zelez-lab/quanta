//! Browser-side smoke test for step 050 + step 079 + B⁰.
//!
//! Builds the simplest possible end-to-end Quanta-on-WebGPU demo: an
//! `add_one` kernel that increments every element of a `u32` buffer. Run
//! it from a real browser tab — there's no headless harness, by design.
//!
//! ## Build
//!
//! ```sh
//! ./scripts/build-web.sh web_add_one
//! ```
//!
//! Serve `examples/web_add_one/index.html` over HTTPS (or
//! http://localhost) and open in a WebGPU-capable browser.
//!
//! ## What the test asserts
//!
//! - Input buffer: 64 × `u32` initialized to `[0, 1, 2, …, 63]`.
//! - After dispatching `add_one` with one workgroup of size 64,
//!   the buffer must equal `[1, 2, 3, …, 64]`.
//!
//! Result is handed back to JS via the wasm imports
//! `quanta_complete_bytes` (success) or `quanta_complete_err` (failure).
//! No `wasm-bindgen` runtime is involved.

#![cfg(target_arch = "wasm32")]

use quanta::GpuDevice;
use quanta::webgpu::spawn_local;
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};

unsafe extern "C" {
    fn quanta_complete_bytes(task: u32, ptr: *const u8, len: usize);
    fn quanta_complete_err(task: u32, ptr: *const u8, len: usize);
}

fn build_add_one_kernel() -> KernelDef {
    KernelDef {
        name: "add_one".into(),
        params: vec![KernelParam::FieldWrite {
            name: "buf".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::U32(1),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 4,
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

    let n = 64usize;
    let bytes = n * core::mem::size_of::<u32>();

    let buf = dev
        .field_alloc(bytes, quanta::FieldUsage::default_compute())
        .map_err(|e| format!("field_alloc: {:?}", e))?;

    let mut input = Vec::with_capacity(bytes);
    for i in 0u32..(n as u32) {
        input.extend_from_slice(&i.to_le_bytes());
    }
    dev.field_write_bytes(buf, &input)
        .map_err(|e| format!("field_write: {:?}", e))?;

    let kernel = build_add_one_kernel();
    let kernel_bytes = serialize_kernel(&kernel);
    let mut wave = dev
        .wave_jit(&kernel_bytes)
        .map_err(|e| format!("wave_jit: {:?}", e))?;
    wave.bind_handle(0, buf);

    let _pulse = dev
        .wave_dispatch(&wave, [1, 1, 1])
        .map_err(|e| format!("wave_dispatch: {:?}", e))?;

    dev.field_read_bytes_async(buf, bytes)
        .await
        .map_err(|e| format!("read_back: {:?}", e))
}

/// Smoke-test entry. JS-side harness calls
/// `wasm.web_add_one_run(task)` with a freshly minted task id; this
/// function spawns the async test and replies via
/// `quanta_complete_bytes` / `quanta_complete_err`.
#[unsafe(no_mangle)]
pub extern "C" fn web_add_one_run(task: u32) {
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
