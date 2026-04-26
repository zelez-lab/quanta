//! Browser-side smoke test for step 050 + step 079.
//!
//! Builds the simplest possible end-to-end Quanta-on-WebGPU demo: an
//! `add_one` kernel that increments every element of a `u32` buffer. Run
//! it from a real browser tab — there's no headless harness, by design.
//! The point is to prove that:
//!
//! 1. `quanta::webgpu::WebgpuDevice::new_async` acquires a real
//!    `GPUDevice` via `navigator.gpu.requestAdapter` / `requestDevice`.
//! 2. `wave_jit` runs `emit_wgsl_jit` and feeds the result to
//!    `device.createShaderModule({ code })`.
//! 3. `wave_dispatch` enqueues a real compute pass.
//! 4. `field_read_bytes_async` round-trips data back out via `mapAsync`.
//!
//! ## Build
//!
//! ```sh
//! cargo build --target wasm32-unknown-unknown --release \
//!     -p web-add-one
//! wasm-bindgen --target web --out-dir examples/web_add_one/pkg \
//!     target/wasm32-unknown-unknown/release/web_add_one.wasm
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
//! Failures bubble up to the browser console as JS-shaped Result errors;
//! pass returns the read-back vector as a `Uint8Array` to JS.

#![cfg(target_arch = "wasm32")]

use quanta::GpuDevice;
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use wasm_bindgen::prelude::*;

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

/// Run the smoke test, returning the buffer contents (Vec<u8> of 64 u32s
/// little-endian) on success.
#[wasm_bindgen]
pub async fn run_add_one() -> Result<Vec<u8>, JsValue> {
    // Use the typed device directly — we need the async read-back method.
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("new_async: {:?}", e)))?;

    let n = 64usize;
    let bytes = n * core::mem::size_of::<u32>();

    let buf = dev
        .field_alloc(bytes, quanta::FieldUsage::default_compute())
        .map_err(|e| JsValue::from_str(&format!("field_alloc: {:?}", e)))?;

    let mut input = Vec::with_capacity(bytes);
    for i in 0u32..(n as u32) {
        input.extend_from_slice(&i.to_le_bytes());
    }
    dev.field_write_bytes(buf, &input)
        .map_err(|e| JsValue::from_str(&format!("field_write: {:?}", e)))?;

    let kernel = build_add_one_kernel();
    let kernel_bytes = serialize_kernel(&kernel);
    let mut wave = dev
        .wave_jit(&kernel_bytes)
        .map_err(|e| JsValue::from_str(&format!("wave_jit: {:?}", e)))?;
    wave.bind_handle(0, buf);

    let _pulse = dev
        .wave_dispatch(&wave, [1, 1, 1])
        .map_err(|e| JsValue::from_str(&format!("wave_dispatch: {:?}", e)))?;

    let out = dev
        .field_read_bytes_async(buf, bytes)
        .await
        .map_err(|e| JsValue::from_str(&format!("read_back: {:?}", e)))?;

    Ok(out)
}
