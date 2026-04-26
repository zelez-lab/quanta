//! Hand-written `extern "C"` blocks for the WebGPU surface Quanta uses.
//!
//! Step 000 of the project rejects `ash`, `metal-rs`, `objc`. The same rule
//! extends to wasm32: **no `web-sys`, no `wgpu`** — both are pre-generated
//! typed binding libraries (`web-sys` is ~150K LOC of WebIDL bindings;
//! `wgpu` is a full GPU abstraction). Every external wrapper crate puts its
//! full API surface inside Quanta's TCB.
//!
//! What lives here is the WebGPU surface Quanta actually uses — the same
//! shape as the project's hand-written Metal headers and `vkSomething`
//! loaders. About 200 lines we own and can audit line-by-line.
//!
//! `wasm-bindgen` itself stays — it's the wasm32 calling-convention bridge
//! (libc-equivalent for Rust ↔ JS FFI), not a wrapper.

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // ── Top-level: navigator.gpu ─────────────────────────────────────────────

    pub type Navigator;

    pub type GpuInstance;
    #[wasm_bindgen(method, js_name = requestAdapter)]
    pub fn request_adapter(this: &GpuInstance) -> js_sys::Promise;

    // ── GPUAdapter ──────────────────────────────────────────────────────────

    pub type GpuAdapter;
    #[wasm_bindgen(method, js_name = requestDevice)]
    pub fn request_device(this: &GpuAdapter) -> js_sys::Promise;

    // ── GPUDevice ───────────────────────────────────────────────────────────

    pub type GpuDevice;
    #[wasm_bindgen(method, getter)]
    pub fn queue(this: &GpuDevice) -> GpuQueue;
    #[wasm_bindgen(method, js_name = createBuffer)]
    pub fn create_buffer(this: &GpuDevice, descriptor: &JsValue) -> GpuBuffer;
    #[wasm_bindgen(method, js_name = createShaderModule)]
    pub fn create_shader_module(this: &GpuDevice, descriptor: &JsValue) -> GpuShaderModule;
    #[wasm_bindgen(method, js_name = createComputePipeline)]
    pub fn create_compute_pipeline(this: &GpuDevice, descriptor: &JsValue) -> GpuComputePipeline;
    #[wasm_bindgen(method, js_name = createRenderPipeline)]
    pub fn create_render_pipeline(this: &GpuDevice, descriptor: &JsValue) -> GpuRenderPipeline;
    #[wasm_bindgen(method, js_name = createBindGroup)]
    pub fn create_bind_group(this: &GpuDevice, descriptor: &JsValue) -> GpuBindGroup;
    #[wasm_bindgen(method, js_name = createCommandEncoder)]
    pub fn create_command_encoder(this: &GpuDevice) -> GpuCommandEncoder;
    #[wasm_bindgen(method, js_name = createTexture)]
    pub fn create_texture(this: &GpuDevice, descriptor: &JsValue) -> GpuTexture;
    #[wasm_bindgen(method, js_name = createSampler)]
    pub fn create_sampler(this: &GpuDevice, descriptor: &JsValue) -> GpuSampler;

    // ── GPUQueue ────────────────────────────────────────────────────────────

    pub type GpuQueue;
    #[wasm_bindgen(method, js_name = writeBuffer)]
    pub fn write_buffer(this: &GpuQueue, buffer: &GpuBuffer, offset: u64, data: &[u8]);
    #[wasm_bindgen(method, js_name = writeTexture)]
    pub fn write_texture(
        this: &GpuQueue,
        destination: &JsValue,
        data: &[u8],
        data_layout: &JsValue,
        size: &JsValue,
    );
    #[wasm_bindgen(method)]
    pub fn submit(this: &GpuQueue, command_buffers: &js_sys::Array);
    #[wasm_bindgen(method, js_name = onSubmittedWorkDone)]
    pub fn on_submitted_work_done(this: &GpuQueue) -> js_sys::Promise;

    // ── GPUBuffer ───────────────────────────────────────────────────────────

    pub type GpuBuffer;
    #[wasm_bindgen(method, js_name = mapAsync)]
    pub fn map_async(this: &GpuBuffer, mode: u32) -> js_sys::Promise;
    #[wasm_bindgen(method, js_name = getMappedRange)]
    pub fn get_mapped_range(this: &GpuBuffer) -> js_sys::ArrayBuffer;
    #[wasm_bindgen(method)]
    pub fn unmap(this: &GpuBuffer);
    #[wasm_bindgen(method)]
    pub fn destroy(this: &GpuBuffer);

    // ── GPUShaderModule, GPUComputePipeline, GPUBindGroup ──────────────────

    pub type GpuShaderModule;
    pub type GpuComputePipeline;
    #[wasm_bindgen(method, js_name = getBindGroupLayout)]
    pub fn get_bind_group_layout(this: &GpuComputePipeline, index: u32) -> JsValue;

    pub type GpuRenderPipeline;
    #[wasm_bindgen(method, js_name = getBindGroupLayout)]
    pub fn render_get_bind_group_layout(this: &GpuRenderPipeline, index: u32) -> JsValue;

    pub type GpuBindGroup;
    pub type GpuTexture;
    #[wasm_bindgen(method, js_name = createView)]
    pub fn create_view(this: &GpuTexture) -> GpuTextureView;
    #[wasm_bindgen(method)]
    pub fn destroy_texture(this: &GpuTexture);
    pub type GpuTextureView;
    pub type GpuSampler;

    // ── GPUCommandEncoder, GPUComputePassEncoder, GPUCommandBuffer ─────────

    pub type GpuCommandEncoder;
    #[wasm_bindgen(method, js_name = beginComputePass)]
    pub fn begin_compute_pass(this: &GpuCommandEncoder) -> GpuComputePassEncoder;
    #[wasm_bindgen(method, js_name = beginRenderPass)]
    pub fn begin_render_pass(
        this: &GpuCommandEncoder,
        descriptor: &JsValue,
    ) -> GpuRenderPassEncoder;
    #[wasm_bindgen(method, js_name = copyBufferToBuffer)]
    pub fn copy_buffer_to_buffer(
        this: &GpuCommandEncoder,
        src: &GpuBuffer,
        src_offset: u64,
        dst: &GpuBuffer,
        dst_offset: u64,
        size: u64,
    );
    #[wasm_bindgen(method, js_name = copyTextureToBuffer)]
    pub fn copy_texture_to_buffer(
        this: &GpuCommandEncoder,
        source: &JsValue,
        destination: &JsValue,
        copy_size: &JsValue,
    );
    #[wasm_bindgen(method)]
    pub fn finish(this: &GpuCommandEncoder) -> GpuCommandBuffer;

    pub type GpuComputePassEncoder;
    #[wasm_bindgen(method, js_name = setPipeline)]
    pub fn set_pipeline(this: &GpuComputePassEncoder, pipeline: &GpuComputePipeline);
    #[wasm_bindgen(method, js_name = setBindGroup)]
    pub fn set_bind_group(this: &GpuComputePassEncoder, index: u32, group: &GpuBindGroup);
    #[wasm_bindgen(method, js_name = dispatchWorkgroups)]
    pub fn dispatch_workgroups(this: &GpuComputePassEncoder, x: u32, y: u32, z: u32);
    #[wasm_bindgen(method)]
    pub fn end(this: &GpuComputePassEncoder);

    pub type GpuRenderPassEncoder;
    #[wasm_bindgen(method, js_name = setPipeline)]
    pub fn rp_set_pipeline(this: &GpuRenderPassEncoder, pipeline: &GpuRenderPipeline);
    #[wasm_bindgen(method, js_name = setBindGroup)]
    pub fn rp_set_bind_group(this: &GpuRenderPassEncoder, index: u32, group: &GpuBindGroup);
    #[wasm_bindgen(method, js_name = setVertexBuffer)]
    pub fn rp_set_vertex_buffer(
        this: &GpuRenderPassEncoder,
        slot: u32,
        buffer: &GpuBuffer,
        offset: u64,
    );
    #[wasm_bindgen(method, js_name = setIndexBuffer)]
    pub fn rp_set_index_buffer(
        this: &GpuRenderPassEncoder,
        buffer: &GpuBuffer,
        format: &str,
        offset: u64,
    );
    #[wasm_bindgen(method)]
    pub fn draw(this: &GpuRenderPassEncoder, vertex_count: u32, instance_count: u32);
    #[wasm_bindgen(method, js_name = drawIndexed)]
    pub fn draw_indexed(this: &GpuRenderPassEncoder, index_count: u32, instance_count: u32);
    #[wasm_bindgen(method, js_name = setViewport)]
    pub fn rp_set_viewport(
        this: &GpuRenderPassEncoder,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    );
    #[wasm_bindgen(method, js_name = setScissorRect)]
    pub fn rp_set_scissor(this: &GpuRenderPassEncoder, x: u32, y: u32, width: u32, height: u32);
    #[wasm_bindgen(method, js_name = end)]
    pub fn rp_end(this: &GpuRenderPassEncoder);

    pub type GpuCommandBuffer;
}

// ── GPU buffer usage flags (mirrors the WebGPU spec). ──────────────────────

#[allow(dead_code)] // surface kept complete; not every flag is used yet.
pub mod buffer_usage {
    pub const MAP_READ: u32 = 0x0001;
    pub const MAP_WRITE: u32 = 0x0002;
    pub const COPY_SRC: u32 = 0x0004;
    pub const COPY_DST: u32 = 0x0008;
    pub const INDEX: u32 = 0x0010;
    pub const VERTEX: u32 = 0x0020;
    pub const UNIFORM: u32 = 0x0040;
    pub const STORAGE: u32 = 0x0080;
    pub const INDIRECT: u32 = 0x0100;
}

#[allow(dead_code)]
pub mod texture_usage {
    pub const COPY_SRC: u32 = 0x01;
    pub const COPY_DST: u32 = 0x02;
    pub const TEXTURE_BINDING: u32 = 0x04;
    pub const STORAGE_BINDING: u32 = 0x08;
    pub const RENDER_ATTACHMENT: u32 = 0x10;
}

#[allow(dead_code)]
pub mod map_mode {
    pub const READ: u32 = 0x0001;
    pub const WRITE: u32 = 0x0002;
}

/// Fetch `navigator.gpu` lazily — `wasm-bindgen`'s `static` form is
/// deprecated in 0.2.x. Reflect lookup on the global object is more
/// portable across host environments.
pub fn gpu() -> Result<GpuInstance, wasm_bindgen::JsValue> {
    let global = js_sys::global();
    let nav = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("navigator"))?;
    let gpu = js_sys::Reflect::get(&nav, &wasm_bindgen::JsValue::from_str("gpu"))?;
    if gpu.is_undefined() || gpu.is_null() {
        return Err(wasm_bindgen::JsValue::from_str(
            "navigator.gpu is undefined — WebGPU not available",
        ));
    }
    Ok(gpu.unchecked_into::<GpuInstance>())
}
