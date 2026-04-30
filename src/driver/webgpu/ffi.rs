//! Quanta-owned WebGPU ABI — bare `extern "C"` imports.
//!
//! Step 000 forbids `web-sys` and `wgpu`. B⁰ (2026-04-28) extends the
//! same rule to `wasm-bindgen`'s runtime crate: this driver speaks to
//! the browser through hand-authored wasm imports defined here, and a
//! TypeScript-authored `web/src/quanta.ts` (and sibling helpers) that
//! compile to `quanta.js` + `*.js`.
//! Together they are the entire FFI TCB on the WebGPU backend — about
//! 500 lines of code we own and audit, instead of the ~30-60 KB
//! wasm-bindgen runtime that previously sat between Rust and JS.
//!
//! ABI shape:
//!
//! - All long-lived JS objects (devices, buffers, pipelines, shader
//!   modules, …) are represented on the Rust side as `u32` handles
//!   into a JS-side handle table. Handle 0 is the null handle.
//! - All strings cross as `(ptr: *const u8, len: usize)` — JS reads
//!   them out of the wasm linear memory via `TextDecoder`.
//! - All `u64` sizes/offsets cross as `f64` (exact up to 2^53, larger
//!   than any plausible WebGPU resource).
//! - All enum-shaped parameters cross as `u32` codes; the JS side maps
//!   them to the WebGPU IDL strings (`"rgba8unorm"` etc.) via tables in
//!   `web/src/codes.ts`. The two sides are kept in lockstep manually
//!   for B⁰; B′ + B″ replace the manual alignment with a generator from
//!   the W3C `webgpu.idl`.
//! - All async ops take a `task: u32` argument. The JS side resolves
//!   the underlying Promise and then calls back into the wasm exports
//!   `quanta_resolve(task, handle)` or `quanta_reject(task)` in
//!   `executor.rs`. There is no equivalent of `JsFuture` — the
//!   minimal Rust executor plus this callback shape is the entire
//!   async story.

#![allow(dead_code)]
// Some imports here are wired for completeness (e.g. depth-stencil
// attachment, `quanta_release`) but only used by future feature work.
// Keeping them in the surface lets the JS side stay in lockstep without
// per-feature cfg gating.

// ── Handle conventions ──────────────────────────────────────────────────────

/// A null/uninitialized handle. The JS side raises an error if it ever
/// receives this for a lookup, so accidental zero values surface loudly.
pub const NULL_HANDLE: u32 = 0;

// ── Buffer usage flags (mirrors the WebGPU spec) ────────────────────────────

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
    pub const QUERY_RESOLVE: u32 = 0x0200;
}

pub mod texture_usage {
    pub const COPY_SRC: u32 = 0x01;
    pub const COPY_DST: u32 = 0x02;
    pub const TEXTURE_BINDING: u32 = 0x04;
    pub const STORAGE_BINDING: u32 = 0x08;
    pub const RENDER_ATTACHMENT: u32 = 0x10;
}

pub mod map_mode {
    pub const READ: u32 = 0x0001;
    pub const WRITE: u32 = 0x0002;
}

// ── Enum codes — these MUST match `web/src/codes.ts` exactly ────────────────

pub mod format {
    pub const RGBA8UNORM: u32 = 0;
    pub const BGRA8UNORM: u32 = 1;
    pub const R8UNORM: u32 = 2;
    pub const R16FLOAT: u32 = 3;
    pub const R32FLOAT: u32 = 4;
    pub const RG32FLOAT: u32 = 5;
    pub const RGBA16FLOAT: u32 = 6;
    pub const RGBA32FLOAT: u32 = 7;
    pub const DEPTH32FLOAT: u32 = 8;
}

pub mod attribute_format {
    pub const FLOAT: u32 = 0;
    pub const FLOAT2: u32 = 1;
    pub const FLOAT3: u32 = 2;
    pub const FLOAT4: u32 = 3;
    pub const SINT: u32 = 4;
    pub const SINT2: u32 = 5;
    pub const SINT3: u32 = 6;
    pub const SINT4: u32 = 7;
    pub const UINT: u32 = 8;
    pub const UINT2: u32 = 9;
    pub const UINT3: u32 = 10;
    pub const UINT4: u32 = 11;
    pub const UNORM8X4: u32 = 12;
}

pub mod topology {
    pub const POINT: u32 = 0;
    pub const LINE: u32 = 1;
    pub const LINE_STRIP: u32 = 2;
    pub const TRIANGLE: u32 = 3;
    pub const TRIANGLE_STRIP: u32 = 4;
}

pub mod cull_mode {
    pub const NONE: u32 = 0;
    pub const FRONT: u32 = 1;
    pub const BACK: u32 = 2;
}

pub mod blend_factor {
    pub const ZERO: u32 = 0;
    pub const ONE: u32 = 1;
    pub const SRC_ALPHA: u32 = 2;
    pub const ONE_MINUS_SRC_ALPHA: u32 = 3;
    pub const DST_ALPHA: u32 = 4;
    pub const ONE_MINUS_DST_ALPHA: u32 = 5;
    pub const SRC_COLOR: u32 = 6;
    pub const ONE_MINUS_SRC_COLOR: u32 = 7;
    pub const DST_COLOR: u32 = 8;
    pub const ONE_MINUS_DST_COLOR: u32 = 9;
}

pub mod blend_op {
    pub const ADD: u32 = 0;
    pub const SUBTRACT: u32 = 1;
    pub const REVERSE_SUBTRACT: u32 = 2;
    pub const MIN: u32 = 3;
    pub const MAX: u32 = 4;
}

pub mod filter {
    pub const NEAREST: u32 = 0;
    pub const LINEAR: u32 = 1;
}

pub mod address {
    pub const CLAMP_TO_EDGE: u32 = 0;
    pub const REPEAT: u32 = 1;
    pub const MIRROR_REPEAT: u32 = 2;
}

pub mod compare {
    /// `0` is the "compare not configured" sentinel for samplers; the
    /// real compare ops start at 1 to keep the sentinel out of band.
    pub const UNSET: u32 = 0;
    pub const NEVER: u32 = 1;
    pub const LESS: u32 = 2;
    pub const EQUAL: u32 = 3;
    pub const LESS_EQUAL: u32 = 4;
    pub const GREATER: u32 = 5;
    pub const NOT_EQUAL: u32 = 6;
    pub const GREATER_EQUAL: u32 = 7;
    pub const ALWAYS: u32 = 8;
}

pub mod step_mode {
    pub const VERTEX: u32 = 0;
    pub const INSTANCE: u32 = 1;
}

pub mod index_format {
    pub const UINT16: u32 = 0;
    pub const UINT32: u32 = 1;
}

pub mod load_op {
    pub const LOAD: u32 = 0;
    pub const CLEAR: u32 = 1;
}

pub mod store_op {
    pub const STORE: u32 = 0;
    pub const DISCARD: u32 = 1;
}

// ── Imports — everything below MUST mirror `web/src/webgpu.ts` ──────────────

unsafe extern "C" {
    // Adapter / device acquisition (async).
    pub fn quanta_request_adapter(task: u32);
    pub fn quanta_request_device(adapter: u32, task: u32);

    // Buffers.
    pub fn quanta_create_buffer(device: u32, size: f64, usage: u32) -> u32;
    pub fn quanta_destroy_buffer(buffer: u32);
    pub fn quanta_write_buffer(
        device: u32,
        buffer: u32,
        offset: f64,
        data_ptr: *const u8,
        data_len: usize,
    );
    pub fn quanta_map_async_read(buffer: u32, task: u32);
    pub fn quanta_get_mapped_range_copy(buffer: u32, dst_ptr: *mut u8, len: usize);
    pub fn quanta_unmap_buffer(buffer: u32);

    // Shader / compute pipeline.
    pub fn quanta_create_shader_module(device: u32, code_ptr: *const u8, code_len: usize) -> u32;
    pub fn quanta_create_compute_pipeline(
        device: u32,
        module: u32,
        entry_ptr: *const u8,
        entry_len: usize,
    ) -> u32;
    pub fn quanta_compute_pipeline_get_bind_group_layout(pipeline: u32, index: u32) -> u32;

    // Render pipeline (descriptor builder).
    pub fn quanta_rp_desc_create() -> u32;
    pub fn quanta_rp_desc_set_vertex(
        desc: u32,
        module: u32,
        entry_ptr: *const u8,
        entry_len: usize,
    );
    pub fn quanta_rp_desc_add_vertex_buffer(desc: u32, stride: u32, step_mode: u32);
    pub fn quanta_rp_desc_add_vertex_attribute(
        desc: u32,
        buf_index: u32,
        format_code: u32,
        offset: u32,
        location: u32,
    );
    pub fn quanta_rp_desc_set_fragment(
        desc: u32,
        module: u32,
        entry_ptr: *const u8,
        entry_len: usize,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn quanta_rp_desc_add_color_target(
        desc: u32,
        format_code: u32,
        blend_enabled: u32,
        src_color: u32,
        dst_color: u32,
        op_color: u32,
        src_alpha: u32,
        dst_alpha: u32,
        op_alpha: u32,
    );
    pub fn quanta_rp_desc_set_primitive(desc: u32, topology_code: u32, cull_mode_code: u32);
    pub fn quanta_rp_desc_set_multisample(desc: u32, count: u32);
    pub fn quanta_rp_desc_set_depth_stencil(
        desc: u32,
        format_code: u32,
        depth_write: u32,
        compare_code: u32,
    );
    pub fn quanta_create_render_pipeline(device: u32, desc: u32) -> u32;
    pub fn quanta_render_pipeline_get_bind_group_layout(pipeline: u32, index: u32) -> u32;

    // Bind group (descriptor builder).
    pub fn quanta_bg_desc_create(layout: u32) -> u32;
    pub fn quanta_bg_desc_add_buffer(desc: u32, binding: u32, buffer: u32);
    pub fn quanta_bg_desc_add_sampler(desc: u32, binding: u32, sampler: u32);
    pub fn quanta_bg_desc_add_texture_view(desc: u32, binding: u32, view: u32);
    pub fn quanta_create_bind_group(device: u32, desc: u32) -> u32;

    // Command encoder.
    pub fn quanta_create_command_encoder(device: u32) -> u32;
    pub fn quanta_encoder_copy_buffer_to_buffer(
        encoder: u32,
        src: u32,
        src_off: f64,
        dst: u32,
        dst_off: f64,
        size: f64,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn quanta_encoder_copy_texture_to_buffer(
        encoder: u32,
        src_texture: u32,
        dst_buffer: u32,
        dst_bytes_per_row: u32,
        dst_rows_per_image: u32,
        width: u32,
        height: u32,
        depth: u32,
    );
    pub fn quanta_encoder_finish(encoder: u32) -> u32;

    // Compute pass.
    pub fn quanta_encoder_begin_compute_pass(encoder: u32) -> u32;
    pub fn quanta_compute_pass_set_pipeline(pass: u32, pipeline: u32);
    pub fn quanta_compute_pass_set_bind_group(pass: u32, index: u32, group: u32);
    pub fn quanta_compute_pass_dispatch(pass: u32, x: u32, y: u32, z: u32);
    pub fn quanta_compute_pass_end(pass: u32);

    // Render pass.
    pub fn quanta_rpass_desc_create() -> u32;
    #[allow(clippy::too_many_arguments)]
    pub fn quanta_rpass_desc_add_color_attachment(
        desc: u32,
        view: u32,
        load_op: u32,
        store_op: u32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    );
    pub fn quanta_rpass_desc_set_depth_attachment(
        desc: u32,
        view: u32,
        load_op: u32,
        store_op: u32,
        clear_depth: f32,
    );
    pub fn quanta_encoder_begin_render_pass(encoder: u32, desc: u32) -> u32;
    pub fn quanta_render_pass_set_pipeline(pass: u32, pipeline: u32);
    pub fn quanta_render_pass_set_bind_group(pass: u32, index: u32, group: u32);
    pub fn quanta_render_pass_set_vertex_buffer(pass: u32, slot: u32, buffer: u32, offset: f64);
    pub fn quanta_render_pass_set_index_buffer(
        pass: u32,
        buffer: u32,
        format_code: u32,
        offset: f64,
    );
    // Occlusion query support (post-step-063 closure). Maps the
    // typed `OcclusionQuery` API to GPUQuerySet + the
    // occlusionQuerySet field on render pass descriptors +
    // beginOcclusionQuery / endOcclusionQuery + resolveQuerySet
    // for asynchronous result readback.
    pub fn quanta_create_query_set(device: u32, count: u32) -> u32;
    pub fn quanta_rpass_desc_set_occlusion_query_set(desc: u32, query_set: u32);
    pub fn quanta_render_pass_begin_occlusion_query(pass: u32, index: u32);
    pub fn quanta_render_pass_end_occlusion_query(pass: u32);
    /// Encode a resolve from the query set into a buffer. The
    /// destination buffer must have COPY_DST + QUERY_RESOLVE
    /// usage. Each query result is 8 bytes (u64).
    pub fn quanta_encoder_resolve_query_set(
        encoder: u32,
        query_set: u32,
        first_query: u32,
        query_count: u32,
        dst_buffer: u32,
        dst_offset: f64,
    );

    pub fn quanta_render_pass_draw(pass: u32, vertex_count: u32, instance_count: u32);
    pub fn quanta_render_pass_draw_indexed(pass: u32, index_count: u32, instance_count: u32);
    pub fn quanta_render_pass_draw_indirect(pass: u32, indirect_buffer: u32, indirect_offset: f64);
    pub fn quanta_render_pass_draw_indexed_indirect(
        pass: u32,
        indirect_buffer: u32,
        indirect_offset: f64,
    );
    pub fn quanta_render_pass_set_viewport(
        pass: u32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        min_depth: f32,
        max_depth: f32,
    );
    pub fn quanta_render_pass_set_scissor(pass: u32, x: u32, y: u32, w: u32, h: u32);
    pub fn quanta_render_pass_set_stencil_reference(pass: u32, reference: u32);
    pub fn quanta_render_pass_end(pass: u32);

    // Render bundle (steps 032 + 033, render path).
    pub fn quanta_create_render_bundle_encoder(
        device: u32,
        color_format_code: u32,
        depth_format_code: u32,
        sample_count: u32,
    ) -> u32;
    pub fn quanta_render_bundle_set_pipeline(encoder: u32, pipeline: u32);
    pub fn quanta_render_bundle_set_bind_group(encoder: u32, index: u32, group: u32);
    pub fn quanta_render_bundle_set_vertex_buffer(
        encoder: u32,
        slot: u32,
        buffer: u32,
        offset: f64,
    );
    pub fn quanta_render_bundle_draw(encoder: u32, vertex_count: u32, instance_count: u32);
    pub fn quanta_render_bundle_finish(encoder: u32) -> u32;
    pub fn quanta_render_pass_execute_bundles(pass: u32, bundles_ptr: *const u32, count: u32);

    // Queue.
    pub fn quanta_queue_submit(device: u32, command_buffer: u32);
    pub fn quanta_queue_on_submitted_work_done(device: u32, task: u32);

    // Textures / samplers.
    #[allow(clippy::too_many_arguments)]
    pub fn quanta_create_texture(
        device: u32,
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        mip_level_count: u32,
        sample_count: u32,
        format_code: u32,
        usage: u32,
    ) -> u32;
    pub fn quanta_texture_create_view(texture: u32) -> u32;
    pub fn quanta_destroy_texture(texture: u32);
    #[allow(clippy::too_many_arguments)]
    pub fn quanta_queue_write_texture(
        device: u32,
        texture: u32,
        data_ptr: *const u8,
        data_len: usize,
        bytes_per_row: u32,
        rows_per_image: u32,
        width: u32,
        height: u32,
        depth: u32,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn quanta_create_sampler(
        device: u32,
        mag_filter: u32,
        min_filter: u32,
        mipmap_filter: u32,
        address_u: u32,
        address_v: u32,
        address_w: u32,
        max_anisotropy: u32,
        compare_code: u32,
    ) -> u32;

    // Universal release for handles without a destroy method (shader
    // modules, pipelines, bind-group layouts, samplers, textures' views).
    pub fn quanta_release(handle: u32);

    // Diagnostic — surfaced from Rust panics / errors before the
    // top-level task even gets to call `quanta_complete_err`. Useful
    // when an init path explodes before there is a task to reject.
    pub fn quanta_console_error(ptr: *const u8, len: usize);
}

/// Wrap a `&str` in a (ptr, len) pair suitable for a string-arg FFI call.
#[inline]
pub fn str_parts(s: &str) -> (*const u8, usize) {
    (s.as_ptr(), s.len())
}

/// Same, for `&[u8]`.
#[inline]
pub fn bytes_parts(b: &[u8]) -> (*const u8, usize) {
    (b.as_ptr(), b.len())
}
