//! Quanta-owned WebGPU ABI — the driver's half of the command tape.
//!
//! Step 000 forbids `web-sys` and `wgpu`. B⁰ (2026-04-28) extends the
//! same rule to `wasm-bindgen`'s runtime crate: this driver speaks to
//! the browser through an ABI defined here, and a TypeScript-authored
//! `web/src/quanta.ts` (and sibling helpers) that compile to
//! `quanta.js` + `*.js`.
//! Together they are the entire FFI TCB on the WebGPU backend — about
//! 500 lines of code we own and audit, instead of the ~30-60 KB
//! wasm-bindgen runtime that previously sat between Rust and JS.
//!
//! Under dija's R10 the module imports NOTHING: every fn below appends
//! a command to the tape (`tape.rs`) and the glue performs the WebGPU
//! call when it drains, after the wasm entry returns.
//!
//! ABI shape:
//!
//! - All long-lived JS objects (devices, buffers, pipelines, shader
//!   modules, …) are represented on the Rust side as `u32` handles
//!   into a JS-side handle table. Handle 0 is the null handle.
//! - All strings cross as `(ptr: *const u8, len: usize)` — copied into
//!   the tape as an inline payload and decoded by `TextDecoder`.
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
//! - A handle-returning fn answers BEFORE its call runs, so a failure
//!   the old imports reported as `NULL_HANDLE` (no webgpu context on
//!   that canvas) now throws glue-side at drain instead.

#![allow(dead_code)]
// Some fns here are wired for completeness (e.g. depth-stencil
// attachment, `quanta_console_error`) but only used by future feature
// work. Keeping them in the surface lets the JS side stay in lockstep
// without per-feature cfg gating.

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

// ── Tape-backed fns (dija R10) — everything below MUST mirror the
// opcode arms in `web/src/tape.ts` ─────────────────────────────────────────
//
// These have the same signatures as the `env` imports they replace, so
// the ~200 driver call sites never change; `unsafe` is kept purely for
// that compatibility. Nothing here calls out: each fn appends its op to
// the tape (`tape.rs`) and the glue performs the call when it drains.
//
// Wire rules — the interpreter matches these positionally:
//
// - The words of an op are its arguments in declaration order, one word
//   each; an `f64` arg is TWO words (`tape::f64_words`) and an `f32` arg
//   is one word (its IEEE-754 bits).
// - A fn that RETURNS a handle mints it locally (`tape::mint_id`) and
//   emits it as the FIRST word; the glue binds its object to that id.
// - `(ptr, len)` pairs are not words at all: the bytes are copied inline
//   as payloads AFTER every fixed word, in argument order. The `words:`
//   comments below spell out each op whose word list is therefore not
//   literally its argument list.

use super::tape::{self, Op};

// ── Adapter / device acquisition ────────────────────────────────────────────

/// Async class: the REQUEST rides the tape; the result returns through
/// the exported `quanta_resolve`/`quanta_reject` setters as today.
pub unsafe fn quanta_request_adapter(task: u32) {
    tape::emit(Op::RequestAdapter, &[task], &[]);
}

pub unsafe fn quanta_request_device(adapter: u32, task: u32) {
    tape::emit(Op::RequestDevice, &[adapter, task], &[]);
}

// ── Buffers ────────────────────────────────────────────────────────────────

/// Creator class: the driver mints the destination id (high bit set)
/// and the glue binds its object to it at drain.
pub unsafe fn quanta_create_buffer(device: u32, size: f64, usage: u32) -> u32 {
    let id = tape::mint_id();
    let [s0, s1] = tape::f64_words(size);
    tape::emit(Op::CreateBuffer, &[id, device, s0, s1, usage], &[]);
    id
}

/// Void-mutator class: fire-and-forget.
pub unsafe fn quanta_destroy_buffer(buffer: u32) {
    tape::emit(Op::DestroyBuffer, &[buffer], &[]);
}

/// words: `(device, buffer, offset:f64)` + payload = the bytes.
pub unsafe fn quanta_write_buffer(
    device: u32,
    buffer: u32,
    offset: f64,
    data_ptr: *const u8,
    data_len: usize,
) {
    let [o0, o1] = tape::f64_words(offset);
    let data = unsafe { core::slice::from_raw_parts(data_ptr, data_len) };
    tape::emit(Op::WriteBuffer, &[device, buffer, o0, o1], &[data]);
}

/// The mapped range never crosses as a separate call: the glue copies
/// the bytes into `dst_ptr`, unmaps, and only THEN resolves `task`, so
/// the awaiting Rust code wakes with its destination already filled.
/// `dst` must stay allocated until the task resolves.
///
/// words: `(buffer, task, dst_ptr, dst_len)`.
pub unsafe fn quanta_map_async_read(buffer: u32, task: u32, dst_ptr: *mut u8, dst_len: usize) {
    tape::emit(
        Op::MapAsyncRead,
        &[buffer, task, dst_ptr as u32, dst_len as u32],
        &[],
    );
}

// ── Shader / compute pipeline ──────────────────────────────────────────────

/// words: `(dst, device)` + payload = WGSL source.
pub unsafe fn quanta_create_shader_module(
    device: u32,
    code_ptr: *const u8,
    code_len: usize,
) -> u32 {
    let id = tape::mint_id();
    let code = unsafe { core::slice::from_raw_parts(code_ptr, code_len) };
    tape::emit(Op::CreateShaderModule, &[id, device], &[code]);
    id
}

/// words: `(dst, device, module)` + payload = entry-point name.
pub unsafe fn quanta_create_compute_pipeline(
    device: u32,
    module: u32,
    entry_ptr: *const u8,
    entry_len: usize,
) -> u32 {
    let id = tape::mint_id();
    let entry = unsafe { core::slice::from_raw_parts(entry_ptr, entry_len) };
    tape::emit(Op::CreateComputePipeline, &[id, device, module], &[entry]);
    id
}

pub unsafe fn quanta_compute_pipeline_get_bind_group_layout(pipeline: u32, index: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(
        Op::ComputePipelineGetBindGroupLayout,
        &[id, pipeline, index],
        &[],
    );
    id
}

// ── Render pipeline (descriptor builder) ───────────────────────────────────

pub unsafe fn quanta_rp_desc_create() -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::RpDescCreate, &[id], &[]);
    id
}

/// words: `(desc, module)` + payload = entry-point name.
pub unsafe fn quanta_rp_desc_set_vertex(
    desc: u32,
    module: u32,
    entry_ptr: *const u8,
    entry_len: usize,
) {
    let entry = unsafe { core::slice::from_raw_parts(entry_ptr, entry_len) };
    tape::emit(Op::RpDescSetVertex, &[desc, module], &[entry]);
}

pub unsafe fn quanta_rp_desc_add_vertex_buffer(desc: u32, stride: u32, step_mode: u32) {
    tape::emit(Op::RpDescAddVertexBuffer, &[desc, stride, step_mode], &[]);
}

pub unsafe fn quanta_rp_desc_add_vertex_attribute(
    desc: u32,
    buf_index: u32,
    format_code: u32,
    offset: u32,
    location: u32,
) {
    tape::emit(
        Op::RpDescAddVertexAttribute,
        &[desc, buf_index, format_code, offset, location],
        &[],
    );
}

/// words: `(desc, module)` + payload = entry-point name.
pub unsafe fn quanta_rp_desc_set_fragment(
    desc: u32,
    module: u32,
    entry_ptr: *const u8,
    entry_len: usize,
) {
    let entry = unsafe { core::slice::from_raw_parts(entry_ptr, entry_len) };
    tape::emit(Op::RpDescSetFragment, &[desc, module], &[entry]);
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn quanta_rp_desc_add_color_target(
    desc: u32,
    format_code: u32,
    blend_enabled: u32,
    src_color: u32,
    dst_color: u32,
    op_color: u32,
    src_alpha: u32,
    dst_alpha: u32,
    op_alpha: u32,
) {
    tape::emit(
        Op::RpDescAddColorTarget,
        &[
            desc,
            format_code,
            blend_enabled,
            src_color,
            dst_color,
            op_color,
            src_alpha,
            dst_alpha,
            op_alpha,
        ],
        &[],
    );
}

pub unsafe fn quanta_rp_desc_set_primitive(desc: u32, topology_code: u32, cull_mode_code: u32) {
    tape::emit(
        Op::RpDescSetPrimitive,
        &[desc, topology_code, cull_mode_code],
        &[],
    );
}

pub unsafe fn quanta_rp_desc_set_multisample(desc: u32, count: u32) {
    tape::emit(Op::RpDescSetMultisample, &[desc, count], &[]);
}

pub unsafe fn quanta_rp_desc_set_depth_stencil(
    desc: u32,
    format_code: u32,
    depth_write: u32,
    compare_code: u32,
) {
    tape::emit(
        Op::RpDescSetDepthStencil,
        &[desc, format_code, depth_write, compare_code],
        &[],
    );
}

pub unsafe fn quanta_create_render_pipeline(device: u32, desc: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::CreateRenderPipeline, &[id, device, desc], &[]);
    id
}

pub unsafe fn quanta_render_pipeline_get_bind_group_layout(pipeline: u32, index: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(
        Op::RenderPipelineGetBindGroupLayout,
        &[id, pipeline, index],
        &[],
    );
    id
}

// ── Bind group (descriptor builder) ────────────────────────────────────────

pub unsafe fn quanta_bg_desc_create(layout: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::BgDescCreate, &[id, layout], &[]);
    id
}

pub unsafe fn quanta_bg_desc_add_buffer(desc: u32, binding: u32, buffer: u32) {
    tape::emit(Op::BgDescAddBuffer, &[desc, binding, buffer], &[]);
}

pub unsafe fn quanta_bg_desc_add_sampler(desc: u32, binding: u32, sampler: u32) {
    tape::emit(Op::BgDescAddSampler, &[desc, binding, sampler], &[]);
}

pub unsafe fn quanta_bg_desc_add_texture_view(desc: u32, binding: u32, view: u32) {
    tape::emit(Op::BgDescAddTextureView, &[desc, binding, view], &[]);
}

pub unsafe fn quanta_create_bind_group(device: u32, desc: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::CreateBindGroup, &[id, device, desc], &[]);
    id
}

// ── Command encoder ────────────────────────────────────────────────────────

pub unsafe fn quanta_create_command_encoder(device: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::CreateCommandEncoder, &[id, device], &[]);
    id
}

/// words: `(encoder, src, src_off:f64, dst, dst_off:f64, size:f64)`.
pub unsafe fn quanta_encoder_copy_buffer_to_buffer(
    encoder: u32,
    src: u32,
    src_off: f64,
    dst: u32,
    dst_off: f64,
    size: f64,
) {
    let [so0, so1] = tape::f64_words(src_off);
    let [do0, do1] = tape::f64_words(dst_off);
    let [sz0, sz1] = tape::f64_words(size);
    tape::emit(
        Op::EncoderCopyBufferToBuffer,
        &[encoder, src, so0, so1, dst, do0, do1, sz0, sz1],
        &[],
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn quanta_encoder_copy_texture_to_buffer(
    encoder: u32,
    src_texture: u32,
    dst_buffer: u32,
    dst_bytes_per_row: u32,
    dst_rows_per_image: u32,
    width: u32,
    height: u32,
    depth: u32,
) {
    tape::emit(
        Op::EncoderCopyTextureToBuffer,
        &[
            encoder,
            src_texture,
            dst_buffer,
            dst_bytes_per_row,
            dst_rows_per_image,
            width,
            height,
            depth,
        ],
        &[],
    );
}

pub unsafe fn quanta_encoder_finish(encoder: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::EncoderFinish, &[id, encoder], &[]);
    id
}

// ── Compute pass ───────────────────────────────────────────────────────────

pub unsafe fn quanta_encoder_begin_compute_pass(encoder: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::EncoderBeginComputePass, &[id, encoder], &[]);
    id
}

pub unsafe fn quanta_compute_pass_set_pipeline(pass: u32, pipeline: u32) {
    tape::emit(Op::ComputePassSetPipeline, &[pass, pipeline], &[]);
}

pub unsafe fn quanta_compute_pass_set_bind_group(pass: u32, index: u32, group: u32) {
    tape::emit(Op::ComputePassSetBindGroup, &[pass, index, group], &[]);
}

pub unsafe fn quanta_compute_pass_dispatch(pass: u32, x: u32, y: u32, z: u32) {
    tape::emit(Op::ComputePassDispatch, &[pass, x, y, z], &[]);
}

pub unsafe fn quanta_compute_pass_end(pass: u32) {
    tape::emit(Op::ComputePassEnd, &[pass], &[]);
}

// ── Render pass ────────────────────────────────────────────────────────────

pub unsafe fn quanta_rpass_desc_create() -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::RpassDescCreate, &[id], &[]);
    id
}

/// `resolve_view` is WebGPU's native `resolveTarget` — the view of
/// a single-sample destination the multisampled attachment
/// resolves into at pass end. `NULL_HANDLE` = no resolve.
///
/// words: `(desc, view, load_op, store_op, resolve_view, r, g, b, a)`,
/// the clear color one `f32` bit pattern per word.
#[allow(clippy::too_many_arguments)]
pub unsafe fn quanta_rpass_desc_add_color_attachment(
    desc: u32,
    view: u32,
    load_op: u32,
    store_op: u32,
    resolve_view: u32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) {
    tape::emit(
        Op::RpassDescAddColorAttachment,
        &[
            desc,
            view,
            load_op,
            store_op,
            resolve_view,
            r.to_bits(),
            g.to_bits(),
            b.to_bits(),
            a.to_bits(),
        ],
        &[],
    );
}

pub unsafe fn quanta_rpass_desc_set_depth_attachment(
    desc: u32,
    view: u32,
    load_op: u32,
    store_op: u32,
    clear_depth: f32,
) {
    tape::emit(
        Op::RpassDescSetDepthAttachment,
        &[desc, view, load_op, store_op, clear_depth.to_bits()],
        &[],
    );
}

pub unsafe fn quanta_encoder_begin_render_pass(encoder: u32, desc: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::EncoderBeginRenderPass, &[id, encoder, desc], &[]);
    id
}

pub unsafe fn quanta_render_pass_set_pipeline(pass: u32, pipeline: u32) {
    tape::emit(Op::RenderPassSetPipeline, &[pass, pipeline], &[]);
}

pub unsafe fn quanta_render_pass_set_bind_group(pass: u32, index: u32, group: u32) {
    tape::emit(Op::RenderPassSetBindGroup, &[pass, index, group], &[]);
}

/// words: `(pass, slot, buffer, offset:f64)`.
pub unsafe fn quanta_render_pass_set_vertex_buffer(pass: u32, slot: u32, buffer: u32, offset: f64) {
    let [o0, o1] = tape::f64_words(offset);
    tape::emit(
        Op::RenderPassSetVertexBuffer,
        &[pass, slot, buffer, o0, o1],
        &[],
    );
}

/// words: `(pass, buffer, format_code, offset:f64)`.
pub unsafe fn quanta_render_pass_set_index_buffer(
    pass: u32,
    buffer: u32,
    format_code: u32,
    offset: f64,
) {
    let [o0, o1] = tape::f64_words(offset);
    tape::emit(
        Op::RenderPassSetIndexBuffer,
        &[pass, buffer, format_code, o0, o1],
        &[],
    );
}

// Occlusion query support (post-step-063 closure). Maps the
// typed `OcclusionQuery` API to GPUQuerySet + the
// occlusionQuerySet field on render pass descriptors +
// beginOcclusionQuery / endOcclusionQuery + resolveQuerySet
// for asynchronous result readback.

pub unsafe fn quanta_create_query_set(device: u32, count: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::CreateQuerySet, &[id, device, count], &[]);
    id
}

pub unsafe fn quanta_rpass_desc_set_occlusion_query_set(desc: u32, query_set: u32) {
    tape::emit(Op::RpassDescSetOcclusionQuerySet, &[desc, query_set], &[]);
}

pub unsafe fn quanta_render_pass_begin_occlusion_query(pass: u32, index: u32) {
    tape::emit(Op::RenderPassBeginOcclusionQuery, &[pass, index], &[]);
}

pub unsafe fn quanta_render_pass_end_occlusion_query(pass: u32) {
    tape::emit(Op::RenderPassEndOcclusionQuery, &[pass], &[]);
}

/// Encode a resolve from the query set into a buffer. The
/// destination buffer must have COPY_DST + QUERY_RESOLVE
/// usage. Each query result is 8 bytes (u64).
///
/// words: `(encoder, query_set, first_query, query_count, dst_buffer,
/// dst_offset:f64)`.
pub unsafe fn quanta_encoder_resolve_query_set(
    encoder: u32,
    query_set: u32,
    first_query: u32,
    query_count: u32,
    dst_buffer: u32,
    dst_offset: f64,
) {
    let [o0, o1] = tape::f64_words(dst_offset);
    tape::emit(
        Op::EncoderResolveQuerySet,
        &[
            encoder,
            query_set,
            first_query,
            query_count,
            dst_buffer,
            o0,
            o1,
        ],
        &[],
    );
}

pub unsafe fn quanta_render_pass_draw(pass: u32, vertex_count: u32, instance_count: u32) {
    tape::emit(
        Op::RenderPassDraw,
        &[pass, vertex_count, instance_count],
        &[],
    );
}

pub unsafe fn quanta_render_pass_draw_indexed(pass: u32, index_count: u32, instance_count: u32) {
    tape::emit(
        Op::RenderPassDrawIndexed,
        &[pass, index_count, instance_count],
        &[],
    );
}

/// words: `(pass, indirect_buffer, indirect_offset:f64)`.
pub unsafe fn quanta_render_pass_draw_indirect(
    pass: u32,
    indirect_buffer: u32,
    indirect_offset: f64,
) {
    let [o0, o1] = tape::f64_words(indirect_offset);
    tape::emit(
        Op::RenderPassDrawIndirect,
        &[pass, indirect_buffer, o0, o1],
        &[],
    );
}

/// words: `(pass, indirect_buffer, indirect_offset:f64)`.
pub unsafe fn quanta_render_pass_draw_indexed_indirect(
    pass: u32,
    indirect_buffer: u32,
    indirect_offset: f64,
) {
    let [o0, o1] = tape::f64_words(indirect_offset);
    tape::emit(
        Op::RenderPassDrawIndexedIndirect,
        &[pass, indirect_buffer, o0, o1],
        &[],
    );
}

/// words: `(pass, x, y, w, h, min_depth, max_depth)`, one `f32` bit
/// pattern per word after `pass`.
pub unsafe fn quanta_render_pass_set_viewport(
    pass: u32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    min_depth: f32,
    max_depth: f32,
) {
    tape::emit(
        Op::RenderPassSetViewport,
        &[
            pass,
            x.to_bits(),
            y.to_bits(),
            w.to_bits(),
            h.to_bits(),
            min_depth.to_bits(),
            max_depth.to_bits(),
        ],
        &[],
    );
}

pub unsafe fn quanta_render_pass_set_scissor(pass: u32, x: u32, y: u32, w: u32, h: u32) {
    tape::emit(Op::RenderPassSetScissor, &[pass, x, y, w, h], &[]);
}

pub unsafe fn quanta_render_pass_set_stencil_reference(pass: u32, reference: u32) {
    tape::emit(Op::RenderPassSetStencilReference, &[pass, reference], &[]);
}

pub unsafe fn quanta_render_pass_end(pass: u32) {
    tape::emit(Op::RenderPassEnd, &[pass], &[]);
}

// ── Render bundle (steps 032 + 033, render path) ───────────────────────────

pub unsafe fn quanta_create_render_bundle_encoder(
    device: u32,
    color_format_code: u32,
    depth_format_code: u32,
    sample_count: u32,
) -> u32 {
    let id = tape::mint_id();
    tape::emit(
        Op::CreateRenderBundleEncoder,
        &[
            id,
            device,
            color_format_code,
            depth_format_code,
            sample_count,
        ],
        &[],
    );
    id
}

pub unsafe fn quanta_render_bundle_set_pipeline(encoder: u32, pipeline: u32) {
    tape::emit(Op::RenderBundleSetPipeline, &[encoder, pipeline], &[]);
}

pub unsafe fn quanta_render_bundle_set_bind_group(encoder: u32, index: u32, group: u32) {
    tape::emit(Op::RenderBundleSetBindGroup, &[encoder, index, group], &[]);
}

/// words: `(encoder, slot, buffer, offset:f64)`.
pub unsafe fn quanta_render_bundle_set_vertex_buffer(
    encoder: u32,
    slot: u32,
    buffer: u32,
    offset: f64,
) {
    let [o0, o1] = tape::f64_words(offset);
    tape::emit(
        Op::RenderBundleSetVertexBuffer,
        &[encoder, slot, buffer, o0, o1],
        &[],
    );
}

pub unsafe fn quanta_render_bundle_draw(encoder: u32, vertex_count: u32, instance_count: u32) {
    tape::emit(
        Op::RenderBundleDraw,
        &[encoder, vertex_count, instance_count],
        &[],
    );
}

pub unsafe fn quanta_render_bundle_finish(encoder: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::RenderBundleFinish, &[id, encoder], &[]);
    id
}

/// words: `(pass)` + payload = the `count` bundle handles as LE `u32`s,
/// so the count is the payload length over four.
pub unsafe fn quanta_render_pass_execute_bundles(pass: u32, bundles_ptr: *const u32, count: u32) {
    let bundles =
        unsafe { core::slice::from_raw_parts(bundles_ptr.cast::<u8>(), count as usize * 4) };
    tape::emit(Op::RenderPassExecuteBundles, &[pass], &[bundles]);
}

// ── Queue ──────────────────────────────────────────────────────────────────

pub unsafe fn quanta_queue_submit(device: u32, command_buffer: u32) {
    tape::emit(Op::QueueSubmit, &[device, command_buffer], &[]);
}

pub unsafe fn quanta_queue_on_submitted_work_done(device: u32, task: u32) {
    tape::emit(Op::QueueOnSubmittedWorkDone, &[device, task], &[]);
}

// ── Textures / samplers ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub unsafe fn quanta_create_texture(
    device: u32,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
    sample_count: u32,
    format_code: u32,
    usage: u32,
) -> u32 {
    let id = tape::mint_id();
    tape::emit(
        Op::CreateTexture,
        &[
            id,
            device,
            width,
            height,
            depth_or_array_layers,
            mip_level_count,
            sample_count,
            format_code,
            usage,
        ],
        &[],
    );
    id
}

pub unsafe fn quanta_texture_create_view(texture: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::TextureCreateView, &[id, texture], &[]);
    id
}

pub unsafe fn quanta_destroy_texture(texture: u32) {
    tape::emit(Op::DestroyTexture, &[texture], &[]);
}

/// words: `(device, texture, origin_x, origin_y, bytes_per_row,
/// rows_per_image, width, height, depth)` + payload = the pixel bytes.
/// The source `(ptr, len)` sits in the MIDDLE of the argument list but
/// its payload rides at the end, like every other payload op.
#[allow(clippy::too_many_arguments)]
pub unsafe fn quanta_queue_write_texture(
    device: u32,
    texture: u32,
    origin_x: u32,
    origin_y: u32,
    data_ptr: *const u8,
    data_len: usize,
    bytes_per_row: u32,
    rows_per_image: u32,
    width: u32,
    height: u32,
    depth: u32,
) {
    let data = unsafe { core::slice::from_raw_parts(data_ptr, data_len) };
    tape::emit(
        Op::QueueWriteTexture,
        &[
            device,
            texture,
            origin_x,
            origin_y,
            bytes_per_row,
            rows_per_image,
            width,
            height,
            depth,
        ],
        &[data],
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn quanta_create_sampler(
    device: u32,
    mag_filter: u32,
    min_filter: u32,
    mipmap_filter: u32,
    address_u: u32,
    address_v: u32,
    address_w: u32,
    max_anisotropy: u32,
    compare_code: u32,
) -> u32 {
    let id = tape::mint_id();
    tape::emit(
        Op::CreateSampler,
        &[
            id,
            device,
            mag_filter,
            min_filter,
            mipmap_filter,
            address_u,
            address_v,
            address_w,
            max_anisotropy,
            compare_code,
        ],
        &[],
    );
    id
}

// ── Canvas presentation (step 096) ─────────────────────────────────────────
// The canvas handle comes from the page (`registerCanvas` on the
// instantiated module — a GLUE-minted id, below the high bit) or from
// `quanta_canvas_create_offscreen` for headless surfaces. The context
// handle is a `GPUCanvasContext`. There is deliberately no present op:
// the browser composites the current texture when the task ends —
// queue submission order is the present ordering.

/// Sync-query class: reads state the glue PUSHED at instantiation
/// (`__quanta_env_init`) instead of calling out. `navigator.gpu !==
/// undefined`, safe before any device exists — the runtime-capability
/// pre-flight (R3).
pub unsafe fn quanta_webgpu_available() -> u32 {
    tape::env_available()
}

/// Create a driver-owned `OffscreenCanvas` (headless surfaces).
pub unsafe fn quanta_canvas_create_offscreen(width: u32, height: u32) -> u32 {
    let id = tape::mint_id();
    tape::set_canvas_dims(id, width, height);
    tape::emit(Op::CanvasCreateOffscreen, &[id, width, height], &[]);
    id
}

/// `canvas.getContext("webgpu")`. A canvas that already handed out a
/// 2d/webgl context has no webgpu context to give: the old import
/// answered `NULL_HANDLE` and the driver raised `NotSupported`, but a
/// deferred call has no answer to return — the glue throws at drain
/// instead.
pub unsafe fn quanta_canvas_context_create(canvas: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::CanvasContextCreate, &[id, canvas], &[]);
    id
}

/// Set the canvas backing size (the `drawableSize` analogue) and
/// `context.configure({device, format, usage, alphaMode:"opaque"})`.
pub unsafe fn quanta_canvas_context_configure(
    context: u32,
    canvas: u32,
    device: u32,
    format_code: u32,
    usage: u32,
    width: u32,
    height: u32,
) {
    tape::set_canvas_dims(canvas, width, height);
    tape::emit(
        Op::CanvasContextConfigure,
        &[context, canvas, device, format_code, usage, width, height],
        &[],
    );
}

pub unsafe fn quanta_canvas_context_unconfigure(context: u32) {
    tape::emit(Op::CanvasContextUnconfigure, &[context], &[]);
}

/// `context.getCurrentTexture()` — the per-frame color target.
pub unsafe fn quanta_canvas_get_current_texture(context: u32) -> u32 {
    let id = tape::mint_id();
    tape::emit(Op::CanvasGetCurrentTexture, &[id, context], &[]);
    id
}

/// Canvas backing-store extent — the acquire-time out-of-date poll
/// and `surface_current_extent` read these. Pushed by the glue
/// (`__quanta_canvas_dims`) at registration and by every configure.
pub unsafe fn quanta_canvas_width(canvas: u32) -> u32 {
    tape::canvas_dims(canvas).0
}

pub unsafe fn quanta_canvas_height(canvas: u32) -> u32 {
    tape::canvas_dims(canvas).1
}

/// `navigator.gpu.getPreferredCanvasFormat()` as a format code
/// (`format::BGRA8UNORM` or `format::RGBA8UNORM`).
pub unsafe fn quanta_canvas_preferred_format() -> u32 {
    tape::env_preferred_format()
}

// ── Universal handle release (for handles without a destroy method) ────────
// Shader modules, pipelines, bind-group layouts, samplers, texture views.

pub unsafe fn quanta_release(handle: u32) {
    tape::emit(Op::Release, &[handle], &[]);
}

// ── Diagnostics ────────────────────────────────────────────────────────────

/// Surfaced from Rust panics / errors before the top-level task even
/// gets to call [`super::complete_err`]. Useful when an init path
/// explodes before there is a task to reject.
///
/// words: none + payload = the utf8 message.
pub unsafe fn quanta_console_error(ptr: *const u8, len: usize) {
    let msg = unsafe { core::slice::from_raw_parts(ptr, len) };
    tape::emit(Op::ConsoleError, &[], &[msg]);
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
