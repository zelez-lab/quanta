// WebGPU FFI imports — the JS half of every wasm import declared in
// `src/driver/webgpu/ffi.rs`. The corresponding Rust extern "C" block
// must mirror this object's shape exactly (function names + arity).
//
// All long-lived JS objects (devices, buffers, pipelines, …) live in
// the shared `HandleTable` and cross the wasm boundary as `u32`. All
// strings cross as (ptr, len) into wasm linear memory.
import { readUtf8, viewBytes } from "./strings.js";
import { bindTask } from "./tasks.js";
import { formatName, attributeFormatName, topologyName, cullModeName, blendFactorName, blendOpName, filterName, addressName, compareName, stepModeName, indexFormatName, loadOpName, storeOpName, } from "./codes.js";
import "./webgpu-types.js";
const COMPARE_UNSET = 0;
export function makeImports(state) {
    // Internal helper — every async import calls this to hand off a JS
    // Promise to the wasm executor. Constructing the closure inline keeps
    // type inference clean.
    function async_(task, p, mapHandle) {
        const e = state.exports;
        if (e === null) {
            throw new Error("quanta glue: async import called before wasm exports were wired");
        }
        bindTask(e, task, p, mapHandle);
    }
    function readString(ptr, len) {
        state.syncCalls++;
        return readUtf8(state.memory, ptr, len);
    }
    // u64 sizes cross the FFI as `f64`. JS numbers are exact integers up
    // to 2^53; WebGPU sizes are well below that.
    function size(n) {
        return n;
    }
    return {
        // ── adapter / device acquisition ────────────────────────────────────────
        quanta_request_adapter(task) {
            const gpu = navigator.gpu;
            if (gpu === undefined) {
                async_(task, Promise.resolve(null), () => 0);
                return;
            }
            async_(task, gpu.requestAdapter(), (a) => a === null ? 0 : state.handles.alloc(a));
        },
        quanta_request_device(adapter, task) {
            const a = state.handles.get(adapter);
            async_(task, a.requestDevice(), (d) => state.handles.alloc(d));
        },
        // ── buffers ────────────────────────────────────────────────────────────
        quanta_create_buffer(device, size_f64, usage) {
            const dev = state.handles.get(device);
            const buf = dev.createBuffer({ size: size(size_f64), usage });
            return state.handles.alloc(buf);
        },
        quanta_destroy_buffer(buffer) {
            const buf = state.handles.get(buffer);
            buf.destroy();
            state.handles.release(buffer);
        },
        quanta_write_buffer(device, buffer, offset_f64, data_ptr, data_len) {
            const dev = state.handles.get(device);
            const buf = state.handles.get(buffer);
            // viewBytes is a borrowed view; writeBuffer copies synchronously.
            dev.queue.writeBuffer(buf, size(offset_f64), viewBytes(state.memory, data_ptr, data_len));
        },
        quanta_map_async_read(buffer, task) {
            const buf = state.handles.get(buffer);
            // GPUMapMode.READ = 0x0001
            async_(task, buf.mapAsync(0x0001), () => 0);
        },
        quanta_get_mapped_range_copy(buffer, dst_ptr, len) {
            const buf = state.handles.get(buffer);
            const range = buf.getMappedRange();
            const src = new Uint8Array(range, 0, len);
            const dst = new Uint8Array(state.memory.buffer, dst_ptr, len);
            dst.set(src);
        },
        quanta_unmap_buffer(buffer) {
            const buf = state.handles.get(buffer);
            buf.unmap();
        },
        // ── shader / compute pipeline ──────────────────────────────────────────
        quanta_create_shader_module(device, code_ptr, code_len) {
            const dev = state.handles.get(device);
            const code = readString(code_ptr, code_len);
            const m = dev.createShaderModule({ code });
            return state.handles.alloc(m);
        },
        quanta_create_compute_pipeline(device, module_h, entry_ptr, entry_len) {
            const dev = state.handles.get(device);
            const m = state.handles.get(module_h);
            const entryPoint = readString(entry_ptr, entry_len);
            const p = dev.createComputePipeline({
                layout: "auto",
                compute: { module: m, entryPoint },
            });
            return state.handles.alloc(p);
        },
        quanta_compute_pipeline_get_bind_group_layout(pipeline, index) {
            const p = state.handles.get(pipeline);
            return state.handles.alloc(p.getBindGroupLayout(index));
        },
        // ── render pipeline (builder pattern) ──────────────────────────────────
        quanta_rp_desc_create() {
            const desc = {
                layout: "auto",
                vertex: null,
                fragment: null,
                primitive: { topology: "triangle-list", cullMode: "none" },
                multisample: { count: 1 },
                depthStencil: null,
                vertexBuffers: [],
                colorTargets: [],
            };
            return state.handles.alloc(desc);
        },
        quanta_rp_desc_set_vertex(desc_h, module_h, entry_ptr, entry_len) {
            const desc = state.handles.get(desc_h);
            const m = state.handles.get(module_h);
            desc.vertex = { module: m, entryPoint: readString(entry_ptr, entry_len) };
        },
        quanta_rp_desc_add_vertex_buffer(desc_h, stride, step_mode) {
            const desc = state.handles.get(desc_h);
            desc.vertexBuffers.push({
                arrayStride: stride,
                stepMode: stepModeName(step_mode),
                attributes: [],
            });
        },
        quanta_rp_desc_add_vertex_attribute(desc_h, buf_index, format_code, offset, location) {
            const desc = state.handles.get(desc_h);
            const buf = desc.vertexBuffers[buf_index];
            if (buf === undefined) {
                throw new Error(`quanta glue: vertex attribute on unknown buffer index ${buf_index}`);
            }
            buf.attributes.push({
                format: attributeFormatName(format_code),
                offset,
                shaderLocation: location,
            });
        },
        quanta_rp_desc_set_fragment(desc_h, module_h, entry_ptr, entry_len) {
            const desc = state.handles.get(desc_h);
            const m = state.handles.get(module_h);
            desc.fragment = {
                module: m,
                entryPoint: readString(entry_ptr, entry_len),
                targets: desc.colorTargets,
            };
        },
        quanta_rp_desc_add_color_target(desc_h, format_code, blend_enabled, src_color, dst_color, op_color, src_alpha, dst_alpha, op_alpha) {
            const desc = state.handles.get(desc_h);
            const target = { format: formatName(format_code) };
            if (blend_enabled !== 0) {
                target.blend = {
                    color: {
                        srcFactor: blendFactorName(src_color),
                        dstFactor: blendFactorName(dst_color),
                        operation: blendOpName(op_color),
                    },
                    alpha: {
                        srcFactor: blendFactorName(src_alpha),
                        dstFactor: blendFactorName(dst_alpha),
                        operation: blendOpName(op_alpha),
                    },
                };
            }
            desc.colorTargets.push(target);
            // If fragment was already set, ensure its `targets` array points
            // to the up-to-date list (we share by reference, so this is a
            // no-op as long as fragment was set after the first push). Keep
            // this branch defensive.
            if (desc.fragment !== null && desc.fragment.targets !== desc.colorTargets) {
                desc.fragment.targets = desc.colorTargets;
            }
        },
        quanta_rp_desc_set_primitive(desc_h, topology_code, cull_mode_code) {
            const desc = state.handles.get(desc_h);
            desc.primitive = {
                topology: topologyName(topology_code),
                cullMode: cullModeName(cull_mode_code),
            };
        },
        quanta_rp_desc_set_multisample(desc_h, count) {
            const desc = state.handles.get(desc_h);
            desc.multisample = { count };
        },
        quanta_rp_desc_set_depth_stencil(desc_h, format_code, depth_write, compare_code) {
            const desc = state.handles.get(desc_h);
            desc.depthStencil = {
                format: formatName(format_code),
                depthWriteEnabled: depth_write !== 0,
                depthCompare: compareName(compare_code),
            };
        },
        quanta_create_render_pipeline(device, desc_h) {
            const dev = state.handles.get(device);
            const desc = state.handles.get(desc_h);
            // Stitch vertex buffers into the vertex stage; descriptor is
            // built lazily here to avoid mutating the JS object every time
            // a vertex buffer gets added.
            const vertexStage = desc.vertex === null
                ? null
                : { ...desc.vertex, buffers: desc.vertexBuffers };
            const pipelineDesc = {
                layout: desc.layout,
                vertex: vertexStage,
                primitive: desc.primitive,
                multisample: desc.multisample,
            };
            if (desc.fragment !== null)
                pipelineDesc.fragment = desc.fragment;
            if (desc.depthStencil !== null)
                pipelineDesc.depthStencil = desc.depthStencil;
            const p = dev.createRenderPipeline(pipelineDesc);
            state.handles.release(desc_h);
            return state.handles.alloc(p);
        },
        quanta_render_pipeline_get_bind_group_layout(pipeline, index) {
            const p = state.handles.get(pipeline);
            return state.handles.alloc(p.getBindGroupLayout(index));
        },
        // ── bind group (builder pattern) ───────────────────────────────────────
        quanta_bg_desc_create(layout) {
            const l = state.handles.get(layout);
            const desc = { layout: l, entries: [] };
            return state.handles.alloc(desc);
        },
        quanta_bg_desc_add_buffer(desc_h, binding, buffer) {
            const desc = state.handles.get(desc_h);
            const buf = state.handles.get(buffer);
            desc.entries.push({ binding, resource: { buffer: buf } });
        },
        quanta_bg_desc_add_sampler(desc_h, binding, sampler) {
            const desc = state.handles.get(desc_h);
            const s = state.handles.get(sampler);
            desc.entries.push({ binding, resource: s });
        },
        quanta_bg_desc_add_texture_view(desc_h, binding, view) {
            const desc = state.handles.get(desc_h);
            const v = state.handles.get(view);
            desc.entries.push({ binding, resource: v });
        },
        quanta_create_bind_group(device, desc_h) {
            const dev = state.handles.get(device);
            const desc = state.handles.get(desc_h);
            const bg = dev.createBindGroup(desc);
            state.handles.release(desc_h);
            return state.handles.alloc(bg);
        },
        // ── command encoder ────────────────────────────────────────────────────
        quanta_create_command_encoder(device) {
            const dev = state.handles.get(device);
            return state.handles.alloc(dev.createCommandEncoder());
        },
        quanta_encoder_copy_buffer_to_buffer(encoder, src, src_off, dst, dst_off, n) {
            const enc = state.handles.get(encoder);
            const s = state.handles.get(src);
            const d = state.handles.get(dst);
            enc.copyBufferToBuffer(s, size(src_off), d, size(dst_off), size(n));
        },
        quanta_encoder_copy_texture_to_buffer(encoder, src_texture, dst_buffer, dst_bytes_per_row, dst_rows_per_image, width, height, depth) {
            const enc = state.handles.get(encoder);
            const t = state.handles.get(src_texture);
            const b = state.handles.get(dst_buffer);
            enc.copyTextureToBuffer({ texture: t }, {
                buffer: b,
                bytesPerRow: dst_bytes_per_row,
                rowsPerImage: dst_rows_per_image,
            }, { width, height, depthOrArrayLayers: depth });
        },
        quanta_encoder_finish(encoder) {
            const enc = state.handles.get(encoder);
            const cmd = enc.finish();
            state.handles.release(encoder);
            return state.handles.alloc(cmd);
        },
        // ── compute pass ───────────────────────────────────────────────────────
        quanta_encoder_begin_compute_pass(encoder) {
            const enc = state.handles.get(encoder);
            return state.handles.alloc(enc.beginComputePass());
        },
        quanta_compute_pass_set_pipeline(pass, pipeline) {
            const cp = state.handles.get(pass);
            const p = state.handles.get(pipeline);
            cp.setPipeline(p);
        },
        quanta_compute_pass_set_bind_group(pass, index, group) {
            const cp = state.handles.get(pass);
            const g = state.handles.get(group);
            cp.setBindGroup(index, g);
        },
        quanta_compute_pass_dispatch(pass, x, y, z) {
            const cp = state.handles.get(pass);
            cp.dispatchWorkgroups(x, y, z);
        },
        quanta_compute_pass_end(pass) {
            const cp = state.handles.get(pass);
            cp.end();
            state.handles.release(pass);
        },
        // ── render pass (descriptor builder + execute) ─────────────────────────
        quanta_rpass_desc_create() {
            const desc = {
                colorAttachments: [],
                depthStencilAttachment: null,
            };
            return state.handles.alloc(desc);
        },
        quanta_rpass_desc_add_color_attachment(desc_h, view, load_op, store_op, resolve_view, r, g, b, a) {
            const desc = state.handles.get(desc_h);
            const v = state.handles.get(view);
            const att = {
                view: v,
                loadOp: loadOpName(load_op),
                storeOp: storeOpName(store_op),
                clearValue: { r, g, b, a },
            };
            if (resolve_view !== 0) {
                att.resolveTarget = state.handles.get(resolve_view);
            }
            desc.colorAttachments.push(att);
        },
        quanta_rpass_desc_set_depth_attachment(desc_h, view, load_op, store_op, clear_depth) {
            const desc = state.handles.get(desc_h);
            const v = state.handles.get(view);
            desc.depthStencilAttachment = {
                view: v,
                depthLoadOp: loadOpName(load_op),
                depthStoreOp: storeOpName(store_op),
                depthClearValue: clear_depth,
            };
        },
        quanta_encoder_begin_render_pass(encoder, desc_h) {
            const enc = state.handles.get(encoder);
            const desc = state.handles.get(desc_h);
            const passDesc = { colorAttachments: desc.colorAttachments };
            if (desc.depthStencilAttachment !== null) {
                passDesc.depthStencilAttachment = desc.depthStencilAttachment;
            }
            if (desc.occlusionQuerySet !== undefined) {
                passDesc.occlusionQuerySet = desc.occlusionQuerySet;
            }
            const rp = enc.beginRenderPass(passDesc);
            state.handles.release(desc_h);
            return state.handles.alloc(rp);
        },
        // Occlusion queries (post-step-063 closure).
        quanta_create_query_set(device, count) {
            const dev = state.handles.get(device);
            const qs = dev.createQuerySet({ type: "occlusion", count });
            return state.handles.alloc(qs);
        },
        quanta_rpass_desc_set_occlusion_query_set(desc_h, query_set) {
            const desc = state.handles.get(desc_h);
            desc.occlusionQuerySet = state.handles.get(query_set);
        },
        quanta_render_pass_begin_occlusion_query(pass, index) {
            const rp = state.handles.get(pass);
            rp.beginOcclusionQuery(index);
        },
        quanta_render_pass_end_occlusion_query(pass) {
            const rp = state.handles.get(pass);
            rp.endOcclusionQuery();
        },
        quanta_encoder_resolve_query_set(encoder, query_set, first_query, query_count, dst_buffer, dst_offset) {
            const enc = state.handles.get(encoder);
            const qs = state.handles.get(query_set);
            const dst = state.handles.get(dst_buffer);
            enc.resolveQuerySet(qs, first_query, query_count, dst, dst_offset);
        },
        quanta_render_pass_set_pipeline(pass, pipeline) {
            const rp = state.handles.get(pass);
            const p = state.handles.get(pipeline);
            rp.setPipeline(p);
        },
        quanta_render_pass_set_bind_group(pass, index, group) {
            const rp = state.handles.get(pass);
            const g = state.handles.get(group);
            rp.setBindGroup(index, g);
        },
        quanta_render_pass_set_vertex_buffer(pass, slot, buffer, offset) {
            const rp = state.handles.get(pass);
            const b = state.handles.get(buffer);
            rp.setVertexBuffer(slot, b, size(offset));
        },
        quanta_render_pass_set_index_buffer(pass, buffer, format_code, offset) {
            const rp = state.handles.get(pass);
            const b = state.handles.get(buffer);
            rp.setIndexBuffer(b, indexFormatName(format_code), size(offset));
        },
        quanta_render_pass_draw(pass, vertex_count, instance_count) {
            const rp = state.handles.get(pass);
            rp.draw(vertex_count, instance_count);
        },
        quanta_render_pass_draw_indexed(pass, index_count, instance_count) {
            const rp = state.handles.get(pass);
            rp.drawIndexed(index_count, instance_count);
        },
        quanta_render_pass_draw_indirect(pass, indirect_buffer, indirect_offset) {
            const rp = state.handles.get(pass);
            const buf = state.handles.get(indirect_buffer);
            rp.drawIndirect(buf, indirect_offset);
        },
        quanta_render_pass_draw_indexed_indirect(pass, indirect_buffer, indirect_offset) {
            const rp = state.handles.get(pass);
            const buf = state.handles.get(indirect_buffer);
            rp.drawIndexedIndirect(buf, indirect_offset);
        },
        quanta_render_pass_set_viewport(pass, x, y, w, h, min_d, max_d) {
            const rp = state.handles.get(pass);
            rp.setViewport(x, y, w, h, min_d, max_d);
        },
        quanta_render_pass_set_scissor(pass, x, y, w, h) {
            const rp = state.handles.get(pass);
            rp.setScissorRect(x, y, w, h);
        },
        quanta_render_pass_set_stencil_reference(pass, reference) {
            const rp = state.handles.get(pass);
            rp.setStencilReference(reference);
        },
        quanta_render_pass_end(pass) {
            const rp = state.handles.get(pass);
            rp.end();
            state.handles.release(pass);
        },
        // ── render bundles (steps 032 + 033) ──────────────────────────────────
        quanta_create_render_bundle_encoder(device, color_format_code, depth_format_code, sample_count) {
            const dev = state.handles.get(device);
            const desc = {
                colorFormats: [formatName(color_format_code)],
                sampleCount: sample_count > 0 ? sample_count : 1,
            };
            if (depth_format_code !== 0) {
                desc.depthStencilFormat = formatName(depth_format_code);
            }
            const enc = dev.createRenderBundleEncoder(desc);
            return state.handles.alloc(enc);
        },
        quanta_render_bundle_set_pipeline(encoder, pipeline) {
            const enc = state.handles.get(encoder);
            const p = state.handles.get(pipeline);
            enc.setPipeline(p);
        },
        quanta_render_bundle_set_bind_group(encoder, index, group) {
            const enc = state.handles.get(encoder);
            const g = state.handles.get(group);
            enc.setBindGroup(index, g);
        },
        quanta_render_bundle_set_vertex_buffer(encoder, slot, buffer, offset) {
            const enc = state.handles.get(encoder);
            const b = state.handles.get(buffer);
            enc.setVertexBuffer(slot, b, size(offset));
        },
        quanta_render_bundle_draw(encoder, vertex_count, instance_count) {
            const enc = state.handles.get(encoder);
            enc.draw(vertex_count, instance_count);
        },
        quanta_render_bundle_finish(encoder) {
            const enc = state.handles.get(encoder);
            const bundle = enc.finish();
            state.handles.release(encoder);
            return state.handles.alloc(bundle);
        },
        quanta_render_pass_execute_bundles(pass, bundles_ptr, count) {
            const rp = state.handles.get(pass);
            const view = new Uint32Array(state.memory.buffer, bundles_ptr, count);
            const arr = [];
            for (let i = 0; i < count; i++) {
                arr.push(state.handles.get(view[i]));
            }
            rp.executeBundles(arr);
        },
        // ── queue ──────────────────────────────────────────────────────────────
        quanta_queue_submit(device, command_buffer) {
            const dev = state.handles.get(device);
            const cb = state.handles.get(command_buffer);
            dev.queue.submit([cb]);
            state.handles.release(command_buffer);
        },
        quanta_queue_on_submitted_work_done(device, task) {
            const dev = state.handles.get(device);
            async_(task, dev.queue.onSubmittedWorkDone(), () => 0);
        },
        // ── textures / samplers ────────────────────────────────────────────────
        quanta_create_texture(device, width, height, depth_or_array_layers, mip_level_count, sample_count, format_code, usage) {
            const dev = state.handles.get(device);
            const tex = dev.createTexture({
                size: { width, height, depthOrArrayLayers: depth_or_array_layers },
                mipLevelCount: mip_level_count,
                sampleCount: sample_count,
                format: formatName(format_code),
                usage,
            });
            return state.handles.alloc(tex);
        },
        quanta_texture_create_view(texture) {
            const t = state.handles.get(texture);
            return state.handles.alloc(t.createView());
        },
        quanta_destroy_texture(texture) {
            const t = state.handles.get(texture);
            t.destroy();
            state.handles.release(texture);
        },
        quanta_queue_write_texture(device, texture, data_ptr, data_len, bytes_per_row, rows_per_image, width, height, depth) {
            const dev = state.handles.get(device);
            const t = state.handles.get(texture);
            dev.queue.writeTexture({ texture: t }, viewBytes(state.memory, data_ptr, data_len), { offset: 0, bytesPerRow: bytes_per_row, rowsPerImage: rows_per_image }, { width, height, depthOrArrayLayers: depth });
        },
        quanta_create_sampler(device, mag_filter, min_filter, mipmap_filter, address_u, address_v, address_w, max_anisotropy, compare_code) {
            const dev = state.handles.get(device);
            const desc = {
                magFilter: filterName(mag_filter),
                minFilter: filterName(min_filter),
                mipmapFilter: filterName(mipmap_filter),
                addressModeU: addressName(address_u),
                addressModeV: addressName(address_v),
                addressModeW: addressName(address_w),
            };
            if (max_anisotropy > 1)
                desc.maxAnisotropy = max_anisotropy;
            if (compare_code !== COMPARE_UNSET)
                desc.compare = compareName(compare_code);
            const s = dev.createSampler(desc);
            return state.handles.alloc(s);
        },
        // ── canvas presentation (step 096) ─────────────────────────────────────
        // The canvas handle is embedder-registered (`registerCanvas` on the
        // instantiated module) or created here for headless surfaces. There
        // is no present import on purpose: the compositor shows the current
        // texture when the task returns to the event loop.
        quanta_webgpu_available() {
            return navigator.gpu !== undefined ? 1 : 0;
        },
        quanta_canvas_create_offscreen(width, height) {
            return state.handles.alloc(new OffscreenCanvas(width, height));
        },
        quanta_canvas_context_create(canvas) {
            const c = state.handles.get(canvas);
            // "webgpu" is missing from lib.dom's getContext overloads; the
            // cast is the entire accommodation.
            const ctx = c.getContext("webgpu");
            return ctx === null ? 0 : state.handles.alloc(ctx);
        },
        quanta_canvas_context_configure(context, canvas, device, format_code, usage, width, height) {
            const ctx = state.handles.get(context);
            const c = state.handles.get(canvas);
            const dev = state.handles.get(device);
            // Drive the backing-store size (the drawableSize analogue). CSS
            // layout size stays the embedder's.
            c.width = width;
            c.height = height;
            ctx.configure({
                device: dev,
                format: formatName(format_code),
                usage,
                alphaMode: "opaque",
            });
        },
        quanta_canvas_context_unconfigure(context) {
            state.handles.get(context).unconfigure();
        },
        quanta_canvas_get_current_texture(context) {
            const ctx = state.handles.get(context);
            return state.handles.alloc(ctx.getCurrentTexture());
        },
        quanta_canvas_width(canvas) {
            return state.handles.get(canvas).width;
        },
        quanta_canvas_height(canvas) {
            return state.handles.get(canvas).height;
        },
        quanta_canvas_preferred_format() {
            // Mirrors ffi.rs `format`: rgba8unorm = 0, bgra8unorm = 1.
            return navigator.gpu.getPreferredCanvasFormat() === "rgba8unorm" ? 0 : 1;
        },
        // ── universal handle release (for handles without a destroy method) ────
        quanta_release(handle) {
            state.handles.release(handle);
        },
        // ── debug ──────────────────────────────────────────────────────────────
        quanta_console_error(ptr, len) {
            console.error(readUtf8(state.memory, ptr, len));
        },
    };
}
