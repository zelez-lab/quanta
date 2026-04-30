// WebGPU FFI imports — the JS half of every wasm import declared in
// `src/driver/webgpu/ffi.rs`. The corresponding Rust extern "C" block
// must mirror this object's shape exactly (function names + arity).
//
// All long-lived JS objects (devices, buffers, pipelines, …) live in
// the shared `HandleTable` and cross the wasm boundary as `u32`. All
// strings cross as (ptr, len) into wasm linear memory.

import { HandleTable } from "./handles.js";
import { readUtf8, viewBytes } from "./strings.js";
import { bindTask, type WasmExports } from "./tasks.js";
import {
  formatName,
  attributeFormatName,
  topologyName,
  cullModeName,
  blendFactorName,
  blendOpName,
  filterName,
  addressName,
  compareName,
  stepModeName,
  indexFormatName,
  loadOpName,
  storeOpName,
} from "./codes.js";

import "./webgpu-types.js";

/**
 * Mutable shared state the imports close over. `exports` is filled in
 * after `WebAssembly.instantiate` returns — until then, only the
 * synchronous imports are safe to call.
 */
export interface GlueState {
  memory: WebAssembly.Memory;
  exports: WasmExports | null;
  handles: HandleTable;
  /** Diagnostic counter; incremented every time wasm calls a sync FFI. */
  syncCalls: number;
}

interface RenderPipelineDescriptor {
  layout: "auto";
  vertex: any | null;
  fragment: any | null;
  primitive: { topology: string; cullMode: string };
  multisample: { count: number };
  depthStencil: any | null;
  vertexBuffers: { arrayStride: number; stepMode: string; attributes: any[] }[];
  colorTargets: any[];
}

interface BindGroupDescriptor {
  layout: GPUBindGroupLayout;
  entries: any[];
}

interface RenderPassDescriptor {
  colorAttachments: any[];
  depthStencilAttachment: any | null;
}

const COMPARE_UNSET = 0;

export function makeImports(state: GlueState): WebAssembly.ModuleImports {
  // Internal helper — every async import calls this to hand off a JS
  // Promise to the wasm executor. Constructing the closure inline keeps
  // type inference clean.
  function async_<T>(task: number, p: Promise<T>, mapHandle: (v: T) => number): void {
    const e = state.exports;
    if (e === null) {
      throw new Error("quanta glue: async import called before wasm exports were wired");
    }
    bindTask(e, task, p, mapHandle);
  }

  function readString(ptr: number, len: number): string {
    state.syncCalls++;
    return readUtf8(state.memory, ptr, len);
  }

  // u64 sizes cross the FFI as `f64`. JS numbers are exact integers up
  // to 2^53; WebGPU sizes are well below that.
  function size(n: number): number {
    return n;
  }

  return {
    // ── adapter / device acquisition ────────────────────────────────────────

    quanta_request_adapter(task: number): void {
      const gpu = navigator.gpu;
      if (gpu === undefined) {
        async_(task, Promise.resolve(null), () => 0);
        return;
      }
      async_(task, gpu.requestAdapter(), (a) =>
        a === null ? 0 : state.handles.alloc(a),
      );
    },

    quanta_request_device(adapter: number, task: number): void {
      const a = state.handles.get<GPUAdapter>(adapter);
      async_(task, a.requestDevice(), (d) => state.handles.alloc(d));
    },

    // ── buffers ────────────────────────────────────────────────────────────

    quanta_create_buffer(device: number, size_f64: number, usage: number): number {
      const dev = state.handles.get<GPUDevice>(device);
      const buf = dev.createBuffer({ size: size(size_f64), usage });
      return state.handles.alloc(buf);
    },

    quanta_destroy_buffer(buffer: number): void {
      const buf = state.handles.get<GPUBuffer>(buffer);
      buf.destroy();
      state.handles.release(buffer);
    },

    quanta_write_buffer(
      device: number,
      buffer: number,
      offset_f64: number,
      data_ptr: number,
      data_len: number,
    ): void {
      const dev = state.handles.get<GPUDevice>(device);
      const buf = state.handles.get<GPUBuffer>(buffer);
      // viewBytes is a borrowed view; writeBuffer copies synchronously.
      dev.queue.writeBuffer(buf, size(offset_f64), viewBytes(state.memory, data_ptr, data_len));
    },

    quanta_map_async_read(buffer: number, task: number): void {
      const buf = state.handles.get<GPUBuffer>(buffer);
      // GPUMapMode.READ = 0x0001
      async_(task, buf.mapAsync(0x0001), () => 0);
    },

    quanta_get_mapped_range_copy(
      buffer: number,
      dst_ptr: number,
      len: number,
    ): void {
      const buf = state.handles.get<GPUBuffer>(buffer);
      const range = buf.getMappedRange();
      const src = new Uint8Array(range, 0, len);
      const dst = new Uint8Array(state.memory.buffer, dst_ptr, len);
      dst.set(src);
    },

    quanta_unmap_buffer(buffer: number): void {
      const buf = state.handles.get<GPUBuffer>(buffer);
      buf.unmap();
    },

    // ── shader / compute pipeline ──────────────────────────────────────────

    quanta_create_shader_module(
      device: number,
      code_ptr: number,
      code_len: number,
    ): number {
      const dev = state.handles.get<GPUDevice>(device);
      const code = readString(code_ptr, code_len);
      const m = dev.createShaderModule({ code });
      return state.handles.alloc(m);
    },

    quanta_create_compute_pipeline(
      device: number,
      module_h: number,
      entry_ptr: number,
      entry_len: number,
    ): number {
      const dev = state.handles.get<GPUDevice>(device);
      const m = state.handles.get<GPUShaderModule>(module_h);
      const entryPoint = readString(entry_ptr, entry_len);
      const p = dev.createComputePipeline({
        layout: "auto",
        compute: { module: m, entryPoint },
      });
      return state.handles.alloc(p);
    },

    quanta_compute_pipeline_get_bind_group_layout(
      pipeline: number,
      index: number,
    ): number {
      const p = state.handles.get<GPUComputePipeline>(pipeline);
      return state.handles.alloc(p.getBindGroupLayout(index));
    },

    // ── render pipeline (builder pattern) ──────────────────────────────────

    quanta_rp_desc_create(): number {
      const desc: RenderPipelineDescriptor = {
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

    quanta_rp_desc_set_vertex(
      desc_h: number,
      module_h: number,
      entry_ptr: number,
      entry_len: number,
    ): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      const m = state.handles.get<GPUShaderModule>(module_h);
      desc.vertex = { module: m, entryPoint: readString(entry_ptr, entry_len) };
    },

    quanta_rp_desc_add_vertex_buffer(
      desc_h: number,
      stride: number,
      step_mode: number,
    ): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      desc.vertexBuffers.push({
        arrayStride: stride,
        stepMode: stepModeName(step_mode),
        attributes: [],
      });
    },

    quanta_rp_desc_add_vertex_attribute(
      desc_h: number,
      buf_index: number,
      format_code: number,
      offset: number,
      location: number,
    ): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      const buf = desc.vertexBuffers[buf_index];
      if (buf === undefined) {
        throw new Error(
          `quanta glue: vertex attribute on unknown buffer index ${buf_index}`,
        );
      }
      buf.attributes.push({
        format: attributeFormatName(format_code),
        offset,
        shaderLocation: location,
      });
    },

    quanta_rp_desc_set_fragment(
      desc_h: number,
      module_h: number,
      entry_ptr: number,
      entry_len: number,
    ): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      const m = state.handles.get<GPUShaderModule>(module_h);
      desc.fragment = {
        module: m,
        entryPoint: readString(entry_ptr, entry_len),
        targets: desc.colorTargets,
      };
    },

    quanta_rp_desc_add_color_target(
      desc_h: number,
      format_code: number,
      blend_enabled: number,
      src_color: number,
      dst_color: number,
      op_color: number,
      src_alpha: number,
      dst_alpha: number,
      op_alpha: number,
    ): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      const target: any = { format: formatName(format_code) };
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

    quanta_rp_desc_set_primitive(
      desc_h: number,
      topology_code: number,
      cull_mode_code: number,
    ): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      desc.primitive = {
        topology: topologyName(topology_code),
        cullMode: cullModeName(cull_mode_code),
      };
    },

    quanta_rp_desc_set_multisample(desc_h: number, count: number): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      desc.multisample = { count };
    },

    quanta_rp_desc_set_depth_stencil(
      desc_h: number,
      format_code: number,
      depth_write: number,
      compare_code: number,
    ): void {
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);
      desc.depthStencil = {
        format: formatName(format_code),
        depthWriteEnabled: depth_write !== 0,
        depthCompare: compareName(compare_code),
      };
    },

    quanta_create_render_pipeline(device: number, desc_h: number): number {
      const dev = state.handles.get<GPUDevice>(device);
      const desc = state.handles.get<RenderPipelineDescriptor>(desc_h);

      // Stitch vertex buffers into the vertex stage; descriptor is
      // built lazily here to avoid mutating the JS object every time
      // a vertex buffer gets added.
      const vertexStage =
        desc.vertex === null
          ? null
          : { ...desc.vertex, buffers: desc.vertexBuffers };

      const pipelineDesc: any = {
        layout: desc.layout,
        vertex: vertexStage,
        primitive: desc.primitive,
        multisample: desc.multisample,
      };
      if (desc.fragment !== null) pipelineDesc.fragment = desc.fragment;
      if (desc.depthStencil !== null) pipelineDesc.depthStencil = desc.depthStencil;

      const p = dev.createRenderPipeline(pipelineDesc);
      state.handles.release(desc_h);
      return state.handles.alloc(p);
    },

    quanta_render_pipeline_get_bind_group_layout(
      pipeline: number,
      index: number,
    ): number {
      const p = state.handles.get<GPURenderPipeline>(pipeline);
      return state.handles.alloc(p.getBindGroupLayout(index));
    },

    // ── bind group (builder pattern) ───────────────────────────────────────

    quanta_bg_desc_create(layout: number): number {
      const l = state.handles.get<GPUBindGroupLayout>(layout);
      const desc: BindGroupDescriptor = { layout: l, entries: [] };
      return state.handles.alloc(desc);
    },

    quanta_bg_desc_add_buffer(
      desc_h: number,
      binding: number,
      buffer: number,
    ): void {
      const desc = state.handles.get<BindGroupDescriptor>(desc_h);
      const buf = state.handles.get<GPUBuffer>(buffer);
      desc.entries.push({ binding, resource: { buffer: buf } });
    },

    quanta_bg_desc_add_sampler(
      desc_h: number,
      binding: number,
      sampler: number,
    ): void {
      const desc = state.handles.get<BindGroupDescriptor>(desc_h);
      const s = state.handles.get<GPUSampler>(sampler);
      desc.entries.push({ binding, resource: s });
    },

    quanta_bg_desc_add_texture_view(
      desc_h: number,
      binding: number,
      view: number,
    ): void {
      const desc = state.handles.get<BindGroupDescriptor>(desc_h);
      const v = state.handles.get<GPUTextureView>(view);
      desc.entries.push({ binding, resource: v });
    },

    quanta_create_bind_group(device: number, desc_h: number): number {
      const dev = state.handles.get<GPUDevice>(device);
      const desc = state.handles.get<BindGroupDescriptor>(desc_h);
      const bg = dev.createBindGroup(desc);
      state.handles.release(desc_h);
      return state.handles.alloc(bg);
    },

    // ── command encoder ────────────────────────────────────────────────────

    quanta_create_command_encoder(device: number): number {
      const dev = state.handles.get<GPUDevice>(device);
      return state.handles.alloc(dev.createCommandEncoder());
    },

    quanta_encoder_copy_buffer_to_buffer(
      encoder: number,
      src: number,
      src_off: number,
      dst: number,
      dst_off: number,
      n: number,
    ): void {
      const enc = state.handles.get<GPUCommandEncoder>(encoder);
      const s = state.handles.get<GPUBuffer>(src);
      const d = state.handles.get<GPUBuffer>(dst);
      enc.copyBufferToBuffer(s, size(src_off), d, size(dst_off), size(n));
    },

    quanta_encoder_copy_texture_to_buffer(
      encoder: number,
      src_texture: number,
      dst_buffer: number,
      dst_bytes_per_row: number,
      dst_rows_per_image: number,
      width: number,
      height: number,
      depth: number,
    ): void {
      const enc = state.handles.get<GPUCommandEncoder>(encoder);
      const t = state.handles.get<GPUTexture>(src_texture);
      const b = state.handles.get<GPUBuffer>(dst_buffer);
      enc.copyTextureToBuffer(
        { texture: t },
        {
          buffer: b,
          bytesPerRow: dst_bytes_per_row,
          rowsPerImage: dst_rows_per_image,
        },
        { width, height, depthOrArrayLayers: depth },
      );
    },

    quanta_encoder_finish(encoder: number): number {
      const enc = state.handles.get<GPUCommandEncoder>(encoder);
      const cmd = enc.finish();
      state.handles.release(encoder);
      return state.handles.alloc(cmd);
    },

    // ── compute pass ───────────────────────────────────────────────────────

    quanta_encoder_begin_compute_pass(encoder: number): number {
      const enc = state.handles.get<GPUCommandEncoder>(encoder);
      return state.handles.alloc(enc.beginComputePass());
    },

    quanta_compute_pass_set_pipeline(pass: number, pipeline: number): void {
      const cp = state.handles.get<GPUComputePassEncoder>(pass);
      const p = state.handles.get<GPUComputePipeline>(pipeline);
      cp.setPipeline(p);
    },

    quanta_compute_pass_set_bind_group(
      pass: number,
      index: number,
      group: number,
    ): void {
      const cp = state.handles.get<GPUComputePassEncoder>(pass);
      const g = state.handles.get<GPUBindGroup>(group);
      cp.setBindGroup(index, g);
    },

    quanta_compute_pass_dispatch(
      pass: number,
      x: number,
      y: number,
      z: number,
    ): void {
      const cp = state.handles.get<GPUComputePassEncoder>(pass);
      cp.dispatchWorkgroups(x, y, z);
    },

    quanta_compute_pass_end(pass: number): void {
      const cp = state.handles.get<GPUComputePassEncoder>(pass);
      cp.end();
      state.handles.release(pass);
    },

    // ── render pass (descriptor builder + execute) ─────────────────────────

    quanta_rpass_desc_create(): number {
      const desc: RenderPassDescriptor = {
        colorAttachments: [],
        depthStencilAttachment: null,
      };
      return state.handles.alloc(desc);
    },

    quanta_rpass_desc_add_color_attachment(
      desc_h: number,
      view: number,
      load_op: number,
      store_op: number,
      r: number,
      g: number,
      b: number,
      a: number,
    ): void {
      const desc = state.handles.get<RenderPassDescriptor>(desc_h);
      const v = state.handles.get<GPUTextureView>(view);
      desc.colorAttachments.push({
        view: v,
        loadOp: loadOpName(load_op),
        storeOp: storeOpName(store_op),
        clearValue: { r, g, b, a },
      });
    },

    quanta_rpass_desc_set_depth_attachment(
      desc_h: number,
      view: number,
      load_op: number,
      store_op: number,
      clear_depth: number,
    ): void {
      const desc = state.handles.get<RenderPassDescriptor>(desc_h);
      const v = state.handles.get<GPUTextureView>(view);
      desc.depthStencilAttachment = {
        view: v,
        depthLoadOp: loadOpName(load_op),
        depthStoreOp: storeOpName(store_op),
        depthClearValue: clear_depth,
      };
    },

    quanta_encoder_begin_render_pass(encoder: number, desc_h: number): number {
      const enc = state.handles.get<GPUCommandEncoder>(encoder);
      const desc = state.handles.get<RenderPassDescriptor>(desc_h);
      const passDesc: any = { colorAttachments: desc.colorAttachments };
      if (desc.depthStencilAttachment !== null) {
        passDesc.depthStencilAttachment = desc.depthStencilAttachment;
      }
      const rp = enc.beginRenderPass(passDesc);
      state.handles.release(desc_h);
      return state.handles.alloc(rp);
    },

    quanta_render_pass_set_pipeline(pass: number, pipeline: number): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      const p = state.handles.get<GPURenderPipeline>(pipeline);
      rp.setPipeline(p);
    },

    quanta_render_pass_set_bind_group(
      pass: number,
      index: number,
      group: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      const g = state.handles.get<GPUBindGroup>(group);
      rp.setBindGroup(index, g);
    },

    quanta_render_pass_set_vertex_buffer(
      pass: number,
      slot: number,
      buffer: number,
      offset: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      const b = state.handles.get<GPUBuffer>(buffer);
      rp.setVertexBuffer(slot, b, size(offset));
    },

    quanta_render_pass_set_index_buffer(
      pass: number,
      buffer: number,
      format_code: number,
      offset: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      const b = state.handles.get<GPUBuffer>(buffer);
      rp.setIndexBuffer(b, indexFormatName(format_code), size(offset));
    },

    quanta_render_pass_draw(
      pass: number,
      vertex_count: number,
      instance_count: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      rp.draw(vertex_count, instance_count);
    },

    quanta_render_pass_draw_indexed(
      pass: number,
      index_count: number,
      instance_count: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      rp.drawIndexed(index_count, instance_count);
    },

    quanta_render_pass_draw_indirect(
      pass: number,
      indirect_buffer: number,
      indirect_offset: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      const buf = state.handles.get<GPUBuffer>(indirect_buffer);
      rp.drawIndirect(buf, indirect_offset);
    },

    quanta_render_pass_draw_indexed_indirect(
      pass: number,
      indirect_buffer: number,
      indirect_offset: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      const buf = state.handles.get<GPUBuffer>(indirect_buffer);
      rp.drawIndexedIndirect(buf, indirect_offset);
    },

    quanta_render_pass_set_viewport(
      pass: number,
      x: number,
      y: number,
      w: number,
      h: number,
      min_d: number,
      max_d: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      rp.setViewport(x, y, w, h, min_d, max_d);
    },

    quanta_render_pass_set_scissor(
      pass: number,
      x: number,
      y: number,
      w: number,
      h: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      rp.setScissorRect(x, y, w, h);
    },

    quanta_render_pass_set_stencil_reference(pass: number, reference: number): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      rp.setStencilReference(reference);
    },

    quanta_render_pass_end(pass: number): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      rp.end();
      state.handles.release(pass);
    },

    // ── render bundles (steps 032 + 033) ──────────────────────────────────

    quanta_create_render_bundle_encoder(
      device: number,
      color_format_code: number,
      depth_format_code: number,
      sample_count: number,
    ): number {
      const dev = state.handles.get<GPUDevice>(device);
      const desc: GPURenderBundleEncoderDescriptor = {
        colorFormats: [formatName(color_format_code) as GPUTextureFormat],
        sampleCount: sample_count > 0 ? sample_count : 1,
      };
      if (depth_format_code !== 0) {
        desc.depthStencilFormat = formatName(depth_format_code) as GPUTextureFormat;
      }
      const enc = dev.createRenderBundleEncoder(desc);
      return state.handles.alloc(enc);
    },

    quanta_render_bundle_set_pipeline(encoder: number, pipeline: number): void {
      const enc = state.handles.get<GPURenderBundleEncoder>(encoder);
      const p = state.handles.get<GPURenderPipeline>(pipeline);
      enc.setPipeline(p);
    },

    quanta_render_bundle_set_bind_group(
      encoder: number,
      index: number,
      group: number,
    ): void {
      const enc = state.handles.get<GPURenderBundleEncoder>(encoder);
      const g = state.handles.get<GPUBindGroup>(group);
      enc.setBindGroup(index, g);
    },

    quanta_render_bundle_set_vertex_buffer(
      encoder: number,
      slot: number,
      buffer: number,
      offset: number,
    ): void {
      const enc = state.handles.get<GPURenderBundleEncoder>(encoder);
      const b = state.handles.get<GPUBuffer>(buffer);
      enc.setVertexBuffer(slot, b, size(offset));
    },

    quanta_render_bundle_draw(
      encoder: number,
      vertex_count: number,
      instance_count: number,
    ): void {
      const enc = state.handles.get<GPURenderBundleEncoder>(encoder);
      enc.draw(vertex_count, instance_count);
    },

    quanta_render_bundle_finish(encoder: number): number {
      const enc = state.handles.get<GPURenderBundleEncoder>(encoder);
      const bundle = enc.finish();
      state.handles.release(encoder);
      return state.handles.alloc(bundle);
    },

    quanta_render_pass_execute_bundles(
      pass: number,
      bundles_ptr: number,
      count: number,
    ): void {
      const rp = state.handles.get<GPURenderPassEncoder>(pass);
      const view = new Uint32Array(state.memory.buffer, bundles_ptr, count);
      const arr: GPURenderBundle[] = [];
      for (let i = 0; i < count; i++) {
        arr.push(state.handles.get<GPURenderBundle>(view[i]!));
      }
      rp.executeBundles(arr);
    },

    // ── queue ──────────────────────────────────────────────────────────────

    quanta_queue_submit(device: number, command_buffer: number): void {
      const dev = state.handles.get<GPUDevice>(device);
      const cb = state.handles.get<GPUCommandBuffer>(command_buffer);
      dev.queue.submit([cb]);
      state.handles.release(command_buffer);
    },

    quanta_queue_on_submitted_work_done(device: number, task: number): void {
      const dev = state.handles.get<GPUDevice>(device);
      async_(task, dev.queue.onSubmittedWorkDone(), () => 0);
    },

    // ── textures / samplers ────────────────────────────────────────────────

    quanta_create_texture(
      device: number,
      width: number,
      height: number,
      depth_or_array_layers: number,
      mip_level_count: number,
      sample_count: number,
      format_code: number,
      usage: number,
    ): number {
      const dev = state.handles.get<GPUDevice>(device);
      const tex = dev.createTexture({
        size: { width, height, depthOrArrayLayers: depth_or_array_layers },
        mipLevelCount: mip_level_count,
        sampleCount: sample_count,
        format: formatName(format_code),
        usage,
      });
      return state.handles.alloc(tex);
    },

    quanta_texture_create_view(texture: number): number {
      const t = state.handles.get<GPUTexture>(texture);
      return state.handles.alloc(t.createView());
    },

    quanta_destroy_texture(texture: number): void {
      const t = state.handles.get<GPUTexture>(texture);
      t.destroy();
      state.handles.release(texture);
    },

    quanta_queue_write_texture(
      device: number,
      texture: number,
      data_ptr: number,
      data_len: number,
      bytes_per_row: number,
      rows_per_image: number,
      width: number,
      height: number,
      depth: number,
    ): void {
      const dev = state.handles.get<GPUDevice>(device);
      const t = state.handles.get<GPUTexture>(texture);
      dev.queue.writeTexture(
        { texture: t },
        viewBytes(state.memory, data_ptr, data_len),
        { offset: 0, bytesPerRow: bytes_per_row, rowsPerImage: rows_per_image },
        { width, height, depthOrArrayLayers: depth },
      );
    },

    quanta_create_sampler(
      device: number,
      mag_filter: number,
      min_filter: number,
      mipmap_filter: number,
      address_u: number,
      address_v: number,
      address_w: number,
      max_anisotropy: number,
      compare_code: number,
    ): number {
      const dev = state.handles.get<GPUDevice>(device);
      const desc: any = {
        magFilter: filterName(mag_filter),
        minFilter: filterName(min_filter),
        mipmapFilter: filterName(mipmap_filter),
        addressModeU: addressName(address_u),
        addressModeV: addressName(address_v),
        addressModeW: addressName(address_w),
      };
      if (max_anisotropy > 1) desc.maxAnisotropy = max_anisotropy;
      if (compare_code !== COMPARE_UNSET) desc.compare = compareName(compare_code);
      const s = dev.createSampler(desc);
      return state.handles.alloc(s);
    },

    // ── universal handle release (for handles without a destroy method) ────

    quanta_release(handle: number): void {
      state.handles.release(handle);
    },

    // ── debug ──────────────────────────────────────────────────────────────

    quanta_console_error(ptr: number, len: number): void {
      console.error(readUtf8(state.memory, ptr, len));
    },
  };
}
