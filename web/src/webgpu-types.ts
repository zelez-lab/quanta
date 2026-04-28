// Minimal WebGPU type declarations.
//
// Quanta deliberately does NOT depend on `@webgpu/types`. This file
// declares exactly the surface our `glue.ts` touches — about 50 lines of
// hand-written types we own and audit, instead of the ~10K lines that
// ship with the npm package. Under B″ this file will be replaced with
// types generated from the W3C `webgpu.idl`; for B⁰ we hand-author it.
//
// All types are loose: methods accept `any` for descriptor objects.
// This is intentional — the descriptors are built by JS-side `Object`
// literals at the call site, and adding precise types would require
// re-deriving the entire WebIDL hierarchy. The Rust side carries the
// strict typing on the IDL fields.
//
// We use `declare global` so the WebGPU types are available without an
// `import` in every consumer module, matching the way `lib.dom.d.ts`
// surfaces standard browser APIs.

declare global {
  interface Navigator {
    readonly gpu: GPU;
  }

  interface GPU {
    requestAdapter(): Promise<GPUAdapter | null>;
  }

  interface GPUAdapter {
    requestDevice(): Promise<GPUDevice>;
  }

  interface GPUDevice {
    readonly queue: GPUQueue;
    createBuffer(desc: any): GPUBuffer;
    createTexture(desc: any): GPUTexture;
    createSampler(desc: any): GPUSampler;
    createShaderModule(desc: any): GPUShaderModule;
    createComputePipeline(desc: any): GPUComputePipeline;
    createRenderPipeline(desc: any): GPURenderPipeline;
    createBindGroup(desc: any): GPUBindGroup;
    createCommandEncoder(): GPUCommandEncoder;
  }

  interface GPUQueue {
    submit(buffers: GPUCommandBuffer[]): void;
    // `data` accepts any TypedArray view; we relax the type because the
    // wasm linear memory's buffer is `ArrayBufferLike` (could also be
    // `SharedArrayBuffer`) and `BufferSource` from lib.dom narrows to
    // plain `ArrayBuffer` only.
    writeBuffer(buf: GPUBuffer, off: number, data: ArrayBufferView): void;
    writeTexture(dst: any, data: ArrayBufferView, layout: any, size: any): void;
    onSubmittedWorkDone(): Promise<undefined>;
  }

  interface GPUBuffer {
    mapAsync(mode: number): Promise<undefined>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  interface GPUTexture {
    createView(): GPUTextureView;
    destroy(): void;
  }

  interface GPUTextureView {}
  interface GPUSampler {}
  interface GPUShaderModule {}

  interface GPUComputePipeline {
    getBindGroupLayout(idx: number): GPUBindGroupLayout;
  }

  interface GPURenderPipeline {
    getBindGroupLayout(idx: number): GPUBindGroupLayout;
  }

  interface GPUBindGroupLayout {}
  interface GPUBindGroup {}
  interface GPUCommandBuffer {}

  interface GPUCommandEncoder {
    beginComputePass(): GPUComputePassEncoder;
    beginRenderPass(desc: any): GPURenderPassEncoder;
    copyBufferToBuffer(src: GPUBuffer, srcOff: number, dst: GPUBuffer, dstOff: number, size: number): void;
    copyTextureToBuffer(src: any, dst: any, size: any): void;
    finish(): GPUCommandBuffer;
  }

  interface GPUComputePassEncoder {
    setPipeline(p: GPUComputePipeline): void;
    setBindGroup(idx: number, group: GPUBindGroup): void;
    dispatchWorkgroups(x: number, y: number, z: number): void;
    end(): void;
  }

  interface GPURenderPassEncoder {
    setPipeline(p: GPURenderPipeline): void;
    setBindGroup(idx: number, group: GPUBindGroup): void;
    setVertexBuffer(slot: number, buf: GPUBuffer, offset: number): void;
    setIndexBuffer(buf: GPUBuffer, format: string, offset: number): void;
    draw(vertexCount: number, instanceCount: number): void;
    drawIndexed(indexCount: number, instanceCount: number): void;
    setViewport(x: number, y: number, w: number, h: number, minD: number, maxD: number): void;
    setScissorRect(x: number, y: number, w: number, h: number): void;
    end(): void;
  }
}

export {};
