// The command tape — the JS half of the zero-import boundary (dija R10).
//
// The wasm module imports NOTHING. Instead of calling out through
// `env:*`, the driver appends commands to a tape in its linear memory
// (`src/driver/webgpu/tape.rs`); this module DRAINS that tape after
// every wasm entry returns and performs the WebGPU calls. Async results
// still travel back through the exported setters (`quanta_resolve` /
// `quanta_reject`), and the sync queries the driver used to make
// (`navigator.gpu`, canvas extents, preferred format) are PUSHED into
// wasm instead (`__quanta_env_init` / `__quanta_canvas_dims`).
//
// Wire format (version 1 — asserted byte-for-byte by the conformance
// harness): little-endian `u32` words, `[MAGIC][VERSION]` then a
// sequence of `[opcode][args…]`. An `f64` arg is two words and an `f32`
// arg is one (their IEEE-754 bit patterns, so both read straight out of
// the DataView). Byte payloads are inline after the fixed words:
// `[len][bytes… padded to a word]`.
//
// Every arm below is the body of the `env` import it replaces — the
// handle table, the code tables and the task plumbing are unchanged.
// Ids with the high bit set were minted by the driver (`tape::mint_id`)
// and are bound here; ids below it were minted by `registerCanvas`.
// One table serves both, so no arm special-cases the difference.
import { decodeUtf8 } from "./strings.js";
import { bindTask } from "./tasks.js";
import { COMPARE_UNSET } from "./webgpu.js";
import { formatName, attributeFormatName, topologyName, cullModeName, blendFactorName, blendOpName, filterName, addressName, compareName, stepModeName, indexFormatName, loadOpName, storeOpName, } from "./codes.js";
import "./webgpu-types.js";
/** `"QTAP"` little-endian — the first word of every non-empty tape. */
const TAPE_MAGIC = 0x50415451;
const TAPE_VERSION = 1;
/** `GPUMapMode.READ`. */
const MAP_MODE_READ = 0x0001;
/**
 * Opcodes — mirrors the `Op` enum in `src/driver/webgpu/tape.rs` value
 * for value. Never renumber: append and bump `TAPE_VERSION` instead.
 */
const OP = {
    RequestAdapter: 0x01,
    RequestDevice: 0x02,
    CreateBuffer: 0x10,
    DestroyBuffer: 0x11,
    WriteBuffer: 0x12,
    MapAsyncRead: 0x13,
    UnmapBuffer: 0x14,
    CreateShaderModule: 0x20,
    CreateComputePipeline: 0x21,
    ComputePipelineGetBindGroupLayout: 0x22,
    RpDescCreate: 0x30,
    RpDescSetVertex: 0x31,
    RpDescAddVertexBuffer: 0x32,
    RpDescAddVertexAttribute: 0x33,
    RpDescSetFragment: 0x34,
    RpDescAddColorTarget: 0x35,
    RpDescSetPrimitive: 0x36,
    RpDescSetMultisample: 0x37,
    RpDescSetDepthStencil: 0x38,
    CreateRenderPipeline: 0x39,
    RenderPipelineGetBindGroupLayout: 0x3a,
    BgDescCreate: 0x40,
    BgDescAddBuffer: 0x41,
    BgDescAddSampler: 0x42,
    BgDescAddTextureView: 0x43,
    CreateBindGroup: 0x44,
    CreateCommandEncoder: 0x50,
    EncoderCopyBufferToBuffer: 0x51,
    EncoderCopyTextureToBuffer: 0x52,
    EncoderFinish: 0x53,
    EncoderBeginComputePass: 0x60,
    ComputePassSetPipeline: 0x61,
    ComputePassSetBindGroup: 0x62,
    ComputePassDispatch: 0x63,
    ComputePassEnd: 0x64,
    RpassDescCreate: 0x70,
    RpassDescAddColorAttachment: 0x71,
    RpassDescSetDepthAttachment: 0x72,
    EncoderBeginRenderPass: 0x73,
    RenderPassSetPipeline: 0x74,
    RenderPassSetBindGroup: 0x75,
    RenderPassSetVertexBuffer: 0x76,
    RenderPassSetIndexBuffer: 0x77,
    RenderPassDraw: 0x78,
    RenderPassDrawIndexed: 0x79,
    RenderPassDrawIndirect: 0x7a,
    RenderPassDrawIndexedIndirect: 0x7b,
    RenderPassSetViewport: 0x7c,
    RenderPassSetScissor: 0x7d,
    RenderPassSetStencilReference: 0x7e,
    RenderPassEnd: 0x7f,
    CreateQuerySet: 0x90,
    RpassDescSetOcclusionQuerySet: 0x91,
    RenderPassBeginOcclusionQuery: 0x92,
    RenderPassEndOcclusionQuery: 0x93,
    EncoderResolveQuerySet: 0x94,
    CreateRenderBundleEncoder: 0xa0,
    RenderBundleSetPipeline: 0xa1,
    RenderBundleSetBindGroup: 0xa2,
    RenderBundleSetVertexBuffer: 0xa3,
    RenderBundleDraw: 0xa4,
    RenderBundleFinish: 0xa5,
    RenderPassExecuteBundles: 0xa6,
    QueueSubmit: 0xb0,
    QueueOnSubmittedWorkDone: 0xb1,
    CreateTexture: 0xc0,
    TextureCreateView: 0xc1,
    DestroyTexture: 0xc2,
    QueueWriteTexture: 0xc3,
    CreateSampler: 0xc4,
    CanvasCreateOffscreen: 0xd0,
    CanvasContextCreate: 0xd1,
    CanvasContextConfigure: 0xd2,
    CanvasContextUnconfigure: 0xd3,
    CanvasGetCurrentTexture: 0xd4,
    CompleteBytes: 0xe0,
    CompleteErr: 0xe1,
    Release: 0xf0,
    ConsoleError: 0xf1,
};
/** Cursor over one drained tape. All reads are little-endian. */
class Reader {
    view;
    bytes;
    pos = 0;
    constructor(bytes) {
        this.bytes = bytes;
        this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    }
    done() {
        return this.pos >= this.bytes.byteLength;
    }
    u32() {
        const v = this.view.getUint32(this.pos, true);
        this.pos += 4;
        return v;
    }
    /** A `u64` size/offset: two words holding the `f64` bit pattern. */
    f64() {
        const v = this.view.getFloat64(this.pos, true);
        this.pos += 8;
        return v;
    }
    f32() {
        const v = this.view.getFloat32(this.pos, true);
        this.pos += 4;
        return v;
    }
    /**
     * An inline `[len][bytes][pad]` payload, borrowed from the drained
     * copy — valid for the rest of the drain, and independent of wasm
     * memory growth. Copy it (`.slice()`) to hand it to a caller.
     */
    payload() {
        const len = this.u32();
        const start = this.bytes.byteOffset + this.pos;
        this.pos += (len + 3) & ~3;
        return new Uint8Array(this.bytes.buffer, start, len);
    }
    text() {
        return decodeUtf8(this.payload());
    }
}
function requireExports(state) {
    const e = state.exports;
    if (e === null) {
        throw new Error("quanta glue: tape drained before wasm exports were wired");
    }
    return e;
}
/**
 * Hand a JS Promise to the wasm executor. Resolution re-enters wasm, so
 * it goes through `state.enter` — the continuation appends to the tape
 * and that tape needs draining too.
 */
function async_(state, task, p, mapHandle) {
    bindTask(requireExports(state), task, p, mapHandle, state.enter);
}
/**
 * Perform every command the wasm side queued, then clear the tape.
 *
 * Called after each wasm entry returns (see `GlueState.enter`). Throws
 * on a malformed tape or a bad handle — the same loud failure the
 * direct imports gave — but always clears first, so a half-executed
 * tape is never replayed.
 */
export function drainTape(state) {
    const e = requireExports(state);
    const len = e.__quanta_tape_len();
    if (len === 0)
        return;
    const ptr = e.__quanta_tape_ptr();
    // Copy before interpreting: an arm may re-enter wasm (pushing canvas
    // dims), and a memory that grows detaches every live view of it.
    const bytes = new Uint8Array(state.memory.buffer, ptr, len).slice();
    try {
        interpret(state, new Reader(bytes));
    }
    finally {
        e.__quanta_tape_clear();
        state.drains++;
    }
}
function interpret(state, r) {
    const magic = r.u32();
    if (magic !== TAPE_MAGIC) {
        throw new Error(`quanta glue: tape magic ${magic.toString(16)} is not QTAP`);
    }
    const version = r.u32();
    if (version !== TAPE_VERSION) {
        throw new Error(`quanta glue: tape version ${version}, this glue speaks ${TAPE_VERSION}`);
    }
    const handles = state.handles;
    while (!r.done()) {
        const op = r.u32();
        switch (op) {
            // ── adapter / device acquisition ────────────────────────────────────
            case OP.RequestAdapter: {
                const task = r.u32();
                const gpu = navigator.gpu;
                if (gpu === undefined) {
                    async_(state, task, Promise.resolve(null), () => 0);
                    break;
                }
                async_(state, task, gpu.requestAdapter(), (a) => a === null ? 0 : handles.alloc(a));
                break;
            }
            case OP.RequestDevice: {
                const adapter = r.u32();
                const task = r.u32();
                const a = handles.get(adapter);
                async_(state, task, a.requestDevice(), (d) => handles.alloc(d));
                break;
            }
            // ── buffers ────────────────────────────────────────────────────────
            case OP.CreateBuffer: {
                const dst = r.u32();
                const device = r.u32();
                const size = r.f64();
                const usage = r.u32();
                const dev = handles.get(device);
                handles.bind(dst, dev.createBuffer({ size, usage }));
                break;
            }
            case OP.DestroyBuffer: {
                const buffer = r.u32();
                handles.get(buffer).destroy();
                handles.release(buffer);
                break;
            }
            case OP.WriteBuffer: {
                const device = r.u32();
                const buffer = r.u32();
                const offset = r.f64();
                const data = r.payload();
                const dev = handles.get(device);
                const buf = handles.get(buffer);
                // `data` is a borrowed view of the drained tape; writeBuffer
                // copies synchronously.
                dev.queue.writeBuffer(buf, offset, data);
                break;
            }
            case OP.MapAsyncRead: {
                const buffer = r.u32();
                const task = r.u32();
                const dstPtr = r.u32();
                const dstLen = r.u32();
                const buf = handles.get(buffer);
                const e = requireExports(state);
                buf.mapAsync(MAP_MODE_READ).then(() => {
                    const src = new Uint8Array(buf.getMappedRange(), 0, dstLen);
                    // Re-view: memory may have grown while the map was in flight.
                    new Uint8Array(state.memory.buffer, dstPtr, dstLen).set(src);
                    buf.unmap();
                    // Only now is the destination readable — resolve last.
                    state.enter(() => e.quanta_resolve(task, 0));
                }, (err) => {
                    console.error("quanta glue: mapAsync rejected", err);
                    state.enter(() => e.quanta_reject(task));
                });
                break;
            }
            case OP.UnmapBuffer: {
                const buffer = r.u32();
                handles.get(buffer).unmap();
                break;
            }
            // ── shader / compute pipeline ──────────────────────────────────────
            case OP.CreateShaderModule: {
                const dst = r.u32();
                const device = r.u32();
                const code = r.text();
                const dev = handles.get(device);
                handles.bind(dst, dev.createShaderModule({ code }));
                break;
            }
            case OP.CreateComputePipeline: {
                const dst = r.u32();
                const device = r.u32();
                const module_h = r.u32();
                const entryPoint = r.text();
                const dev = handles.get(device);
                const m = handles.get(module_h);
                handles.bind(dst, dev.createComputePipeline({
                    layout: "auto",
                    compute: { module: m, entryPoint },
                }));
                break;
            }
            case OP.ComputePipelineGetBindGroupLayout: {
                const dst = r.u32();
                const pipeline = r.u32();
                const index = r.u32();
                const p = handles.get(pipeline);
                handles.bind(dst, p.getBindGroupLayout(index));
                break;
            }
            // ── render pipeline (builder pattern) ──────────────────────────────
            case OP.RpDescCreate: {
                const dst = r.u32();
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
                handles.bind(dst, desc);
                break;
            }
            case OP.RpDescSetVertex: {
                const desc_h = r.u32();
                const module_h = r.u32();
                const entryPoint = r.text();
                const desc = handles.get(desc_h);
                const m = handles.get(module_h);
                desc.vertex = { module: m, entryPoint };
                break;
            }
            case OP.RpDescAddVertexBuffer: {
                const desc_h = r.u32();
                const stride = r.u32();
                const stepMode = r.u32();
                const desc = handles.get(desc_h);
                desc.vertexBuffers.push({
                    arrayStride: stride,
                    stepMode: stepModeName(stepMode),
                    attributes: [],
                });
                break;
            }
            case OP.RpDescAddVertexAttribute: {
                const desc_h = r.u32();
                const bufIndex = r.u32();
                const formatCode = r.u32();
                const offset = r.u32();
                const location = r.u32();
                const desc = handles.get(desc_h);
                const buf = desc.vertexBuffers[bufIndex];
                if (buf === undefined) {
                    throw new Error(`quanta glue: vertex attribute on unknown buffer index ${bufIndex}`);
                }
                buf.attributes.push({
                    format: attributeFormatName(formatCode),
                    offset,
                    shaderLocation: location,
                });
                break;
            }
            case OP.RpDescSetFragment: {
                const desc_h = r.u32();
                const module_h = r.u32();
                const entryPoint = r.text();
                const desc = handles.get(desc_h);
                const m = handles.get(module_h);
                desc.fragment = { module: m, entryPoint, targets: desc.colorTargets };
                break;
            }
            case OP.RpDescAddColorTarget: {
                const desc_h = r.u32();
                const formatCode = r.u32();
                const blendEnabled = r.u32();
                const srcColor = r.u32();
                const dstColor = r.u32();
                const opColor = r.u32();
                const srcAlpha = r.u32();
                const dstAlpha = r.u32();
                const opAlpha = r.u32();
                const desc = handles.get(desc_h);
                const target = { format: formatName(formatCode) };
                if (blendEnabled !== 0) {
                    target.blend = {
                        color: {
                            srcFactor: blendFactorName(srcColor),
                            dstFactor: blendFactorName(dstColor),
                            operation: blendOpName(opColor),
                        },
                        alpha: {
                            srcFactor: blendFactorName(srcAlpha),
                            dstFactor: blendFactorName(dstAlpha),
                            operation: blendOpName(opAlpha),
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
                break;
            }
            case OP.RpDescSetPrimitive: {
                const desc_h = r.u32();
                const topologyCode = r.u32();
                const cullModeCode = r.u32();
                const desc = handles.get(desc_h);
                desc.primitive = {
                    topology: topologyName(topologyCode),
                    cullMode: cullModeName(cullModeCode),
                };
                break;
            }
            case OP.RpDescSetMultisample: {
                const desc_h = r.u32();
                const count = r.u32();
                handles.get(desc_h).multisample = { count };
                break;
            }
            case OP.RpDescSetDepthStencil: {
                const desc_h = r.u32();
                const formatCode = r.u32();
                const depthWrite = r.u32();
                const compareCode = r.u32();
                const desc = handles.get(desc_h);
                desc.depthStencil = {
                    format: formatName(formatCode),
                    depthWriteEnabled: depthWrite !== 0,
                    depthCompare: compareName(compareCode),
                };
                break;
            }
            case OP.CreateRenderPipeline: {
                const dst = r.u32();
                const device = r.u32();
                const desc_h = r.u32();
                const dev = handles.get(device);
                const desc = handles.get(desc_h);
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
                handles.bind(dst, dev.createRenderPipeline(pipelineDesc));
                handles.release(desc_h);
                break;
            }
            case OP.RenderPipelineGetBindGroupLayout: {
                const dst = r.u32();
                const pipeline = r.u32();
                const index = r.u32();
                const p = handles.get(pipeline);
                handles.bind(dst, p.getBindGroupLayout(index));
                break;
            }
            // ── bind group (builder pattern) ───────────────────────────────────
            case OP.BgDescCreate: {
                const dst = r.u32();
                const layout = r.u32();
                const l = handles.get(layout);
                const desc = { layout: l, entries: [] };
                handles.bind(dst, desc);
                break;
            }
            case OP.BgDescAddBuffer: {
                const desc_h = r.u32();
                const binding = r.u32();
                const buffer = r.u32();
                const desc = handles.get(desc_h);
                const buf = handles.get(buffer);
                desc.entries.push({ binding, resource: { buffer: buf } });
                break;
            }
            case OP.BgDescAddSampler: {
                const desc_h = r.u32();
                const binding = r.u32();
                const sampler = r.u32();
                const desc = handles.get(desc_h);
                desc.entries.push({ binding, resource: handles.get(sampler) });
                break;
            }
            case OP.BgDescAddTextureView: {
                const desc_h = r.u32();
                const binding = r.u32();
                const view = r.u32();
                const desc = handles.get(desc_h);
                desc.entries.push({ binding, resource: handles.get(view) });
                break;
            }
            case OP.CreateBindGroup: {
                const dst = r.u32();
                const device = r.u32();
                const desc_h = r.u32();
                const dev = handles.get(device);
                const desc = handles.get(desc_h);
                handles.bind(dst, dev.createBindGroup(desc));
                handles.release(desc_h);
                break;
            }
            // ── command encoder ────────────────────────────────────────────────
            case OP.CreateCommandEncoder: {
                const dst = r.u32();
                const device = r.u32();
                handles.bind(dst, handles.get(device).createCommandEncoder());
                break;
            }
            case OP.EncoderCopyBufferToBuffer: {
                const encoder = r.u32();
                const src = r.u32();
                const srcOff = r.f64();
                const dst = r.u32();
                const dstOff = r.f64();
                const size = r.f64();
                const enc = handles.get(encoder);
                const s = handles.get(src);
                const d = handles.get(dst);
                enc.copyBufferToBuffer(s, srcOff, d, dstOff, size);
                break;
            }
            case OP.EncoderCopyTextureToBuffer: {
                const encoder = r.u32();
                const srcTexture = r.u32();
                const dstBuffer = r.u32();
                const bytesPerRow = r.u32();
                const rowsPerImage = r.u32();
                const width = r.u32();
                const height = r.u32();
                const depth = r.u32();
                const enc = handles.get(encoder);
                const t = handles.get(srcTexture);
                const b = handles.get(dstBuffer);
                enc.copyTextureToBuffer({ texture: t }, { buffer: b, bytesPerRow, rowsPerImage }, { width, height, depthOrArrayLayers: depth });
                break;
            }
            case OP.EncoderFinish: {
                const dst = r.u32();
                const encoder = r.u32();
                const enc = handles.get(encoder);
                handles.bind(dst, enc.finish());
                handles.release(encoder);
                break;
            }
            // ── compute pass ───────────────────────────────────────────────────
            case OP.EncoderBeginComputePass: {
                const dst = r.u32();
                const encoder = r.u32();
                const enc = handles.get(encoder);
                handles.bind(dst, enc.beginComputePass());
                break;
            }
            case OP.ComputePassSetPipeline: {
                const pass = r.u32();
                const pipeline = r.u32();
                const cp = handles.get(pass);
                cp.setPipeline(handles.get(pipeline));
                break;
            }
            case OP.ComputePassSetBindGroup: {
                const pass = r.u32();
                const index = r.u32();
                const group = r.u32();
                const cp = handles.get(pass);
                cp.setBindGroup(index, handles.get(group));
                break;
            }
            case OP.ComputePassDispatch: {
                const pass = r.u32();
                const x = r.u32();
                const y = r.u32();
                const z = r.u32();
                handles.get(pass).dispatchWorkgroups(x, y, z);
                break;
            }
            case OP.ComputePassEnd: {
                const pass = r.u32();
                handles.get(pass).end();
                handles.release(pass);
                break;
            }
            // ── render pass (descriptor builder + execute) ─────────────────────
            case OP.RpassDescCreate: {
                const dst = r.u32();
                const desc = {
                    colorAttachments: [],
                    depthStencilAttachment: null,
                };
                handles.bind(dst, desc);
                break;
            }
            case OP.RpassDescAddColorAttachment: {
                const desc_h = r.u32();
                const view = r.u32();
                const loadOp = r.u32();
                const storeOp = r.u32();
                const resolveView = r.u32();
                const red = r.f32();
                const green = r.f32();
                const blue = r.f32();
                const alpha = r.f32();
                const desc = handles.get(desc_h);
                const att = {
                    view: handles.get(view),
                    loadOp: loadOpName(loadOp),
                    storeOp: storeOpName(storeOp),
                    clearValue: { r: red, g: green, b: blue, a: alpha },
                };
                if (resolveView !== 0) {
                    att.resolveTarget = handles.get(resolveView);
                }
                desc.colorAttachments.push(att);
                break;
            }
            case OP.RpassDescSetDepthAttachment: {
                const desc_h = r.u32();
                const view = r.u32();
                const loadOp = r.u32();
                const storeOp = r.u32();
                const clearDepth = r.f32();
                const desc = handles.get(desc_h);
                desc.depthStencilAttachment = {
                    view: handles.get(view),
                    depthLoadOp: loadOpName(loadOp),
                    depthStoreOp: storeOpName(storeOp),
                    depthClearValue: clearDepth,
                };
                break;
            }
            case OP.EncoderBeginRenderPass: {
                const dst = r.u32();
                const encoder = r.u32();
                const desc_h = r.u32();
                const enc = handles.get(encoder);
                const desc = handles.get(desc_h);
                const passDesc = { colorAttachments: desc.colorAttachments };
                if (desc.depthStencilAttachment !== null) {
                    passDesc.depthStencilAttachment = desc.depthStencilAttachment;
                }
                if (desc.occlusionQuerySet !== undefined) {
                    passDesc.occlusionQuerySet = desc.occlusionQuerySet;
                }
                handles.bind(dst, enc.beginRenderPass(passDesc));
                handles.release(desc_h);
                break;
            }
            case OP.RenderPassSetPipeline: {
                const pass = r.u32();
                const pipeline = r.u32();
                const rp = handles.get(pass);
                rp.setPipeline(handles.get(pipeline));
                break;
            }
            case OP.RenderPassSetBindGroup: {
                const pass = r.u32();
                const index = r.u32();
                const group = r.u32();
                const rp = handles.get(pass);
                rp.setBindGroup(index, handles.get(group));
                break;
            }
            case OP.RenderPassSetVertexBuffer: {
                const pass = r.u32();
                const slot = r.u32();
                const buffer = r.u32();
                const offset = r.f64();
                const rp = handles.get(pass);
                rp.setVertexBuffer(slot, handles.get(buffer), offset);
                break;
            }
            case OP.RenderPassSetIndexBuffer: {
                const pass = r.u32();
                const buffer = r.u32();
                const formatCode = r.u32();
                const offset = r.f64();
                const rp = handles.get(pass);
                rp.setIndexBuffer(handles.get(buffer), indexFormatName(formatCode), offset);
                break;
            }
            case OP.RenderPassDraw: {
                const pass = r.u32();
                const vertexCount = r.u32();
                const instanceCount = r.u32();
                handles.get(pass).draw(vertexCount, instanceCount);
                break;
            }
            case OP.RenderPassDrawIndexed: {
                const pass = r.u32();
                const indexCount = r.u32();
                const instanceCount = r.u32();
                handles
                    .get(pass)
                    .drawIndexed(indexCount, instanceCount);
                break;
            }
            case OP.RenderPassDrawIndirect: {
                const pass = r.u32();
                const indirectBuffer = r.u32();
                const indirectOffset = r.f64();
                const rp = handles.get(pass);
                rp.drawIndirect(handles.get(indirectBuffer), indirectOffset);
                break;
            }
            case OP.RenderPassDrawIndexedIndirect: {
                const pass = r.u32();
                const indirectBuffer = r.u32();
                const indirectOffset = r.f64();
                const rp = handles.get(pass);
                rp.drawIndexedIndirect(handles.get(indirectBuffer), indirectOffset);
                break;
            }
            case OP.RenderPassSetViewport: {
                const pass = r.u32();
                const x = r.f32();
                const y = r.f32();
                const w = r.f32();
                const h = r.f32();
                const minDepth = r.f32();
                const maxDepth = r.f32();
                const rp = handles.get(pass);
                rp.setViewport(x, y, w, h, minDepth, maxDepth);
                break;
            }
            case OP.RenderPassSetScissor: {
                const pass = r.u32();
                const x = r.u32();
                const y = r.u32();
                const w = r.u32();
                const h = r.u32();
                handles.get(pass).setScissorRect(x, y, w, h);
                break;
            }
            case OP.RenderPassSetStencilReference: {
                const pass = r.u32();
                const reference = r.u32();
                handles.get(pass).setStencilReference(reference);
                break;
            }
            case OP.RenderPassEnd: {
                const pass = r.u32();
                handles.get(pass).end();
                handles.release(pass);
                break;
            }
            // ── occlusion queries (post-step-063 closure) ──────────────────────
            case OP.CreateQuerySet: {
                const dst = r.u32();
                const device = r.u32();
                const count = r.u32();
                const dev = handles.get(device);
                handles.bind(dst, dev.createQuerySet({ type: "occlusion", count }));
                break;
            }
            case OP.RpassDescSetOcclusionQuerySet: {
                const desc_h = r.u32();
                const querySet = r.u32();
                const desc = handles.get(desc_h);
                desc.occlusionQuerySet = handles.get(querySet);
                break;
            }
            case OP.RenderPassBeginOcclusionQuery: {
                const pass = r.u32();
                const index = r.u32();
                handles.get(pass).beginOcclusionQuery(index);
                break;
            }
            case OP.RenderPassEndOcclusionQuery: {
                const pass = r.u32();
                handles.get(pass).endOcclusionQuery();
                break;
            }
            case OP.EncoderResolveQuerySet: {
                const encoder = r.u32();
                const querySet = r.u32();
                const firstQuery = r.u32();
                const queryCount = r.u32();
                const dstBuffer = r.u32();
                const dstOffset = r.f64();
                const enc = handles.get(encoder);
                const qs = handles.get(querySet);
                const dst = handles.get(dstBuffer);
                enc.resolveQuerySet(qs, firstQuery, queryCount, dst, dstOffset);
                break;
            }
            // ── render bundles (steps 032 + 033) ──────────────────────────────
            case OP.CreateRenderBundleEncoder: {
                const dst = r.u32();
                const device = r.u32();
                const colorFormatCode = r.u32();
                const depthFormatCode = r.u32();
                const sampleCount = r.u32();
                const dev = handles.get(device);
                const desc = {
                    colorFormats: [formatName(colorFormatCode)],
                    sampleCount: sampleCount > 0 ? sampleCount : 1,
                };
                if (depthFormatCode !== 0) {
                    desc.depthStencilFormat = formatName(depthFormatCode);
                }
                handles.bind(dst, dev.createRenderBundleEncoder(desc));
                break;
            }
            case OP.RenderBundleSetPipeline: {
                const encoder = r.u32();
                const pipeline = r.u32();
                const enc = handles.get(encoder);
                enc.setPipeline(handles.get(pipeline));
                break;
            }
            case OP.RenderBundleSetBindGroup: {
                const encoder = r.u32();
                const index = r.u32();
                const group = r.u32();
                const enc = handles.get(encoder);
                enc.setBindGroup(index, handles.get(group));
                break;
            }
            case OP.RenderBundleSetVertexBuffer: {
                const encoder = r.u32();
                const slot = r.u32();
                const buffer = r.u32();
                const offset = r.f64();
                const enc = handles.get(encoder);
                enc.setVertexBuffer(slot, handles.get(buffer), offset);
                break;
            }
            case OP.RenderBundleDraw: {
                const encoder = r.u32();
                const vertexCount = r.u32();
                const instanceCount = r.u32();
                const enc = handles.get(encoder);
                enc.draw(vertexCount, instanceCount);
                break;
            }
            case OP.RenderBundleFinish: {
                const dst = r.u32();
                const encoder = r.u32();
                const enc = handles.get(encoder);
                handles.bind(dst, enc.finish());
                handles.release(encoder);
                break;
            }
            case OP.RenderPassExecuteBundles: {
                const pass = r.u32();
                // Payload = the bundle handles as LE u32s.
                const ids = r.payload();
                const view = new Uint32Array(ids.buffer, ids.byteOffset, ids.byteLength / 4);
                const arr = [];
                for (const id of view) {
                    arr.push(handles.get(id));
                }
                handles.get(pass).executeBundles(arr);
                break;
            }
            // ── queue ──────────────────────────────────────────────────────────
            case OP.QueueSubmit: {
                const device = r.u32();
                const commandBuffer = r.u32();
                const dev = handles.get(device);
                dev.queue.submit([handles.get(commandBuffer)]);
                handles.release(commandBuffer);
                break;
            }
            case OP.QueueOnSubmittedWorkDone: {
                const device = r.u32();
                const task = r.u32();
                const dev = handles.get(device);
                async_(state, task, dev.queue.onSubmittedWorkDone(), () => 0);
                break;
            }
            // ── textures / samplers ────────────────────────────────────────────
            case OP.CreateTexture: {
                const dst = r.u32();
                const device = r.u32();
                const width = r.u32();
                const height = r.u32();
                const depthOrArrayLayers = r.u32();
                const mipLevelCount = r.u32();
                const sampleCount = r.u32();
                const formatCode = r.u32();
                const usage = r.u32();
                const dev = handles.get(device);
                handles.bind(dst, dev.createTexture({
                    size: { width, height, depthOrArrayLayers },
                    mipLevelCount,
                    sampleCount,
                    format: formatName(formatCode),
                    usage,
                }));
                break;
            }
            case OP.TextureCreateView: {
                const dst = r.u32();
                const texture = r.u32();
                handles.bind(dst, handles.get(texture).createView());
                break;
            }
            case OP.DestroyTexture: {
                const texture = r.u32();
                handles.get(texture).destroy();
                handles.release(texture);
                break;
            }
            case OP.QueueWriteTexture: {
                const device = r.u32();
                const texture = r.u32();
                const originX = r.u32();
                const originY = r.u32();
                const bytesPerRow = r.u32();
                const rowsPerImage = r.u32();
                const width = r.u32();
                const height = r.u32();
                const depth = r.u32();
                const data = r.payload();
                const dev = handles.get(device);
                const t = handles.get(texture);
                dev.queue.writeTexture({ texture: t, origin: { x: originX, y: originY, z: 0 } }, data, { offset: 0, bytesPerRow, rowsPerImage }, { width, height, depthOrArrayLayers: depth });
                break;
            }
            case OP.CreateSampler: {
                const dst = r.u32();
                const device = r.u32();
                const magFilter = r.u32();
                const minFilter = r.u32();
                const mipmapFilter = r.u32();
                const addressU = r.u32();
                const addressV = r.u32();
                const addressW = r.u32();
                const maxAnisotropy = r.u32();
                const compareCode = r.u32();
                const dev = handles.get(device);
                const desc = {
                    magFilter: filterName(magFilter),
                    minFilter: filterName(minFilter),
                    mipmapFilter: filterName(mipmapFilter),
                    addressModeU: addressName(addressU),
                    addressModeV: addressName(addressV),
                    addressModeW: addressName(addressW),
                };
                if (maxAnisotropy > 1)
                    desc.maxAnisotropy = maxAnisotropy;
                if (compareCode !== COMPARE_UNSET)
                    desc.compare = compareName(compareCode);
                handles.bind(dst, dev.createSampler(desc));
                break;
            }
            // ── canvas presentation (step 096) ─────────────────────────────────
            // There is no present arm on purpose: the compositor shows the
            // current texture when the task returns to the event loop.
            case OP.CanvasCreateOffscreen: {
                const dst = r.u32();
                const width = r.u32();
                const height = r.u32();
                handles.bind(dst, new OffscreenCanvas(width, height));
                break;
            }
            case OP.CanvasContextCreate: {
                const dst = r.u32();
                const canvas = r.u32();
                const c = handles.get(canvas);
                // "webgpu" is missing from lib.dom's getContext overloads; the
                // cast is the entire accommodation.
                const ctx = c.getContext("webgpu");
                if (ctx === null) {
                    throw new Error("quanta glue: the canvas could not provide a webgpu context — " +
                        "WebGPU is absent, or the canvas already handed out a 2d/webgl context");
                }
                handles.bind(dst, ctx);
                break;
            }
            case OP.CanvasContextConfigure: {
                const context = r.u32();
                const canvas = r.u32();
                const device = r.u32();
                const formatCode = r.u32();
                const usage = r.u32();
                const width = r.u32();
                const height = r.u32();
                const ctx = handles.get(context);
                const c = handles.get(canvas);
                const dev = handles.get(device);
                // Drive the backing-store size (the drawableSize analogue). CSS
                // layout size stays the embedder's.
                c.width = width;
                c.height = height;
                requireExports(state).__quanta_canvas_dims(canvas, c.width, c.height);
                ctx.configure({
                    device: dev,
                    format: formatName(formatCode),
                    usage,
                    alphaMode: "opaque",
                });
                break;
            }
            case OP.CanvasContextUnconfigure: {
                const context = r.u32();
                handles.get(context).unconfigure();
                break;
            }
            case OP.CanvasGetCurrentTexture: {
                const dst = r.u32();
                const context = r.u32();
                const ctx = handles.get(context);
                handles.bind(dst, ctx.getCurrentTexture());
                break;
            }
            // ── top-level completion ───────────────────────────────────────────
            case OP.CompleteBytes: {
                const task = r.u32();
                const bytes = r.payload().slice();
                const t = state.topLevelTasks.get(task);
                if (t === undefined) {
                    console.error(`quanta glue: unknown top-level task ${task}`);
                    break;
                }
                state.topLevelTasks.delete(task);
                t.resolve(bytes);
                break;
            }
            case OP.CompleteErr: {
                const task = r.u32();
                const message = r.text();
                const t = state.topLevelTasks.get(task);
                if (t === undefined) {
                    console.error(`quanta glue: unknown top-level task ${task}`);
                    break;
                }
                state.topLevelTasks.delete(task);
                t.reject(new Error(message));
                break;
            }
            // ── universal handle release + diagnostics ─────────────────────────
            case OP.Release: {
                handles.release(r.u32());
                break;
            }
            case OP.ConsoleError: {
                console.error(r.text());
                break;
            }
            default:
                throw new Error(`quanta glue: unknown tape opcode 0x${op.toString(16)}`);
        }
    }
}
