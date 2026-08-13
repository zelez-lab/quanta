//! The command tape — the zero-import boundary (dija R10).
//!
//! Instead of calling out through `env:*` imports, the driver appends
//! commands to a tape in linear memory; the glue DRAINS the tape after
//! every wasm entry returns and performs the WebGPU calls. Async
//! results still arrive through the exported setters
//! (`quanta_resolve` / `quanta_reject` — the `init_poll` precedent).
//!
//! ## Wire format (version 1 — dija asserts this byte-for-byte)
//!
//! Little-endian u32 words. The tape begins with `[MAGIC][VERSION]`
//! and is a sequence of ops: `[opcode][args…]`. Fixed args are one
//! word each (`f64` args are TWO words, the LE bit pattern). Byte
//! payloads are inline: `[len][bytes… padded to a word]`. The glue
//! reads `__quanta_tape_ptr()/__quanta_tape_len()`, interprets, then
//! calls `__quanta_tape_clear()`.
//!
//! ## Id spaces
//!
//! Ops that CREATE glue-side objects carry a driver-minted destination
//! id with the high bit set (`0x8000_0000 | n`). Glue-minted ids
//! (`registerCanvas`) stay below the high bit — one handle table
//! serves both, and the `registerCanvas` contract does not move.

use alloc::vec::Vec;
use core::cell::RefCell;

pub(crate) const TAPE_MAGIC: u32 = 0x5041_5451; // "QTAP" LE
pub(crate) const TAPE_VERSION: u32 = 1;
const DRIVER_ID_BIT: u32 = 0x8000_0000;

/// Tape opcodes — the wire contract. Explicit discriminants; NEVER
/// reorder or reuse (append + bump [`TAPE_VERSION`] instead).
#[repr(u32)]
#[derive(Clone, Copy)]
pub(crate) enum Op {
    // Async acquisition (results return via quanta_resolve/reject).
    RequestAdapter = 0x01,
    RequestDevice = 0x02,
    // Buffers.
    CreateBuffer = 0x10,
    DestroyBuffer = 0x11,
    WriteBuffer = 0x12,  // payload
    MapAsyncRead = 0x13, // (buffer, task, dst_ptr, dst_len): glue copies at resolve, unmaps, then resolves
    // Reserved: `MapAsyncRead` unmaps at completion, so nothing emits
    // this today. The interpreter still honors it (the wire contract is
    // total) and the slot must not be reused.
    #[allow(dead_code)]
    UnmapBuffer = 0x14,
    // Shaders / compute pipelines.
    CreateShaderModule = 0x20,    // payload = WGSL source
    CreateComputePipeline = 0x21, // payload = entry-point name
    ComputePipelineGetBindGroupLayout = 0x22,
    // Render-pipeline descriptor building.
    RpDescCreate = 0x30,
    RpDescSetVertex = 0x31, // payload = entry name
    RpDescAddVertexBuffer = 0x32,
    RpDescAddVertexAttribute = 0x33,
    RpDescSetFragment = 0x34, // payload = entry name
    RpDescAddColorTarget = 0x35,
    RpDescSetPrimitive = 0x36,
    RpDescSetMultisample = 0x37,
    RpDescSetDepthStencil = 0x38,
    CreateRenderPipeline = 0x39,
    RenderPipelineGetBindGroupLayout = 0x3A,
    // Bind groups.
    BgDescCreate = 0x40,
    BgDescAddBuffer = 0x41,
    BgDescAddSampler = 0x42,
    BgDescAddTextureView = 0x43,
    CreateBindGroup = 0x44,
    // Encoders + copies.
    CreateCommandEncoder = 0x50,
    EncoderCopyBufferToBuffer = 0x51,
    EncoderCopyTextureToBuffer = 0x52,
    EncoderFinish = 0x53,
    // Compute passes.
    EncoderBeginComputePass = 0x60,
    ComputePassSetPipeline = 0x61,
    ComputePassSetBindGroup = 0x62,
    ComputePassDispatch = 0x63,
    ComputePassEnd = 0x64,
    // Render-pass descriptor + pass ops.
    RpassDescCreate = 0x70,
    RpassDescAddColorAttachment = 0x71,
    RpassDescSetDepthAttachment = 0x72,
    EncoderBeginRenderPass = 0x73,
    RenderPassSetPipeline = 0x74,
    RenderPassSetBindGroup = 0x75,
    RenderPassSetVertexBuffer = 0x76,
    RenderPassSetIndexBuffer = 0x77,
    RenderPassDraw = 0x78,
    RenderPassDrawIndexed = 0x79,
    RenderPassDrawIndirect = 0x7A,
    RenderPassDrawIndexedIndirect = 0x7B,
    RenderPassSetViewport = 0x7C,
    RenderPassSetScissor = 0x7D,
    RenderPassSetStencilReference = 0x7E,
    RenderPassEnd = 0x7F,
    // Queries.
    CreateQuerySet = 0x90,
    RpassDescSetOcclusionQuerySet = 0x91,
    RenderPassBeginOcclusionQuery = 0x92,
    RenderPassEndOcclusionQuery = 0x93,
    EncoderResolveQuerySet = 0x94,
    // Render bundles.
    CreateRenderBundleEncoder = 0xA0,
    RenderBundleSetPipeline = 0xA1,
    RenderBundleSetBindGroup = 0xA2,
    RenderBundleSetVertexBuffer = 0xA3,
    RenderBundleDraw = 0xA4,
    RenderBundleFinish = 0xA5,
    RenderPassExecuteBundles = 0xA6, // payload = bundle id words
    // Queue.
    QueueSubmit = 0xB0,
    QueueOnSubmittedWorkDone = 0xB1,
    // Textures + samplers.
    CreateTexture = 0xC0,
    TextureCreateView = 0xC1,
    DestroyTexture = 0xC2,
    QueueWriteTexture = 0xC3, // payload
    CreateSampler = 0xC4,
    // Canvas / surface (canvas ids are glue-minted, passed opaquely).
    CanvasCreateOffscreen = 0xD0,
    CanvasContextCreate = 0xD1,
    CanvasContextConfigure = 0xD2,
    CanvasContextUnconfigure = 0xD3,
    CanvasGetCurrentTexture = 0xD4,
    // Top-level completion (replaces the quanta_complete_* imports).
    CompleteBytes = 0xE0, // payload
    CompleteErr = 0xE1,   // payload = utf8 message
    // Handle lifetime + diagnostics.
    Release = 0xF0,
    ConsoleError = 0xF1, // payload = utf8 message
}

struct TapeState {
    buf: Vec<u8>,
    next_id: u32,
}

impl TapeState {
    const fn new() -> Self {
        TapeState {
            buf: Vec::new(),
            next_id: 1,
        }
    }
}

// wasm32 is single-threaded; a thread_local RefCell is the whole story.
std::thread_local! {
    static TAPE: RefCell<TapeState> = const { RefCell::new(TapeState::new()) };
}

fn with<R>(f: impl FnOnce(&mut TapeState) -> R) -> R {
    TAPE.with(|t| f(&mut t.borrow_mut()))
}

/// Mint a driver-side handle id (high bit set — never collides with
/// glue-minted `registerCanvas` ids).
pub(crate) fn mint_id() -> u32 {
    with(|t| {
        let id = DRIVER_ID_BIT | t.next_id;
        t.next_id += 1;
        id
    })
}

fn push_word(buf: &mut Vec<u8>, w: u32) {
    buf.extend_from_slice(&w.to_le_bytes());
}

/// Append one op: fixed word args, then byte payloads (each preceded
/// by its length word, padded to word alignment).
pub(crate) fn emit(op: Op, words: &[u32], payloads: &[&[u8]]) {
    with(|t| {
        if t.buf.is_empty() {
            push_word(&mut t.buf, TAPE_MAGIC);
            push_word(&mut t.buf, TAPE_VERSION);
        }
        push_word(&mut t.buf, op as u32);
        for &w in words {
            push_word(&mut t.buf, w);
        }
        for p in payloads {
            push_word(&mut t.buf, p.len() as u32);
            t.buf.extend_from_slice(p);
            let pad = (4 - (p.len() % 4)) % 4;
            t.buf.extend_from_slice(&[0u8; 3][..pad]);
        }
    })
}

/// The two-word LE encoding of an `f64` arg.
pub(crate) fn f64_words(v: f64) -> [u32; 2] {
    let bits = v.to_bits();
    [bits as u32, (bits >> 32) as u32]
}

// ── The drain interface (called by the glue) ────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn __quanta_tape_ptr() -> *const u8 {
    with(|t| t.buf.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn __quanta_tape_len() -> usize {
    with(|t| t.buf.len())
}

#[unsafe(no_mangle)]
pub extern "C" fn __quanta_tape_clear() {
    with(|t| t.buf.clear())
}

// ── Push-state (replaces the sync-query imports) ────────────────────

struct EnvState {
    available: u32,
    preferred_format: u32,
}

std::thread_local! {
    static ENV: RefCell<EnvState> = const {
        RefCell::new(EnvState { available: 0, preferred_format: 0 })
    };
}

// Canvas dims pushed by the glue (`registerCanvas` + every resize it
// owns under R1). Keyed by the glue-minted canvas id.
std::thread_local! {
    static CANVAS_DIMS: RefCell<alloc::collections::BTreeMap<u32, (u32, u32)>> =
        const { RefCell::new(alloc::collections::BTreeMap::new()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn __quanta_env_init(available: u32, preferred_format: u32) {
    ENV.with(|e| {
        let mut e = e.borrow_mut();
        e.available = available;
        e.preferred_format = preferred_format;
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn __quanta_canvas_dims(canvas: u32, width: u32, height: u32) {
    CANVAS_DIMS.with(|c| {
        c.borrow_mut().insert(canvas, (width, height));
    });
}

pub(crate) fn env_available() -> u32 {
    ENV.with(|e| e.borrow().available)
}

pub(crate) fn env_preferred_format() -> u32 {
    ENV.with(|e| e.borrow().preferred_format)
}

pub(crate) fn canvas_dims(canvas: u32) -> (u32, u32) {
    CANVAS_DIMS.with(|c| c.borrow().get(&canvas).copied().unwrap_or((0, 0)))
}

/// Record dims the DRIVER itself just set (R1: `configure` owns the
/// backing store). The glue pushes the same values at drain, but a
/// configure and the acquire that checks its extent routinely happen
/// inside ONE wasm entry — the driver-side write is what keeps that
/// pair reading the same dims it does with direct calls.
pub(crate) fn set_canvas_dims(canvas: u32, width: u32, height: u32) {
    CANVAS_DIMS.with(|c| {
        c.borrow_mut().insert(canvas, (width, height));
    });
}

// ── Top-level completion ────────────────────────────────────────────

/// Hand a top-level task's result bytes back to the host — the tape
/// spelling of the old `quanta_complete_bytes` import. The glue
/// resolves the Promise `runReturningBytes` returned.
pub fn complete_bytes(task: u32, bytes: &[u8]) {
    emit(Op::CompleteBytes, &[task], &[bytes]);
}

/// Reject a top-level task with a message — the tape spelling of the
/// old `quanta_complete_err` import.
pub fn complete_err(task: u32, message: &str) {
    emit(Op::CompleteErr, &[task], &[message.as_bytes()]);
}
