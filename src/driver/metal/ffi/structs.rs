//! `#[repr(C)]` struct definitions for Metal FFI.

use core::ffi::c_void;

use super::constants::{Id, NSUInteger};

// ─── Geometry structs ───────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MTLSize {
    pub width: NSUInteger,
    pub height: NSUInteger,
    pub depth: NSUInteger,
}

impl MTLSize {
    pub fn new(width: u64, height: u64, depth: u64) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MTLOrigin {
    pub x: NSUInteger,
    pub y: NSUInteger,
    pub z: NSUInteger,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MTLRegion {
    pub origin: MTLOrigin,
    pub size: MTLSize,
}

impl MTLRegion {
    pub fn new_2d(x: u64, y: u64, width: u64, height: u64) -> Self {
        Self {
            origin: MTLOrigin { x, y, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MTLScissorRect {
    pub x: u64,
    pub y: u64,
    pub width: u64,
    pub height: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MTLViewport {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: f64,
    pub height: f64,
    pub znear: f64,
    pub zfar: f64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MTLClearColor {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
    pub alpha: f64,
}

impl MTLClearColor {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self {
            red: r,
            green: g,
            blue: b,
            alpha: a,
        }
    }
}

// ─── Texture view helpers ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NSRange {
    pub location: u64,
    pub length: u64,
}

// ─── ObjC block ABI (for addCompletedHandler:) ────────────────────────────

/// Minimal ObjC block descriptor (no copy/dispose — stack block, no captures
/// that need ARC. We use a global descriptor with a static invoke function.)
#[repr(C)]
pub struct BlockDescriptor {
    pub reserved: u64,
    pub size: u64,
}

/// ObjC block layout for `void (^)(id)` — one argument, no return.
/// The `context` field carries our Rust data (semaphore pointer).
#[repr(C)]
pub struct CompletionBlock {
    pub isa: *const c_void,
    pub flags: i32,
    pub reserved: i32,
    pub invoke: unsafe extern "C" fn(*mut CompletionBlock, Id),
    pub descriptor: *const BlockDescriptor,
    pub semaphore: *mut c_void,
}

/// Global block descriptor (static, shared by all completion blocks).
pub(crate) static COMPLETION_BLOCK_DESCRIPTOR: BlockDescriptor = BlockDescriptor {
    reserved: 0,
    size: core::mem::size_of::<CompletionBlock>() as u64,
};

/// Invoke function for the completion block: signals the semaphore.
pub(crate) unsafe extern "C" fn completion_block_invoke(block: *mut CompletionBlock, _cmd_buf: Id) {
    super::extern_fns::dispatch_semaphore_signal((*block).semaphore);
}
