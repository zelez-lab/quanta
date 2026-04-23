//! Raw Metal and Objective-C FFI bindings.
//!
//! Minimal ObjC runtime + Metal API surface required by the Quanta GPU driver.
//! No external crate dependencies — just `extern "C"` calls to the system frameworks.

#![allow(unsafe_op_in_unsafe_fn, dead_code, clippy::upper_case_acronyms)]

use core::ffi::c_void;
use core::mem;

// ─── Types ───────────────────────────────────────────────────────────────────

pub type Id = *mut c_void;
pub type Sel = *mut c_void;
pub type Class = *mut c_void;
pub type NSUInteger = u64;
pub type BOOL = i8;

pub const NIL: Id = core::ptr::null_mut();
pub const YES: BOOL = 1;
#[allow(dead_code)]
pub const NO: BOOL = 0;

// ─── Metal pixel formats ────────────────────────────────────────────────────

pub const MTL_PIXEL_FORMAT_R8_UNORM: NSUInteger = 10;
pub const MTL_PIXEL_FORMAT_R16_FLOAT: NSUInteger = 25;
pub const MTL_PIXEL_FORMAT_R32_FLOAT: NSUInteger = 55;
pub const MTL_PIXEL_FORMAT_RG32_FLOAT: NSUInteger = 63;
pub const MTL_PIXEL_FORMAT_RGBA8_UNORM: NSUInteger = 70;
pub const MTL_PIXEL_FORMAT_BGRA8_UNORM: NSUInteger = 80;
pub const MTL_PIXEL_FORMAT_RGBA16_FLOAT: NSUInteger = 115;
pub const MTL_PIXEL_FORMAT_RGBA32_FLOAT: NSUInteger = 125;
pub const MTL_PIXEL_FORMAT_DEPTH32_FLOAT: NSUInteger = 252;

// Compressed formats
pub const MTL_PIXEL_FORMAT_BC1_RGBA: NSUInteger = 130;
pub const MTL_PIXEL_FORMAT_BC3_RGBA: NSUInteger = 132;
pub const MTL_PIXEL_FORMAT_BC5_RG_SNORM: NSUInteger = 135;
pub const MTL_PIXEL_FORMAT_BC7_RGBA_UNORM: NSUInteger = 140;
pub const MTL_PIXEL_FORMAT_ASTC_4X4_LDR: NSUInteger = 204;
pub const MTL_PIXEL_FORMAT_ASTC_6X6_LDR: NSUInteger = 208;
pub const MTL_PIXEL_FORMAT_ASTC_8X8_LDR: NSUInteger = 212;
pub const MTL_PIXEL_FORMAT_ETC2_RGB8: NSUInteger = 180;
pub const MTL_PIXEL_FORMAT_EAC_RGBA8: NSUInteger = 178;

// ─── Metal resource options ─────────────────────────────────────────────────

pub const MTL_RESOURCE_STORAGE_MODE_SHARED: NSUInteger = 0 << 4;
pub const MTL_RESOURCE_STORAGE_MODE_PRIVATE: NSUInteger = 2 << 4;

// ─── Metal storage modes (for texture descriptors) ──────────────────────────

pub const MTL_STORAGE_MODE_SHARED: NSUInteger = 0;
pub const MTL_STORAGE_MODE_PRIVATE: NSUInteger = 2;

// ─── Metal texture usage ────────────────────────────────────────────────────

pub const MTL_TEXTURE_USAGE_SHADER_READ: NSUInteger = 0x01;
pub const MTL_TEXTURE_USAGE_SHADER_WRITE: NSUInteger = 0x02;
pub const MTL_TEXTURE_USAGE_RENDER_TARGET: NSUInteger = 0x04;

// ─── Metal texture types ────────────────────────────────────────────────────

pub const MTL_TEXTURE_TYPE_2D: NSUInteger = 2;
pub const MTL_TEXTURE_TYPE_2D_ARRAY: NSUInteger = 3;
pub const MTL_TEXTURE_TYPE_2D_MULTISAMPLE: NSUInteger = 4;
pub const MTL_TEXTURE_TYPE_CUBE: NSUInteger = 5;
pub const MTL_TEXTURE_TYPE_CUBE_ARRAY: NSUInteger = 6;
pub const MTL_TEXTURE_TYPE_3D: NSUInteger = 7;

// ─── Metal load/store actions ───────────────────────────────────────────────

pub const MTL_LOAD_ACTION_DONT_CARE: NSUInteger = 0;
pub const MTL_LOAD_ACTION_LOAD: NSUInteger = 1;
pub const MTL_LOAD_ACTION_CLEAR: NSUInteger = 2;
pub const MTL_STORE_ACTION_DONT_CARE: NSUInteger = 0;
pub const MTL_STORE_ACTION_STORE: NSUInteger = 1;
pub const MTL_STORE_ACTION_MULTISAMPLE_RESOLVE: NSUInteger = 2;

// ─── Metal primitive types ──────────────────────────────────────────────────

pub const MTL_PRIMITIVE_TYPE_TRIANGLE: NSUInteger = 3;

// ─── Metal index types ──────────────────────────────────────────────────────

pub const MTL_INDEX_TYPE_UINT32: NSUInteger = 1;

// ─── Metal compare functions ────────────────────────────────────────────────

pub const MTL_COMPARE_NEVER: NSUInteger = 0;
pub const MTL_COMPARE_LESS: NSUInteger = 1;
pub const MTL_COMPARE_EQUAL: NSUInteger = 2;
pub const MTL_COMPARE_LESS_EQUAL: NSUInteger = 3;
pub const MTL_COMPARE_GREATER: NSUInteger = 4;
pub const MTL_COMPARE_NOT_EQUAL: NSUInteger = 5;
pub const MTL_COMPARE_GREATER_EQUAL: NSUInteger = 6;
pub const MTL_COMPARE_ALWAYS: NSUInteger = 7;

// ─── Metal stencil operations ───────────────────────────────────────────────

pub const MTL_STENCIL_OP_KEEP: NSUInteger = 0;
pub const MTL_STENCIL_OP_ZERO: NSUInteger = 1;
pub const MTL_STENCIL_OP_REPLACE: NSUInteger = 2;
pub const MTL_STENCIL_OP_INCREMENT_CLAMP: NSUInteger = 3;
pub const MTL_STENCIL_OP_DECREMENT_CLAMP: NSUInteger = 4;
pub const MTL_STENCIL_OP_INVERT: NSUInteger = 5;
pub const MTL_STENCIL_OP_INCREMENT_WRAP: NSUInteger = 6;
pub const MTL_STENCIL_OP_DECREMENT_WRAP: NSUInteger = 7;

// ─── Metal blend factors ────────────────────────────────────────────────────

pub const MTL_BLEND_FACTOR_ZERO: NSUInteger = 0;
pub const MTL_BLEND_FACTOR_ONE: NSUInteger = 1;
pub const MTL_BLEND_FACTOR_SRC_COLOR: NSUInteger = 2;
pub const MTL_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: NSUInteger = 3;
pub const MTL_BLEND_FACTOR_SRC_ALPHA: NSUInteger = 4;
pub const MTL_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: NSUInteger = 5;
pub const MTL_BLEND_FACTOR_DST_COLOR: NSUInteger = 6;
pub const MTL_BLEND_FACTOR_ONE_MINUS_DST_COLOR: NSUInteger = 7;
pub const MTL_BLEND_FACTOR_DST_ALPHA: NSUInteger = 8;
pub const MTL_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: NSUInteger = 9;

// ─── Metal blend operations ─────────────────────────────────────────────────

pub const MTL_BLEND_OP_ADD: NSUInteger = 0;
pub const MTL_BLEND_OP_SUBTRACT: NSUInteger = 1;
pub const MTL_BLEND_OP_REVERSE_SUBTRACT: NSUInteger = 2;
pub const MTL_BLEND_OP_MIN: NSUInteger = 3;
pub const MTL_BLEND_OP_MAX: NSUInteger = 4;

// ─── Metal sampler min/mag filter ───────────────────────────────────────────

pub const MTL_SAMPLER_MIN_MAG_FILTER_NEAREST: NSUInteger = 0;
pub const MTL_SAMPLER_MIN_MAG_FILTER_LINEAR: NSUInteger = 1;

// ─── Metal sampler mip filter ───────────────────────────────────────────────

pub const MTL_SAMPLER_MIP_FILTER_NEAREST: NSUInteger = 1;
pub const MTL_SAMPLER_MIP_FILTER_LINEAR: NSUInteger = 2;

// ─── Metal sampler address modes ────────────────────────────────────────────

pub const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: NSUInteger = 0;
pub const MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: NSUInteger = 1;
pub const MTL_SAMPLER_ADDRESS_MODE_REPEAT: NSUInteger = 2;
pub const MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT: NSUInteger = 3;

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

// ─── Extern bindings ─────────────────────────────────────────────────────────

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    pub fn objc_getClass(name: *const u8) -> Class;
    pub fn sel_registerName(name: *const u8) -> Sel;
    pub fn objc_msgSend();
}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    pub fn MTLCreateSystemDefaultDevice() -> Id;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Get an ObjC class by name (null-terminated).
pub fn cls(name: &[u8]) -> Class {
    debug_assert!(
        name.last() == Some(&0),
        "class name must be null-terminated"
    );
    unsafe { objc_getClass(name.as_ptr()) }
}

/// Get an ObjC selector by name (null-terminated).
pub fn sel(name: &[u8]) -> Sel {
    debug_assert!(name.last() == Some(&0), "selector must be null-terminated");
    unsafe { sel_registerName(name.as_ptr()) }
}

/// Create an NSString from a null-terminated byte slice.
pub fn nsstring(s: &[u8]) -> Id {
    debug_assert!(s.last() == Some(&0), "string must be null-terminated");
    let f: unsafe extern "C" fn(Id, Sel, *const u8) -> Id =
        unsafe { mem::transmute(objc_msgSend as *const c_void) };
    unsafe {
        f(
            cls(b"NSString\0") as Id,
            sel(b"stringWithUTF8String:\0"),
            s.as_ptr(),
        )
    }
}

// ─── msg_send helpers ───────────────────────────────────────────────────────

/// Send message with no arguments, returning Id.
pub unsafe fn msg_id(obj: Id, name: &[u8]) -> Id {
    let f: unsafe extern "C" fn(Id, Sel) -> Id = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name))
}

/// Send message with no arguments, returning void.
pub unsafe fn msg_void(obj: Id, name: &[u8]) {
    let f: unsafe extern "C" fn(Id, Sel) = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name))
}

/// Send message with no arguments, returning u64.
pub unsafe fn msg_u64(obj: Id, name: &[u8]) -> u64 {
    let f: unsafe extern "C" fn(Id, Sel) -> u64 = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name))
}

/// Send message with no arguments, returning *mut u8 (for buffer contents).
pub unsafe fn msg_ptr(obj: Id, name: &[u8]) -> *mut u8 {
    let f: unsafe extern "C" fn(Id, Sel) -> *mut u8 = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name))
}

/// Send message with one Id argument, returning void.
pub unsafe fn msg_void_id(obj: Id, name: &[u8], v: Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id) = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), v)
}

/// Send message with one u64 argument, returning void.
pub unsafe fn msg_void_u64(obj: Id, name: &[u8], v: u64) {
    let f: unsafe extern "C" fn(Id, Sel, u64) = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), v)
}

/// Send message with one u32 argument, returning void.
pub unsafe fn msg_void_u32(obj: Id, name: &[u8], v: u32) {
    let f: unsafe extern "C" fn(Id, Sel, u32) = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), v)
}

/// Send message with one f64 argument, returning void.
pub unsafe fn msg_void_f64(obj: Id, name: &[u8], v: f64) {
    let f: unsafe extern "C" fn(Id, Sel, f64) = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), v)
}

/// Send message with one bool argument, returning void.
pub unsafe fn msg_void_bool(obj: Id, name: &[u8], v: bool) {
    let f: unsafe extern "C" fn(Id, Sel, BOOL) = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), if v { YES } else { NO })
}

/// Send message with one Id argument, returning Id.
pub unsafe fn msg_id_id(obj: Id, name: &[u8], v: Id) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id) -> Id = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), v)
}

/// Send message with one u64 argument, returning Id.
pub unsafe fn msg_id_u64(obj: Id, name: &[u8], v: u64) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, u64) -> Id = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), v)
}

/// Send message returning MTLSize (for max_threads_per_threadgroup).
pub unsafe fn msg_mtlsize(obj: Id, name: &[u8]) -> MTLSize {
    let f: unsafe extern "C" fn(Id, Sel) -> MTLSize = mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name))
}

/// newBufferWithBytes:length:options: -> Id
pub unsafe fn msg_new_buffer_with_bytes(
    device: Id,
    ptr: *const c_void,
    len: u64,
    options: NSUInteger,
) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, u64, NSUInteger) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        device,
        sel(b"newBufferWithBytes:length:options:\0"),
        ptr,
        len,
        options,
    )
}

/// newBufferWithLength:options: -> Id
pub unsafe fn msg_new_buffer(device: Id, len: u64, options: NSUInteger) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, u64, NSUInteger) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    f(device, sel(b"newBufferWithLength:options:\0"), len, options)
}

/// newLibraryWithSource:options:error: -> Id
pub unsafe fn msg_new_library_with_source(device: Id, source: Id, options: Id) -> (Id, Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    let mut error: Id = NIL;
    let lib = f(
        device,
        sel(b"newLibraryWithSource:options:error:\0"),
        source,
        options,
        &mut error,
    );
    (lib, error)
}

/// newLibraryWithData:error: -> Id
pub unsafe fn msg_new_library_with_data(device: Id, data: *const c_void, len: u64) -> (Id, Id) {
    // Create a dispatch_data_t from the raw bytes
    let dispatch_data = dispatch_data_create(
        data,
        len as usize,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    let mut error: Id = NIL;
    let lib = f(
        device,
        sel(b"newLibraryWithData:error:\0"),
        dispatch_data as Id,
        &mut error,
    );
    (lib, error)
}

/// newComputePipelineStateWithFunction:error: -> Id
pub unsafe fn msg_new_compute_pipeline(device: Id, func: Id) -> (Id, Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    let mut error: Id = NIL;
    let pipeline = f(
        device,
        sel(b"newComputePipelineStateWithFunction:error:\0"),
        func,
        &mut error,
    );
    (pipeline, error)
}

/// newRenderPipelineStateWithDescriptor:error: -> Id
pub unsafe fn msg_new_render_pipeline(device: Id, desc: Id) -> (Id, Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    let mut error: Id = NIL;
    let pipeline = f(
        device,
        sel(b"newRenderPipelineStateWithDescriptor:error:\0"),
        desc,
        &mut error,
    );
    (pipeline, error)
}

/// setBuffer:offset:atIndex: on compute/render encoder
pub unsafe fn msg_set_buffer(encoder: Id, sel_name: &[u8], buffer: Id, offset: u64, index: u64) {
    let f: unsafe extern "C" fn(Id, Sel, Id, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(sel_name), buffer, offset, index)
}

/// setBytes:length:atIndex: on encoder
pub unsafe fn msg_set_bytes(
    encoder: Id,
    sel_name: &[u8],
    ptr: *const c_void,
    len: u64,
    index: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(sel_name), ptr, len, index)
}

/// setTexture:atIndex: on encoder
pub unsafe fn msg_set_texture(encoder: Id, sel_name: &[u8], texture: Id, index: u64) {
    let f: unsafe extern "C" fn(Id, Sel, Id, u64) = mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(sel_name), texture, index)
}

/// setSamplerState:atIndex: on encoder
pub unsafe fn msg_set_sampler(encoder: Id, sel_name: &[u8], sampler: Id, index: u64) {
    let f: unsafe extern "C" fn(Id, Sel, Id, u64) = mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(sel_name), sampler, index)
}

/// dispatchThreads:threadsPerThreadgroup: on compute encoder
pub unsafe fn msg_dispatch_threads(encoder: Id, grid: MTLSize, group: MTLSize) {
    let f: unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"dispatchThreads:threadsPerThreadgroup:\0"),
        grid,
        group,
    )
}

/// dispatchThreadgroups:threadsPerThreadgroup: on compute encoder
pub unsafe fn msg_dispatch_threadgroups(encoder: Id, groups: MTLSize, group_size: MTLSize) {
    let f: unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
        groups,
        group_size,
    )
}

/// dispatchThreadgroups:threadsPerThreadgroup: (indirect) on compute encoder
pub unsafe fn msg_dispatch_threadgroups_indirect(
    encoder: Id,
    buffer: Id,
    offset: u64,
    group: MTLSize,
) {
    let f: unsafe extern "C" fn(Id, Sel, Id, u64, MTLSize) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(
            b"dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:\0",
        ),
        buffer,
        offset,
        group,
    )
}

/// drawPrimitives:vertexStart:vertexCount:
pub unsafe fn msg_draw_primitives(encoder: Id, ptype: NSUInteger, start: u64, count: u64) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"drawPrimitives:vertexStart:vertexCount:\0"),
        ptype,
        start,
        count,
    )
}

/// drawPrimitives:vertexStart:vertexCount:instanceCount:
pub unsafe fn msg_draw_primitives_instanced(
    encoder: Id,
    ptype: NSUInteger,
    start: u64,
    count: u64,
    instances: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, u64, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"drawPrimitives:vertexStart:vertexCount:instanceCount:\0"),
        ptype,
        start,
        count,
        instances,
    )
}

/// drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:
pub unsafe fn msg_draw_indexed(
    encoder: Id,
    ptype: NSUInteger,
    index_count: u64,
    index_type: NSUInteger,
    index_buffer: Id,
    offset: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, u64, NSUInteger, Id, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\0"),
        ptype,
        index_count,
        index_type,
        index_buffer,
        offset,
    )
}

/// drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
pub unsafe fn msg_draw_indexed_instanced(
    encoder: Id,
    ptype: NSUInteger,
    index_count: u64,
    index_type: NSUInteger,
    index_buffer: Id,
    offset: u64,
    instances: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, u64, NSUInteger, Id, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:\0"),
        ptype,
        index_count,
        index_type,
        index_buffer,
        offset,
        instances,
    )
}

/// drawPrimitives:indirectBuffer:indirectBufferOffset:
pub unsafe fn msg_draw_primitives_indirect(
    encoder: Id,
    ptype: NSUInteger,
    buffer: Id,
    offset: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, Id, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"drawPrimitives:indirectBuffer:indirectBufferOffset:\0"),
        ptype,
        buffer,
        offset,
    )
}

/// drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:
pub unsafe fn msg_draw_indexed_indirect(
    encoder: Id,
    ptype: NSUInteger,
    index_type: NSUInteger,
    index_buffer: Id,
    index_offset: u64,
    indirect_buffer: Id,
    indirect_offset: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, NSUInteger, Id, u64, Id, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:\0"),
        ptype,
        index_type,
        index_buffer,
        index_offset,
        indirect_buffer,
        indirect_offset,
    )
}

/// setScissorRect: on render encoder
pub unsafe fn msg_set_scissor_rect(encoder: Id, rect: MTLScissorRect) {
    let f: unsafe extern "C" fn(Id, Sel, MTLScissorRect) =
        mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(b"setScissorRect:\0"), rect)
}

/// setViewport: on render encoder
pub unsafe fn msg_set_viewport(encoder: Id, viewport: MTLViewport) {
    let f: unsafe extern "C" fn(Id, Sel, MTLViewport) =
        mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(b"setViewport:\0"), viewport)
}

/// setStencilReferenceValue: on render encoder
pub unsafe fn msg_set_stencil_ref(encoder: Id, value: u32) {
    let f: unsafe extern "C" fn(Id, Sel, u32) = mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(b"setStencilReferenceValue:\0"), value)
}

/// copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:
pub unsafe fn msg_copy_buffer(
    blit: Id,
    src: Id,
    src_offset: u64,
    dst: Id,
    dst_offset: u64,
    size: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, Id, u64, Id, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        blit,
        sel(b"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:\0"),
        src,
        src_offset,
        dst,
        dst_offset,
        size,
    )
}

/// replaceRegion:mipmapLevel:withBytes:bytesPerRow: on texture
pub unsafe fn msg_replace_region(
    texture: Id,
    region: MTLRegion,
    level: u64,
    bytes: *const c_void,
    bytes_per_row: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, MTLRegion, u64, *const c_void, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        texture,
        sel(b"replaceRegion:mipmapLevel:withBytes:bytesPerRow:\0"),
        region,
        level,
        bytes,
        bytes_per_row,
    )
}

/// getBytes:bytesPerRow:fromRegion:mipmapLevel: on texture
pub unsafe fn msg_get_bytes(
    texture: Id,
    ptr: *mut c_void,
    bytes_per_row: u64,
    region: MTLRegion,
    level: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, *mut c_void, u64, MTLRegion, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        texture,
        sel(b"getBytes:bytesPerRow:fromRegion:mipmapLevel:\0"),
        ptr,
        bytes_per_row,
        region,
        level,
    )
}

/// newRenderCommandEncoderWithDescriptor: on command buffer
pub unsafe fn msg_new_render_encoder(cmd: Id, desc: Id) -> Id {
    msg_id_id(cmd, b"renderCommandEncoderWithDescriptor:\0", desc)
}

/// setClearColor: on color attachment
pub unsafe fn msg_set_clear_color(attachment: Id, color: MTLClearColor) {
    let f: unsafe extern "C" fn(Id, Sel, MTLClearColor) =
        mem::transmute(objc_msgSend as *const c_void);
    f(attachment, sel(b"setClearColor:\0"), color)
}

/// functionNames property on MTLLibrary — returns NSArray of NSString
pub unsafe fn msg_function_names(library: Id) -> Id {
    msg_id(library, b"functionNames\0")
}

/// NSArray count
pub unsafe fn msg_array_count(array: Id) -> u64 {
    msg_u64(array, b"count\0")
}

/// NSArray objectAtIndex:
pub unsafe fn msg_array_object_at(array: Id, index: u64) -> Id {
    msg_id_u64(array, b"objectAtIndex:\0", index)
}

/// NSString UTF8String
pub unsafe fn msg_utf8_string(nsstring: Id) -> *const u8 {
    let f: unsafe extern "C" fn(Id, Sel) -> *const u8 =
        mem::transmute(objc_msgSend as *const c_void);
    f(nsstring, sel(b"UTF8String\0"))
}

/// Get function from library by name: newFunctionWithName:
pub unsafe fn msg_get_function(library: Id, name: Id) -> Id {
    msg_id_id(library, b"newFunctionWithName:\0", name)
}

/// Push debug group on encoder
pub unsafe fn msg_push_debug_group(encoder: Id, label: &str) {
    let mut bytes: alloc::vec::Vec<u8> = label.bytes().collect();
    bytes.push(0);
    let ns = nsstring(&bytes);
    msg_void_id(encoder, b"pushDebugGroup:\0", ns)
}

/// Pop debug group on encoder
pub unsafe fn msg_pop_debug_group(encoder: Id) {
    msg_void(encoder, b"popDebugGroup\0")
}

/// generateMipmapsForTexture: on blit encoder
pub unsafe fn msg_generate_mipmaps(blit: Id, texture: Id) {
    msg_void_id(blit, b"generateMipmapsForTexture:\0", texture)
}

/// sampleTimestamps:gpuTimestamp: on device
pub unsafe fn msg_sample_timestamps(device: Id, cpu_ts: *mut u64, gpu_ts: *mut u64) {
    let f: unsafe extern "C" fn(Id, Sel, *mut u64, *mut u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        device,
        sel(b"sampleTimestamps:gpuTimestamp:\0"),
        cpu_ts,
        gpu_ts,
    );
}

// ─── MTLVertexFormat ───────────────────────────────────────────────────────

pub const MTL_VERTEX_FORMAT_FLOAT: NSUInteger = 28;
pub const MTL_VERTEX_FORMAT_FLOAT2: NSUInteger = 29;
pub const MTL_VERTEX_FORMAT_FLOAT3: NSUInteger = 30;
pub const MTL_VERTEX_FORMAT_FLOAT4: NSUInteger = 31;
pub const MTL_VERTEX_FORMAT_INT: NSUInteger = 32;
pub const MTL_VERTEX_FORMAT_INT2: NSUInteger = 33;
pub const MTL_VERTEX_FORMAT_INT3: NSUInteger = 34;
pub const MTL_VERTEX_FORMAT_INT4: NSUInteger = 35;
pub const MTL_VERTEX_FORMAT_UINT: NSUInteger = 36;
pub const MTL_VERTEX_FORMAT_UINT2: NSUInteger = 37;
pub const MTL_VERTEX_FORMAT_UINT3: NSUInteger = 38;
pub const MTL_VERTEX_FORMAT_UINT4: NSUInteger = 39;
pub const MTL_VERTEX_FORMAT_UCHAR4_NORMALIZED: NSUInteger = 4;

// ─── MTLVertexStepFunction ─────────────────────────────────────────────────

pub const MTL_VERTEX_STEP_FUNCTION_PER_VERTEX: NSUInteger = 1;
pub const MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE: NSUInteger = 2;

// ─── Metal data types (for function constants) ────────────────────────────

pub const MTL_DATA_TYPE_FLOAT: NSUInteger = 3;
pub const MTL_DATA_TYPE_INT: NSUInteger = 29;
pub const MTL_DATA_TYPE_UINT: NSUInteger = 30;
pub const MTL_DATA_TYPE_BOOL: NSUInteger = 53;

// ─── Function constant helpers ─────────────────────────────────────────────

/// MTLFunctionConstantValues setConstantValue:type:atIndex:
pub unsafe fn msg_set_constant_value(
    fcv: Id,
    value_ptr: *const c_void,
    ty: NSUInteger,
    index: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, NSUInteger, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        fcv,
        sel(b"setConstantValue:type:atIndex:\0"),
        value_ptr,
        ty,
        index,
    );
}

/// MTLLibrary newFunctionWithName:constantValues:error:
pub unsafe fn msg_new_function_with_constants(library: Id, name: Id, constants: Id) -> (Id, Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    let mut error: Id = NIL;
    let func = f(
        library,
        sel(b"newFunctionWithName:constantValues:error:\0"),
        name,
        constants,
        &mut error,
    );
    (func, error)
}

// ─── Texture view helpers ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NSRange {
    pub location: u64,
    pub length: u64,
}

/// newTextureViewWithPixelFormat:textureType:levels:slices:
pub unsafe fn msg_new_texture_view(
    texture: Id,
    format: NSUInteger,
    tex_type: NSUInteger,
    levels: NSRange,
    slices: NSRange,
) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, NSUInteger, NSRange, NSRange) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        texture,
        sel(b"newTextureViewWithPixelFormat:textureType:levels:slices:\0"),
        format,
        tex_type,
        levels,
        slices,
    )
}

// ─── Metal visibility result modes (occlusion queries) ─────────────────────

pub const MTL_VISIBILITY_RESULT_MODE_DISABLED: NSUInteger = 0;
pub const MTL_VISIBILITY_RESULT_MODE_BOOLEAN: NSUInteger = 1;
pub const MTL_VISIBILITY_RESULT_MODE_COUNTING: NSUInteger = 2;

/// setVisibilityResultMode:offset: on render encoder
pub unsafe fn msg_set_visibility_result_mode(encoder: Id, mode: NSUInteger, offset: u64) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"setVisibilityResultMode:offset:\0"),
        mode,
        offset,
    )
}

// ─── dispatch_data ──────────────────────────────────────────────────────────

#[link(name = "System", kind = "dylib")]
unsafe extern "C" {
    fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *mut c_void,
        destructor: *mut c_void,
    ) -> *mut c_void;
}
