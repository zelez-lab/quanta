//! `extern "C"` function declarations and ObjC message-send helpers.

use core::ffi::c_void;
use core::mem;

use alloc::boxed::Box;

use super::constants::{BOOL, Class, Id, NIL, NO, NSUInteger, Sel, YES};
use super::structs::{
    COMPLETION_BLOCK_DESCRIPTOR, CompletionBlock, MTLClearColor, MTLRegion, MTLScissorRect,
    MTLSize, MTLViewport, NSRange, completion_block_invoke,
};

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

// ─── dispatch_data ──────────────────────────────────────────────────────────

#[link(name = "System", kind = "dylib")]
unsafe extern "C" {
    pub fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *mut c_void,
        destructor: *mut c_void,
    ) -> *mut c_void;
    pub fn dispatch_semaphore_create(value: isize) -> *mut c_void;
    pub fn dispatch_semaphore_signal(dsema: *mut c_void) -> isize;
    pub fn dispatch_semaphore_wait(dsema: *mut c_void, timeout: u64) -> isize;
    pub fn dispatch_release(object: *mut c_void);
}

// _NSConcreteGlobalBlock — the ISA for global (heap-safe) blocks.
unsafe extern "C" {
    static _NSConcreteGlobalBlock: *const c_void;
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

/// Create a heap-allocated completion block that signals `semaphore` when invoked.
/// Returns a leaked Box pointer — Metal retains the block, and we clean up
/// the semaphore in the Pulse wait_fn.
pub fn make_completion_block(semaphore: *mut c_void) -> *mut CompletionBlock {
    let block = Box::new(CompletionBlock {
        isa: unsafe { _NSConcreteGlobalBlock },
        flags: (1 << 28), // BLOCK_IS_GLOBAL
        reserved: 0,
        invoke: completion_block_invoke,
        descriptor: &COMPLETION_BLOCK_DESCRIPTOR,
        semaphore,
    });
    Box::into_raw(block)
}

/// Send addCompletedHandler: to a command buffer with a block.
pub unsafe fn msg_add_completed_handler(cmd: Id, block: *mut CompletionBlock) {
    let f: unsafe extern "C" fn(Id, Sel, *mut CompletionBlock) =
        mem::transmute(objc_msgSend as *const c_void);
    f(cmd, sel(b"addCompletedHandler:\0"), block);
}
