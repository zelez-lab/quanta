//! `extern "C"` function declarations and ObjC message-send helpers.

use core::ffi::c_void;
use core::mem;

use alloc::boxed::Box;

use super::constants::{BOOL, Class, Id, NIL, NO, NSUInteger, Sel, YES};
use super::structs::{
    COMPLETION_BLOCK_DESCRIPTOR, CompletionBlock, MTLClearColor, MTLOrigin, MTLRegion,
    MTLScissorRect, MTLSize, MTLViewport, NSRange, completion_block_invoke,
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

/// Send message taking one MTLSize, returning Id (e.g.
/// `[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:`).
pub unsafe fn msg_id_mtlsize(obj: Id, name: &[u8], size: MTLSize) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, MTLSize) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), size)
}

/// Send message returning a `*mut f32` (e.g.
/// `MTLRasterizationRateLayerDescriptor.horizontalSampleStorage`).
pub unsafe fn msg_ptr_f32(obj: Id, name: &[u8]) -> *mut f32 {
    let f: unsafe extern "C" fn(Id, Sel) -> *mut f32 =
        mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name))
}

/// Send message taking one NSUInteger, returning BOOL — used for
/// `MTLDevice.supportsRasterizationRateMapWithLayerCount:`.
pub unsafe fn msg_bool_u64(obj: Id, name: &[u8], v: u64) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, u64) -> BOOL =
        mem::transmute(objc_msgSend as *const c_void);
    f(obj, sel(name), v) != NO
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

/// newComputePipelineStateWithFunction(/Descriptor):error: -> Id
///
/// Tries to create a pipeline with `supportIndirectCommandBuffers =
/// YES` first. That flag is required for the resulting pipeline to
/// be usable in `MTLIndirectComputeCommand.setComputePipelineState:`
/// — without it, calling `setComputePipelineState:` on an
/// MTLIndirectComputeCommand is undefined behavior on Apple Silicon
/// (the observable failure is a SIGSEGV in `record_dispatch`,
/// steps 032 + 033).
///
/// Some compute functions (notably ones that read textures) cannot
/// be compiled with the ICB flag set — Metal rejects them with
/// "Compute function cannot be used with indirect command buffers".
/// For those, the pipeline can't be used in an ICB *and* the flag
/// must NOT be set, so we fall back to the flagless form. The
/// resulting pipeline still works for direct dispatch.
pub unsafe fn msg_new_compute_pipeline(device: Id, func: Id) -> (Id, Id) {
    // First attempt: descriptor with supportIndirectCommandBuffers = YES.
    let cls_id = cls(b"MTLComputePipelineDescriptor\0") as Id;
    let alloc: Id = msg_id(cls_id, b"alloc\0");
    let desc: Id = msg_id(alloc, b"init\0");
    msg_void_id(desc, b"setComputeFunction:\0", func);
    msg_void_bool(desc, b"setSupportIndirectCommandBuffers:\0", true);

    let f_desc: unsafe extern "C" fn(Id, Sel, Id, NSUInteger, *mut Id, *mut Id) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    let mut error: Id = NIL;
    let mut reflection: Id = NIL;
    let pipeline = f_desc(
        device,
        sel(b"newComputePipelineStateWithDescriptor:options:reflection:error:\0"),
        desc,
        0, // MTLPipelineOption.none
        &mut reflection,
        &mut error,
    );
    msg_void(desc, b"release\0");

    if !pipeline.is_null() {
        return (pipeline, error);
    }

    // Fallback: function-only form, no ICB support. The error from
    // the first attempt is discarded since the second form is the
    // user-facing path.
    let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    let mut error2: Id = NIL;
    let pipeline = f(
        device,
        sel(b"newComputePipelineStateWithFunction:error:\0"),
        func,
        &mut error2,
    );
    (pipeline, error2)
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

/// copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:
/// toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:
/// on a blit command encoder. Copies one 2D texture region (slice 0,
/// level 0) into a linear buffer — the readback path for GPU-resident
/// (private) render targets, which `getBytes` cannot touch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn msg_copy_texture_to_buffer(
    blit: Id,
    texture: Id,
    origin: MTLOrigin,
    size: MTLSize,
    buffer: Id,
    dst_offset: u64,
    dst_bytes_per_row: u64,
    dst_bytes_per_image: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, Id, u64, u64, MTLOrigin, MTLSize, Id, u64, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        blit,
        sel(b"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:\0"),
        texture,
        0, // sourceSlice
        0, // sourceLevel
        origin,
        size,
        buffer,
        dst_offset,
        dst_bytes_per_row,
        dst_bytes_per_image,
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

// ─── MTLIndirectCommandBuffer (steps 032 + 033) ────────────────────────────

/// Build an MTLIndirectCommandBufferDescriptor, configure for compute
/// dispatch, and return it (autoreleased — caller does not own it).
pub unsafe fn msg_new_icb_descriptor(
    command_types: NSUInteger,
    max_kernel_buffer_bind_count: NSUInteger,
) -> Id {
    let cls_id = cls(b"MTLIndirectCommandBufferDescriptor\0") as Id;
    let alloc: Id = msg_id(cls_id, b"alloc\0");
    let desc: Id = msg_id(alloc, b"init\0");
    msg_void_u64(desc, b"setCommandTypes:\0", command_types);
    msg_void_u64(
        desc,
        b"setMaxKernelBufferBindCount:\0",
        max_kernel_buffer_bind_count,
    );
    // Inherit pipeline + buffers from the parent encoder so we don't
    // have to re-pin every resource per command.
    msg_void_bool(desc, b"setInheritPipelineState:\0", false);
    msg_void_bool(desc, b"setInheritBuffers:\0", false);
    desc
}

/// `[device newIndirectCommandBufferWithDescriptor:maxCommandCount:options:]`
pub unsafe fn msg_new_icb(
    device: Id,
    descriptor: Id,
    max_command_count: NSUInteger,
    options: NSUInteger,
) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id, NSUInteger, NSUInteger) -> Id =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        device,
        sel(b"newIndirectCommandBufferWithDescriptor:maxCommandCount:options:\0"),
        descriptor,
        max_command_count,
        options,
    )
}

/// `[icb indirectComputeCommandAtIndex:]` — returns the command slot
/// the caller writes pipeline + bindings + dispatch into.
pub unsafe fn msg_icb_compute_command_at_index(icb: Id, index: NSUInteger) -> Id {
    msg_id_u64(icb, b"indirectComputeCommandAtIndex:\0", index)
}

/// `[indirectComputeCommand setComputePipelineState:]`
pub unsafe fn msg_icc_set_compute_pipeline(cmd: Id, pipeline: Id) {
    msg_void_id(cmd, b"setComputePipelineState:\0", pipeline)
}

/// `[indirectComputeCommand setKernelBuffer:offset:atIndex:]`
pub unsafe fn msg_icc_set_kernel_buffer(cmd: Id, buffer: Id, offset: u64, index: u64) {
    msg_set_buffer(
        cmd,
        b"setKernelBuffer:offset:atIndex:\0",
        buffer,
        offset,
        index,
    )
}

/// `[indirectComputeCommand concurrentDispatchThreadgroups:threadsPerThreadgroup:]`
pub unsafe fn msg_icc_concurrent_dispatch_threadgroups(
    cmd: Id,
    groups: MTLSize,
    group_size: MTLSize,
) {
    let f: unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        cmd,
        sel(b"concurrentDispatchThreadgroups:threadsPerThreadgroup:\0"),
        groups,
        group_size,
    )
}

/// `[indirectComputeCommand setBarrier]` — marks this command as a
/// barrier point: subsequent commands in the same `executeCommandsInBuffer`
/// won't begin executing until this one completes. Without it,
/// `MTLIndirectCommandTypeConcurrentDispatch` commands within an ICB
/// can run concurrently and observe stale data from earlier commands
/// in the same execute call.
pub unsafe fn msg_icc_set_barrier(cmd: Id) {
    msg_void(cmd, b"setBarrier\0")
}

/// `[encoder executeCommandsInBuffer:withRange:]` — executes a range
/// of recorded commands from an ICB on the active compute encoder.
pub unsafe fn msg_execute_commands_in_buffer(encoder: Id, icb: Id, range: NSRange) {
    let f: unsafe extern "C" fn(Id, Sel, Id, NSRange) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        encoder,
        sel(b"executeCommandsInBuffer:withRange:\0"),
        icb,
        range,
    )
}

/// `[encoder useResource:usage:]` — declare a resource used by ICB
/// commands so the GPU resource hazard tracker sees it. `usage` is
/// MTLResourceUsageRead = 1, Write = 2, Sample = 4.
pub unsafe fn msg_use_resource(encoder: Id, resource: Id, usage: NSUInteger) {
    let f: unsafe extern "C" fn(Id, Sel, Id, NSUInteger) =
        mem::transmute(objc_msgSend as *const c_void);
    f(encoder, sel(b"useResource:usage:\0"), resource, usage)
}

/// `[icb indirectRenderCommandAtIndex:]` — render-path counterpart
/// to `indirectComputeCommandAtIndex:`. Returns the
/// MTLIndirectRenderCommand slot the caller writes pipeline +
/// vertex buffers + draw into.
pub unsafe fn msg_icb_render_command_at_index(icb: Id, index: NSUInteger) -> Id {
    msg_id_u64(icb, b"indirectRenderCommandAtIndex:\0", index)
}

/// `[indirectRenderCommand setRenderPipelineState:]`
pub unsafe fn msg_irc_set_render_pipeline(cmd: Id, pipeline: Id) {
    msg_void_id(cmd, b"setRenderPipelineState:\0", pipeline)
}

/// `[indirectRenderCommand setVertexBuffer:offset:atIndex:]`
pub unsafe fn msg_irc_set_vertex_buffer(cmd: Id, buffer: Id, offset: u64, index: u64) {
    msg_set_buffer(
        cmd,
        b"setVertexBuffer:offset:atIndex:\0",
        buffer,
        offset,
        index,
    )
}

/// `[indirectRenderCommand drawPrimitives:vertexStart:vertexCount:
///   instanceCount:baseInstance:]` — record a non-indexed draw
/// into the ICB slot.
pub unsafe fn msg_irc_draw_primitives(
    cmd: Id,
    primitive_type: NSUInteger,
    vertex_start: u64,
    vertex_count: u64,
    instance_count: u64,
    base_instance: u64,
) {
    let f: unsafe extern "C" fn(Id, Sel, NSUInteger, u64, u64, u64, u64) =
        mem::transmute(objc_msgSend as *const c_void);
    f(
        cmd,
        sel(b"drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:\0"),
        primitive_type,
        vertex_start,
        vertex_count,
        instance_count,
        base_instance,
    )
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
