//! Metal and Objective-C type aliases and constant definitions.

use core::ffi::c_void;

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

// ─── Metal visibility result modes (occlusion queries) ─────────────────────

pub const MTL_VISIBILITY_RESULT_MODE_DISABLED: NSUInteger = 0;
pub const MTL_VISIBILITY_RESULT_MODE_BOOLEAN: NSUInteger = 1;
pub const MTL_VISIBILITY_RESULT_MODE_COUNTING: NSUInteger = 2;

/// Infinite timeout for dispatch_semaphore_wait.
pub const DISPATCH_TIME_FOREVER: u64 = !0;

// ─── MTLIndirectCommandType (steps 032 + 033) ──────────────────────────────
//
// Bitmask passed to MTLIndirectCommandBufferDescriptor.commandTypes.
// We use ConcurrentDispatchThreads to lower compute dispatches into
// the ICB; render variants are documented for the future render-path
// follow-up.

#[allow(dead_code)]
pub const MTL_INDIRECT_COMMAND_TYPE_DRAW: NSUInteger = 1 << 0;
#[allow(dead_code)]
pub const MTL_INDIRECT_COMMAND_TYPE_DRAW_INDEXED: NSUInteger = 1 << 1;
#[allow(dead_code)]
pub const MTL_INDIRECT_COMMAND_TYPE_DRAW_PATCHES: NSUInteger = 1 << 2;
#[allow(dead_code)]
pub const MTL_INDIRECT_COMMAND_TYPE_DRAW_INDEXED_PATCHES: NSUInteger = 1 << 3;
pub const MTL_INDIRECT_COMMAND_TYPE_CONCURRENT_DISPATCH: NSUInteger = 1 << 5;
#[allow(dead_code)]
pub const MTL_INDIRECT_COMMAND_TYPE_CONCURRENT_DISPATCH_THREADS: NSUInteger = 1 << 6;
