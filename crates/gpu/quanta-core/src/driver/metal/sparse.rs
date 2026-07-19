//! Native Metal sparse-texture residency.
//!
//! Apple Silicon family 7+ exposes sparse textures via:
//! - `MTLDevice.sparseTileSizeWithTextureType:pixelFormat:sampleCount:`
//!   for the per-tile pixel granularity.
//! - `MTLHeap` of type `placement` with private storage as the
//!   backing physical-page reserve.
//! - `MTLTexture` allocated from the heap (uses heap memory
//!   lazily — pages are committed only when explicitly mapped).
//! - `MTLResourceStateCommandEncoder.updateTextureMapping:mode:
//!   region:mipLevel:slice:` to map / unmap individual tiles
//!   from a command buffer.
//!
//! `sparse_map_tile` and `sparse_unmap_tile` both go through
//! `sparse_update_tile`, which submits a one-shot resource state
//! command buffer and waits for completion. The `_backing` arg
//! the typed wrapper threads through (T7602) is unused on Metal:
//! the heap supplies the physical pages, not a caller-owned
//! buffer. The wrapper still records its (mip, x, y) → backing
//! mapping in its own HashMap, satisfying the contract.

use alloc::format;
use core::ffi::c_void;

use crate::{QuantaError, TextureDesc, TextureUsage};

use super::device::{MetalDevice, MetalSparseTexture};
use super::device_impl::format_to_metal;
use super::ffi;

impl MetalDevice {
    pub(crate) fn sparse_texture_create_native(
        &self,
        desc: &TextureDesc,
    ) -> Result<u64, QuantaError> {
        // Sparse residency native today supports 2D color textures
        // only. 3D / cube / array sparse textures need a slice-aware
        // tile-key shape (sparse_update_tile passes slice = 0); wire
        // them once the 2D path is stable. Also gate multi-mip —
        // partial-residency across mips needs the mip-tail handling
        // that mirrors Vulkan's image-sparse-memory-requirements.
        if !matches!(desc.kind, crate::TextureKind::D2) {
            return Err(QuantaError::not_supported(
                "Metal sparse residency native: only 2D textures are supported today (3D / Cube / Array pending)",
            ));
        }
        if desc.mip_levels > 1 {
            return Err(QuantaError::not_supported(
                "Metal sparse residency native: only single-mip textures are supported today",
            ));
        }
        unsafe {
            let format_code = format_to_metal(desc.format);
            // Tile size in pixels. Per Apple, this depends on
            // texture type, format, and sample count.
            let f: unsafe extern "C" fn(
                ffi::Id,
                ffi::Sel,
                ffi::NSUInteger, // textureType
                ffi::NSUInteger, // pixelFormat
                ffi::NSUInteger, // sampleCount
            ) -> ffi::MTLSize = core::mem::transmute(ffi::objc_msgSend as *const c_void);
            let tile = f(
                self.device,
                ffi::sel(b"sparseTileSizeWithTextureType:pixelFormat:sampleCount:\0"),
                ffi::MTL_TEXTURE_TYPE_2D,
                format_code,
                desc.sample_count.max(1) as ffi::NSUInteger,
            );
            if tile.width == 0 || tile.height == 0 {
                return Err(QuantaError::not_supported(
                    "Metal returned zero sparse tile size — sparse residency may not be supported for this format",
                ));
            }

            // Build the texture descriptor first — we need to ask
            // the device for its heap size + alignment requirement
            // before we can size the heap correctly.
            let mtl_desc = ffi::msg_id(ffi::cls(b"MTLTextureDescriptor\0") as ffi::Id, b"new\0");
            ffi::msg_void_u64(mtl_desc, b"setWidth:\0", desc.width as u64);
            ffi::msg_void_u64(mtl_desc, b"setHeight:\0", desc.height as u64);
            ffi::msg_void_u64(mtl_desc, b"setPixelFormat:\0", format_code);
            ffi::msg_void_u64(
                mtl_desc,
                b"setSampleCount:\0",
                desc.sample_count.max(1) as u64,
            );
            let mut usage: ffi::NSUInteger = 0;
            if desc.usage.has(TextureUsage::SHADER_READ) {
                usage |= ffi::MTL_TEXTURE_USAGE_SHADER_READ;
            }
            if desc.usage.has(TextureUsage::STORAGE) {
                // Same grant as texture_create_impl: STORAGE licenses both
                // access::read and access::read_write texel binding.
                usage |= ffi::MTL_TEXTURE_USAGE_SHADER_READ | ffi::MTL_TEXTURE_USAGE_SHADER_WRITE;
            }
            if desc.usage.has(TextureUsage::RENDER_TARGET) {
                usage |= ffi::MTL_TEXTURE_USAGE_RENDER_TARGET;
            }
            if usage == 0 {
                usage = ffi::MTL_TEXTURE_USAGE_SHADER_READ;
            }
            ffi::msg_void_u64(mtl_desc, b"setUsage:\0", usage);
            ffi::msg_void_u64(
                mtl_desc,
                b"setStorageMode:\0",
                ffi::MTL_STORAGE_MODE_PRIVATE,
            );
            ffi::msg_void_u64(
                mtl_desc,
                b"setHazardTrackingMode:\0",
                1, // MTLHazardTrackingMode.untracked (matches placement-heap requirement)
            );
            if desc.mip_levels > 1 {
                ffi::msg_void_u64(mtl_desc, b"setMipmapLevelCount:\0", desc.mip_levels as u64);
            }

            // Ask the device how big the heap must be to host this
            // sparse texture. `heapTextureSizeAndAlignWithDescriptor:`
            // returns a `MTLSizeAndAlign { size, align }` (two
            // NSUInteger fields). Aligning to `align` and adding
            // `size` gives the minimum heap footprint.
            #[repr(C)]
            #[derive(Default, Copy, Clone)]
            struct MTLSizeAndAlign {
                size: u64,
                align: u64,
            }
            let f_align: unsafe extern "C" fn(ffi::Id, ffi::Sel, ffi::Id) -> MTLSizeAndAlign =
                core::mem::transmute(ffi::objc_msgSend as *const c_void);
            let san = f_align(
                self.device,
                ffi::sel(b"heapTextureSizeAndAlignWithDescriptor:\0"),
                mtl_desc,
            );
            if san.size == 0 {
                ffi::msg_void(mtl_desc, b"release\0");
                return Err(QuantaError::not_supported(
                    "Metal device couldn't size a heap for the requested sparse texture",
                ));
            }
            let heap_size = pad_to(san.size, san.align.max(1));

            // Build the heap descriptor.
            let heap_desc = ffi::msg_id(ffi::cls(b"MTLHeapDescriptor\0") as ffi::Id, b"new\0");
            ffi::msg_void_u64(heap_desc, b"setSize:\0", heap_size);
            ffi::msg_void_u64(heap_desc, b"setType:\0", ffi::MTL_HEAP_TYPE_PLACEMENT);
            ffi::msg_void_u64(
                heap_desc,
                b"setStorageMode:\0",
                ffi::MTL_STORAGE_MODE_PRIVATE,
            );
            ffi::msg_void_u64(
                heap_desc,
                b"setHazardTrackingMode:\0",
                1, // .untracked — matches the texture descriptor below
            );
            let heap = ffi::msg_id_id(self.device, b"newHeapWithDescriptor:\0", heap_desc);
            ffi::msg_void(heap_desc, b"release\0");
            if heap.is_null() {
                ffi::msg_void(mtl_desc, b"release\0");
                return Err(QuantaError::out_of_memory().with_context(&format!(
                    "newHeapWithDescriptor (sparse, size {heap_size}) returned null"
                )));
            }

            // Allocate texture from the placement heap. Placement
            // heaps require an explicit offset (vs automatic heaps
            // which manage placement themselves) — selector is
            // `newTextureWithDescriptor:offset:`. Metal also
            // requires the descriptor's hazard tracking + storage
            // mode to match the heap.
            ffi::msg_void_u64(
                mtl_desc,
                b"setHazardTrackingMode:\0",
                1, // MTLHazardTrackingMode.untracked
            );
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, ffi::Id, ffi::NSUInteger) -> ffi::Id =
                core::mem::transmute(ffi::objc_msgSend as *const c_void);
            let texture = f(
                heap,
                ffi::sel(b"newTextureWithDescriptor:offset:\0"),
                mtl_desc,
                0, // offset
            );
            ffi::msg_void(mtl_desc, b"release\0");
            if texture.is_null() {
                ffi::msg_void(heap, b"release\0");
                return Err(QuantaError::out_of_memory().with_context(
                    "heap.newTextureWithDescriptor returned null for sparse texture",
                ));
            }

            let handle = self.alloc_handle();
            self.textures
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, texture);
            self.sparse_textures
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(
                    handle,
                    MetalSparseTexture {
                        heap,
                        tile_w: tile.width,
                        tile_h: tile.height,
                    },
                );

            Ok(handle)
        }
    }

    pub(crate) fn sparse_update_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
        map: bool,
    ) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mtl_tex = *textures
            .get(&texture)
            .ok_or_else(|| QuantaError::not_found("sparse texture handle not found"))?;
        drop(textures);

        let sparse = self
            .sparse_textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let entry = sparse.get(&texture).ok_or_else(|| {
            QuantaError::invalid_param("sparse_update_tile called on a non-sparse texture")
        })?;
        let tile_w = entry.tile_w;
        let tile_h = entry.tile_h;
        drop(sparse);

        let region = ffi::MTLRegion {
            origin: ffi::MTLOrigin {
                x: (x as u64) * tile_w,
                y: (y as u64) * tile_h,
                z: 0,
            },
            size: ffi::MTLSize {
                width: tile_w,
                height: tile_h,
                depth: 1,
            },
        };
        let mode: ffi::NSUInteger = if map {
            ffi::MTL_SPARSE_TEXTURE_MAPPING_MODE_MAP
        } else {
            ffi::MTL_SPARSE_TEXTURE_MAPPING_MODE_UNMAP
        };

        unsafe {
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let encoder = ffi::msg_id(cmd, b"resourceStateCommandEncoder\0");
            if encoder.is_null() {
                return Err(QuantaError::not_supported(
                    "resourceStateCommandEncoder is unavailable on this device",
                ));
            }
            // updateTextureMapping:mode:region:mipLevel:slice:
            // 5-arg method: id<MTLTexture>, NSUInteger, MTLRegion,
            // NSUInteger, NSUInteger.
            let f: unsafe extern "C" fn(
                ffi::Id,
                ffi::Sel,
                ffi::Id,
                ffi::NSUInteger,
                ffi::MTLRegion,
                ffi::NSUInteger,
                ffi::NSUInteger,
            ) = core::mem::transmute(ffi::objc_msgSend as *const c_void);
            f(
                encoder,
                ffi::sel(b"updateTextureMapping:mode:region:mipLevel:slice:\0"),
                mtl_tex,
                mode,
                region,
                mip as ffi::NSUInteger,
                0, // slice (single-layer 2D for now)
            );
            ffi::msg_void(encoder, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");
        }
        Ok(())
    }
}

#[inline]
fn pad_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}
