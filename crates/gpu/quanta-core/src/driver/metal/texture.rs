//! Texture and sampler operations for Metal.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{QuantaError, Texture, TextureDesc, TextureUsage};

use super::MetalDevice;
use super::ffi;
use super::{
    address_to_metal, compare_op_to_metal, filter_to_metal, format_bytes_per_pixel, format_to_metal,
};

impl MetalDevice {
    pub(crate) fn texture_create_impl(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        unsafe {
            let mtl_desc = ffi::msg_id(ffi::cls(b"MTLTextureDescriptor\0") as ffi::Id, b"new\0");
            ffi::msg_void_u64(mtl_desc, b"setWidth:\0", desc.width as u64);
            ffi::msg_void_u64(mtl_desc, b"setHeight:\0", desc.height as u64);
            ffi::msg_void_u64(mtl_desc, b"setPixelFormat:\0", format_to_metal(desc.format));
            ffi::msg_void_u64(mtl_desc, b"setSampleCount:\0", desc.sample_count as u64);

            let mut usage: ffi::NSUInteger = 0;
            if desc.usage.has(TextureUsage::SHADER_READ) {
                usage |= ffi::MTL_TEXTURE_USAGE_SHADER_READ;
            }
            if desc.usage.has(TextureUsage::SHADER_WRITE) {
                usage |= ffi::MTL_TEXTURE_USAGE_SHADER_WRITE;
            }
            if desc.usage.has(TextureUsage::RENDER_TARGET) {
                usage |= ffi::MTL_TEXTURE_USAGE_RENDER_TARGET;
            }
            if usage == 0 {
                usage = ffi::MTL_TEXTURE_USAGE_SHADER_READ;
            }
            ffi::msg_void_u64(mtl_desc, b"setUsage:\0", usage);

            // Storage mode. Render targets are GPU-resident:
            // `StorageModePrivate` unconditionally, whether or not they are
            // also sampled (sampling a private texture is fully legal). The
            // iOS simulator *requires* private storage for render targets —
            // a shared render target composites as silently empty there —
            // and private is the correct, faster placement on every Metal
            // GPU. CPU access to a private render target goes through the
            // staging-blit read path below; direct CPU writes are rejected
            // (see `texture_write_region_impl`). Non-render-target textures
            // stay shared so `getBytes`/`replaceRegion` round-trip directly
            // (the glyph-atlas `write_region` hot path depends on it).
            if desc.usage.has(TextureUsage::RENDER_TARGET) {
                ffi::msg_void_u64(
                    mtl_desc,
                    b"setStorageMode:\0",
                    ffi::MTL_STORAGE_MODE_PRIVATE,
                );
            } else {
                ffi::msg_void_u64(mtl_desc, b"setStorageMode:\0", ffi::MTL_STORAGE_MODE_SHARED);
            }

            // Texture type
            use crate::TextureKind;
            match desc.kind {
                TextureKind::D2 if desc.sample_count > 1 => {
                    ffi::msg_void_u64(
                        mtl_desc,
                        b"setTextureType:\0",
                        ffi::MTL_TEXTURE_TYPE_2D_MULTISAMPLE,
                    );
                }
                TextureKind::D2 => {} // default
                TextureKind::D3 => {
                    ffi::msg_void_u64(mtl_desc, b"setTextureType:\0", ffi::MTL_TEXTURE_TYPE_3D);
                    ffi::msg_void_u64(mtl_desc, b"setDepth:\0", desc.depth as u64);
                }
                TextureKind::Cube => {
                    ffi::msg_void_u64(mtl_desc, b"setTextureType:\0", ffi::MTL_TEXTURE_TYPE_CUBE);
                }
                TextureKind::Array2D => {
                    ffi::msg_void_u64(
                        mtl_desc,
                        b"setTextureType:\0",
                        ffi::MTL_TEXTURE_TYPE_2D_ARRAY,
                    );
                    ffi::msg_void_u64(mtl_desc, b"setArrayLength:\0", desc.array_length as u64);
                }
                TextureKind::ArrayCube => {
                    ffi::msg_void_u64(
                        mtl_desc,
                        b"setTextureType:\0",
                        ffi::MTL_TEXTURE_TYPE_CUBE_ARRAY,
                    );
                    ffi::msg_void_u64(mtl_desc, b"setArrayLength:\0", desc.array_length as u64);
                }
            }

            if desc.mip_levels > 1 {
                ffi::msg_void_u64(mtl_desc, b"setMipmapLevelCount:\0", desc.mip_levels as u64);
            } else if desc.mip_levels == 0 {
                let max_dim = desc.width.max(desc.height);
                let levels = (max_dim as f32).log2().floor() as u64 + 1;
                ffi::msg_void_u64(mtl_desc, b"setMipmapLevelCount:\0", levels);
            }

            let tex = ffi::msg_id_id(self.device, b"newTextureWithDescriptor:\0", mtl_desc);
            let handle = self.alloc_handle();
            self.textures
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, tex);
            self.texture_formats
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, desc.format);
            self.texture_usages
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, desc.usage);

            Ok(Texture {
                handle,
                width: desc.width,
                height: desc.height,
                format: desc.format,
                device: None,
                live: true,
            })
        }
    }

    pub(crate) fn texture_write_impl(
        &self,
        texture: &Texture,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        self.texture_write_region_impl(texture, (0, 0), (texture.width(), texture.height()), data)
    }

    pub(crate) fn texture_write_region_impl(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        size: (u32, u32),
        data: &[u8],
    ) -> Result<(), QuantaError> {
        // Render targets are GPU-resident (private storage) on Metal; a CPU
        // `replaceRegion` against private storage does not land, so reject it
        // with a clear pointer to the supported paths rather than silently
        // dropping the write. Uploading pixels into a render target needs a
        // staging blit (write to a shared buffer, blit into the texture) —
        // not yet wired — so callers should write to a non-render-target
        // texture or render the content in instead.
        if self.texture_is_render_target(texture.handle()) {
            return Err(QuantaError::not_supported(
                "render targets are GPU-resident (private storage) on Metal; \
                 CPU writes need a staging blit — write to a non-render-target \
                 texture or render into it instead",
            )
            .with_context(&format!("texture_write: handle {}", texture.handle())));
        }
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_write: handle {}", texture.handle()))
        })?;
        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let region = ffi::MTLRegion::new_2d(
            origin.0 as u64,
            origin.1 as u64,
            size.0 as u64,
            size.1 as u64,
        );
        // bytesPerRow describes the SOURCE data: one tightly packed
        // region row, not a full texture row.
        let bytes_per_row = size.0 as u64 * bytes_per_pixel as u64;
        unsafe {
            ffi::msg_replace_region(*tex, region, 0, data.as_ptr() as *const _, bytes_per_row);
        }
        Ok(())
    }

    /// True if the texture handle was created with `RENDER_TARGET` usage
    /// and therefore lives in `StorageModePrivate` — the CPU cannot reach
    /// it with `getBytes`/`replaceRegion`, so reads route through a staging
    /// blit and direct writes are rejected. A missing handle answers
    /// `false` (the caller's own handle check produces the real error).
    fn texture_is_render_target(&self, handle: u64) -> bool {
        self.texture_usages
            .read()
            .ok()
            .and_then(|u| u.get(&handle).copied())
            .is_some_and(|usage| usage.has(TextureUsage::RENDER_TARGET))
    }

    pub(crate) fn texture_read_impl(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        // Render targets are private (GPU-resident) storage, which `getBytes`
        // cannot touch — copy the texture into a shared staging buffer on a
        // blit encoder, then read the buffer. Shared (non-render-target)
        // textures keep the direct `getBytes` read. Readback is thus
        // usage-driven, not conditioned on the exact storage decision, so it
        // covers both the render-target class and any sampled render target.
        if self.texture_is_render_target(texture.handle()) {
            return self.texture_read_via_staging(texture);
        }
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_read: handle {}", texture.handle()))
        })?;
        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let size = (texture.width() * texture.height()) as usize * bytes_per_pixel;
        let mut result = vec![0u8; size];
        let region = ffi::MTLRegion::new_2d(0, 0, texture.width() as u64, texture.height() as u64);
        let bytes_per_row = texture.width() as u64 * bytes_per_pixel as u64;
        unsafe {
            ffi::msg_get_bytes(
                *tex,
                result.as_mut_ptr() as *mut _,
                bytes_per_row,
                region,
                0,
            );
        }
        Ok(result)
    }

    /// Read a private (render-target) texture back to the CPU via a
    /// texture→buffer blit into a shared staging buffer.
    ///
    /// `copyFromTexture:toBuffer:` wants the destination row pitch aligned to
    /// the format's linear alignment (up to 256 bytes on some Metal GPUs), so
    /// the staging buffer is allocated with a padded pitch and the padding is
    /// stripped as the tightly-packed result is assembled. The blit is
    /// committed and waited on before the buffer is read; higher layers are
    /// still responsible for ordering the producing render/dispatch ahead of
    /// this call (see `Texture::read`).
    fn texture_read_via_staging(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = *textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_read: handle {}", texture.handle()))
        })?;
        drop(textures);

        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let width = texture.width() as usize;
        let height = texture.height() as usize;
        let tight_row = width * bytes_per_pixel;
        // Align the staging row pitch up to 256 bytes — the conservative
        // Metal linear-texture alignment that holds on every GPU family.
        const ROW_ALIGN: usize = 256;
        let padded_row = tight_row.div_ceil(ROW_ALIGN) * ROW_ALIGN;
        let staging_len = padded_row * height;

        let mut result = vec![0u8; tight_row * height];
        unsafe {
            let staging = ffi::msg_new_buffer(
                self.device,
                staging_len as u64,
                ffi::MTL_RESOURCE_STORAGE_MODE_SHARED,
            );
            if staging.is_null() {
                return Err(QuantaError::out_of_memory().with_context(
                    "texture_read: staging buffer allocation for render-target readback failed",
                ));
            }
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let blit = ffi::msg_id(cmd, b"blitCommandEncoder\0");
            ffi::msg_copy_texture_to_buffer(
                blit,
                tex,
                ffi::MTLOrigin { x: 0, y: 0, z: 0 },
                ffi::MTLSize::new(texture.width() as u64, texture.height() as u64, 1),
                staging,
                0,
                padded_row as u64,
                staging_len as u64,
            );
            ffi::msg_void(blit, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");

            let contents = ffi::msg_ptr(staging, b"contents\0");
            // Strip the row padding: copy `tight_row` bytes out of each
            // `padded_row`-strided source row.
            for y in 0..height {
                core::ptr::copy_nonoverlapping(
                    contents.add(y * padded_row),
                    result.as_mut_ptr().add(y * tight_row),
                    tight_row,
                );
            }
            ffi::msg_void(staging, b"release\0");
        }
        Ok(result)
    }

    pub(crate) fn sampler_create_impl(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        unsafe {
            let sd = ffi::msg_id(ffi::cls(b"MTLSamplerDescriptor\0") as ffi::Id, b"new\0");
            ffi::msg_void_u64(sd, b"setMinFilter:\0", filter_to_metal(desc.min_filter));
            ffi::msg_void_u64(sd, b"setMagFilter:\0", filter_to_metal(desc.mag_filter));
            let mip_filter = match desc.mip_filter {
                crate::texture::Filter::Nearest => ffi::MTL_SAMPLER_MIP_FILTER_NEAREST,
                crate::texture::Filter::Linear => ffi::MTL_SAMPLER_MIP_FILTER_LINEAR,
            };
            ffi::msg_void_u64(sd, b"setMipFilter:\0", mip_filter);
            ffi::msg_void_u64(sd, b"setSAddressMode:\0", address_to_metal(desc.address_u));
            ffi::msg_void_u64(sd, b"setTAddressMode:\0", address_to_metal(desc.address_v));
            ffi::msg_void_u64(sd, b"setMaxAnisotropy:\0", desc.max_anisotropy as u64);

            // Comparison function for depth/shadow samplers
            if let Some(cmp) = desc.compare {
                ffi::msg_void_u64(sd, b"setCompareFunction:\0", compare_op_to_metal(cmp));
            }

            let sampler = ffi::msg_id_id(self.device, b"newSamplerStateWithDescriptor:\0", sd);
            let handle = self.alloc_handle();
            self.samplers
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, sampler);
            Ok(crate::Sampler {
                handle,
                device: None,
                live: true,
            })
        }
    }

    pub(crate) fn generate_mipmaps_impl(&self, texture: &Texture) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("generate_mipmaps: handle {}", texture.handle()))
        })?;
        unsafe {
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let blit = ffi::msg_id(cmd, b"blitCommandEncoder\0");
            ffi::msg_generate_mipmaps(blit, *tex);
            ffi::msg_void(blit, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");
        }
        Ok(())
    }
}
