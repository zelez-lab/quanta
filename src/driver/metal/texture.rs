//! Texture and sampler operations for Metal.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{QuantaError, Texture, TextureDesc, TextureUsage};

use super::MetalDevice;
use super::ffi;
use super::{address_to_metal, filter_to_metal, format_bytes_per_pixel, format_to_metal};

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

            // Storage mode
            if desc.usage.has(TextureUsage::RENDER_TARGET)
                && !desc.usage.has(TextureUsage::SHADER_READ)
            {
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
            let handle = self.alloc_handle()?;
            self.textures
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, tex);

            Ok(Texture {
                handle,
                width: desc.width,
                height: desc.height,
                format: desc.format,
                drop_fn: None,
            })
        }
    }

    pub(crate) fn texture_write_impl(
        &self,
        texture: &Texture,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_write: handle {}", texture.handle()))
        })?;
        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let region = ffi::MTLRegion::new_2d(0, 0, texture.width() as u64, texture.height() as u64);
        let bytes_per_row = texture.width() as u64 * bytes_per_pixel as u64;
        unsafe {
            ffi::msg_replace_region(*tex, region, 0, data.as_ptr() as *const _, bytes_per_row);
        }
        Ok(())
    }

    pub(crate) fn texture_read_impl(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let textures = self
            .textures
            .lock()
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

    pub(crate) fn sampler_create_impl(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        unsafe {
            let sd = ffi::msg_id(ffi::cls(b"MTLSamplerDescriptor\0") as ffi::Id, b"new\0");
            ffi::msg_void_u64(sd, b"setMinFilter:\0", filter_to_metal(desc.min_filter));
            ffi::msg_void_u64(sd, b"setMagFilter:\0", filter_to_metal(desc.mag_filter));
            let mip_filter = match desc.mip_filter {
                crate::render_pass::Filter::Nearest => ffi::MTL_SAMPLER_MIP_FILTER_NEAREST,
                crate::render_pass::Filter::Linear => ffi::MTL_SAMPLER_MIP_FILTER_LINEAR,
            };
            ffi::msg_void_u64(sd, b"setMipFilter:\0", mip_filter);
            ffi::msg_void_u64(sd, b"setSAddressMode:\0", address_to_metal(desc.address_u));
            ffi::msg_void_u64(sd, b"setTAddressMode:\0", address_to_metal(desc.address_v));
            ffi::msg_void_u64(sd, b"setMaxAnisotropy:\0", desc.max_anisotropy as u64);
            let sampler = ffi::msg_id_id(self.device, b"newSamplerStateWithDescriptor:\0", sd);
            let handle = self.alloc_handle()?;
            self.samplers
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, sampler);
            Ok(crate::Sampler {
                handle,
                drop_fn: None,
            })
        }
    }

    pub(crate) fn generate_mipmaps_impl(&self, texture: &Texture) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .lock()
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
