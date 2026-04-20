//! Texture and sampler operations for Metal.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{QuantaError, Texture, TextureDesc, TextureUsage};
use metal as mtl;

use super::MetalDevice;
use super::{address_to_metal, filter_to_metal, format_bytes_per_pixel, format_to_metal};

impl MetalDevice {
    pub(crate) fn texture_create_impl(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let mtl_desc = mtl::TextureDescriptor::new();
        mtl_desc.set_width(desc.width as u64);
        mtl_desc.set_height(desc.height as u64);
        mtl_desc.set_pixel_format(format_to_metal(desc.format));
        mtl_desc.set_sample_count(desc.sample_count as u64);

        let mut usage = mtl::MTLTextureUsage::empty();
        if desc.usage.has(TextureUsage::SHADER_READ) {
            usage |= mtl::MTLTextureUsage::ShaderRead;
        }
        if desc.usage.has(TextureUsage::SHADER_WRITE) {
            usage |= mtl::MTLTextureUsage::ShaderWrite;
        }
        if desc.usage.has(TextureUsage::RENDER_TARGET) {
            usage |= mtl::MTLTextureUsage::RenderTarget;
        }
        if usage.is_empty() {
            usage = mtl::MTLTextureUsage::ShaderRead;
        }
        mtl_desc.set_usage(usage);

        // Storage mode: Private for render-only, Shared if CPU needs access
        if desc.usage.has(TextureUsage::RENDER_TARGET) && !desc.usage.has(TextureUsage::SHADER_READ)
        {
            mtl_desc.set_storage_mode(mtl::MTLStorageMode::Private);
        } else {
            mtl_desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        }

        // Texture type
        use crate::TextureKind;
        match desc.kind {
            TextureKind::D2 if desc.sample_count > 1 => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::D2Multisample);
            }
            TextureKind::D2 => {} // default
            TextureKind::D3 => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::D3);
                mtl_desc.set_depth(desc.depth as u64);
            }
            TextureKind::Cube => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::Cube);
            }
            TextureKind::Array2D => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::D2Array);
                mtl_desc.set_array_length(desc.array_length as u64);
            }
            TextureKind::ArrayCube => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::CubeArray);
                mtl_desc.set_array_length(desc.array_length as u64);
            }
        }

        if desc.mip_levels > 1 {
            mtl_desc.set_mipmap_level_count(desc.mip_levels as u64);
        } else if desc.mip_levels == 0 {
            // Auto-calculate: floor(log2(max(w,h))) + 1
            let max_dim = desc.width.max(desc.height);
            let levels = (max_dim as f32).log2().floor() as u64 + 1;
            mtl_desc.set_mipmap_level_count(levels);
        }

        let tex = self.device.new_texture(&mtl_desc);
        let handle = self.alloc_handle();
        self.textures.lock().unwrap().insert(handle, tex);

        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            drop_fn: None,
        })
    }

    pub(crate) fn texture_write_impl(
        &self,
        texture: &Texture,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_write: handle {}", texture.handle()))
        })?;
        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let region = mtl::MTLRegion::new_2d(0, 0, texture.width() as u64, texture.height() as u64);
        let bytes_per_row = texture.width() as u64 * bytes_per_pixel as u64;
        tex.replace_region(region, 0, data.as_ptr() as *const _, bytes_per_row);
        Ok(())
    }

    pub(crate) fn texture_read_impl(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_read: handle {}", texture.handle()))
        })?;
        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let size = (texture.width() * texture.height()) as usize * bytes_per_pixel;
        let mut result = vec![0u8; size];
        let region = mtl::MTLRegion::new_2d(0, 0, texture.width() as u64, texture.height() as u64);
        let bytes_per_row = texture.width() as u64 * bytes_per_pixel as u64;
        tex.get_bytes(result.as_mut_ptr() as *mut _, bytes_per_row, region, 0);
        Ok(result)
    }

    pub(crate) fn sampler_create_impl(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let sd = mtl::SamplerDescriptor::new();
        sd.set_min_filter(filter_to_metal(desc.min_filter));
        sd.set_mag_filter(filter_to_metal(desc.mag_filter));
        sd.set_mip_filter(match desc.mip_filter {
            crate::render_pass::Filter::Nearest => mtl::MTLSamplerMipFilter::Nearest,
            crate::render_pass::Filter::Linear => mtl::MTLSamplerMipFilter::Linear,
        });
        sd.set_address_mode_s(address_to_metal(desc.address_u));
        sd.set_address_mode_t(address_to_metal(desc.address_v));
        sd.set_max_anisotropy(desc.max_anisotropy as u64);
        let _sampler = self.device.new_sampler(&sd);
        let handle = self.alloc_handle();
        // TODO: store sampler for binding
        Ok(crate::Sampler {
            handle,
            drop_fn: None,
        })
    }

    pub(crate) fn generate_mipmaps_impl(&self, texture: &Texture) -> Result<(), QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("generate_mipmaps: handle {}", texture.handle()))
        })?;
        let cmd = self.queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.generate_mipmaps(tex);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }
}
