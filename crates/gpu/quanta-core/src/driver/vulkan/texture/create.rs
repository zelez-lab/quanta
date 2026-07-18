//! Texture and sampler creation for Vulkan.

use alloc::format;

use crate::{Format, QuantaError, Texture, TextureDesc, TextureUsage};

use super::super::ffi;
use super::super::{VkTexture, VulkanDevice, format_to_vulkan, sample_count_to_vk};

impl VulkanDevice {
    pub(crate) fn texture_create_impl(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let vk_format = format_to_vulkan(desc.format);

        let mut vk_usage =
            ffi::VK_IMAGE_USAGE_TRANSFER_SRC_BIT | ffi::VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if desc.usage.has(TextureUsage::SHADER_READ) {
            vk_usage |= ffi::VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if desc.usage.has(TextureUsage::SHADER_WRITE) {
            vk_usage |= ffi::VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if desc.usage.has(TextureUsage::RENDER_TARGET) {
            if matches!(desc.format, Format::Depth32Float) {
                vk_usage |= ffi::VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            } else {
                vk_usage |= ffi::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            }
        }

        let image_info = ffi::VkImageCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            image_type: ffi::VK_IMAGE_TYPE_2D,
            format: vk_format,
            extent: ffi::VkExtent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth.max(1),
            },
            mip_levels: desc.mip_levels.max(1),
            array_layers: desc.array_length.max(1),
            samples: sample_count_to_vk(desc.sample_count),
            tiling: ffi::VK_IMAGE_TILING_OPTIMAL,
            usage: vk_usage,
            sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
            initial_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
        };

        let mut image = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateImage(self.device, &image_info, core::ptr::null(), &mut image) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetImageMemoryRequirements(self.device, image, &mut mem_reqs) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            ffi::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )?;
        let alloc = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
        };
        let mut memory = ffi::null_handle();
        let result =
            unsafe { ffi::vkAllocateMemory(self.device, &alloc, core::ptr::null(), &mut memory) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }
        let result = unsafe { ffi::vkBindImageMemory(self.device, image, memory, 0) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let aspect = if matches!(desc.format, Format::Depth32Float) {
            ffi::VK_IMAGE_ASPECT_DEPTH_BIT
        } else {
            ffi::VK_IMAGE_ASPECT_COLOR_BIT
        };

        let view_info = ffi::VkImageViewCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            image,
            view_type: ffi::VK_IMAGE_VIEW_TYPE_2D,
            format: vk_format,
            components: ffi::VkComponentMapping::default(),
            subresource_range: ffi::VkImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: desc.mip_levels.max(1),
                base_array_layer: 0,
                layer_count: desc.array_length.max(1),
            },
        };

        let mut view = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateImageView(self.device, &view_info, core::ptr::null(), &mut view)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let handle = self.alloc_handle();
        self.textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                VkTexture {
                    image,
                    view,
                    memory,
                    width: desc.width,
                    height: desc.height,
                    format: vk_format,
                    mip_levels: desc.mip_levels.max(1),
                    current_layout: std::sync::atomic::AtomicU32::new(
                        ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                    ),
                },
            );

        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            sample_count: desc.sample_count,
            device: None,
            live: true,
        })
    }

    /// Create a sparse-residency `VkImage` — step 063 slice 21.
    ///
    /// Differs from `texture_create_impl` in three ways:
    /// 1. The image flags include `VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
    ///    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT` so the driver knows
    ///    memory backing arrives later via `vkQueueBindSparse`.
    /// 2. No `vkAllocateMemory` / `vkBindImageMemory` — sparse images
    ///    forbid that path. Tile memory is allocated lazily inside
    ///    `sparse_map_tile`.
    /// 3. No image-view creation — without bound memory, sampling
    ///    or rendering would fail. The view can be lazily created
    ///    when the first tile is mapped.
    ///
    /// The returned `VkTexture` has `memory = null_handle()` and
    /// `view = null_handle()`. The destroy path tolerates both.
    pub(crate) fn sparse_image_create_impl(
        &self,
        desc: &TextureDesc,
    ) -> Result<Texture, QuantaError> {
        // Sparse residency native today supports 2D color textures
        // only. 3D / array / cube sparse textures use a different
        // tile-key shape (need mip + slice/depth in the binding
        // registry) and per-face alignment rules; wire them once
        // the 2D path is stable. Also gate multi-mip — partial
        // residency across mips needs the mip-tail metadata that
        // `vkGetImageSparseMemoryRequirements` returns.
        if !matches!(desc.kind, crate::TextureKind::D2) {
            return Err(QuantaError::not_supported(
                "Vulkan sparse residency native: only 2D textures are supported today (3D / Array / Cube pending)",
            ));
        }
        if desc.mip_levels > 1 {
            return Err(QuantaError::not_supported(
                "Vulkan sparse residency native: only single-mip textures are supported today (mip-tail handling pending)",
            ));
        }
        let vk_format = format_to_vulkan(desc.format);

        let mut vk_usage =
            ffi::VK_IMAGE_USAGE_TRANSFER_SRC_BIT | ffi::VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if desc.usage.has(TextureUsage::SHADER_READ) {
            vk_usage |= ffi::VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if desc.usage.has(TextureUsage::SHADER_WRITE) {
            vk_usage |= ffi::VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if desc.usage.has(TextureUsage::RENDER_TARGET) {
            if matches!(desc.format, Format::Depth32Float) {
                vk_usage |= ffi::VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            } else {
                vk_usage |= ffi::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            }
        }

        let image_info = ffi::VkImageCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_IMAGE_CREATE_SPARSE_BINDING_BIT
                | ffi::VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
            image_type: ffi::VK_IMAGE_TYPE_2D,
            format: vk_format,
            extent: ffi::VkExtent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth.max(1),
            },
            mip_levels: desc.mip_levels.max(1),
            array_layers: desc.array_length.max(1),
            samples: sample_count_to_vk(desc.sample_count),
            tiling: ffi::VK_IMAGE_TILING_OPTIMAL,
            usage: vk_usage,
            sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
            initial_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
        };

        let mut image = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateImage(self.device, &image_info, core::ptr::null(), &mut image) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory()
                .with_context(&format!("sparse vkCreateImage VkResult {}", result)));
        }

        let handle = self.alloc_handle();
        self.textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                VkTexture {
                    image,
                    view: ffi::null_handle(),
                    memory: ffi::null_handle(),
                    width: desc.width,
                    height: desc.height,
                    format: vk_format,
                    mip_levels: desc.mip_levels.max(1),
                    current_layout: std::sync::atomic::AtomicU32::new(
                        ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                    ),
                },
            );

        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            sample_count: desc.sample_count,
            device: None,
            live: true,
        })
    }

    pub(crate) fn sampler_create_impl(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let (compare_enable, compare_op) = if let Some(cmp) = desc.compare {
            (1u32, super::super::compare_op_to_vk(cmp))
        } else {
            (0u32, 0u32)
        };
        let info = ffi::VkSamplerCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            mag_filter: super::super::filter_to_vk(desc.mag_filter),
            min_filter: super::super::filter_to_vk(desc.min_filter),
            mipmap_mode: match desc.mip_filter {
                crate::texture::Filter::Nearest => ffi::VK_SAMPLER_MIPMAP_MODE_NEAREST,
                crate::texture::Filter::Linear => ffi::VK_SAMPLER_MIPMAP_MODE_LINEAR,
            },
            address_mode_u: super::super::address_to_vk(desc.address_u),
            address_mode_v: super::super::address_to_vk(desc.address_v),
            address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            mip_lod_bias: 0.0,
            anisotropy_enable: if desc.max_anisotropy > 1 { 1 } else { 0 },
            max_anisotropy: desc.max_anisotropy as f32,
            compare_enable,
            compare_op,
            min_lod: 0.0,
            max_lod: ffi::VK_LOD_CLAMP_NONE,
            border_color: 0,
            unnormalized_coordinates: 0,
        };
        let mut sampler = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateSampler(self.device, &info, core::ptr::null(), &mut sampler) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::invalid_param("sampler creation failed")
                .with_context(&format!("create_sampler: VkResult {}", result)));
        }
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
