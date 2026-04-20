//! Texture operations for Vulkan.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::{Format, QuantaError, Texture, TextureDesc, TextureUsage};

use super::ffi;
use super::{
    VkTexture, VulkanDevice, format_bytes_per_pixel_vk, format_to_vulkan, sample_count_to_vk,
};

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
        self.textures.lock().unwrap().insert(
            handle,
            VkTexture {
                image,
                view,
                memory,
                width: desc.width,
                height: desc.height,
                format: vk_format,
            },
        );

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

        // Create staging buffer
        let staging_info = ffi::VkBufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size: data.len() as u64,
            usage: ffi::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
        };
        let mut staging_buf = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateBuffer(
                self.device,
                &staging_info,
                core::ptr::null(),
                &mut staging_buf,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetBufferMemoryRequirements(self.device, staging_buf, &mut mem_reqs) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )?;
        let alloc = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
        };
        let mut staging_mem = ffi::null_handle();
        let result = unsafe {
            ffi::vkAllocateMemory(self.device, &alloc, core::ptr::null(), &mut staging_mem)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }
        unsafe {
            let r = ffi::vkBindBufferMemory(self.device, staging_buf, staging_mem, 0);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::out_of_memory());
            }
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let r = ffi::vkMapMemory(self.device, staging_mem, 0, data.len() as u64, 0, &mut ptr);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::invalid_param("map failed")
                    .with_context("texture_write: staging map"));
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            ffi::vkUnmapMemory(self.device, staging_mem);
        }

        // Transition image layout + copy
        let cmd = self.alloc_command_buffer()?;
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }

            // Transition: UNDEFINED -> TRANSFER_DST
            let barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: tex.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            ffi::vkCmdPipelineBarrier(
                cmd,
                ffi::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &barrier,
            );

            // Copy buffer -> image
            let region = ffi::VkBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: ffi::VkImageSubresourceLayers {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: ffi::VkOffset3D { x: 0, y: 0, z: 0 },
                image_extent: ffi::VkExtent3D {
                    width: tex.width,
                    height: tex.height,
                    depth: 1,
                },
            };
            ffi::vkCmdCopyBufferToImage(
                cmd,
                staging_buf,
                tex.image,
                ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &region,
            );

            // Transition: TRANSFER_DST -> SHADER_READ
            let barrier2 = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                new_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: tex.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            ffi::vkCmdPipelineBarrier(
                cmd,
                ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                ffi::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &barrier2,
            );

            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(textures);
        self.submit_and_wait(cmd)?;

        // Clean up staging
        unsafe {
            ffi::vkDestroyBuffer(self.device, staging_buf, core::ptr::null());
            ffi::vkFreeMemory(self.device, staging_mem, core::ptr::null());
        }
        Ok(())
    }

    pub(crate) fn texture_read_impl(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_read: handle {}", texture.handle()))
        })?;

        let bpp = format_bytes_per_pixel_vk(texture.format());
        let size = (tex.width * tex.height) as usize * bpp;

        // Create staging buffer
        let staging_info = ffi::VkBufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size: size as u64,
            usage: ffi::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
        };
        let mut staging_buf = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateBuffer(
                self.device,
                &staging_info,
                core::ptr::null(),
                &mut staging_buf,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetBufferMemoryRequirements(self.device, staging_buf, &mut mem_reqs) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )?;
        let alloc = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
        };
        let mut staging_mem = ffi::null_handle();
        let result = unsafe {
            ffi::vkAllocateMemory(self.device, &alloc, core::ptr::null(), &mut staging_mem)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }
        let result = unsafe { ffi::vkBindBufferMemory(self.device, staging_buf, staging_mem, 0) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        // Transition + copy
        let cmd = self.alloc_command_buffer()?;
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }

            let barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT,
                dst_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: tex.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            ffi::vkCmdPipelineBarrier(
                cmd,
                ffi::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &barrier,
            );

            let region = ffi::VkBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: ffi::VkImageSubresourceLayers {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: ffi::VkOffset3D { x: 0, y: 0, z: 0 },
                image_extent: ffi::VkExtent3D {
                    width: tex.width,
                    height: tex.height,
                    depth: 1,
                },
            };
            ffi::vkCmdCopyImageToBuffer(
                cmd,
                tex.image,
                ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                staging_buf,
                1,
                &region,
            );

            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(textures);
        self.submit_and_wait(cmd)?;

        // Read from staging
        let mut result = vec![0u8; size];
        unsafe {
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let r = ffi::vkMapMemory(self.device, staging_mem, 0, size as u64, 0, &mut ptr);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::invalid_param("map failed")
                    .with_context("texture_read: staging map"));
            }
            std::ptr::copy_nonoverlapping(ptr as *const u8, result.as_mut_ptr(), size);
            ffi::vkUnmapMemory(self.device, staging_mem);
            ffi::vkDestroyBuffer(self.device, staging_buf, core::ptr::null());
            ffi::vkFreeMemory(self.device, staging_mem, core::ptr::null());
        }
        Ok(result)
    }

    pub(crate) fn sampler_create_impl(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let info = ffi::VkSamplerCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            mag_filter: super::filter_to_vk(desc.mag_filter),
            min_filter: super::filter_to_vk(desc.min_filter),
            mipmap_mode: match desc.mip_filter {
                crate::render_pass::Filter::Nearest => ffi::VK_SAMPLER_MIPMAP_MODE_NEAREST,
                crate::render_pass::Filter::Linear => ffi::VK_SAMPLER_MIPMAP_MODE_LINEAR,
            },
            address_mode_u: super::address_to_vk(desc.address_u),
            address_mode_v: super::address_to_vk(desc.address_v),
            address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            mip_lod_bias: 0.0,
            anisotropy_enable: if desc.max_anisotropy > 1 { 1 } else { 0 },
            max_anisotropy: desc.max_anisotropy as f32,
            compare_enable: 0,
            compare_op: 0,
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
        self.samplers.lock().unwrap().insert(handle, sampler);
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

        let mut mip_width = tex.width as i32;
        let mut mip_height = tex.height as i32;
        let mip_levels = (mip_width.max(mip_height) as f32).log2().floor() as u32 + 1;

        let cmd = self.alloc_command_buffer()?;
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }

            for i in 1..mip_levels {
                // Transition level i-1 to TRANSFER_SRC
                let barrier_src = ffi::VkImageMemoryBarrier {
                    s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    p_next: core::ptr::null(),
                    src_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                    dst_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                    old_layout: if i == 1 {
                        ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    } else {
                        ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                    },
                    new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    image: tex.image,
                    subresource_range: ffi::VkImageSubresourceRange {
                        aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                        base_mip_level: i - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };
                ffi::vkCmdPipelineBarrier(
                    cmd,
                    ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    1,
                    &barrier_src,
                );

                // Transition level i to TRANSFER_DST
                let barrier_dst = ffi::VkImageMemoryBarrier {
                    s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    p_next: core::ptr::null(),
                    src_access_mask: 0,
                    dst_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                    old_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                    new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    image: tex.image,
                    subresource_range: ffi::VkImageSubresourceRange {
                        aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                        base_mip_level: i,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };
                ffi::vkCmdPipelineBarrier(
                    cmd,
                    ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    1,
                    &barrier_dst,
                );

                let next_width = (mip_width / 2).max(1);
                let next_height = (mip_height / 2).max(1);

                let blit = ffi::VkImageBlit {
                    src_subresource: ffi::VkImageSubresourceLayers {
                        aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                        mip_level: i - 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [
                        ffi::VkOffset3D { x: 0, y: 0, z: 0 },
                        ffi::VkOffset3D {
                            x: mip_width,
                            y: mip_height,
                            z: 1,
                        },
                    ],
                    dst_subresource: ffi::VkImageSubresourceLayers {
                        aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                        mip_level: i,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [
                        ffi::VkOffset3D { x: 0, y: 0, z: 0 },
                        ffi::VkOffset3D {
                            x: next_width,
                            y: next_height,
                            z: 1,
                        },
                    ],
                };

                ffi::vkCmdBlitImage(
                    cmd,
                    tex.image,
                    ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    tex.image,
                    ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &blit,
                    ffi::VK_FILTER_LINEAR,
                );

                mip_width = next_width;
                mip_height = next_height;
            }

            // Transition all levels to SHADER_READ
            let final_barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                dst_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                new_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: tex.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            ffi::vkCmdPipelineBarrier(
                cmd,
                ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                ffi::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &final_barrier,
            );

            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(textures);
        self.submit_and_wait(cmd)
    }
}
