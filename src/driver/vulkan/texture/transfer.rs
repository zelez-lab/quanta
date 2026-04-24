//! Texture read, write, copy, and mipmap generation for Vulkan.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::{QuantaError, Texture};

use super::super::ffi;
use super::super::{VulkanDevice, format_bytes_per_pixel_vk};

impl VulkanDevice {
    pub(crate) fn texture_write_impl(
        &self,
        texture: &Texture,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
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
        // Track layout: texture_write leaves image in SHADER_READ_ONLY_OPTIMAL
        tex.current_layout.store(
            ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            std::sync::atomic::Ordering::Relaxed,
        );
        drop(textures);
        self.submit_and_wait(cmd)?.wait()?;

        // Clean up staging
        unsafe {
            ffi::vkDestroyBuffer(self.device, staging_buf, core::ptr::null());
            ffi::vkFreeMemory(self.device, staging_mem, core::ptr::null());
        }
        Ok(())
    }

    pub(crate) fn texture_read_impl(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
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

            // Use tracked layout instead of assuming SHADER_READ_ONLY
            let actual_layout = tex
                .current_layout
                .load(std::sync::atomic::Ordering::Relaxed);
            let barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT | ffi::VK_ACCESS_MEMORY_WRITE_BIT,
                dst_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                old_layout: actual_layout,
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
        self.submit_and_wait(cmd)?.wait()?;

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

    pub(crate) fn generate_mipmaps_impl(&self, texture: &Texture) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("generate_mipmaps: handle {}", texture.handle()))
        })?;

        let mut mip_width = tex.width as i32;
        let mut mip_height = tex.height as i32;
        // Use the image's actual mip level count, not a computed value.
        let mip_levels = tex.mip_levels;
        if mip_levels <= 1 {
            return Ok(()); // Nothing to generate — image has only 1 mip level
        }

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
        self.submit_and_wait(cmd).and_then(|mut p| p.wait())
    }
}
