//! Timestamp queries, MSAA resolve, and helper functions.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::QuantaError;

use super::super::VulkanDevice;
use super::super::ffi;

impl VulkanDevice {
    // === Timestamp queries (Step 011) ===

    pub(crate) fn timestamp_query_create_impl(&self, count: u32) -> Result<u64, QuantaError> {
        let pool_info = ffi::VkQueryPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            query_type: ffi::VK_QUERY_TYPE_TIMESTAMP,
            query_count: count,
            pipeline_statistics: 0,
        };
        let mut pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateQueryPool(self.device, &pool_info, core::ptr::null(), &mut pool)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::invalid_param("query pool creation failed")
                .with_context(&format!("timestamp_query_create: VkResult {result}")));
        }
        let handle = self.alloc_handle();
        self.query_pools
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, super::super::VkQueryPool { pool, count });
        Ok(handle)
    }

    pub(crate) fn timestamp_write_impl(
        &self,
        query_handle: u64,
        index: u32,
    ) -> Result<(), QuantaError> {
        let pools = self
            .query_pools
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let qp = pools.get(&query_handle).ok_or_else(|| {
            QuantaError::not_found("query pool not found")
                .with_context(&format!("timestamp_write: handle {query_handle}"))
        })?;

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
            ffi::vkCmdResetQueryPool(cmd, qp.pool, index, 1);
            ffi::vkCmdWriteTimestamp(
                cmd,
                ffi::VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                qp.pool,
                index,
            );
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(pools);
        self.submit_and_wait(cmd).and_then(|mut p| p.wait())
    }

    pub(crate) fn timestamp_query_read_impl(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        let pools = self
            .query_pools
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let qp = pools.get(&handle).ok_or_else(|| {
            QuantaError::not_found("query pool not found")
                .with_context(&format!("timestamp_query_read: handle {handle}"))
        })?;

        let count = qp.count as usize;
        let mut results = vec![0u64; count];
        let result = unsafe {
            ffi::vkGetQueryPoolResults(
                self.device,
                qp.pool,
                0,
                qp.count,
                count * core::mem::size_of::<u64>(),
                results.as_mut_ptr() as *mut c_void,
                core::mem::size_of::<u64>() as u64,
                ffi::VK_QUERY_RESULT_64_BIT | ffi::VK_QUERY_RESULT_WAIT_BIT,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::invalid_param("query read failed")
                .with_context(&format!("timestamp_query_read: VkResult {result}")));
        }
        Ok(results)
    }

    // === MSAA Resolve (Step 012) ===

    pub(crate) fn resolve_texture_impl(
        &self,
        src_handle: u64,
        dst_handle: u64,
    ) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let src = textures.get(&src_handle).ok_or_else(|| {
            QuantaError::not_found("source texture not found")
                .with_context(&format!("resolve_texture: src handle {src_handle}"))
        })?;
        let dst = textures.get(&dst_handle).ok_or_else(|| {
            QuantaError::not_found("destination texture not found")
                .with_context(&format!("resolve_texture: dst handle {dst_handle}"))
        })?;

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

            // Transition src to TRANSFER_SRC
            let barrier_src = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: src.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            // Transition dst to TRANSFER_DST
            let barrier_dst = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: dst.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let barriers = [barrier_src, barrier_dst];
            ffi::vkCmdPipelineBarrier(
                cmd,
                ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                2,
                barriers.as_ptr(),
            );

            // Resolve
            let region = ffi::VkImageResolve {
                src_subresource: ffi::VkImageSubresourceLayers {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offset: ffi::VkOffset3D { x: 0, y: 0, z: 0 },
                dst_subresource: ffi::VkImageSubresourceLayers {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offset: ffi::VkOffset3D { x: 0, y: 0, z: 0 },
                extent: ffi::VkExtent3D {
                    width: src.width,
                    height: src.height,
                    depth: 1,
                },
            };
            ffi::vkCmdResolveImage(
                cmd,
                src.image,
                ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                dst.image,
                ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &region,
            );

            // Transition both back to shader-read
            let barrier_src_back = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                dst_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                new_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: src.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let barrier_dst_back = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                new_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: dst.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let barriers_back = [barrier_src_back, barrier_dst_back];
            ffi::vkCmdPipelineBarrier(
                cmd,
                ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                ffi::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                2,
                barriers_back.as_ptr(),
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

pub(super) fn attr_format_to_vulkan(fmt: crate::AttributeFormat) -> u32 {
    match fmt {
        crate::AttributeFormat::Float => ffi::VK_FORMAT_R32_SFLOAT,
        crate::AttributeFormat::Float2 => ffi::VK_FORMAT_R32G32_SFLOAT,
        crate::AttributeFormat::Float3 => ffi::VK_FORMAT_R32G32B32_SFLOAT,
        crate::AttributeFormat::Float4 => ffi::VK_FORMAT_R32G32B32A32_SFLOAT,
        crate::AttributeFormat::Int => ffi::VK_FORMAT_R32_SINT,
        crate::AttributeFormat::Int2 => ffi::VK_FORMAT_R32G32_SINT,
        crate::AttributeFormat::Int3 => ffi::VK_FORMAT_R32G32B32_SINT,
        crate::AttributeFormat::Int4 => ffi::VK_FORMAT_R32G32B32A32_SINT,
        crate::AttributeFormat::UInt => ffi::VK_FORMAT_R32_UINT,
        crate::AttributeFormat::UInt2 => ffi::VK_FORMAT_R32G32_UINT,
        crate::AttributeFormat::UInt3 => ffi::VK_FORMAT_R32G32B32_UINT,
        crate::AttributeFormat::UInt4 => ffi::VK_FORMAT_R32G32B32A32_UINT,
        crate::AttributeFormat::UByte4Norm => ffi::VK_FORMAT_R8G8B8A8_UNORM,
    }
}
