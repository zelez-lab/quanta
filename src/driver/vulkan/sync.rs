//! Memory barriers and resource state transitions for Vulkan.
//!
//! Uses Synchronization2 API (Vulkan 1.3 core) via raw FFI.

use crate::{QuantaError, ResourceState, Texture};

use super::VulkanDevice;
use super::ffi;

impl VulkanDevice {
    pub(crate) fn barrier_impl(&self) -> Result<(), QuantaError> {
        let cmd = self.alloc_command_buffer()?;
        let begin_info = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin_info);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let memory_barrier = ffi::VkMemoryBarrier2 {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            p_next: core::ptr::null(),
            src_stage_mask: ffi::VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            src_access_mask: ffi::VK_ACCESS_2_MEMORY_WRITE_BIT,
            dst_stage_mask: ffi::VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            dst_access_mask: ffi::VK_ACCESS_2_MEMORY_READ_BIT | ffi::VK_ACCESS_2_MEMORY_WRITE_BIT,
        };

        let dep_info = ffi::VkDependencyInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            p_next: core::ptr::null(),
            dependency_flags: 0,
            memory_barrier_count: 1,
            p_memory_barriers: &memory_barrier,
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: core::ptr::null(),
            image_memory_barrier_count: 0,
            p_image_memory_barriers: core::ptr::null(),
        };
        unsafe {
            ffi::vkCmdPipelineBarrier2(cmd, &dep_info);
        }

        unsafe {
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        self.submit_and_wait(cmd)
    }

    pub(crate) fn barrier_buffer_impl(
        &self,
        handle: u64,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers
            .get(&handle)
            .ok_or_else(|| QuantaError::invalid_param("buffer not found"))?;

        let cmd = self.alloc_command_buffer()?;
        let begin_info = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin_info);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let buffer_barrier = ffi::VkBufferMemoryBarrier2 {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            p_next: core::ptr::null(),
            src_stage_mask: state_to_stage(from),
            src_access_mask: state_to_access_write(from),
            dst_stage_mask: state_to_stage(to),
            dst_access_mask: state_to_access_read(to),
            src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
            buffer: buf.buffer,
            offset: 0,
            size: ffi::VK_WHOLE_SIZE,
        };

        let dep_info = ffi::VkDependencyInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            p_next: core::ptr::null(),
            dependency_flags: 0,
            memory_barrier_count: 0,
            p_memory_barriers: core::ptr::null(),
            buffer_memory_barrier_count: 1,
            p_buffer_memory_barriers: &buffer_barrier,
            image_memory_barrier_count: 0,
            p_image_memory_barriers: core::ptr::null(),
        };
        unsafe {
            ffi::vkCmdPipelineBarrier2(cmd, &dep_info);
        }

        unsafe {
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(buffers);
        self.submit_and_wait(cmd)
    }

    pub(crate) fn barrier_texture_impl(
        &self,
        texture: &Texture,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures
            .get(&texture.handle())
            .ok_or_else(|| QuantaError::invalid_param("texture not found"))?;

        let cmd = self.alloc_command_buffer()?;
        let begin_info = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin_info);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let image_barrier = ffi::VkImageMemoryBarrier2 {
            s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            p_next: core::ptr::null(),
            src_stage_mask: state_to_stage(from),
            src_access_mask: state_to_access_write(from),
            dst_stage_mask: state_to_stage(to),
            dst_access_mask: state_to_access_read(to),
            old_layout: state_to_layout(from),
            new_layout: state_to_layout(to),
            src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
            image: tex.image,
            subresource_range: ffi::VkImageSubresourceRange {
                aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                base_mip_level: 0,
                level_count: ffi::VK_REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: ffi::VK_REMAINING_ARRAY_LAYERS,
            },
        };

        let dep_info = ffi::VkDependencyInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            p_next: core::ptr::null(),
            dependency_flags: 0,
            memory_barrier_count: 0,
            p_memory_barriers: core::ptr::null(),
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: core::ptr::null(),
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &image_barrier,
        };
        unsafe {
            ffi::vkCmdPipelineBarrier2(cmd, &dep_info);
        }

        unsafe {
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(textures);
        self.submit_and_wait(cmd)
    }
}

// ============================================================================
// ResourceState -> Vulkan mapping helpers
// ============================================================================

fn state_to_layout(state: ResourceState) -> u32 {
    match state {
        ResourceState::General => ffi::VK_IMAGE_LAYOUT_GENERAL,
        ResourceState::ComputeWrite | ResourceState::ComputeRead => ffi::VK_IMAGE_LAYOUT_GENERAL,
        ResourceState::RenderTarget => ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ResourceState::DepthStencil => ffi::VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ResourceState::ShaderRead => ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        ResourceState::TransferSrc => ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        ResourceState::TransferDst => ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        ResourceState::Present => ffi::VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    }
}

fn state_to_stage(state: ResourceState) -> u64 {
    match state {
        ResourceState::ComputeWrite | ResourceState::ComputeRead => {
            ffi::VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
        }
        ResourceState::RenderTarget => ffi::VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        ResourceState::DepthStencil => {
            ffi::VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                | ffi::VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
        }
        ResourceState::ShaderRead => {
            ffi::VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                | ffi::VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
        }
        ResourceState::TransferSrc | ResourceState::TransferDst => {
            ffi::VK_PIPELINE_STAGE_2_TRANSFER_BIT
        }
        ResourceState::General | ResourceState::Present => {
            ffi::VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
        }
    }
}

fn state_to_access_write(state: ResourceState) -> u64 {
    match state {
        ResourceState::ComputeWrite => ffi::VK_ACCESS_2_SHADER_WRITE_BIT,
        ResourceState::RenderTarget => ffi::VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        ResourceState::DepthStencil => ffi::VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        ResourceState::TransferDst => ffi::VK_ACCESS_2_TRANSFER_WRITE_BIT,
        ResourceState::General => ffi::VK_ACCESS_2_MEMORY_WRITE_BIT,
        ResourceState::ComputeRead
        | ResourceState::ShaderRead
        | ResourceState::TransferSrc
        | ResourceState::Present => ffi::VK_ACCESS_2_NONE,
    }
}

fn state_to_access_read(state: ResourceState) -> u64 {
    match state {
        ResourceState::ComputeRead | ResourceState::ComputeWrite => {
            ffi::VK_ACCESS_2_SHADER_READ_BIT
        }
        ResourceState::ShaderRead => ffi::VK_ACCESS_2_SHADER_READ_BIT,
        ResourceState::RenderTarget => ffi::VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
        ResourceState::DepthStencil => ffi::VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        ResourceState::TransferSrc => ffi::VK_ACCESS_2_TRANSFER_READ_BIT,
        ResourceState::TransferDst => ffi::VK_ACCESS_2_TRANSFER_WRITE_BIT,
        ResourceState::Present => ffi::VK_ACCESS_2_NONE,
        ResourceState::General => {
            ffi::VK_ACCESS_2_MEMORY_READ_BIT | ffi::VK_ACCESS_2_MEMORY_WRITE_BIT
        }
    }
}
