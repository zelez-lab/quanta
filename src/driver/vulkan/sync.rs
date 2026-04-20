//! Memory barriers and resource state transitions for Vulkan.
//!
//! Vulkan requires explicit synchronization between pipeline stages.
//! These implementations insert `VkMemoryBarrier2`, `VkBufferMemoryBarrier2`,
//! and `VkImageMemoryBarrier2` via the Synchronization2 API (Vulkan 1.3 core).

use crate::{QuantaError, ResourceState, Texture};
use ash::vk;

use super::VulkanDevice;

impl VulkanDevice {
    pub(crate) fn barrier_impl(&self) -> Result<(), QuantaError> {
        let cmd = self.alloc_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|_| QuantaError::submit_failed())?;
        }

        let memory_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);

        let dep_info =
            vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&memory_barrier));
        unsafe {
            self.device.cmd_pipeline_barrier2(cmd, &dep_info);
        }

        unsafe {
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        self.submit_and_wait(cmd)
    }

    pub(crate) fn barrier_buffer_impl(
        &self,
        handle: u64,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(&handle)
            .ok_or_else(|| QuantaError::invalid_param("buffer not found"))?;

        let cmd = self.alloc_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|_| QuantaError::submit_failed())?;
        }

        let buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(buf.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .src_stage_mask(state_to_stage(from))
            .src_access_mask(state_to_access_write(from))
            .dst_stage_mask(state_to_stage(to))
            .dst_access_mask(state_to_access_read(to));

        let dep_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(core::slice::from_ref(&buffer_barrier));
        unsafe {
            self.device.cmd_pipeline_barrier2(cmd, &dep_info);
        }

        unsafe {
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        // Drop the lock before submitting to avoid holding it across GPU wait.
        drop(buffers);
        self.submit_and_wait(cmd)
    }

    pub(crate) fn barrier_texture_impl(
        &self,
        texture: &Texture,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures
            .get(&texture.handle())
            .ok_or_else(|| QuantaError::invalid_param("texture not found"))?;

        let cmd = self.alloc_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|_| QuantaError::submit_failed())?;
        }

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .image(tex.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
            .old_layout(state_to_layout(from))
            .new_layout(state_to_layout(to))
            .src_stage_mask(state_to_stage(from))
            .src_access_mask(state_to_access_write(from))
            .dst_stage_mask(state_to_stage(to))
            .dst_access_mask(state_to_access_read(to));

        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(core::slice::from_ref(&image_barrier));
        unsafe {
            self.device.cmd_pipeline_barrier2(cmd, &dep_info);
        }

        unsafe {
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        // Drop the lock before submitting to avoid holding it across GPU wait.
        drop(textures);
        self.submit_and_wait(cmd)
    }
}

// ============================================================================
// ResourceState → Vulkan mapping helpers
// ============================================================================

fn state_to_layout(state: ResourceState) -> vk::ImageLayout {
    match state {
        ResourceState::General => vk::ImageLayout::GENERAL,
        ResourceState::ComputeWrite | ResourceState::ComputeRead => vk::ImageLayout::GENERAL,
        ResourceState::RenderTarget => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ResourceState::DepthStencil => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ResourceState::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ResourceState::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ResourceState::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ResourceState::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

fn state_to_stage(state: ResourceState) -> vk::PipelineStageFlags2 {
    match state {
        ResourceState::ComputeWrite | ResourceState::ComputeRead => {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        }
        ResourceState::RenderTarget => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        ResourceState::DepthStencil => {
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
        }
        ResourceState::ShaderRead => {
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER
        }
        ResourceState::TransferSrc | ResourceState::TransferDst => {
            vk::PipelineStageFlags2::TRANSFER
        }
        ResourceState::General | ResourceState::Present => vk::PipelineStageFlags2::ALL_COMMANDS,
    }
}

fn state_to_access_write(state: ResourceState) -> vk::AccessFlags2 {
    match state {
        ResourceState::ComputeWrite => vk::AccessFlags2::SHADER_WRITE,
        ResourceState::RenderTarget => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ResourceState::DepthStencil => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ResourceState::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
        ResourceState::General => vk::AccessFlags2::MEMORY_WRITE,
        // Read-only states produce no writes.
        ResourceState::ComputeRead
        | ResourceState::ShaderRead
        | ResourceState::TransferSrc
        | ResourceState::Present => vk::AccessFlags2::NONE,
    }
}

fn state_to_access_read(state: ResourceState) -> vk::AccessFlags2 {
    match state {
        ResourceState::ComputeRead | ResourceState::ComputeWrite => vk::AccessFlags2::SHADER_READ,
        ResourceState::ShaderRead => vk::AccessFlags2::SHADER_READ,
        ResourceState::RenderTarget => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
        ResourceState::DepthStencil => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
        ResourceState::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
        ResourceState::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
        ResourceState::Present => vk::AccessFlags2::NONE,
        ResourceState::General => vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    }
}
