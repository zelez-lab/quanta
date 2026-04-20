//! Texture operations for Vulkan.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{Format, QuantaError, Texture, TextureDesc, TextureUsage};
use ash::vk;

use super::{
    VkTexture, VulkanDevice, format_bytes_per_pixel_vk, format_to_vulkan, sample_count_to_vk,
};

impl VulkanDevice {
    pub(crate) fn texture_create_impl(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let vk_format = format_to_vulkan(desc.format);

        let mut vk_usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if desc.usage.has(TextureUsage::SHADER_READ) {
            vk_usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage.has(TextureUsage::SHADER_WRITE) {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.has(TextureUsage::RENDER_TARGET) {
            if matches!(desc.format, Format::Depth32Float) {
                vk_usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            } else {
                vk_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }
        }

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth.max(1),
            })
            .mip_levels(desc.mip_levels.max(1))
            .array_layers(desc.array_length.max(1))
            .samples(sample_count_to_vk(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            self.device
                .create_image(&image_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);
        let memory = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|_| QuantaError::out_of_memory())?;
        }

        let aspect = if matches!(desc.format, Format::Depth32Float) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: desc.mip_levels.max(1),
                base_array_layer: 0,
                layer_count: desc.array_length.max(1),
            });

        let view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };

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
        let staging_info = vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buf = unsafe {
            self.device
                .create_buffer(&staging_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buf) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);
        let staging_mem = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        unsafe {
            self.device
                .bind_buffer_memory(staging_buf, staging_mem, 0)
                .map_err(|_| QuantaError::out_of_memory())?;
            let ptr = self
                .device
                .map_memory(
                    staging_mem,
                    0,
                    data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|_| {
                    QuantaError::invalid_param("map failed")
                        .with_context("texture_write: staging map")
                })? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(staging_mem);
        }

        // Transition image layout + copy
        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;

            // Transition: UNDEFINED -> TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Copy buffer -> image
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: tex.width,
                    height: tex.height,
                    depth: 1,
                });
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition: TRANSFER_DST -> SHADER_READ
            let barrier2 = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier2],
            );

            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(textures);
        self.submit_and_wait(cmd)?;

        // Clean up staging
        unsafe {
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);
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
        let staging_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buf = unsafe {
            self.device
                .create_buffer(&staging_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buf) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);
        let staging_mem = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        unsafe {
            self.device
                .bind_buffer_memory(staging_buf, staging_mem, 0)
                .map_err(|_| QuantaError::out_of_memory())?;
        }

        // Transition + copy
        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: tex.width,
                    height: tex.height,
                    depth: 1,
                });
            self.device.cmd_copy_image_to_buffer(
                cmd,
                tex.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buf,
                &[region],
            );

            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(textures);
        self.submit_and_wait(cmd)?;

        // Read from staging
        let mut result = vec![0u8; size];
        unsafe {
            let ptr = self
                .device
                .map_memory(staging_mem, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(|_| {
                    QuantaError::invalid_param("map failed")
                        .with_context("texture_read: staging map")
                })? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size);
            self.device.unmap_memory(staging_mem);
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);
        }
        Ok(result)
    }

    pub(crate) fn sampler_create_impl(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let info = vk::SamplerCreateInfo::default()
            .min_filter(super::filter_to_vk(desc.min_filter))
            .mag_filter(super::filter_to_vk(desc.mag_filter))
            .mipmap_mode(match desc.mip_filter {
                crate::render_pass::Filter::Nearest => vk::SamplerMipmapMode::NEAREST,
                crate::render_pass::Filter::Linear => vk::SamplerMipmapMode::LINEAR,
            })
            .address_mode_u(super::address_to_vk(desc.address_u))
            .address_mode_v(super::address_to_vk(desc.address_v))
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_anisotropy(desc.max_anisotropy as f32)
            .anisotropy_enable(desc.max_anisotropy > 1)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = unsafe {
            self.device.create_sampler(&info, None).map_err(|e| {
                QuantaError::invalid_param("sampler creation failed")
                    .with_context(&format!("create_sampler: {:?}", e))
            })?
        };
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
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;

            for i in 1..mip_levels {
                // Transition level i-1 to TRANSFER_SRC
                let barrier_src = vk::ImageMemoryBarrier::default()
                    .old_layout(if i == 1 {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else {
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    })
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(tex.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_src],
                );

                // Transition level i to TRANSFER_DST
                let barrier_dst = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(tex.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_dst],
                );

                let next_width = (mip_width / 2).max(1);
                let next_height = (mip_height / 2).max(1);

                let blit = vk::ImageBlit::default()
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: mip_width,
                            y: mip_height,
                            z: 1,
                        },
                    ])
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i - 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: next_width,
                            y: next_height,
                            z: 1,
                        },
                    ])
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.cmd_blit_image(
                    cmd,
                    tex.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    tex.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );

                mip_width = next_width;
                mip_height = next_height;
            }

            // Transition all levels to SHADER_READ
            let final_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[final_barrier],
            );

            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(textures);
        self.submit_and_wait(cmd)
    }
}
