//! Timestamp queries, MSAA resolve, and helper functions.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::QuantaError;

use super::super::VulkanDevice;
use super::super::ffi;
use super::super::image_rest_state;

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

    /// Resolve a multisampled `src` into single-sample `dst`.
    /// `vkCmdResolveImage` requires identical formats
    /// (VUID-vkCmdResolveImage-srcImage-01386 — a real device loss on
    /// Iris Xe when violated), but the format mismatch is structural for
    /// the windowed effect path: the swapchain scene is BGRA8 while
    /// compute texel effects are RGBA8-only (SPIR-V has no BGRA8 storage
    /// format). So a differing-format resolve goes through a cached
    /// same-format single-sample temp — resolve src→temp, then a
    /// format-converting `vkCmdBlitImage` temp→dst (1:1 extent, NEAREST,
    /// so the blit is exact; blit handles the channel swizzle).
    #[cfg(feature = "render")]
    pub(crate) fn resolve_texture_impl(
        &self,
        src_handle: u64,
        dst_handle: u64,
    ) -> Result<(), QuantaError> {
        // Conversion decision in its own lock scope: temp creation below
        // takes the textures WRITE lock, so it must not overlap the read
        // lock held for recording.
        let convert = {
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
            if src.format == dst.format {
                None
            } else {
                Some((src.format, src.width, src.height))
            }
        };
        let temp_handle = match convert {
            None => None,
            Some((vk_format, w, h)) => Some(self.resolve_convert_temp(vk_format, w, h)?),
        };

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
        let temp = match temp_handle {
            Some(h) => Some(textures.get(&h).ok_or_else(|| {
                QuantaError::internal("resolve conversion temp missing from registry")
            })?),
            None => None,
        };

        // The images must license the transfer layouts the resolve puts
        // them in. Our own textures always carry both TRANSFER bits (see
        // texture_create_impl), but a swapchain image only carries what
        // the surface capabilities offered — resolving into one without
        // TRANSFER_DST is VUID-vkCmdResolveImage-dstImage-06764, a real
        // device-loss on Intel. Fail loudly instead of recording an
        // invalid command.
        if src.usage & ffi::VK_IMAGE_USAGE_TRANSFER_SRC_BIT == 0 {
            return Err(QuantaError::not_supported(
                "resolve source image was created without TRANSFER_SRC usage \
                 (this surface's capabilities do not offer it)",
            ));
        }
        if dst.usage & ffi::VK_IMAGE_USAGE_TRANSFER_DST_BIT == 0 {
            return Err(QuantaError::not_supported(
                "resolve destination image was created without TRANSFER_DST \
                 usage (this surface's capabilities do not offer it) — \
                 resolve through the render pass's resolve attachment instead",
            ));
        }

        // Route both transitions through the TRACKED layout rather than
        // assuming the source is always in COLOR_ATTACHMENT_OPTIMAL — a
        // resolve source that was previously sampled (SHADER_READ_ONLY) or
        // written by compute (GENERAL) would otherwise mismatch and trip
        // VUID-VkImageMemoryBarrier-oldLayout-01197/01211. The destination
        // is fully overwritten by the resolve, so it discards from
        // UNDEFINED.
        let src_old = src
            .current_layout
            .load(std::sync::atomic::Ordering::Relaxed);
        // Access/stage that must complete before the transfer read. From
        // UNDEFINED there is nothing to wait on; from a real layout, wait
        // on all prior commands (the source may have been produced by a
        // render pass or a shader read still draining the queue).
        let (src_access, src_stage) = if src_old == ffi::VK_IMAGE_LAYOUT_UNDEFINED {
            (0u32, ffi::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
        } else {
            (
                ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | ffi::VK_ACCESS_SHADER_READ_BIT,
                ffi::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            )
        };

        // Where each image settles AFTER the resolve. Derived from usage
        // so the transition is always licensed — SHADER_READ_ONLY needs
        // SAMPLED (VUID-VkImageMemoryBarrier-oldLayout-01211): a
        // render-target-only image (an MSAA intermediate, a swapchain
        // image) rests as an attachment instead, which also keeps the
        // pre-present transition's COLOR_ATTACHMENT assumption true when
        // the destination is an acquired frame.
        let (src_rest, src_rest_access, src_rest_stage) = image_rest_state(src.usage);
        let (dst_rest, dst_rest_access, dst_rest_stage) = image_rest_state(dst.usage);

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

            // Transition src to TRANSFER_SRC from its tracked layout.
            let barrier_src = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: src_access,
                dst_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                old_layout: src_old,
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
            // Transition dst to TRANSFER_DST. The resolve overwrites the
            // whole image, so discarding from UNDEFINED is correct and
            // valid regardless of the dst's prior contents.
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
            // The conversion temp is fully overwritten by the resolve —
            // discard from UNDEFINED like the destination. Built longhand
            // (not struct-update): VkImageMemoryBarrier is not Copy, and
            // barrier_dst still has to move into the barrier list.
            let temp_barrier = temp.map(|t| ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: t.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            });
            let mut barriers = vec![barrier_src, barrier_dst];
            if let Some(b) = temp_barrier {
                barriers.push(b);
            }
            ffi::vkCmdPipelineBarrier(
                cmd,
                // srcStageMask must cover whatever produced the source in
                // its tracked layout (all-commands from a real layout,
                // top-of-pipe when discarding from UNDEFINED).
                src_stage,
                ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                barriers.len() as u32,
                barriers.as_ptr(),
            );

            // Resolve — into the same-format temp when converting, else
            // straight into the destination.
            let mid_image = temp.map(|t| t.image).unwrap_or(dst.image);
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
                mid_image,
                ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &region,
            );

            if let Some(t) = temp {
                // Flip the temp to TRANSFER_SRC behind the resolve write,
                // then blit it into the real destination — the blit is
                // what performs the format conversion (channel swizzle);
                // 1:1 extents with NEAREST make it exact.
                let barrier_temp_flip = ffi::VkImageMemoryBarrier {
                    s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    p_next: core::ptr::null(),
                    src_access_mask: ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                    dst_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                    old_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    new_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    image: t.image,
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
                    ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    1,
                    &barrier_temp_flip,
                );
                let subresource = ffi::VkImageSubresourceLayers {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let extent = [
                    ffi::VkOffset3D { x: 0, y: 0, z: 0 },
                    ffi::VkOffset3D {
                        x: src.width as i32,
                        y: src.height as i32,
                        z: 1,
                    },
                ];
                let blit = ffi::VkImageBlit {
                    src_subresource: subresource,
                    src_offsets: extent,
                    dst_subresource: subresource,
                    dst_offsets: extent,
                };
                ffi::vkCmdBlitImage(
                    cmd,
                    t.image,
                    ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dst.image,
                    ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &blit,
                    ffi::VK_FILTER_NEAREST,
                );
            }

            // Transition both out of the transfer layouts, each to its
            // usage-derived rest state (computed above).
            let barrier_src_back = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: ffi::VK_ACCESS_TRANSFER_READ_BIT,
                dst_access_mask: src_rest_access,
                old_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                new_layout: src_rest,
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
                dst_access_mask: dst_rest_access,
                old_layout: ffi::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                new_layout: dst_rest,
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
                src_rest_stage | dst_rest_stage,
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
        // Record the rest layouts so the NEXT transition on either
        // texture — a later resolve, a sub-region upload, a present —
        // starts from the correct oldLayout instead of a stale
        // assumption. The temp rests in TRANSFER_SRC (the next
        // conversion discards it from UNDEFINED anyway, but the tracked
        // value stays truthful).
        src.current_layout
            .store(src_rest, std::sync::atomic::Ordering::Relaxed);
        dst.current_layout
            .store(dst_rest, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = temp {
            t.current_layout.store(
                ffi::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        drop(textures);
        self.submit_and_wait(cmd).and_then(|mut p| p.wait())
    }

    /// Get-or-create the cached single-sample intermediate for a
    /// format-converting resolve, keyed by (vk_format, width, height).
    /// A stale temp is replaced through `texture_destroy`, whose
    /// fence-deferred retirement makes the swap safe even with the old
    /// temp still referenced by an in-flight frame.
    #[cfg(feature = "render")]
    fn resolve_convert_temp(&self, vk_format: u32, w: u32, h: u32) -> Result<u64, QuantaError> {
        {
            let cache = self
                .resolve_temp
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            if let Some((f, cw, ch, handle)) = *cache
                && f == vk_format
                && cw == w
                && ch == h
            {
                return Ok(handle);
            }
        }
        let api_format = api_format_from_vk(vk_format).ok_or_else(|| {
            QuantaError::not_supported(
                "resolve_texture: source and destination formats differ, and the source \
                 format has no wired conversion path — resolve into a texture of the \
                 source's format instead",
            )
        })?;
        // Create OUTSIDE the cache lock (texture_create_impl takes the
        // textures write lock; keep lock scopes disjoint). Every created
        // texture carries both TRANSFER bits, which is all the temp needs.
        let tex = self.texture_create_impl(&crate::TextureDesc::new(w, h, api_format))?;
        let handle = tex.handle;
        let old = {
            let mut cache = self
                .resolve_temp
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let old = cache.take();
            *cache = Some((vk_format, w, h, handle));
            old
        };
        if let Some((_, _, _, old_handle)) = old {
            use crate::GpuDevice;
            let _ = self.texture_destroy(old_handle);
        }
        Ok(handle)
    }
}

/// Inverse of `format_to_vulkan` for the color formats a converting
/// resolve can target. Depth and compressed formats intentionally have
/// no arm — a resolve between those and anything else is malformed.
#[cfg(feature = "render")]
fn api_format_from_vk(vk: u32) -> Option<crate::Format> {
    use crate::Format;
    Some(match vk {
        x if x == ffi::VK_FORMAT_R8G8B8A8_UNORM => Format::RGBA8,
        x if x == ffi::VK_FORMAT_B8G8R8A8_UNORM => Format::BGRA8,
        x if x == ffi::VK_FORMAT_R8_UNORM => Format::R8,
        x if x == ffi::VK_FORMAT_R16_SFLOAT => Format::R16Float,
        x if x == ffi::VK_FORMAT_R32_SFLOAT => Format::R32Float,
        x if x == ffi::VK_FORMAT_R32G32_SFLOAT => Format::RG32Float,
        x if x == ffi::VK_FORMAT_R16G16B16A16_SFLOAT => Format::RGBA16Float,
        x if x == ffi::VK_FORMAT_R32G32B32A32_SFLOAT => Format::RGBA32Float,
        _ => return None,
    })
}

#[cfg(feature = "render")]
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
