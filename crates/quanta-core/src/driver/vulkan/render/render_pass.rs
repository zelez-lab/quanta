//! Render pass begin/end and draw command recording.

use alloc::{boxed::Box, format, vec, vec::Vec};
use core::ffi::c_void;

use crate::render_pass::RenderOp;
use crate::{LoadOp, Pulse, QuantaError, RenderPass, StoreOp, Texture};

use super::super::VulkanDevice;
use super::super::ffi;

impl VulkanDevice {
    pub(crate) fn render_begin_impl(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
            color_targets: Vec::new(),
            depth_target: None,
        })
    }

    pub(crate) fn render_end_impl(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let pipeline_handle = pass.ops.iter().find_map(|op| {
            if let RenderOp::SetPipeline(h) = op {
                Some(*h)
            } else {
                None
            }
        });

        let render_pipelines = self
            .render_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let samplers = self
            .samplers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;

        // Fail loudly on any dead handle BEFORE recording starts — a
        // silently skipped bind renders wrong (classic cause: a Field
        // dropped before pulse()).
        {
            use crate::render_pass::HandleKind;
            pass.validate_handles(|kind, h| match kind {
                HandleKind::Buffer => buffers.contains_key(&h),
                HandleKind::Texture => textures.contains_key(&h),
                HandleKind::Pipeline => render_pipelines.contains_key(&h),
                HandleKind::OcclusionQuery => self
                    .query_pools
                    .read()
                    .map(|p| p.contains_key(&h))
                    .unwrap_or(false),
            })?;
        }

        let target_tex = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::not_found("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        // Determine if we have MRT color targets or just the single target.
        let has_mrt = !pass.color_targets.is_empty();

        let (vk_render_pass, pipeline_ref) = if let Some(ph) = pipeline_handle {
            let rp = render_pipelines.get(&ph).ok_or_else(|| {
                QuantaError::not_found("pipeline not found")
                    .with_context(&format!("render_end: pipeline handle {}", ph))
            })?;
            (rp.render_pass, Some(rp))
        } else if has_mrt {
            // MRT: create a transient render pass with per-target load/store ops.
            let mut attachments = Vec::new();
            let mut color_refs = Vec::new();
            let mut resolve_refs = Vec::new();
            let mut has_resolve = false;

            for (i, ct) in pass.color_targets.iter().enumerate() {
                let ct_tex = textures.get(&ct.texture).ok_or_else(|| {
                    QuantaError::not_found("color target texture not found")
                        .with_context(&format!("render_end: color target {i}"))
                })?;
                let load_op = match ct.load_op {
                    LoadOp::Clear(_) => ffi::VK_ATTACHMENT_LOAD_OP_CLEAR,
                    LoadOp::Load => ffi::VK_ATTACHMENT_LOAD_OP_LOAD,
                    LoadOp::DontCare => ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                };
                let (store_op, resolve_handle) = match ct.store_op {
                    StoreOp::Store => (ffi::VK_ATTACHMENT_STORE_OP_STORE, None),
                    StoreOp::DontCare => (ffi::VK_ATTACHMENT_STORE_OP_DONT_CARE, None),
                    StoreOp::Resolve(h) => (ffi::VK_ATTACHMENT_STORE_OP_STORE, Some(h)),
                };
                let initial_layout = match ct.load_op {
                    LoadOp::Load => ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    _ => ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                };
                attachments.push(ffi::VkAttachmentDescription {
                    flags: 0,
                    format: ct_tex.format,
                    samples: ffi::VK_SAMPLE_COUNT_1_BIT,
                    load_op,
                    store_op,
                    stencil_load_op: ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    stencil_store_op: ffi::VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    initial_layout,
                    final_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });
                color_refs.push(ffi::VkAttachmentReference {
                    attachment: i as u32,
                    layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });
                if let Some(rh) = resolve_handle {
                    has_resolve = true;
                    let resolve_tex = textures.get(&rh.0).ok_or_else(|| {
                        QuantaError::not_found("resolve target texture not found")
                            .with_context(&format!("render_end: resolve target for attachment {i}"))
                    })?;
                    let resolve_idx = attachments.len() as u32;
                    attachments.push(ffi::VkAttachmentDescription {
                        flags: 0,
                        format: resolve_tex.format,
                        samples: ffi::VK_SAMPLE_COUNT_1_BIT,
                        load_op: ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                        store_op: ffi::VK_ATTACHMENT_STORE_OP_STORE,
                        stencil_load_op: ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                        stencil_store_op: ffi::VK_ATTACHMENT_STORE_OP_DONT_CARE,
                        initial_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                        final_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    });
                    resolve_refs.push(ffi::VkAttachmentReference {
                        attachment: resolve_idx,
                        layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    });
                } else {
                    resolve_refs.push(ffi::VkAttachmentReference {
                        attachment: !0u32, // VK_ATTACHMENT_UNUSED
                        layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                    });
                }
            }
            let p_resolve = if has_resolve {
                resolve_refs.as_ptr()
            } else {
                core::ptr::null()
            };
            let subpass = ffi::VkSubpassDescription {
                flags: 0,
                pipeline_bind_point: ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: core::ptr::null(),
                color_attachment_count: color_refs.len() as u32,
                p_color_attachments: color_refs.as_ptr(),
                p_resolve_attachments: p_resolve,
                p_depth_stencil_attachment: core::ptr::null(),
                preserve_attachment_count: 0,
                p_preserve_attachments: core::ptr::null(),
            };
            let rp_info = ffi::VkRenderPassCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                subpass_count: 1,
                p_subpasses: &subpass,
                dependency_count: 0,
                p_dependencies: core::ptr::null(),
            };
            let mut transient_rp = ffi::null_handle();
            let result = unsafe {
                ffi::vkCreateRenderPass(self.device, &rp_info, core::ptr::null(), &mut transient_rp)
            };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            (transient_rp, None)
        } else {
            // Create a transient render pass for clear-only usage.
            let color_attachment = ffi::VkAttachmentDescription {
                flags: 0,
                format: target_tex.format,
                samples: ffi::VK_SAMPLE_COUNT_1_BIT,
                load_op: ffi::VK_ATTACHMENT_LOAD_OP_CLEAR,
                store_op: ffi::VK_ATTACHMENT_STORE_OP_STORE,
                stencil_load_op: ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencil_store_op: ffi::VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initial_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                final_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
            let color_ref = ffi::VkAttachmentReference {
                attachment: 0,
                layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
            let subpass = ffi::VkSubpassDescription {
                flags: 0,
                pipeline_bind_point: ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                input_attachment_count: 0,
                p_input_attachments: core::ptr::null(),
                color_attachment_count: 1,
                p_color_attachments: &color_ref,
                p_resolve_attachments: core::ptr::null(),
                p_depth_stencil_attachment: core::ptr::null(),
                preserve_attachment_count: 0,
                p_preserve_attachments: core::ptr::null(),
            };
            let rp_info = ffi::VkRenderPassCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                attachment_count: 1,
                p_attachments: &color_attachment,
                subpass_count: 1,
                p_subpasses: &subpass,
                dependency_count: 0,
                p_dependencies: core::ptr::null(),
            };
            let mut transient_rp = ffi::null_handle();
            let result = unsafe {
                ffi::vkCreateRenderPass(self.device, &rp_info, core::ptr::null(), &mut transient_rp)
            };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            (transient_rp, None)
        };

        // Create framebuffer — MRT uses multiple image views.
        let fb_attachments: Vec<ffi::VkImageView> = if has_mrt {
            let mut views = Vec::new();
            for ct in &pass.color_targets {
                if let Some(tex) = textures.get(&ct.texture) {
                    views.push(tex.view);
                }
                // If this target has a resolve attachment, add the resolve view too.
                if let StoreOp::Resolve(rh) = ct.store_op
                    && let Some(tex) = textures.get(&rh.0)
                {
                    views.push(tex.view);
                }
            }
            views
        } else {
            vec![target_tex.view]
        };
        let fb_info = ffi::VkFramebufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            render_pass: vk_render_pass,
            attachment_count: fb_attachments.len() as u32,
            p_attachments: fb_attachments.as_ptr(),
            width: target_tex.width,
            height: target_tex.height,
            layers: 1,
        };
        let mut framebuffer = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateFramebuffer(self.device, &fb_info, core::ptr::null(), &mut framebuffer)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        // --- Descriptor set allocation and update ---
        let descriptor_pool;
        let descriptor_set;

        if let Some(rp) = pipeline_ref {
            // NOTE(descriptor-pool churn): this creates a fresh
            // VkDescriptorPool per pass. The device's
            // `descriptor_pool_cache` cannot be reused here as-is: its
            // pools are compute-shaped (16 storage-buffer descriptors,
            // no samplers) while render passes need 8 storage + 8
            // combined-image-sampler, and the cache field is a plain
            // `Mutex` (not `Arc`), so a pool can't be returned from the
            // 'static completion closure once render_end is async.
            // Proper reuse needs a render-shaped, Arc-backed cache —
            // deferred to a dedicated change.
            let pool_sizes = [
                ffi::VkDescriptorPoolSize {
                    ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptor_count: 8,
                },
                ffi::VkDescriptorPoolSize {
                    ty: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 8,
                },
            ];
            let pool_info = ffi::VkDescriptorPoolCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                max_sets: 1,
                pool_size_count: 2,
                p_pool_sizes: pool_sizes.as_ptr(),
            };
            let mut pool = ffi::null_handle();
            let result = unsafe {
                ffi::vkCreateDescriptorPool(self.device, &pool_info, core::ptr::null(), &mut pool)
            };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            descriptor_pool = Some(pool);

            let alloc_info = ffi::VkDescriptorSetAllocateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: core::ptr::null(),
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &rp.descriptor_set_layout,
            };
            let mut ds = ffi::null_handle();
            let result =
                unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            descriptor_set = Some(ds);

            // Collect buffer/image info for descriptor writes.
            let mut buffer_infos: Vec<(u32, ffi::VkDescriptorBufferInfo)> = Vec::new();
            let mut image_infos: Vec<(u32, ffi::VkDescriptorImageInfo)> = Vec::new();
            let mut sampler_for_slot: [Option<ffi::VkSampler>; 8] = [None; 8];

            // Default sampler
            let default_sampler_info = ffi::VkSamplerCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                mag_filter: ffi::VK_FILTER_LINEAR,
                min_filter: ffi::VK_FILTER_LINEAR,
                mipmap_mode: ffi::VK_SAMPLER_MIPMAP_MODE_LINEAR,
                address_mode_u: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                address_mode_v: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                mip_lod_bias: 0.0,
                anisotropy_enable: 0,
                max_anisotropy: 1.0,
                compare_enable: 0,
                compare_op: 0,
                min_lod: 0.0,
                max_lod: ffi::VK_LOD_CLAMP_NONE,
                border_color: 0,
                unnormalized_coordinates: 0,
            };
            let mut default_sampler = ffi::null_handle();
            unsafe {
                ffi::vkCreateSampler(
                    self.device,
                    &default_sampler_info,
                    core::ptr::null(),
                    &mut default_sampler,
                );
            }

            // First pass: collect sampler assignments
            for op in &pass.ops {
                if let RenderOp::SetSampler { slot, sampler } = op {
                    let idx = *slot as usize;
                    if idx < 8 {
                        let info = ffi::VkSamplerCreateInfo {
                            s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                            p_next: core::ptr::null(),
                            flags: 0,
                            mag_filter: super::super::filter_to_vk(sampler.mag_filter),
                            min_filter: super::super::filter_to_vk(sampler.min_filter),
                            mipmap_mode: match sampler.mip_filter {
                                crate::texture::Filter::Nearest => {
                                    ffi::VK_SAMPLER_MIPMAP_MODE_NEAREST
                                }
                                crate::texture::Filter::Linear => {
                                    ffi::VK_SAMPLER_MIPMAP_MODE_LINEAR
                                }
                            },
                            address_mode_u: super::super::address_to_vk(sampler.address_u),
                            address_mode_v: super::super::address_to_vk(sampler.address_v),
                            address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                            mip_lod_bias: 0.0,
                            anisotropy_enable: if sampler.max_anisotropy > 1 { 1 } else { 0 },
                            max_anisotropy: sampler.max_anisotropy as f32,
                            compare_enable: 0,
                            compare_op: 0,
                            min_lod: 0.0,
                            max_lod: ffi::VK_LOD_CLAMP_NONE,
                            border_color: 0,
                            unnormalized_coordinates: 0,
                        };
                        let mut s = ffi::null_handle();
                        let r = unsafe {
                            ffi::vkCreateSampler(self.device, &info, core::ptr::null(), &mut s)
                        };
                        if r == ffi::VK_SUCCESS {
                            sampler_for_slot[idx] = Some(s);
                        }
                    }
                }
            }

            // Second pass: buffer and image bindings
            for op in &pass.ops {
                match op {
                    RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                        if let Some(buf) = buffers.get(handle) {
                            buffer_infos.push((
                                *slot,
                                ffi::VkDescriptorBufferInfo {
                                    buffer: buf.buffer,
                                    offset: 0,
                                    range: ffi::VK_WHOLE_SIZE,
                                },
                            ));
                        }
                    }
                    RenderOp::SetTexture { slot, handle } => {
                        if let Some(tex) = textures.get(handle) {
                            let idx = *slot as usize;
                            let sampler = if idx < 8 {
                                sampler_for_slot[idx].unwrap_or(default_sampler)
                            } else {
                                default_sampler
                            };
                            image_infos.push((
                                *slot,
                                ffi::VkDescriptorImageInfo {
                                    sampler,
                                    image_view: tex.view,
                                    image_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                },
                            ));
                        }
                    }
                    _ => {}
                }
            }

            // Build descriptor writes
            let mut writes: Vec<ffi::VkWriteDescriptorSet> = Vec::new();
            for (slot, info) in &buffer_infos {
                writes.push(ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: *slot,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: core::ptr::null(),
                    p_buffer_info: info,
                    p_texel_buffer_view: core::ptr::null(),
                });
            }
            for (slot, info) in &image_infos {
                writes.push(ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: 8 + *slot,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    p_image_info: info,
                    p_buffer_info: core::ptr::null(),
                    p_texel_buffer_view: core::ptr::null(),
                });
            }

            if !writes.is_empty() {
                unsafe {
                    ffi::vkUpdateDescriptorSets(
                        self.device,
                        writes.len() as u32,
                        writes.as_ptr(),
                        0,
                        core::ptr::null(),
                    );
                }
            }
        } else {
            descriptor_pool = None;
            descriptor_set = None;
        }

        // Clear values — one per attachment (MRT: per color target + resolve slots).
        let clear_values: Vec<ffi::VkClearValue> = if has_mrt {
            let mut cvs = Vec::new();
            for ct in &pass.color_targets {
                let cv = match ct.load_op {
                    LoadOp::Clear(c) => ffi::VkClearValue {
                        color: ffi::VkClearColorValue {
                            float32: [c.r, c.g, c.b, c.a],
                        },
                    },
                    _ => ffi::VkClearValue {
                        color: ffi::VkClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                };
                cvs.push(cv);
                // Resolve attachments need a clear value slot too.
                if let StoreOp::Resolve(_) = ct.store_op {
                    cvs.push(ffi::VkClearValue {
                        color: ffi::VkClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    });
                }
            }
            cvs
        } else {
            let clear_color = pass
                .ops
                .iter()
                .find_map(|op| {
                    if let RenderOp::Clear(c) = op {
                        Some(ffi::VkClearValue {
                            color: ffi::VkClearColorValue {
                                float32: [c.r, c.g, c.b, c.a],
                            },
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or(ffi::VkClearValue {
                    color: ffi::VkClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                });
            vec![clear_color]
        };

        // Allocate command buffer and begin recording.
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

            // Transition target image to COLOR_ATTACHMENT_OPTIMAL.
            let barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: target_tex.image,
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
                ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &barrier,
            );

            // Begin render pass.
            let rp_begin = ffi::VkRenderPassBeginInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                p_next: core::ptr::null(),
                render_pass: vk_render_pass,
                framebuffer,
                render_area: ffi::VkRect2D {
                    offset: ffi::VkOffset2D { x: 0, y: 0 },
                    extent: ffi::VkExtent2D {
                        width: target_tex.width,
                        height: target_tex.height,
                    },
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
            };
            // If any op is `ExecuteRenderBundle`, the pass must be
            // begun with SECONDARY_COMMAND_BUFFERS contents (Vulkan
            // forbids mixing inline + secondary inside one
            // subpass). Pre-validate that inline draws and bundle
            // execute don't coexist in the same pass.
            let uses_bundles = pass
                .ops
                .iter()
                .any(|op| matches!(op, RenderOp::ExecuteRenderBundle { .. }));
            if uses_bundles {
                let has_inline_draw = pass.ops.iter().any(|op| {
                    matches!(
                        op,
                        RenderOp::Draw { .. }
                            | RenderOp::DrawIndexed { .. }
                            | RenderOp::DrawIndirect { .. }
                            | RenderOp::DrawIndexedIndirect { .. }
                    )
                });
                if has_inline_draw {
                    return Err(QuantaError::invalid_param(
                        "Vulkan render: cannot mix inline draws with execute_bundle in one render pass",
                    ));
                }
            }
            let subpass_contents = if uses_bundles {
                ffi::VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
            } else {
                ffi::VK_SUBPASS_CONTENTS_INLINE
            };
            ffi::vkCmdBeginRenderPass(cmd, &rp_begin, subpass_contents);

            let mut current_index_buffer: Option<ffi::VkBuffer> = None;

            // Encode each RenderOp.
            for op in &pass.ops {
                match op {
                    RenderOp::SetPipeline(handle) => {
                        if let Some(rp) = render_pipelines.get(handle) {
                            ffi::vkCmdBindPipeline(
                                cmd,
                                ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                                rp.pipeline,
                            );
                            if let Some(ds) = descriptor_set {
                                ffi::vkCmdBindDescriptorSets(
                                    cmd,
                                    ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    rp.layout,
                                    0,
                                    1,
                                    &ds,
                                    0,
                                    core::ptr::null(),
                                );
                            }
                        }
                    }

                    RenderOp::BindVertices {
                        slot,
                        handle,
                        offset,
                    } => {
                        if let Some(buf) = buffers.get(handle) {
                            let offsets = [*offset];
                            ffi::vkCmdBindVertexBuffers(
                                cmd,
                                *slot,
                                1,
                                &buf.buffer,
                                offsets.as_ptr(),
                            );
                        }
                    }

                    RenderOp::BindIndices { handle, offset } => {
                        if let Some(buf) = buffers.get(handle) {
                            ffi::vkCmdBindIndexBuffer(
                                cmd,
                                buf.buffer,
                                *offset,
                                ffi::VK_INDEX_TYPE_UINT32,
                            );
                            current_index_buffer = Some(buf.buffer);
                        }
                    }

                    RenderOp::SetField { .. }
                    | RenderOp::SetUniform { .. }
                    | RenderOp::SetTexture { .. }
                    | RenderOp::SetSampler { .. } => {
                        // Already handled via descriptor set update above.
                    }

                    RenderOp::SetValue { slot, data } => {
                        // The pipeline layout declares exactly one
                        // [0,128) VERTEX|FRAGMENT push range (8 slots
                        // x 16 bytes); pushing outside it or with a
                        // non-multiple-of-4 size is invalid API usage
                        // (VUID-vkCmdPushConstants-offset-01795 /
                        // size-00369) — fail the pass loudly instead.
                        let offset = *slot as usize * 16;
                        if *slot >= 8 || data.len() % 4 != 0 || offset + data.len() > 128 {
                            return Err(QuantaError::invalid_param(format!(
                                "SetValue slot {} with {} bytes exceeds the Vulkan \
                                 push-constant contract (slots 0-7, 4-byte-aligned, \
                                 slot*16 + size <= 128)",
                                slot,
                                data.len()
                            )));
                        }
                        if let Some(rp) = pipeline_ref {
                            ffi::vkCmdPushConstants(
                                cmd,
                                rp.layout,
                                ffi::VK_SHADER_STAGE_VERTEX_BIT | ffi::VK_SHADER_STAGE_FRAGMENT_BIT,
                                *slot * 16,
                                data.len() as u32,
                                data.as_ptr() as *const c_void,
                            );
                        }
                    }

                    RenderOp::Draw {
                        vertex_count,
                        instance_count,
                    } => {
                        ffi::vkCmdDraw(cmd, *vertex_count, *instance_count, 0, 0);
                    }

                    RenderOp::DrawIndexed {
                        index_count,
                        instance_count,
                    } => {
                        ffi::vkCmdDrawIndexed(cmd, *index_count, *instance_count, 0, 0, 0);
                    }

                    RenderOp::SetViewport {
                        x,
                        y,
                        width,
                        height,
                        min_depth,
                        max_depth,
                    } => {
                        let viewport = ffi::VkViewport {
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                            min_depth: *min_depth,
                            max_depth: *max_depth,
                        };
                        ffi::vkCmdSetViewport(cmd, 0, 1, &viewport);
                        // Set default scissor to match viewport (required for dynamic state)
                        let scissor = ffi::VkRect2D {
                            offset: ffi::VkOffset2D {
                                x: *x as i32,
                                y: *y as i32,
                            },
                            extent: ffi::VkExtent2D {
                                width: *width as u32,
                                height: *height as u32,
                            },
                        };
                        ffi::vkCmdSetScissor(cmd, 0, 1, &scissor);
                    }

                    RenderOp::SetScissor {
                        x,
                        y,
                        width,
                        height,
                    } => {
                        let scissor = ffi::VkRect2D {
                            offset: ffi::VkOffset2D {
                                x: *x as i32,
                                y: *y as i32,
                            },
                            extent: ffi::VkExtent2D {
                                width: *width,
                                height: *height,
                            },
                        };
                        ffi::vkCmdSetScissor(cmd, 0, 1, &scissor);
                    }

                    RenderOp::SetStencilRef(value) => {
                        ffi::vkCmdSetStencilReference(
                            cmd,
                            ffi::VK_STENCIL_FACE_FRONT_AND_BACK,
                            *value,
                        );
                    }

                    RenderOp::DrawIndirect {
                        buffer_handle,
                        offset,
                    } => {
                        if let Some(buf) = buffers.get(buffer_handle) {
                            ffi::vkCmdDrawIndirect(cmd, buf.buffer, *offset, 1, 0);
                        }
                    }

                    RenderOp::DrawIndexedIndirect {
                        buffer_handle,
                        offset,
                        index_handle,
                    } => {
                        if let Some(idx_buf) = buffers.get(index_handle) {
                            let needs_rebind = current_index_buffer
                                .map(|b| b != idx_buf.buffer)
                                .unwrap_or(true);
                            if needs_rebind {
                                ffi::vkCmdBindIndexBuffer(
                                    cmd,
                                    idx_buf.buffer,
                                    0,
                                    ffi::VK_INDEX_TYPE_UINT32,
                                );
                                current_index_buffer = Some(idx_buf.buffer);
                            }
                        }
                        if let Some(buf) = buffers.get(buffer_handle) {
                            ffi::vkCmdDrawIndexedIndirect(cmd, buf.buffer, *offset, 1, 0);
                        }
                    }

                    RenderOp::Clear(_) | RenderOp::ClearDepth(_) | RenderOp::ClearStencil(_) => {}
                    RenderOp::DebugPush(_) | RenderOp::DebugPop => {}

                    // Occlusion queries (M3.3)
                    RenderOp::BeginOcclusionQuery { handle, index } => {
                        let pools = self
                            .query_pools
                            .read()
                            .map_err(|_| QuantaError::internal("lock poisoned"))?;
                        if let Some(qp) = pools.get(handle) {
                            ffi::vkCmdResetQueryPool(cmd, qp.pool, *index, 1);
                            ffi::vkCmdBeginQuery(cmd, qp.pool, *index, 0);
                        }
                    }
                    RenderOp::EndOcclusionQuery { handle, index } => {
                        let pools = self
                            .query_pools
                            .read()
                            .map_err(|_| QuantaError::internal("lock poisoned"))?;
                        if let Some(qp) = pools.get(handle) {
                            ffi::vkCmdEndQuery(cmd, qp.pool, *index);
                        }
                    }

                    // VRS native lowering (step 063). When
                    // VK_KHR_fragment_shading_rate was enabled at
                    // device creation and `vkGetDeviceProcAddr`
                    // returned a non-null function pointer, lower
                    // SetShadingRate to vkCmdSetFragmentShadingRateKHR.
                    // SetShadingRateImage (texel-driven rates) is a
                    // separate native track — keep it deferred.
                    RenderOp::SetShadingRate(rate) => {
                        if let Some(set_rate) = self.vrs_set_rate_fn {
                            // Slice 14 — validate against the
                            // hardware-supported rate list cached
                            // at device discovery. Catches an
                            // unsupported rate before the driver
                            // surfaces it as a generic validation
                            // error inside the command buffer.
                            let want = (rate.x_axis(), rate.y_axis());
                            if !self.supported_shading_rates.contains(&want) {
                                ffi::vkCmdEndRenderPass(cmd);
                                return Err(QuantaError::not_supported(
                                    "Vulkan render encoder: requested shading rate is not in the device's supported-rate list",
                                ));
                            }
                            let extent = ffi::VkExtent2D {
                                width: rate.x_axis(),
                                height: rate.y_axis(),
                            };
                            // Pipeline-rate KEEP / KEEP — combine
                            // the per-draw rate with itself, which
                            // yields the requested rate verbatim.
                            // This matches the per-draw semantic of
                            // `RenderOp::SetShadingRate(r)`.
                            let combiner_ops: [u32; 2] = [
                                ffi::VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                                ffi::VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                            ];
                            set_rate(cmd, &extent, combiner_ops.as_ptr());
                        } else {
                            ffi::vkCmdEndRenderPass(cmd);
                            return Err(QuantaError::not_supported(
                                "Vulkan render encoder: VK_KHR_fragment_shading_rate is not available on this device",
                            ));
                        }
                    }
                    RenderOp::SetShadingRateImage { .. } => {
                        ffi::vkCmdEndRenderPass(cmd);
                        return Err(QuantaError::not_supported(
                            "Vulkan render encoder: shading-rate-image (texel-driven VRS) deferred",
                        ));
                    }
                    RenderOp::ExecuteRenderBundle {
                        bundle_handle,
                        count,
                    } => {
                        let bundles = self
                            .render_bundles
                            .read()
                            .map_err(|_| QuantaError::internal("lock poisoned"))?;
                        let bundle = bundles.get(bundle_handle).ok_or_else(|| {
                            QuantaError::not_found("render bundle handle not found")
                        })?;
                        if *count > bundle.recorded {
                            ffi::vkCmdEndRenderPass(cmd);
                            return Err(QuantaError::invalid_param(
                                "execute_bundle count exceeds recorded length",
                            ));
                        }
                        if *count > 0 {
                            ffi::vkCmdExecuteCommands(cmd, *count, bundle.secondaries.as_ptr());
                        }
                    }
                }
            }

            ffi::vkCmdEndRenderPass(cmd);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let transient_rp = if pipeline_handle.is_none() {
            Some(vk_render_pass)
        } else {
            None
        };
        drop(samplers);
        drop(buffers);
        // The render pass's attachments end in COLOR_ATTACHMENT_OPTIMAL
        // (the hardcoded final layout); record that so a later
        // transition (pre-present, sub-region upload) starts from the
        // right layout.
        if let Some(t) = textures.get(&pass.handle) {
            t.current_layout.store(
                ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        for ct in &pass.color_targets {
            if let Some(tex) = textures.get(&ct.texture) {
                tex.current_layout.store(
                    ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }
        drop(textures);
        drop(render_pipelines);

        // Submit WITHOUT blocking. `submit_and_wait` only records the
        // queue submission and hands back a Pulse whose wait_fn blocks
        // on the fence — the same async machinery the compute path
        // rides. The CPU is free to encode the next frame while the
        // GPU executes this pass; the caller waits when it needs the
        // results.
        let submit_pulse = match self.submit_and_wait(cmd) {
            Ok(p) => p,
            Err(e) => {
                // The submission never reached the queue, so the GPU
                // holds no reference to the per-pass objects — destroy
                // them immediately.
                unsafe {
                    ffi::vkDestroyFramebuffer(self.device, framebuffer, core::ptr::null());
                    if let Some(rp) = transient_rp {
                        ffi::vkDestroyRenderPass(self.device, rp, core::ptr::null());
                    }
                    if let Some(pool) = descriptor_pool {
                        ffi::vkDestroyDescriptorPool(self.device, pool, core::ptr::null());
                    }
                }
                return Err(e);
            }
        };

        // Per-pass objects (framebuffer, clear-only/MRT transient
        // render pass, descriptor pool + its set) are referenced by
        // the command buffer now executing on the GPU. Their
        // destruction is deferred to `RenderPassCleanup::drop`, which
        // waits the submission fence FIRST — whether the caller waits
        // the pulse or drops it unwaited.
        let cleanup = RenderPassCleanup {
            submit_pulse,
            device: self.device,
            framebuffer,
            transient_rp,
            descriptor_pool,
        };

        Ok(Pulse {
            handle: self.alloc_handle(),
            completed: false,
            wait_fn: Some(Box::new(move || drop(cleanup))),
        })
    }
}

/// Defers destruction of per-pass Vulkan objects until the GPU has
/// finished executing the submitted render pass.
///
/// Owns the fence-backed `Pulse` from `submit_and_wait` together with
/// the transient objects the in-flight command buffer references. The
/// `Drop` impl waits the fence BEFORE destroying anything, so the
/// destroy calls can never race GPU execution. The guard is captured
/// by the returned pulse's `wait_fn` closure, so cleanup runs exactly
/// once on either path:
/// - the caller calls `Pulse::wait()` → the closure runs, dropping the
///   guard (fence wait, then destroy);
/// - the caller drops the pulse unwaited → the boxed closure is
///   dropped, dropping its capture (same fence wait + destroy — i.e.
///   the old synchronous behavior, never a leak or use-after-free).
struct RenderPassCleanup {
    submit_pulse: Pulse,
    device: ffi::VkDevice,
    framebuffer: ffi::VkFramebuffer,
    transient_rp: Option<ffi::VkRenderPass>,
    descriptor_pool: Option<ffi::VkDescriptorPool>,
}

// Drop only waits the submission fence (legal from any thread) and
// destroys objects this struct exclusively owns — safe to run from
// Pulse::on_complete's waiter thread.
unsafe impl Send for RenderPassCleanup {}

impl Drop for RenderPassCleanup {
    fn drop(&mut self) {
        // Block until the GPU signals the submission fence. This also
        // returns the command buffer to the device's pool (the inner
        // wait_fn does both), so the command buffer cannot be recycled
        // while still executing.
        let _ = self.submit_pulse.wait();
        unsafe {
            ffi::vkDestroyFramebuffer(self.device, self.framebuffer, core::ptr::null());
            if let Some(rp) = self.transient_rp {
                ffi::vkDestroyRenderPass(self.device, rp, core::ptr::null());
            }
            if let Some(pool) = self.descriptor_pool {
                ffi::vkDestroyDescriptorPool(self.device, pool, core::ptr::null());
            }
        }
    }
}
