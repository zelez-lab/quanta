//! Render pass begin/end and draw command recording.

use alloc::{boxed::Box, format, vec, vec::Vec};
use core::ffi::c_void;

use crate::render_pass::RenderOp;
use crate::{LoadOp, Pulse, QuantaError, RenderPass, StoreOp, Texture};

use super::super::VulkanDevice;
use super::super::ffi;
use super::super::sample_count_to_vk;

/// The distinct texture handles a pass binds as SAMPLED sources (every
/// `SetTexture` slot), deduped in first-bind order. Used to transition
/// each source to `SHADER_READ_ONLY_OPTIMAL` before the pass — see the
/// sample-source barrier in `render_end_impl`.
fn sampled_source_handles(pass: &RenderPass) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    for op in &pass.ops {
        if let RenderOp::SetTexture { handle, .. } = op
            && !out.contains(handle)
        {
            out.push(*handle);
        }
    }
    out
}

/// Clamp a scissor rectangle to the render area, yielding a valid
/// `VkRect2D` for `vkCmdSetScissor`.
///
/// Contract (matches the API-level `set_scissor` doc): the offset is
/// clamped to ≥ 0 and the extent shrinks by the clamped-away amount, then
/// the extent is clamped so `offset + extent` never exceeds the render
/// area. A scissor that clamps entirely away becomes `(0, 0, 0, 0)` —
/// nothing draws, and no validation error fires.
///
/// Why this lives on the Vulkan side: `RenderOp::SetScissor` carries
/// `x`/`y` as `u32`, but callers routinely compute a NEGATIVE offset
/// (e.g. a scrolled child clipped above its parent) and cast it in, which
/// arrives here as a large `u32` that `as i32` reads back as negative.
/// Metal's `setScissorRect` silently clamps such a rect to the drawable;
/// Vulkan instead REJECTS a negative offset (VUID-vkCmdSetScissor-x-00595),
/// so identical app code diverged per backend. Clamping here restores
/// parity: the tolerated input becomes a clipped rect on both backends
/// rather than a flood of validation errors on one.
fn clamp_scissor(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    area_w: u32,
    area_h: u32,
) -> ffi::VkRect2D {
    // Read the offset as a signed value: a wrapped-in negative offset
    // (u32 >= 2^31) reads back negative, exactly as it did before the
    // driver saw it.
    let sx = x as i32;
    let sy = y as i32;

    // Per axis: push the offset up to 0, taking the overhang out of the
    // extent. `saturating_sub` on the overhang collapses a fully-off
    // axis to 0.
    let clamp_axis = |off: i32, ext: u32, bound: u32| -> (u32, u32) {
        if off >= 0 {
            let off = off as u32;
            // Trim the extent so off + ext <= bound (a rect starting past
            // the bound yields 0 extent).
            let ext = ext.min(bound.saturating_sub(off));
            (off, ext)
        } else {
            // Negative offset: origin goes to 0, extent loses |off|.
            let overhang = off.unsigned_abs();
            let ext = ext.saturating_sub(overhang).min(bound);
            (0, ext)
        }
    };
    let (ox, ew) = clamp_axis(sx, width, area_w);
    let (oy, eh) = clamp_axis(sy, height, area_h);

    ffi::VkRect2D {
        offset: ffi::VkOffset2D {
            x: ox as i32,
            y: oy as i32,
        },
        extent: ffi::VkExtent2D {
            width: ew,
            height: eh,
        },
    }
}

impl VulkanDevice {
    /// Return the cached VkSampler for `desc`, creating it on first use.
    ///
    /// The cache key is the WHOLE `SamplerDesc`, so distinct descriptors
    /// get distinct samplers and identical descriptors share one — the
    /// pool grows with the number of distinct descriptors, never with
    /// draw or frame count. This replaces the old per-`SetSampler`
    /// `vkCreateSampler` that leaked a sampler on every textured draw and
    /// exhausted the device allocation pool. Samplers live until device
    /// teardown, which drains the cache. A creation failure falls back to
    /// a null handle (the render path substitutes the default sampler),
    /// so a transient failure never poisons the cache with a bad entry.
    #[cfg(feature = "render")]
    pub(super) fn get_or_create_render_sampler(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> ffi::VkSampler {
        // Fast path: shared read lock, hit the existing entry.
        if let Ok(cache) = self.render_sampler_cache.read()
            && let Some(s) = cache.get(desc)
        {
            return *s;
        }
        // Miss: take the write lock and re-check (another thread may have
        // filled it between the read unlock and here), then create once.
        let Ok(mut cache) = self.render_sampler_cache.write() else {
            return ffi::null_handle();
        };
        if let Some(s) = cache.get(desc) {
            return *s;
        }
        let info = super::super::sampler_create_info(desc);
        let mut s = ffi::null_handle();
        let r = unsafe { ffi::vkCreateSampler(self.device, &info, core::ptr::null(), &mut s) };
        if r == ffi::VK_SUCCESS {
            cache.insert(*desc, s);
            s
        } else {
            ffi::null_handle()
        }
    }

    /// Write the CURRENT running slot state into `ds`, then update it.
    ///
    /// The render-side twin of compute's `write_texture_descriptors`: it
    /// serialises one draw's snapshot of the slot arrays into the shared
    /// 16-binding layout — buffers → `STORAGE_BUFFER` at `binding = slot`
    /// (0-7), textures → `COMBINED_IMAGE_SAMPLER` at `binding = 8 + slot`
    /// (8-15) with the slot's sampler and `SHADER_READ_ONLY_OPTIMAL`. A
    /// texture slot with no explicit `SetSampler` falls back to the LINEAR
    /// clamp `default_desc`, resolved through the sampler cache only when
    /// actually needed (so a slot that always carries a sampler never
    /// grows the cache). The info structs live on the stack here and
    /// outlive the single `vkUpdateDescriptorSets` — every pointer in
    /// `writes` refers into `buffer_infos` / `image_infos` below.
    ///
    /// This is exactly the descriptor SHAPE the old whole-pass write
    /// built; only WHEN (per draw) and HOW MANY (one set each) changed.
    #[cfg(feature = "render")]
    fn write_render_descriptors(
        &self,
        set: ffi::VkDescriptorSet,
        texture_for_slot: &[Option<ffi::VkImageView>; 8],
        sampler_for_slot: &[Option<ffi::VkSampler>; 8],
        buffer_for_slot: &[Option<ffi::VkBuffer>; 8],
        default_desc: &crate::texture::SamplerDesc,
    ) {
        let mut buffer_infos: [ffi::VkDescriptorBufferInfo; 8] = unsafe { core::mem::zeroed() };
        let mut image_infos: [ffi::VkDescriptorImageInfo; 8] = unsafe { core::mem::zeroed() };
        let mut writes: [ffi::VkWriteDescriptorSet; 16] = unsafe { core::mem::zeroed() };
        let mut n = 0usize;

        for (slot, buf) in buffer_for_slot.iter().enumerate() {
            if let Some(buffer) = buf {
                buffer_infos[slot] = ffi::VkDescriptorBufferInfo {
                    buffer: *buffer,
                    offset: 0,
                    range: ffi::VK_WHOLE_SIZE,
                };
                writes[n] = ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: set,
                    dst_binding: slot as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: core::ptr::null(),
                    p_buffer_info: &buffer_infos[slot],
                    p_texel_buffer_view: core::ptr::null(),
                };
                n += 1;
            }
        }

        for (slot, view) in texture_for_slot.iter().enumerate() {
            if let Some(image_view) = view {
                // The wrapping closure is required: the fallback resolves
                // the default sampler through the cache lazily, only for a
                // textured slot that lacks an explicit one.
                #[allow(clippy::redundant_closure)]
                let sampler = sampler_for_slot[slot]
                    .unwrap_or_else(|| self.get_or_create_render_sampler(default_desc));
                image_infos[slot] = ffi::VkDescriptorImageInfo {
                    sampler,
                    image_view: *image_view,
                    image_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                };
                writes[n] = ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: set,
                    dst_binding: 8 + slot as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    p_image_info: &image_infos[slot],
                    p_buffer_info: core::ptr::null(),
                    p_texel_buffer_view: core::ptr::null(),
                };
                n += 1;
            }
        }

        if n > 0 {
            unsafe {
                ffi::vkUpdateDescriptorSets(
                    self.device,
                    n as u32,
                    writes.as_ptr(),
                    0,
                    core::ptr::null(),
                );
            }
        }
    }

    /// Allocate one descriptor set from the per-pass `pool`, write the
    /// current running slot state into it, and bind it for the graphics
    /// pipeline — the allocate→write→bind unit that runs immediately
    /// before each `vkCmdDraw*`, so every draw samples exactly the
    /// resources bound before it. `cmd` must be recording.
    #[cfg(feature = "render")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn bind_draw_descriptor_set(
        &self,
        cmd: ffi::VkCommandBuffer,
        pool: ffi::VkDescriptorPool,
        ds_layout: ffi::VkDescriptorSetLayout,
        pipeline_layout: ffi::VkPipelineLayout,
        texture_for_slot: &[Option<ffi::VkImageView>; 8],
        sampler_for_slot: &[Option<ffi::VkSampler>; 8],
        buffer_for_slot: &[Option<ffi::VkBuffer>; 8],
        default_desc: &crate::texture::SamplerDesc,
    ) -> Result<(), QuantaError> {
        let alloc_info = ffi::VkDescriptorSetAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: &ds_layout,
        };
        let mut ds = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }
        self.write_render_descriptors(
            ds,
            texture_for_slot,
            sampler_for_slot,
            buffer_for_slot,
            default_desc,
        );
        unsafe {
            ffi::vkCmdBindDescriptorSets(
                cmd,
                ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout,
                0,
                1,
                &ds,
                0,
                core::ptr::null(),
            );
        }
        Ok(())
    }

    /// Build a single-subpass transient `VkRenderPass` from a ready
    /// attachment list and its color/resolve references, calling
    /// `vkCreateRenderPass` and checking the result.
    ///
    /// The MRT branch and the clear-only branch of `render_end_impl` build
    /// the SAME `VkSubpassDescription` + `VkRenderPassCreateInfo` +
    /// `vkCreateRenderPass` + success-check unit; they differ only in the
    /// attachment count/pointers and the color/resolve reference arrays.
    /// Both pass their already-populated slices here. `p_resolve` is null
    /// for the clear-only path (no resolve attachments) and either
    /// null or the resolve-refs pointer for MRT, decided by the caller.
    #[cfg(feature = "render")]
    fn create_transient_render_pass(
        &self,
        attachments: &[ffi::VkAttachmentDescription],
        color_refs: &[ffi::VkAttachmentReference],
        p_resolve: *const ffi::VkAttachmentReference,
    ) -> Result<ffi::VkRenderPass, QuantaError> {
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
        Ok(transient_rp)
    }

    pub(crate) fn render_begin_impl(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
            value_data: Vec::new(),
            color_targets: Vec::new(),
            depth_target: None,
            primary_format: Some(target.format()),
            primary_samples: Some(target.sample_count()),
            pipeline_shapes: Vec::new(),
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
            // Also fail loudly on a pipeline/target shape mismatch — a
            // phantom or mis-typed attachment that would otherwise
            // silently misrender.
            pass.validate_pass_shape()?;
        }

        let target_tex = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::not_found("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        // Determine if we have MRT color targets or just the single target.
        let has_mrt = !pass.color_targets.is_empty();

        // Whether this pass CLEARS the primary target or must PRESERVE
        // it. Drives the load op, the attachment's initial layout, and
        // the pre-pass barrier below.
        let primary_clears = if has_mrt {
            pass.color_targets
                .iter()
                .find(|ct| ct.texture == pass.handle)
                .map(|ct| !matches!(ct.load_op, LoadOp::Load))
                .unwrap_or(true)
        } else {
            pass.ops.iter().any(|op| matches!(op, RenderOp::Clear(_)))
        };
        // A VIRGIN primary (tracked layout still UNDEFINED — never
        // written) has nothing to preserve even when the pass declares
        // Load: its attachment downgrades to DONT_CARE and its barrier
        // takes the UNDEFINED wildcard. LOADing never-initialized
        // backing faults some drivers (Intel reads uninitialized CCS
        // compression state — STATUS_ACCESS_VIOLATION on the Iris Xe).
        let primary_virgin = target_tex
            .current_layout
            .load(std::sync::atomic::Ordering::Relaxed)
            == ffi::VK_IMAGE_LAYOUT_UNDEFINED;

        // Pipeline lookup — bind state (layout, descriptor shape) and
        // the attachment sample count. The pipeline's BAKED render pass
        // is used only at pipeline creation; it must never begin a
        // pass: its ops are hardcoded (load = CLEAR, initial layout =
        // UNDEFINED), so beginning with it wiped the target on every
        // pipeline-bound pass — a `LoadOp::Load` pass cleared to the
        // fallback (0,0,0,1). The per-pass transient render pass built
        // below carries the DECLARED ops; it stays compatible with the
        // pipeline because render-pass compatibility ignores load/store
        // ops and layouts (formats and sample counts must match — hence
        // `rp.samples` feeding the attachments).
        let pipeline_ref = match pipeline_handle {
            Some(ph) => Some(render_pipelines.get(&ph).ok_or_else(|| {
                QuantaError::not_found("pipeline not found")
                    .with_context(&format!("render_end: pipeline handle {}", ph))
            })?),
            None => None,
        };
        let attachment_samples = pipeline_ref
            .map(|rp| rp.samples)
            .unwrap_or(ffi::VK_SAMPLE_COUNT_1_BIT);

        let vk_render_pass = if has_mrt {
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
                // A Load on a VIRGIN attachment (tracked layout still
                // UNDEFINED — never written) downgrades to DONT_CARE:
                // nothing to preserve, and LOADing never-initialized
                // backing faults some drivers (Intel CCS). Same guard
                // as the single-target path.
                let ct_virgin = ct_tex
                    .current_layout
                    .load(std::sync::atomic::Ordering::Relaxed)
                    == ffi::VK_IMAGE_LAYOUT_UNDEFINED;
                let load_op = if load_op == ffi::VK_ATTACHMENT_LOAD_OP_LOAD && ct_virgin {
                    ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE
                } else {
                    load_op
                };
                let initial_layout = match ct.load_op {
                    LoadOp::Load if !ct_virgin => ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    _ => ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                };
                attachments.push(ffi::VkAttachmentDescription {
                    flags: 0,
                    format: ct_tex.format,
                    // The attachment carries the TARGET's real sample
                    // count. When a pipeline is bound this equals its
                    // rasterization samples — `validate_pass_shape`
                    // already rejected any disagreement — preserving
                    // render-pass compatibility (an MSAA pipeline draws
                    // into MSAA color attachments). When NO pipeline is
                    // bound (a clear-only pass over an MSAA target) the
                    // old pipeline-derived fallback of 1 sample was
                    // simply wrong: the framebuffer's image view is
                    // multisampled and the attachment must say so.
                    samples: sample_count_to_vk(ct.samples),
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
            self.create_transient_render_pass(&attachments, &color_refs, p_resolve)?
        } else {
            // Single-target pass: the load op is derived from the ops —
            // a `Clear` op means CLEAR, otherwise the pass must LOAD
            // (preserve) the target's previous contents. A loading
            // attachment declares its true initial layout; the pre-pass
            // barrier below puts the image there from its tracked
            // layout. EXCEPT a VIRGIN target (see `primary_virgin`
            // above): nothing to preserve → DONT_CARE, semantically
            // identical and it never reads the backing.
            let (load_op, initial_layout) = if primary_clears {
                (
                    ffi::VK_ATTACHMENT_LOAD_OP_CLEAR,
                    ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                )
            } else if primary_virgin {
                (
                    ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                )
            } else {
                (
                    ffi::VK_ATTACHMENT_LOAD_OP_LOAD,
                    ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                )
            };
            let color_attachment = ffi::VkAttachmentDescription {
                flags: 0,
                format: target_tex.format,
                // The target's real sample count (same reasoning as the
                // MRT arm above: equal to the pipeline's rasterization
                // samples whenever one is bound — validated — and the
                // only correct answer for a pipeline-less clear-only
                // pass over an MSAA target). Falls back to the
                // pipeline-derived count if the wrapper could not stamp
                // `primary_samples`.
                samples: pass
                    .primary_samples
                    .map(sample_count_to_vk)
                    .unwrap_or(attachment_samples),
                load_op,
                store_op: ffi::VK_ATTACHMENT_STORE_OP_STORE,
                stencil_load_op: ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencil_store_op: ffi::VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initial_layout,
                final_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
            let color_ref = ffi::VkAttachmentReference {
                attachment: 0,
                layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };
            self.create_transient_render_pass(
                core::slice::from_ref(&color_attachment),
                core::slice::from_ref(&color_ref),
                core::ptr::null(),
            )?
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

        // --- Descriptor pool: one set PER DRAW ---
        //
        // Textures/buffers/samplers are re-bound BETWEEN draws in one pass
        // ([SetTexture(A), Draw, SetTexture(B), Draw]); the op vector
        // preserves that call order. So — mirroring the compute
        // per-dispatch pattern (write_texture_descriptors +
        // acquire_descriptor_pool) — each draw snapshots the running slot
        // state into its OWN freshly-allocated descriptor set, written and
        // bound immediately before the draw. A single whole-pass set (the
        // old shape) collapsed every binding into one flat set, so every
        // draw sampled the LAST texture bound.
        let descriptor_pool;
        let rp_layout;
        let rp_ds_layout;

        // Initial running slot state, seeded into `EncoderState` below
        // and then mutated in op order inside the replay loop. A texture
        // slot holds the VkImageView to sample; a buffer slot holds the
        // VkBuffer to bind; a sampler slot holds the resolved VkSampler.
        // `None` = slot unbound. Each draw reads these to build its set,
        // so a re-bind before the next draw is reflected exactly.
        let texture_for_slot: [Option<ffi::VkImageView>; 8] = [None; 8];
        let sampler_for_slot: [Option<ffi::VkSampler>; 8] = [None; 8];
        let buffer_for_slot: [Option<ffi::VkBuffer>; 8] = [None; 8];

        if let Some(rp) = pipeline_ref {
            // Count the draws so the pool holds one set each. Every draw
            // family that consumes a descriptor set counts.
            let n_draws = pass
                .ops
                .iter()
                .filter(|op| {
                    matches!(
                        op,
                        RenderOp::Draw { .. }
                            | RenderOp::DrawIndexed { .. }
                            | RenderOp::DrawIndirect { .. }
                            | RenderOp::DrawIndexedIndirect { .. }
                    )
                })
                .count();
            // A pipeline-only pass with zero draws still creates a pool
            // (max_sets clamped to 1) but allocates no set — it must not
            // crash.
            let max_sets = core::cmp::max(1, n_draws) as u32;

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
                    descriptor_count: 8 * max_sets,
                },
                ffi::VkDescriptorPoolSize {
                    ty: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 8 * max_sets,
                },
            ];
            let pool_info = ffi::VkDescriptorPoolCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                max_sets,
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
            rp_layout = Some(rp.layout);
            rp_ds_layout = Some(rp.descriptor_set_layout);
        } else {
            descriptor_pool = None;
            rp_layout = None;
            rp_ds_layout = None;
        }

        // Default sampler — LINEAR min/mag/mip, CLAMP_TO_EDGE, no
        // anisotropy. Resolved through the per-device cache like every
        // other sampler, and only when a texture slot actually lacks an
        // explicit SetSampler — a pass that samples nothing (or sets a
        // sampler on every textured slot) must not grow the cache with an
        // entry it never binds. `mip_filter: Linear` here reproduces the
        // historical hardcoded default (which used MIPMAP_MODE_LINEAR),
        // NOT `SamplerDesc::default()` — the latter maps mip to NEAREST.
        let default_desc = crate::texture::SamplerDesc {
            min_filter: crate::texture::Filter::Linear,
            mag_filter: crate::texture::Filter::Linear,
            mip_filter: crate::texture::Filter::Linear,
            address_u: crate::texture::AddressMode::ClampToEdge,
            address_v: crate::texture::AddressMode::ClampToEdge,
            max_anisotropy: 1,
            compare: None,
        };

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

            // Transition target image to COLOR_ATTACHMENT_OPTIMAL. A
            // clearing pass may transition from UNDEFINED (contents die
            // anyway — and it is the universal wildcard). A LOADING
            // pass must transition from the image's TRACKED layout and
            // make prior writes (previous pass, transfer upload)
            // visible — transitioning a preserved target from UNDEFINED
            // legally discards its contents (and Intel really does).
            let (bar_old_layout, bar_src_access, bar_src_stage, bar_dst_access) = if primary_clears
                || primary_virgin
            {
                (
                    ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                    0,
                    ffi::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                )
            } else {
                (
                    target_tex
                        .current_layout
                        .load(std::sync::atomic::Ordering::Relaxed),
                    ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                    ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        | ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                        | ffi::VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                )
            };
            let barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: bar_src_access,
                dst_access_mask: bar_dst_access,
                old_layout: bar_old_layout,
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
                bar_src_stage,
                ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &barrier,
            );

            // A LOADing NON-primary color target needs the same
            // write→attachment dependency the primary got above. Its
            // contents were stored by an earlier submission, and the
            // transient pass declares initial layout
            // COLOR_ATTACHMENT_OPTIMAL for it (no implicit transition,
            // hence no implicit availability) — without an explicit
            // barrier nothing makes those prior writes visible to this
            // pass's load. The builder-managed MSAA intermediate reads
            // its preserved samples through exactly this edge (pass 2
            // of the clear→store / load→resolve flow). Virgin targets
            // (tracked layout UNDEFINED) were downgraded to DONT_CARE
            // when the attachment was declared: nothing to order.
            for ct in &pass.color_targets {
                if ct.texture == pass.handle || !matches!(ct.load_op, LoadOp::Load) {
                    continue;
                }
                let Some(ct_tex) = textures.get(&ct.texture) else {
                    continue;
                };
                let old_layout = ct_tex
                    .current_layout
                    .load(std::sync::atomic::Ordering::Relaxed);
                if old_layout == ffi::VK_IMAGE_LAYOUT_UNDEFINED {
                    continue;
                }
                let load_barrier = ffi::VkImageMemoryBarrier {
                    s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    p_next: core::ptr::null(),
                    src_access_mask: ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                        | ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                    dst_access_mask: ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                        | ffi::VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                    old_layout,
                    new_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                    image: ct_tex.image,
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
                    ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        | ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    0,
                    0,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    1,
                    &load_barrier,
                );
            }

            // Transition every SAMPLED source texture to
            // SHADER_READ_ONLY_OPTIMAL before the pass. `write_render_
            // descriptors` writes SHADER_READ_ONLY_OPTIMAL into each
            // combined-image-sampler descriptor but emits NO barrier —
            // it assumes the image already arrived there. That holds for
            // an uploaded texture or a `resolve_texture` output (both end
            // SHADER_READ_ONLY), but a texture just RENDERED into sits in
            // COLOR_ATTACHMENT_OPTIMAL, and sampling it then is a layout
            // mismatch (VUID-vkCmdDraw-None-09600) that device-loses some
            // drivers (Intel Iris Xe). Consult each source's tracked
            // layout and barrier it from WHATEVER it is → SHADER_READ_ONLY,
            // the same generality `resolve_texture_impl` applies to its
            // source. A texture that is also the render target is skipped
            // (a read-after-write feedback loop is invalid API usage, and
            // it must stay COLOR_ATTACHMENT for the draw).
            for handle in sampled_source_handles(&pass) {
                if handle == pass.handle || pass.color_targets.iter().any(|ct| ct.texture == handle)
                {
                    continue;
                }
                let Some(src) = textures.get(&handle) else {
                    continue;
                };
                let old_layout = src
                    .current_layout
                    .load(std::sync::atomic::Ordering::Relaxed);
                if old_layout == ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL {
                    continue;
                }
                let sample_barrier = ffi::VkImageMemoryBarrier {
                    s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    p_next: core::ptr::null(),
                    src_access_mask: ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                        | ffi::VK_ACCESS_TRANSFER_WRITE_BIT,
                    dst_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT,
                    old_layout,
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
                ffi::vkCmdPipelineBarrier(
                    cmd,
                    ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                        | ffi::VK_PIPELINE_STAGE_TRANSFER_BIT,
                    ffi::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    0,
                    0,
                    core::ptr::null(),
                    0,
                    core::ptr::null(),
                    1,
                    &sample_barrier,
                );
                src.current_layout.store(
                    ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

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

            // Default full-target viewport + scissor, in the same
            // y-flipped convention the explicit SetViewport path uses.
            // Every pipeline declares dynamic viewport/scissor, so a
            // pass that never calls set_viewport would otherwise draw
            // with the state UNSET — undefined behavior
            // (VUID-vkCmdDraw-None-07831/07832) that faults some
            // drivers (STATUS_ACCESS_VIOLATION on Intel Iris Xe).
            // Metal defaults to the full target; this makes the
            // backends agree. A SetViewport/SetScissor op in the pass
            // simply overrides these. Bundle passes record their own
            // dynamic state inside the secondary buffers — the inline
            // defaults here are inert for them.
            if !uses_bundles {
                let default_viewport = ffi::VkViewport {
                    x: 0.0,
                    y: target_tex.height as f32,
                    width: target_tex.width as f32,
                    height: -(target_tex.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let default_scissor = ffi::VkRect2D {
                    offset: ffi::VkOffset2D { x: 0, y: 0 },
                    extent: ffi::VkExtent2D {
                        width: target_tex.width,
                        height: target_tex.height,
                    },
                };
                ffi::vkCmdSetViewport(cmd, 0, 1, &default_viewport);
                ffi::vkCmdSetScissor(cmd, 0, 1, &default_scissor);
            }

            // Running encoder state, mutated in op order inside the
            // op-walk. Seeded from the slot arrays built above so the
            // running snapshot each draw reads reflects every prior bind.
            let mut state = EncoderState {
                texture_for_slot,
                sampler_for_slot,
                buffer_for_slot,
                current_index_buffer: None,
            };
            // Read-only context every op-walk helper reads; borrows the
            // already-locked resource maps and the per-pass layouts for
            // the loop's duration.
            let ctx = DrawContext {
                cmd,
                value_data: &pass.value_data,
                render_pipelines: &render_pipelines,
                buffers: &buffers,
                textures: &textures,
                descriptor_pool,
                rp_ds_layout,
                rp_layout,
                default_desc: &default_desc,
                pipeline_ref,
                target_w: target_tex.width,
                target_h: target_tex.height,
            };

            // Encode each RenderOp — dispatch to the op-family helper.
            for op in &pass.ops {
                match op {
                    RenderOp::SetPipeline(_)
                    | RenderOp::BindVertices { .. }
                    | RenderOp::BindIndices { .. }
                    | RenderOp::SetField { .. }
                    | RenderOp::SetUniform { .. }
                    | RenderOp::SetTexture { .. }
                    | RenderOp::SetSampler { .. }
                    | RenderOp::SetValue { .. } => {
                        self.encode_binding_op(&mut state, &ctx, op)?;
                    }

                    RenderOp::SetViewport { .. }
                    | RenderOp::SetScissor { .. }
                    | RenderOp::SetStencilRef(_) => {
                        self.encode_dynamic_state_op(&ctx, op);
                    }

                    RenderOp::Draw { .. }
                    | RenderOp::DrawIndexed { .. }
                    | RenderOp::DrawIndirect { .. }
                    | RenderOp::DrawIndexedIndirect { .. } => {
                        self.encode_draw_op(&mut state, &ctx, op)?;
                    }

                    RenderOp::Clear(_)
                    | RenderOp::ClearDepth(_)
                    | RenderOp::ClearStencil(_)
                    | RenderOp::DebugPush { .. }
                    | RenderOp::DebugPop
                    | RenderOp::BeginOcclusionQuery { .. }
                    | RenderOp::EndOcclusionQuery { .. }
                    | RenderOp::ExecuteRenderBundle { .. } => {
                        self.encode_query_debug_op(&ctx, op)?;
                    }

                    RenderOp::SetShadingRate(_) | RenderOp::SetShadingRateImage { .. } => {
                        self.encode_vrs_op(&ctx, op)?;
                    }
                }
            }

            ffi::vkCmdEndRenderPass(cmd);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        // The begin pass is ALWAYS per-pass transient now (the
        // pipeline's baked pass never begins a pass), so it is always
        // cleaned up with the framebuffer.
        let transient_rp = Some(vk_render_pass);
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
                destroy_render_pass_objects(
                    self.device,
                    framebuffer,
                    transient_rp,
                    descriptor_pool,
                );
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
            // The cleanup above waits the submit fence when it drops —
            // exactly the deferred device work the keep-alive protects:
            // a consumer holding this pulse past its last Gpu handle
            // must not have the wait dangle on a destroyed VkDevice.
            keep_alive: self.self_ref.pulse_keep_alive(),
        })
    }

    /// Encode a binding / pipeline-state op — the ops that mutate the
    /// running slot state or bind a pipeline / vertex-index buffer /
    /// push-constant range without themselves issuing a draw. The NEXT
    /// draw snapshots the slot state into its own descriptor set, so a
    /// re-bind here reaches exactly the following draw.
    ///
    /// Resolves each handle exactly as the old whole-pass pre-scan did
    /// (same map lookups, same sampler cache).
    #[cfg(feature = "render")]
    fn encode_binding_op(
        &self,
        state: &mut EncoderState,
        ctx: &DrawContext<'_>,
        op: &RenderOp,
    ) -> Result<(), QuantaError> {
        let cmd = ctx.cmd;
        match op {
            RenderOp::SetPipeline(handle) => {
                if let Some(rp) = ctx.render_pipelines.get(handle) {
                    unsafe {
                        ffi::vkCmdBindPipeline(
                            cmd,
                            ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                            rp.pipeline,
                        );
                    }
                    // The descriptor set is bound PER DRAW (below),
                    // not here — a re-bind between draws in this
                    // pass must reach the following draw's set.
                }
            }

            RenderOp::BindVertices {
                slot,
                handle,
                offset,
            } => {
                if let Some(buf) = ctx.buffers.get(handle) {
                    let offsets = [*offset];
                    unsafe {
                        ffi::vkCmdBindVertexBuffers(cmd, *slot, 1, &buf.buffer, offsets.as_ptr());
                    }
                }
            }

            RenderOp::BindIndices { handle, offset } => {
                if let Some(buf) = ctx.buffers.get(handle) {
                    unsafe {
                        ffi::vkCmdBindIndexBuffer(
                            cmd,
                            buf.buffer,
                            *offset,
                            ffi::VK_INDEX_TYPE_UINT32,
                        );
                    }
                    state.current_index_buffer = Some(buf.buffer);
                }
            }

            // Resource binds mutate the running slot state; the
            // NEXT draw snapshots it into its own descriptor set.
            // Resolve each handle exactly as the old whole-pass
            // pre-scan did (same map lookups, same sampler cache).
            RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                let idx = *slot as usize;
                if idx < 8
                    && let Some(buf) = ctx.buffers.get(handle)
                {
                    state.buffer_for_slot[idx] = Some(buf.buffer);
                }
            }
            RenderOp::SetTexture { slot, handle } => {
                let idx = *slot as usize;
                if idx < 8
                    && let Some(tex) = ctx.textures.get(handle)
                {
                    state.texture_for_slot[idx] = Some(tex.view);
                }
            }
            RenderOp::SetSampler { slot, sampler } => {
                let idx = *slot as usize;
                if idx < 8 {
                    // Identical descriptors across draws/frames
                    // reuse one cached sampler, so the sampler pool
                    // stays bounded by distinct descriptors.
                    let s = self.get_or_create_render_sampler(sampler);
                    if !s.is_null() {
                        state.sampler_for_slot[idx] = Some(s);
                    }
                }
            }

            RenderOp::SetValue { slot, offset, len } => {
                // The pipeline layout declares exactly one
                // [0,128) VERTEX|FRAGMENT push range (8 slots
                // x 16 bytes); pushing outside it or with a
                // non-multiple-of-4 size is invalid API usage
                // (VUID-vkCmdPushConstants-offset-01795 /
                // size-00369) — fail the pass loudly instead.
                let pc_offset = *slot as usize * 16;
                if *slot >= 8 || *len % 4 != 0 || pc_offset + *len > 128 {
                    return Err(QuantaError::invalid_param(format!(
                        "SetValue slot {} with {} bytes exceeds the Vulkan \
                         push-constant contract (slots 0-7, 4-byte-aligned, \
                         slot*16 + size <= 128)",
                        slot, len
                    )));
                }
                if let Some(rp) = ctx.pipeline_ref {
                    let data = &ctx.value_data[*offset..*offset + *len];
                    unsafe {
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
            }

            _ => {}
        }
        Ok(())
    }

    /// Encode a dynamic-state op — viewport, scissor, stencil reference.
    /// These record command-buffer dynamic state and touch neither the
    /// slot state nor a descriptor set.
    #[cfg(feature = "render")]
    fn encode_dynamic_state_op(&self, ctx: &DrawContext<'_>, op: &RenderOp) {
        let cmd = ctx.cmd;
        match op {
            RenderOp::SetViewport {
                x,
                y,
                width,
                height,
                min_depth,
                max_depth,
            } => {
                // Negative-viewport y-flip (Vulkan 1.1
                // maintenance1, core — no extension probe; lavapipe
                // and v3dv are both >= 1.1). Vulkan's default NDC is
                // y-down; Metal's and WGSL's are y-up. Emitting the
                // viewport with its origin moved to the bottom edge
                // (y + height) and a NEGATIVE height mirrors the y
                // axis, so a +Y-up clip position lands on the same
                // framebuffer row Metal and WebGPU put it on. This is
                // what makes the same DSL source produce the same
                // pixels on every backend (readback row 0 = the top
                // row). Depth range (min/max_depth) is unchanged.
                let viewport = ffi::VkViewport {
                    x: *x,
                    y: *y + *height,
                    width: *width,
                    height: -*height,
                    min_depth: *min_depth,
                    max_depth: *max_depth,
                };
                // Set default scissor to match viewport (required for
                // dynamic state). Scissor is FRAMEBUFFER-space and is
                // NOT mirrored by the negative viewport, so it keeps
                // the original (un-flipped) x/y/width/height. Route it
                // through the same clamp so a viewport placed with a
                // negative origin can't emit a negative scissor
                // offset. `f32 as i32 as u32` preserves a negative
                // origin as the wrapped-in u32 the clamp decodes (a
                // bare `f32 as u32` would saturate the sign away).
                let scissor = clamp_scissor(
                    *x as i32 as u32,
                    *y as i32 as u32,
                    *width as u32,
                    *height as u32,
                    ctx.target_w,
                    ctx.target_h,
                );
                unsafe {
                    ffi::vkCmdSetViewport(cmd, 0, 1, &viewport);
                    ffi::vkCmdSetScissor(cmd, 0, 1, &scissor);
                }
            }

            RenderOp::SetScissor {
                x,
                y,
                width,
                height,
            } => {
                // Clamp to the render area: a negative (wrapped-in)
                // offset or an oversized rect becomes a valid clipped
                // rect, matching Metal's tolerated behavior instead of
                // tripping VUID-vkCmdSetScissor-x-00595.
                let scissor = clamp_scissor(*x, *y, *width, *height, ctx.target_w, ctx.target_h);
                unsafe {
                    ffi::vkCmdSetScissor(cmd, 0, 1, &scissor);
                }
            }

            RenderOp::SetStencilRef(value) => unsafe {
                ffi::vkCmdSetStencilReference(cmd, ffi::VK_STENCIL_FACE_FRONT_AND_BACK, *value);
            },

            _ => {}
        }
    }

    /// Encode a draw op — the four draw families. Each first binds a
    /// fresh per-draw descriptor set (the running slot snapshot) via the
    /// shared `bind_draw_set` unit, then issues the `vkCmdDraw*`. The
    /// indexed-indirect arm rebinds the index buffer first when it
    /// differs from the currently bound one.
    #[cfg(feature = "render")]
    fn encode_draw_op(
        &self,
        state: &mut EncoderState,
        ctx: &DrawContext<'_>,
        op: &RenderOp,
    ) -> Result<(), QuantaError> {
        let cmd = ctx.cmd;

        // One descriptor set per draw: allocate from the per-pass
        // pool, write the running slot snapshot, bind — then draw.
        // Guarded on the pipeline being present (a pipeline-less pass
        // has no set to bind and issues no draws that consume one).
        macro_rules! bind_draw_set {
            () => {
                if let (Some(pool), Some(dsl), Some(pl)) =
                    (ctx.descriptor_pool, ctx.rp_ds_layout, ctx.rp_layout)
                {
                    self.bind_draw_descriptor_set(
                        cmd,
                        pool,
                        dsl,
                        pl,
                        &state.texture_for_slot,
                        &state.sampler_for_slot,
                        &state.buffer_for_slot,
                        ctx.default_desc,
                    )?;
                }
            };
        }

        unsafe {
            match op {
                RenderOp::Draw {
                    vertex_count,
                    instance_count,
                } => {
                    bind_draw_set!();
                    ffi::vkCmdDraw(cmd, *vertex_count, *instance_count, 0, 0);
                }

                RenderOp::DrawIndexed {
                    index_count,
                    instance_count,
                } => {
                    bind_draw_set!();
                    ffi::vkCmdDrawIndexed(cmd, *index_count, *instance_count, 0, 0, 0);
                }

                RenderOp::DrawIndirect {
                    buffer_handle,
                    offset,
                } => {
                    if let Some(buf) = ctx.buffers.get(buffer_handle) {
                        bind_draw_set!();
                        ffi::vkCmdDrawIndirect(cmd, buf.buffer, *offset, 1, 0);
                    }
                }

                RenderOp::DrawIndexedIndirect {
                    buffer_handle,
                    offset,
                    index_handle,
                } => {
                    if let Some(idx_buf) = ctx.buffers.get(index_handle) {
                        let needs_rebind = state
                            .current_index_buffer
                            .map(|b| b != idx_buf.buffer)
                            .unwrap_or(true);
                        if needs_rebind {
                            ffi::vkCmdBindIndexBuffer(
                                cmd,
                                idx_buf.buffer,
                                0,
                                ffi::VK_INDEX_TYPE_UINT32,
                            );
                            state.current_index_buffer = Some(idx_buf.buffer);
                        }
                    }
                    if let Some(buf) = ctx.buffers.get(buffer_handle) {
                        bind_draw_set!();
                        ffi::vkCmdDrawIndexedIndirect(cmd, buf.buffer, *offset, 1, 0);
                    }
                }

                _ => {}
            }
        }
        Ok(())
    }

    /// Encode a query / debug / clear / bundle op — the ops that touch
    /// query pools, debug markers (currently no-ops), the clear ops
    /// (folded into the render-pass load ops, so no-ops here), and
    /// secondary-command-buffer execution for render bundles. An error
    /// arm ends the render pass before returning, matching the original
    /// inline sequence.
    #[cfg(feature = "render")]
    fn encode_query_debug_op(
        &self,
        ctx: &DrawContext<'_>,
        op: &RenderOp,
    ) -> Result<(), QuantaError> {
        let cmd = ctx.cmd;
        match op {
            RenderOp::Clear(_) | RenderOp::ClearDepth(_) | RenderOp::ClearStencil(_) => {}
            RenderOp::DebugPush { .. } | RenderOp::DebugPop => {}

            // Occlusion queries (M3.3)
            RenderOp::BeginOcclusionQuery { handle, index } => {
                let pools = self
                    .query_pools
                    .read()
                    .map_err(|_| QuantaError::internal("lock poisoned"))?;
                if let Some(qp) = pools.get(handle) {
                    unsafe {
                        ffi::vkCmdResetQueryPool(cmd, qp.pool, *index, 1);
                        ffi::vkCmdBeginQuery(cmd, qp.pool, *index, 0);
                    }
                }
            }
            RenderOp::EndOcclusionQuery { handle, index } => {
                let pools = self
                    .query_pools
                    .read()
                    .map_err(|_| QuantaError::internal("lock poisoned"))?;
                if let Some(qp) = pools.get(handle) {
                    unsafe {
                        ffi::vkCmdEndQuery(cmd, qp.pool, *index);
                    }
                }
            }

            RenderOp::ExecuteRenderBundle {
                bundle_handle,
                count,
            } => {
                let bundles = self
                    .render_bundles
                    .read()
                    .map_err(|_| QuantaError::internal("lock poisoned"))?;
                let bundle = bundles
                    .get(bundle_handle)
                    .ok_or_else(|| QuantaError::not_found("render bundle handle not found"))?;
                if *count > bundle.recorded {
                    unsafe {
                        ffi::vkCmdEndRenderPass(cmd);
                    }
                    return Err(QuantaError::invalid_param(
                        "execute_bundle count exceeds recorded length",
                    ));
                }
                if *count > 0 {
                    unsafe {
                        ffi::vkCmdExecuteCommands(cmd, *count, bundle.secondaries.as_ptr());
                    }
                }
            }

            _ => {}
        }
        Ok(())
    }

    /// Encode a variable-rate-shading op (step 063). `SetShadingRate`
    /// lowers to `vkCmdSetFragmentShadingRateKHR` when the extension is
    /// present and the requested rate is in the device's supported list;
    /// `SetShadingRateImage` (texel-driven rates) is deferred. Every
    /// error arm ends the render pass before returning, exactly as the
    /// original inline sequence did.
    #[cfg(feature = "render")]
    fn encode_vrs_op(&self, ctx: &DrawContext<'_>, op: &RenderOp) -> Result<(), QuantaError> {
        let cmd = ctx.cmd;
        match op {
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
                        unsafe {
                            ffi::vkCmdEndRenderPass(cmd);
                        }
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
                    unsafe {
                        set_rate(cmd, &extent, combiner_ops.as_ptr());
                    }
                } else {
                    unsafe {
                        ffi::vkCmdEndRenderPass(cmd);
                    }
                    return Err(QuantaError::not_supported(
                        "Vulkan render encoder: VK_KHR_fragment_shading_rate is not available on this device",
                    ));
                }
            }
            RenderOp::SetShadingRateImage { .. } => {
                unsafe {
                    ffi::vkCmdEndRenderPass(cmd);
                }
                return Err(QuantaError::not_supported(
                    "Vulkan render encoder: shading-rate-image (texel-driven VRS) deferred",
                ));
            }

            _ => {}
        }
        Ok(())
    }
}

/// Running encoder state threaded through the op-walk in
/// `render_end_impl`, mutated in op order.
///
/// A texture slot holds the `VkImageView` to sample; a buffer slot holds
/// the `VkBuffer` to bind; a sampler slot holds the resolved `VkSampler`.
/// `None` = slot unbound. Each draw reads these to build its own
/// descriptor set, so a re-bind before the next draw is reflected
/// exactly. `current_index_buffer` tracks the last bound index buffer so
/// `DrawIndexedIndirect` only rebinds on a change.
#[cfg(feature = "render")]
struct EncoderState {
    texture_for_slot: [Option<ffi::VkImageView>; 8],
    sampler_for_slot: [Option<ffi::VkSampler>; 8],
    buffer_for_slot: [Option<ffi::VkBuffer>; 8],
    current_index_buffer: Option<ffi::VkBuffer>,
}

/// The read-only context an op-walk helper reads: the recording command
/// buffer, the three resource maps (already locked in `render_end_impl`),
/// the per-pass descriptor pool and its layouts, the default sampler
/// descriptor, the active pipeline (for push constants), and the render
/// area dimensions (for scissor/viewport clamping). Borrowed for the
/// duration of the op-walk; none of these change across ops.
#[cfg(feature = "render")]
struct DrawContext<'a> {
    cmd: ffi::VkCommandBuffer,
    /// The pass's payload arena (`SetValue` bytes live here).
    value_data: &'a [u8],
    render_pipelines: &'a std::collections::HashMap<u64, super::super::VkRenderPipeline>,
    buffers: &'a std::collections::HashMap<u64, super::super::VkBuffer>,
    textures: &'a std::collections::HashMap<u64, super::super::VkTexture>,
    descriptor_pool: Option<ffi::VkDescriptorPool>,
    rp_ds_layout: Option<ffi::VkDescriptorSetLayout>,
    rp_layout: Option<ffi::VkPipelineLayout>,
    default_desc: &'a crate::texture::SamplerDesc,
    pipeline_ref: Option<&'a super::super::VkRenderPipeline>,
    target_w: u32,
    target_h: u32,
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
        destroy_render_pass_objects(
            self.device,
            self.framebuffer,
            self.transient_rp,
            self.descriptor_pool,
        );
    }
}

/// Destroy the per-pass framebuffer plus the optional transient render
/// pass and descriptor pool, in that fixed order.
///
/// The single owner of this destroy sequence: the submit-failure path in
/// `render_end_impl` (the submission never reached the queue, so nothing
/// on the GPU references these) and `RenderPassCleanup::drop` (after the
/// submission fence has been waited) call it with the same handles. The
/// caller is responsible for ensuring the GPU no longer references these
/// objects before invoking it.
#[cfg(feature = "render")]
fn destroy_render_pass_objects(
    device: ffi::VkDevice,
    framebuffer: ffi::VkFramebuffer,
    transient_rp: Option<ffi::VkRenderPass>,
    descriptor_pool: Option<ffi::VkDescriptorPool>,
) {
    unsafe {
        ffi::vkDestroyFramebuffer(device, framebuffer, core::ptr::null());
        if let Some(rp) = transient_rp {
            ffi::vkDestroyRenderPass(device, rp, core::ptr::null());
        }
        if let Some(pool) = descriptor_pool {
            ffi::vkDestroyDescriptorPool(device, pool, core::ptr::null());
        }
    }
}
