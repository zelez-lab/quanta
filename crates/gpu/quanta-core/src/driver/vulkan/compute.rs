//! Compute dispatch operations for Vulkan.

use alloc::format;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::{Pulse, QuantaError, Wave};
use std::collections::HashMap;
use std::ffi::CString;
use std::process::Stdio;

use super::ffi;
use super::{VkComputePipeline, VulkanDevice};

/// Try to optimize SPIR-V binary via spirv-opt if available.
/// Falls back to the original input on any failure (missing binary, crash, etc.).
fn try_optimize_spirv(spirv: &[u8]) -> Vec<u8> {
    let child = std::process::Command::new("spirv-opt")
        .args(["--target-env=vulkan1.3", "-O", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(_) => return spirv.to_vec(),
    };
    // Write SPIR-V to stdin
    if let Some(ref mut stdin) = child.stdin.take() {
        use std::io::Write;
        if stdin.write_all(spirv).is_err() {
            let _ = child.wait();
            return spirv.to_vec();
        }
    }
    match child.wait_with_output() {
        Ok(output) if output.status.success() && !output.stdout.is_empty() => output.stdout,
        _ => spirv.to_vec(),
    }
}

impl VulkanDevice {
    /// JIT-compile a kernel from serialized KernelDef IR.
    ///
    /// Deserializes the IR, emits SPIR-V binary, and creates a Vulkan pipeline.
    #[cfg(feature = "jit")]
    pub(crate) fn wave_jit_impl(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        let kernel = quanta_ir::deserialize_kernel(kernel_def_bytes)
            .map_err(|e| QuantaError::compilation_failed(format!("JIT deserialize: {}", e)))?;

        // Step 082 Layer 4: validate against Vulkan's capability
        // table. Hard NotSupported types (none today on Vulkan —
        // F64/F16 are RequiresFeature, which the validator passes
        // through soft) get rejected here. RequiresFeature types
        // are deferred to the runtime device-caps check
        // (Gpu::supports_*).
        let report = quanta_ir::validate::validate_for(&quanta_ir::caps::VULKAN, &kernel);
        if !report.is_ok() {
            return Err(QuantaError::not_supported(
                "kernel uses unsupported scalar type for Vulkan",
            )
            .with_context(&format!("{}", report)));
        }

        let spirv = quanta_ir::emit_spirv::emit(&kernel)
            .map_err(|e| QuantaError::compilation_failed(format!("JIT SPIR-V emit: {}", e)))?;
        let mut wave = self.wave_impl(&spirv)?;
        // The KernelDef is authoritative for the workgroup size. If the Wave
        // carried a different value, `wave_dispatch_threads` would compute a
        // group count for the wrong local size and silently under-dispatch
        // (the [64,1,1] guess vs quanta-array's LocalSize-1 kernels ran only
        // ⌈n/64⌉ of n threads — zeros for the remaining 63/64 of the output).
        wave.workgroup_size = kernel.workgroup_size;
        Ok(wave)
    }

    pub(crate) fn wave_impl(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        // The compiler produces SPIR-V binary directly -- interpret bytes as u32 words.
        // Check for SPIR-V magic number (0x07230203). If absent, this is likely
        // WGSL text from the fallback emitter — reject with a clear error.
        if kernel.len() < 4 {
            return Err(QuantaError::compilation_failed(
                "kernel binary too short for SPIR-V",
            ));
        }
        let magic = u32::from_le_bytes([kernel[0], kernel[1], kernel[2], kernel[3]]);
        if magic != 0x07230203 {
            return Err(QuantaError::compilation_failed(
                "Vulkan requires SPIR-V binary (magic 0x07230203). Got text shader — \
                 install quanta-compiler or build with LLVM for SPIR-V output.",
            ));
        }
        // LLVM's SPIR-V backend may emit a trailing byte — truncate to word boundary.
        let kernel = &kernel[..kernel.len() & !3];
        // Try spirv-opt optimization pass (no-op if spirv-opt not installed)
        let optimized = try_optimize_spirv(kernel);
        let spirv_words: Vec<u32> = optimized
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Read the module's declared workgroup size so thread-count
        // dispatches (`wave_dispatch_threads`) compute the right group
        // count. Falling back to [64,1,1] keeps the old behavior only for
        // modules that don't declare a literal LocalSize.
        let workgroup_size =
            crate::driver::spirv_meta::local_size(&spirv_words).unwrap_or([64, 1, 1]);

        // Create shader module
        let module_info = ffi::VkShaderModuleCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            code_size: optimized.len(),
            p_code: spirv_words.as_ptr(),
        };
        let mut shader_module = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateShaderModule(
                self.device,
                &module_info,
                core::ptr::null(),
                &mut shader_module,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "shader module: VkResult {}",
                result
            )));
        }

        // Descriptor set layout. Default: 8 storage buffers (stays within
        // maxPerStageDescriptorStorageBuffers on mobile GPUs) — the buffer-only
        // fast path, unchanged. Reflection then overrides/extends slots the
        // kernel binds as textures (storage or sampled images), so the layout
        // matches the module's actual descriptors for AOT and JIT SPIR-V.
        use crate::driver::spirv_meta::DescriptorKind;
        let reflected = crate::driver::spirv_meta::binding_kinds(&spirv_words);
        let max_binding = reflected.iter().map(|&(b, _)| b).max().unwrap_or(0);
        let mut descriptor_kinds =
            alloc::vec![DescriptorKind::StorageBuffer; (max_binding as usize + 1).max(8)];
        for &(binding, kind) in &reflected {
            descriptor_kinds[binding as usize] = kind;
        }
        // Sampled-image slots (`&Sampled2D` reads) bind as COMBINED_IMAGE_SAMPLER
        // with the device compute sampler (F3), exactly like the render path;
        // the dispatch descriptor-write and `prepare_compute_textures` handle
        // that kind. Storage (load/write) slots stay STORAGE_IMAGE in GENERAL.
        let descriptor_set_layout = self.acquire_descriptor_set_layout(&descriptor_kinds)?;

        // Declare a push constant range. Clamp to device limit (128 on mobile, 256 on desktop).
        let push_size = self.max_push_constants_size.min(256);
        let push_range = ffi::VkPushConstantRange {
            stage_flags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_size,
        };
        let pipeline_layout_info = ffi::VkPipelineLayoutCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
        };
        let mut pipeline_layout = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreatePipelineLayout(
                self.device,
                &pipeline_layout_info,
                core::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "pipeline layout: VkResult {}",
                result
            )));
        }

        let entry_name = CString::new("main").unwrap();
        let stage = ffi::VkPipelineShaderStageCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            stage: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: entry_name.as_ptr(),
            p_specialization_info: core::ptr::null(),
        };

        // Folded 1D dispatches issue their remainder row through
        // vkCmdDispatchBase with a non-zero base workgroup, which is
        // only valid on pipelines created with the DISPATCH_BASE flag
        // (core Vulkan 1.1). Set it whenever the entry point resolved
        // so any wave can be folded when its group count exceeds
        // maxComputeWorkGroupCount[0].
        let pipeline_flags = if self.dispatch_base_fn.is_some() {
            ffi::VK_PIPELINE_CREATE_DISPATCH_BASE
        } else {
            0
        };
        let pipeline_info = ffi::VkComputePipelineCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: pipeline_flags,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: ffi::null_handle(),
            base_pipeline_index: -1,
        };

        let mut pipeline = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateComputePipelines(
                self.device,
                self.pipeline_cache,
                1,
                &pipeline_info,
                core::ptr::null(),
                &mut pipeline,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "compute pipeline: VkResult {}",
                result
            )));
        }

        // Clean up shader module (pipeline owns the code now)
        unsafe {
            ffi::vkDestroyShaderModule(self.device, shader_module, core::ptr::null());
        }

        let handle = self.alloc_handle();
        self.compute_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                VkComputePipeline {
                    pipeline,
                    layout: pipeline_layout,
                    descriptor_set_layout,
                    descriptor_kinds,
                },
            );

        Ok(Wave {
            handle,
            bindings: [0u64; 16],
            binding_count: 0,
            texture_bindings: [0u64; 16],
            texture_count: 0,
            storage_texture_kinds: [0; 16],
            push_data: [0u8; 256],
            push_len: 0,
            push_mask: 0,
            workgroup_size,
            device: None,
            live: true,
        })
    }

    pub(crate) fn wave_dispatch_impl(
        &self,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_records_impl(wave, &[([0, 0, 0], groups)])
    }

    /// Dispatch by total thread count, folding oversized 1D dispatches
    /// into a 2D grid. When `ceil(quarks / wg_x)` exceeds the device's
    /// `maxComputeWorkGroupCount[0]`, the groups are split into a
    /// full-rows rectangle of `FOLD_ROW_GROUPS`-wide rows plus a
    /// remainder row issued at base workgroup (0, full_rows) via
    /// `vkCmdDispatchBase` — no waste threads, so unguarded elementwise
    /// kernels stay exact. The SPIR-V emitters bake the matching
    /// linearization into `QuarkId` / `NucleusId` (see
    /// `quanta_ir::dispatch_fold`), so 1D dispatch semantics are
    /// unchanged; the grid is merely physically 2D.
    pub(crate) fn wave_dispatch_threads_impl(
        &self,
        wave: &Wave,
        quarks: u32,
    ) -> Result<Pulse, QuantaError> {
        let wg_x = wave.workgroup_size[0].max(1);
        let groups = quarks.div_ceil(wg_x);
        let limit_x = self.caps.max_groups[0].max(1);
        if groups <= limit_x {
            return self.wave_dispatch_impl(wave, [groups, 1, 1]);
        }

        let row = quanta_ir::dispatch_fold::FOLD_ROW_GROUPS;
        if row > limit_x {
            // Linearization is baked against FOLD_ROW_GROUPS; a device
            // that can't even fit one folded row can't run this shape.
            return Err(QuantaError::not_supported(
                "dispatch group count exceeds maxComputeWorkGroupCount[0] \
                 and the device grid is narrower than the fold row width",
            ));
        }
        if self.dispatch_base_fn.is_none() {
            return Err(QuantaError::not_supported(
                "dispatch group count exceeds maxComputeWorkGroupCount[0] \
                 and vkCmdDispatchBase (Vulkan 1.1) is unavailable",
            ));
        }
        let (full_rows, rem) = quanta_ir::dispatch_fold::fold_groups(groups);
        let rows_total = full_rows + u32::from(rem > 0);
        if rows_total > self.caps.max_groups[1].max(1) {
            return Err(QuantaError::not_supported(
                "dispatch group count exceeds the folded 2D grid capacity \
                 (maxComputeWorkGroupCount[0] * [1])",
            ));
        }

        let mut records: Vec<([u32; 3], [u32; 3])> = Vec::with_capacity(2);
        if full_rows > 0 {
            records.push(([0, 0, 0], [row, full_rows, 1]));
        }
        if rem > 0 {
            records.push(([0, full_rows, 0], [rem, 1, 1]));
        }
        self.wave_dispatch_records_impl(wave, &records)
    }

    /// Shared dispatch body: bind pipeline + descriptors + push
    /// constants once, then record each `(base_workgroup, group_count)`
    /// entry — `vkCmdDispatch` for zero bases, `vkCmdDispatchBase`
    /// otherwise — into a single command buffer / submission. Entries
    /// of a folded dispatch cover disjoint linear ranges, so no
    /// barrier is needed between them.
    /// Validate and prepare each bound texture slot for a compute dispatch.
    ///
    /// Format contract, per texel-slot kind: a slot the kernel declares
    /// `Texture2D<f32>` (`wave.storage_texture_kinds[slot]` 1 = `&mut`,
    /// 3 = `&`) must be bound to an `R32Float` texture, and a
    /// `Texture2D<u32>` slot (2 = `&mut`, 4 = `&`) to an `RGBA8_UNORM`
    /// texture — a mismatch returns `InvalidParam` naming the slot,
    /// expected, and got. `descriptor_kinds` (from SPIR-V reflection) only
    /// says a slot *is* a storage image, not which pixel format it wants;
    /// the wave's kinds array is the expected-format channel it lacks. RGBA8
    /// is a mandatory Vulkan storage format, so — unlike Metal — there is no
    /// feature gate, and read-only vs read-write does not matter here (the
    /// NonWritable split is enforced in the SPIR-V itself). Each valid texel
    /// texture must have been created with `TextureUsage::STORAGE` (read-only
    /// texel access still binds a STORAGE_IMAGE descriptor), checked loudly
    /// rather than left to the validation layer, then is transitioned into
    /// `VK_IMAGE_LAYOUT_GENERAL` so that descriptor is legal at dispatch
    /// time.
    ///
    /// Sampled (`&Sampled2D`) slots have no format constraint (RGBA8 and
    /// R32Float both read); they are moved into `SHADER_READ_ONLY_OPTIMAL`
    /// for their COMBINED_IMAGE_SAMPLER descriptor instead.
    #[cfg(feature = "compute")]
    fn prepare_compute_textures(
        &self,
        wave: &Wave,
        descriptor_kinds: &[crate::driver::spirv_meta::DescriptorKind],
    ) -> Result<(), QuantaError> {
        use crate::driver::spirv_meta::DescriptorKind;
        for slot in 0..wave.texture_count as usize {
            let handle = wave.texture_bindings[slot];
            if handle == 0 {
                continue;
            }
            // Sampled (`&Texture2D`) read slots: the format is not constrained
            // (RGBA8 reads are the existing sampled contract, R32Float works
            // too), so only validate the texture exists, then move it into
            // SHADER_READ_ONLY_OPTIMAL — the layout a COMBINED_IMAGE_SAMPLER
            // descriptor requires. Storage slots fall through to the format
            // check + GENERAL transition below.
            match descriptor_kinds.get(slot) {
                Some(DescriptorKind::SampledImage) => {
                    {
                        let textures = self
                            .textures
                            .read()
                            .map_err(|_| QuantaError::internal("lock poisoned"))?;
                        if !textures.contains_key(&handle) {
                            return Err(QuantaError::not_found("bound compute texture not found")
                                .with_context(&format!("compute texture: slot {slot}")));
                        }
                    }
                    self.transition_texture_handle_shader_read(handle)?;
                    continue;
                }
                Some(DescriptorKind::StorageImage) => {}
                _ => {
                    return Err(QuantaError::invalid_param(
                        "texture bound to a slot the kernel does not use as an image",
                    )
                    .with_context(&format!("compute texture: slot {slot}")));
                }
            }
            let (format, usage) = {
                let textures = self
                    .textures
                    .read()
                    .map_err(|_| QuantaError::internal("lock poisoned"))?;
                let tex = textures.get(&handle).ok_or_else(|| {
                    QuantaError::not_found("bound compute texture not found")
                        .with_context(&format!("compute texture: slot {slot}"))
                })?;
                (tex.format, tex.usage)
            };
            // The scalar-driven format contract, keyed by the wave's per-slot
            // kind: {1,3} ⇔ R32Float, {2,4} ⇔ RGBA8_UNORM. Kind 0 on a
            // reflected storage-image slot is unexpected but non-fatal — fall
            // back to the R32Float expectation so a stale/unstamped wave
            // still validates.
            let (expected_fmt, expected_name) = match wave.storage_texture_kinds[slot] {
                2 | 4 => (ffi::VK_FORMAT_R8G8B8A8_UNORM, "RGBA8_UNORM"),
                _ => (ffi::VK_FORMAT_R32_SFLOAT, "R32Float"),
            };
            if format != expected_fmt {
                return Err(
                    QuantaError::invalid_param("compute texel texture format mismatch")
                        .with_context(&format!(
                            "slot {slot}: expected {expected_name} (VkFormat {expected_fmt}), \
                             got VkFormat {format}"
                        )),
                );
            }
            // A STORAGE_IMAGE descriptor is illegal against an image created
            // without STORAGE usage — read-only texel slots included. Fail
            // loudly here instead of leaving it to the validation layer.
            if usage & ffi::VK_IMAGE_USAGE_STORAGE_BIT == 0 {
                return Err(QuantaError::invalid_param(
                    "texture bound to a texel (`Texture2D`) slot was created without \
                     storage usage",
                )
                .with_context(&format!(
                    "compute texture: slot {slot}: create the texture with \
                     TextureUsage::STORAGE to license texel access (read-only \
                     included)"
                )));
            }
            self.transition_texture_handle_general(handle)?;
        }
        Ok(())
    }

    /// The device's single compute sampler for sampled `&Sampled2D` reads
    /// (`texture_sample_2d`), lazily created and cached. Contract: NEAREST
    /// min/mag/mip, CLAMP_TO_EDGE, no anisotropy/compare, UNNORMALIZED
    /// coordinates — chosen so `sample()` matches the CPU executor's
    /// nearest+clamp texel fetch and to satisfy Vulkan's unnormalized-sampler
    /// rules (equal filters, nearest mip, lod 0..0, non-repeat addressing).
    /// Deliberately not routed through `SamplerDesc`/the render sampler cache:
    /// that path has no unnormalized field and stays render-only.
    #[cfg(feature = "compute")]
    fn get_or_create_compute_sampler(&self) -> Result<ffi::VkSampler, QuantaError> {
        let mut slot = self
            .compute_sampler
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        if !slot.is_null() {
            return Ok(*slot);
        }
        let info = ffi::VkSamplerCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            mag_filter: ffi::VK_FILTER_NEAREST,
            min_filter: ffi::VK_FILTER_NEAREST,
            mipmap_mode: ffi::VK_SAMPLER_MIPMAP_MODE_NEAREST,
            address_mode_u: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            address_mode_v: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            mip_lod_bias: 0.0,
            anisotropy_enable: 0,
            max_anisotropy: 1.0,
            compare_enable: 0,
            compare_op: 0,
            // Unnormalized coordinates require lod clamped to 0 (§ valid usage).
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: 0,
            unnormalized_coordinates: 1,
        };
        let mut sampler = ffi::null_handle();
        let r =
            unsafe { ffi::vkCreateSampler(self.device, &info, core::ptr::null(), &mut sampler) };
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "compute sampler: VkResult {r}"
            )));
        }
        *slot = sampler;
        Ok(sampler)
    }

    /// Append one image descriptor write per bound texture slot, dispatching on
    /// the reflected kind: a sampled (`&Sampled2D`) slot writes a
    /// COMBINED_IMAGE_SAMPLER with the device compute sampler and the view in
    /// SHADER_READ_ONLY_OPTIMAL; a storage slot writes a STORAGE_IMAGE with a
    /// null sampler and the view in GENERAL. Both `wave_dispatch_records_impl`
    /// and `wave_dispatch_indirect_impl` share this, so the two paths can never
    /// disagree on how a sampled binding is written. `image_infos` must outlive
    /// the following `vkUpdateDescriptorSets` (its pointers live in `writes`).
    #[cfg(feature = "compute")]
    #[allow(clippy::too_many_arguments)]
    fn write_texture_descriptors(
        &self,
        wave: &Wave,
        descriptor_kinds: &[crate::driver::spirv_meta::DescriptorKind],
        textures_guard: &HashMap<u64, super::VkTexture>,
        ds: ffi::VkDescriptorSet,
        image_infos: &mut [ffi::VkDescriptorImageInfo; 16],
        writes: &mut [ffi::VkWriteDescriptorSet; 16],
        write_count: &mut usize,
    ) -> Result<(), QuantaError> {
        use crate::driver::spirv_meta::DescriptorKind;
        for slot in 0..wave.texture_count as usize {
            let handle = wave.texture_bindings[slot];
            if handle == 0 {
                continue;
            }
            let Some(tex) = textures_guard.get(&handle) else {
                continue;
            };
            let sampled = matches!(
                descriptor_kinds.get(slot),
                Some(DescriptorKind::SampledImage)
            );
            let (sampler, image_layout, descriptor_type) = if sampled {
                (
                    self.get_or_create_compute_sampler()?,
                    ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                )
            } else {
                (
                    ffi::null_handle(),
                    ffi::VK_IMAGE_LAYOUT_GENERAL,
                    ffi::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                )
            };
            let i = *write_count;
            image_infos[i] = ffi::VkDescriptorImageInfo {
                sampler,
                image_view: tex.view,
                image_layout,
            };
            writes[i] = ffi::VkWriteDescriptorSet {
                s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: core::ptr::null(),
                dst_set: ds,
                dst_binding: slot as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type,
                p_image_info: &image_infos[i],
                p_buffer_info: core::ptr::null(),
                p_texel_buffer_view: core::ptr::null(),
            };
            *write_count += 1;
        }
        Ok(())
    }

    fn wave_dispatch_records_impl(
        &self,
        wave: &Wave,
        records: &[([u32; 3], [u32; 3])],
    ) -> Result<Pulse, QuantaError> {
        let compute_pipelines = self
            .compute_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let cp = compute_pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch: handle {}", wave.handle))
        })?;

        // Acquire descriptor pool from cache (or create new)
        let descriptor_pool = self.acquire_descriptor_pool()?;

        let alloc_info = ffi::VkDescriptorSetAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &cp.descriptor_set_layout,
        };
        let mut ds = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        // Before touching descriptors, transition every bound storage texture
        // into GENERAL (the only layout a STORAGE_IMAGE descriptor accepts) and
        // validate its format against the param's R32Float expectation. This
        // is a self-contained submit+wait, so the layout is settled before the
        // dispatch command buffer runs.
        self.prepare_compute_textures(wave, &cp.descriptor_kinds)?;

        // Update descriptor set with buffer bindings (inline arrays)
        let buffers_guard = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let textures_guard = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mut buffer_infos: [ffi::VkDescriptorBufferInfo; 16] = unsafe { core::mem::zeroed() };
        // Image infos must outlive vkUpdateDescriptorSets — keep them alongside
        // buffer_infos so the pointers in `writes` stay valid until the update.
        let mut image_infos: [ffi::VkDescriptorImageInfo; 16] = unsafe { core::mem::zeroed() };
        let mut writes: [ffi::VkWriteDescriptorSet; 16] = unsafe { core::mem::zeroed() };
        let mut write_count = 0usize;

        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0
                && let Some(buf) = buffers_guard.get(&handle)
            {
                buffer_infos[write_count] = ffi::VkDescriptorBufferInfo {
                    buffer: buf.buffer,
                    offset: 0,
                    range: ffi::VK_WHOLE_SIZE,
                };
                writes[write_count] = ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: slot as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: core::ptr::null(),
                    p_buffer_info: &buffer_infos[write_count],
                    p_texel_buffer_view: core::ptr::null(),
                };
                write_count += 1;
            }
        }

        // Image (texture) bindings, one descriptor per bound slot. A sampled
        // (`&Sampled2D`) slot binds COMBINED_IMAGE_SAMPLER with the device
        // compute sampler and the view in SHADER_READ_ONLY_OPTIMAL; a storage
        // slot binds STORAGE_IMAGE with a null sampler and the view in GENERAL.
        // `prepare_compute_textures` already settled each layout to match.
        self.write_texture_descriptors(
            wave,
            &cp.descriptor_kinds,
            &textures_guard,
            ds,
            &mut image_infos,
            &mut writes,
            &mut write_count,
        )?;

        if write_count > 0 {
            unsafe {
                ffi::vkUpdateDescriptorSets(
                    self.device,
                    write_count as u32,
                    writes.as_ptr(),
                    0,
                    core::ptr::null(),
                );
            }
        }

        // Record and submit command buffer
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
            ffi::vkCmdBindPipeline(cmd, ffi::VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
            ffi::vkCmdBindDescriptorSets(
                cmd,
                ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                cp.layout,
                0,
                1,
                &ds,
                0,
                core::ptr::null(),
            );

            // Push constants from inline buffer
            if wave.push_len > 0 {
                ffi::vkCmdPushConstants(
                    cmd,
                    cp.layout,
                    ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    wave.push_len as u32,
                    wave.push_data.as_ptr() as *const c_void,
                );
            }

            for &(base, counts) in records {
                if base == [0, 0, 0] {
                    ffi::vkCmdDispatch(cmd, counts[0], counts[1], counts[2]);
                } else {
                    // Callers only build non-zero-base records after
                    // checking dispatch_base_fn is resolved.
                    let dispatch_base = self.dispatch_base_fn.ok_or_else(|| {
                        QuantaError::not_supported("vkCmdDispatchBase is unavailable")
                    })?;
                    dispatch_base(
                        cmd, base[0], base[1], base[2], counts[0], counts[1], counts[2],
                    );
                }
            }
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(buffers_guard);
        drop(textures_guard);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?.wait()?;

        // Return descriptor pool to cache for reuse
        self.return_descriptor_pool(descriptor_pool);

        Ok(Pulse {
            handle: self.alloc_handle(),
            completed: true,
            wait_fn: None,
            keep_alive: self.self_ref.pulse_keep_alive(),
        })
    }

    pub(crate) fn wave_dispatch_indirect_impl(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        let compute_pipelines = self
            .compute_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let cp = compute_pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch_indirect: handle {}", wave.handle))
        })?;

        // Acquire descriptor pool from cache (or create new)
        let descriptor_pool = self.acquire_descriptor_pool()?;

        let alloc_info = ffi::VkDescriptorSetAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &cp.descriptor_set_layout,
        };
        let mut ds = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        self.prepare_compute_textures(wave, &cp.descriptor_kinds)?;

        let buffers_guard = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let textures_guard = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mut buffer_infos: [ffi::VkDescriptorBufferInfo; 16] = unsafe { core::mem::zeroed() };
        let mut image_infos: [ffi::VkDescriptorImageInfo; 16] = unsafe { core::mem::zeroed() };
        let mut writes: [ffi::VkWriteDescriptorSet; 16] = unsafe { core::mem::zeroed() };
        let mut write_count = 0usize;

        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0
                && let Some(buf) = buffers_guard.get(&handle)
            {
                buffer_infos[write_count] = ffi::VkDescriptorBufferInfo {
                    buffer: buf.buffer,
                    offset: 0,
                    range: ffi::VK_WHOLE_SIZE,
                };
                writes[write_count] = ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: slot as u32,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: core::ptr::null(),
                    p_buffer_info: &buffer_infos[write_count],
                    p_texel_buffer_view: core::ptr::null(),
                };
                write_count += 1;
            }
        }
        self.write_texture_descriptors(
            wave,
            &cp.descriptor_kinds,
            &textures_guard,
            ds,
            &mut image_infos,
            &mut writes,
            &mut write_count,
        )?;
        if write_count > 0 {
            unsafe {
                ffi::vkUpdateDescriptorSets(
                    self.device,
                    write_count as u32,
                    writes.as_ptr(),
                    0,
                    core::ptr::null(),
                );
            }
        }

        let indirect_buf = buffers_guard.get(&buffer).ok_or_else(|| {
            QuantaError::invalid_param("bad indirect buffer")
                .with_context(&format!("wave_dispatch_indirect: buffer handle {buffer}"))
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
            ffi::vkCmdBindPipeline(cmd, ffi::VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
            ffi::vkCmdBindDescriptorSets(
                cmd,
                ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                cp.layout,
                0,
                1,
                &ds,
                0,
                core::ptr::null(),
            );
            ffi::vkCmdDispatchIndirect(cmd, indirect_buf.buffer, offset);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(buffers_guard);
        drop(textures_guard);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?.wait()?;

        // Return descriptor pool to cache for reuse
        self.return_descriptor_pool(descriptor_pool);

        Ok(Pulse {
            handle: self.alloc_handle(),
            completed: true,
            wait_fn: None,
            keep_alive: self.self_ref.pulse_keep_alive(),
        })
    }
}
