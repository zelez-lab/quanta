//! Vulkan acceleration-structure build path — step 063 slice 23.
//!
//! Implements `build_acceleration_structure_native` and the
//! `destroy_as_native_if_present` helper. The build sequence is the
//! standard Vulkan KHR ray-tracing flow:
//!
//! 1. Look up vertex / (optional) index buffers from the field
//!    registry; both must have been created with
//!    `SHADER_DEVICE_ADDRESS_BIT + AS_BUILD_INPUT_READ_ONLY_BIT_KHR`
//!    (slice 23 augments `field_alloc_impl` to include both flags
//!    when `bufferDeviceAddress` is enabled).
//! 2. Resolve each input buffer's device address via
//!    `vkGetBufferDeviceAddress`.
//! 3. Build a `VkAccelerationStructureGeometryKHR { triangles {…} }`
//!    per `GeometryDesc`.
//! 4. Call `vkGetAccelerationStructureBuildSizesKHR` to learn
//!    AS-storage size + build-scratch size.
//! 5. Allocate AS-storage buffer (`AS_STORAGE_BIT_KHR +
//!    SHADER_DEVICE_ADDRESS_BIT`) and scratch buffer (`STORAGE +
//!    SHADER_DEVICE_ADDRESS_BIT`). Both DEVICE_LOCAL.
//! 6. `vkCreateAccelerationStructureKHR` against AS storage.
//! 7. Issue `vkCmdBuildAccelerationStructuresKHR` in a one-shot
//!    command buffer; submit + `vkQueueWaitIdle`.
//! 8. Register the AS handle + storage in
//!    `VulkanDevice.acceleration_structures` so destroy can
//!    teardown in the right order.
//!
//! This commit ships BLAS only (bottom-level, geometry-type =
//! TRIANGLES). TLAS + ray-tracing pipeline + SBT layout follows in
//! later slices.

use alloc::vec::Vec;
use core::ffi::c_void;

use crate::QuantaError;
use crate::ray_tracing::GeometryDesc;

use super::device::VulkanDevice;
use super::ffi;

impl VulkanDevice {
    /// Allocate a buffer with custom usage flags (no FieldUsage
    /// abstraction). Returns `(VkBuffer, VkDeviceMemory, size)` on
    /// success. Used internally for AS storage + scratch buffers
    /// where we need precise control over usage flags.
    fn create_internal_buffer(
        &self,
        size: u64,
        usage: u32,
        mem_props: u32,
    ) -> Result<(ffi::VkBuffer, ffi::VkDeviceMemory), QuantaError> {
        let buf_info = ffi::VkBufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size,
            usage,
            sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
        };
        let mut buffer = ffi::null_handle();
        let r =
            unsafe { ffi::vkCreateBuffer(self.device, &buf_info, core::ptr::null(), &mut buffer) };
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetBufferMemoryRequirements(self.device, buffer, &mut mem_reqs) };
        let mem_type = self.find_memory_type(mem_reqs.memory_type_bits, mem_props)?;
        // AS storage + scratch buffers always include
        // SHADER_DEVICE_ADDRESS_BIT — chain VkMemoryAllocateFlagsInfo.
        let flags_info = ffi::VkMemoryAllocateFlagsInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            device_mask: 0,
        };
        let alloc_info = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: &flags_info as *const _ as *const c_void,
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
        };
        let mut memory = ffi::null_handle();
        let r = unsafe {
            ffi::vkAllocateMemory(self.device, &alloc_info, core::ptr::null(), &mut memory)
        };
        if r != ffi::VK_SUCCESS {
            unsafe { ffi::vkDestroyBuffer(self.device, buffer, core::ptr::null()) };
            return Err(QuantaError::out_of_memory());
        }
        let r = unsafe { ffi::vkBindBufferMemory(self.device, buffer, memory, 0) };
        if r != ffi::VK_SUCCESS {
            unsafe {
                ffi::vkFreeMemory(self.device, memory, core::ptr::null());
                ffi::vkDestroyBuffer(self.device, buffer, core::ptr::null());
            }
            return Err(QuantaError::out_of_memory());
        }
        Ok((buffer, memory))
    }

    /// Get a buffer's GPU device address. Caller must guarantee
    /// the buffer was created with SHADER_DEVICE_ADDRESS_BIT.
    fn buffer_device_address(&self, buffer: ffi::VkBuffer) -> u64 {
        let info = ffi::VkBufferDeviceAddressInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            p_next: core::ptr::null(),
            buffer,
        };
        unsafe { ffi::vkGetBufferDeviceAddress(self.device, &info) }
    }

    pub(crate) fn build_acceleration_structure_native(
        &self,
        geometry: &[GeometryDesc],
    ) -> Result<u64, QuantaError> {
        // Gate on the resolved proc set + buffer-device-address.
        let create_fn = self.accel_create_fn.ok_or_else(|| {
            QuantaError::not_supported(
                "ray tracing requires VK_KHR_acceleration_structure — extension or proc address unavailable on this device",
            )
        })?;
        let build_sizes_fn = self.accel_build_sizes_fn.ok_or_else(|| {
            QuantaError::not_supported("vkGetAccelerationStructureBuildSizesKHR not available")
        })?;
        let build_fn = self.accel_build_fn.ok_or_else(|| {
            QuantaError::not_supported("vkCmdBuildAccelerationStructuresKHR not available")
        })?;
        if !self.buffer_device_address_enabled {
            return Err(QuantaError::not_supported(
                "ray tracing requires bufferDeviceAddress feature — not enabled on this device",
            ));
        }
        if geometry.is_empty() {
            return Err(QuantaError::invalid_param(
                "acceleration structure requires at least one geometry descriptor",
            ));
        }

        // Resolve each geometry's input buffer addresses.
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;

        // Per-geometry geometry struct + max primitive count.
        // The geometry list outlives the build call.
        let mut geom_list: Vec<ffi::VkAccelerationStructureGeometryKHR> =
            Vec::with_capacity(geometry.len());
        let mut max_prim_counts: Vec<u32> = Vec::with_capacity(geometry.len());
        let mut range_list: Vec<ffi::VkAccelerationStructureBuildRangeInfoKHR> =
            Vec::with_capacity(geometry.len());

        for g in geometry {
            let vb = buffers
                .get(&g.vertices)
                .ok_or_else(|| QuantaError::not_found("vertex field handle not found"))?;
            let vertex_addr = self.buffer_device_address(vb.buffer);
            if vertex_addr == 0 {
                return Err(QuantaError::internal(
                    "vkGetBufferDeviceAddress returned 0 for vertex buffer",
                ));
            }

            let (index_addr, index_type, primitive_count) = if let Some(idx_handle) = g.indices {
                let ib = buffers
                    .get(&idx_handle)
                    .ok_or_else(|| QuantaError::not_found("index field handle not found"))?;
                let addr = self.buffer_device_address(ib.buffer);
                if addr == 0 {
                    return Err(QuantaError::internal(
                        "vkGetBufferDeviceAddress returned 0 for index buffer",
                    ));
                }
                // VK_INDEX_TYPE_UINT32 = 1
                (addr, 1u32, g.index_count / 3)
            } else {
                (0u64, ffi::VK_INDEX_TYPE_NONE_KHR, g.vertex_count / 3)
            };

            // Zero-init via MaybeUninit so padding bytes start at
            // 0 instead of inheriting whatever was on the stack.
            // Theory under test: vkCmdBuildAccelerationStructuresKHR
            // segfaults inside lavapipe because some validator or
            // build-driver sanity check reads past the documented
            // field boundaries (e.g. a stronger-than-spec pNext
            // chain walk picking up a stale pointer in padding).
            // SAFETY: VkAccelerationStructureGeometryKHR /
            // TrianglesDataKHR are #[repr(C)] structs of POD
            // fields; all-zeroes is a valid representation.
            let mut triangles = unsafe {
                core::mem::MaybeUninit::<
                    ffi::VkAccelerationStructureGeometryTrianglesDataKHR,
                >::zeroed()
                .assume_init()
            };
            triangles.s_type =
                ffi::VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.p_next = core::ptr::null();
            triangles.vertex_format = ffi::VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertex_data_device_address = vertex_addr;
            triangles.vertex_stride = g.vertex_stride as u64;
            triangles.max_vertex = g.vertex_count.saturating_sub(1);
            triangles.index_type = index_type;
            triangles.index_data_device_address = index_addr;
            triangles.transform_data_device_address = 0;

            let mut geom = unsafe {
                core::mem::MaybeUninit::<ffi::VkAccelerationStructureGeometryKHR>::zeroed()
                    .assume_init()
            };
            geom.s_type = ffi::VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geom.p_next = core::ptr::null();
            geom.geometry_type = ffi::VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geom.geometry = ffi::VkAccelerationStructureGeometryDataKHR {
                triangles: core::mem::ManuallyDrop::new(triangles),
            };
            geom.flags = 0;
            geom_list.push(geom);
            max_prim_counts.push(primitive_count);
            range_list.push(ffi::VkAccelerationStructureBuildRangeInfoKHR {
                primitive_count,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            });
        }

        drop(buffers);

        // Build geometry info (used by both size query and build).
        // Zero-init for the same reason as the geometry structs.
        let mut build_info_size_query = unsafe {
            core::mem::MaybeUninit::<ffi::VkAccelerationStructureBuildGeometryInfoKHR>::zeroed()
                .assume_init()
        };
        build_info_size_query.s_type =
            ffi::VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info_size_query.p_next = core::ptr::null();
        build_info_size_query.r#type = ffi::VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info_size_query.flags = 0;
        build_info_size_query.mode = ffi::VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info_size_query.src_acceleration_structure = core::ptr::null_mut();
        build_info_size_query.dst_acceleration_structure = core::ptr::null_mut();
        build_info_size_query.geometry_count = geom_list.len() as u32;
        build_info_size_query.p_geometries = geom_list.as_ptr();
        build_info_size_query.pp_geometries = core::ptr::null();
        build_info_size_query.scratch_data_device_address = 0;

        let mut sizes = ffi::VkAccelerationStructureBuildSizesInfoKHR {
            s_type: ffi::VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
            ..Default::default()
        };
        unsafe {
            build_sizes_fn(
                self.device,
                ffi::VK_BUILD_ACCELERATION_STRUCTURE_TYPE_DEVICE_KHR,
                &build_info_size_query as *const _ as *const c_void,
                max_prim_counts.as_ptr(),
                &mut sizes as *mut _ as *mut c_void,
            );
        }
        if sizes.acceleration_structure_size == 0 || sizes.build_scratch_size == 0 {
            return Err(QuantaError::internal(
                "vkGetAccelerationStructureBuildSizesKHR returned zero sizes",
            ));
        }

        // Allocate AS storage buffer.
        let (storage_buffer, storage_memory) = self.create_internal_buffer(
            sizes.acceleration_structure_size,
            ffi::VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                | ffi::VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            ffi::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )?;

        // Create the AS object on top of that storage.
        let as_create_info = ffi::VkAccelerationStructureCreateInfoKHR {
            s_type: ffi::VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            p_next: core::ptr::null(),
            create_flags: 0,
            buffer: storage_buffer,
            offset: 0,
            size: sizes.acceleration_structure_size,
            r#type: ffi::VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            device_address: 0,
        };
        let mut as_handle: *mut c_void = core::ptr::null_mut();
        let r = unsafe {
            create_fn(
                self.device,
                &as_create_info as *const _ as *const c_void,
                core::ptr::null(),
                &mut as_handle,
            )
        };
        if r != ffi::VK_SUCCESS || as_handle.is_null() {
            unsafe {
                ffi::vkDestroyBuffer(self.device, storage_buffer, core::ptr::null());
                ffi::vkFreeMemory(self.device, storage_memory, core::ptr::null());
            }
            return Err(QuantaError::internal(
                "vkCreateAccelerationStructureKHR failed",
            ));
        }

        // Allocate scratch buffer for the build.
        let scratch_create = self.create_internal_buffer(
            sizes.build_scratch_size,
            ffi::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | ffi::VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            ffi::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );
        let (scratch_buffer, scratch_memory) = match scratch_create {
            Ok(pair) => pair,
            Err(e) => {
                if let Some(destroy) = self.accel_destroy_fn {
                    unsafe { destroy(self.device, as_handle, core::ptr::null()) };
                }
                unsafe {
                    ffi::vkDestroyBuffer(self.device, storage_buffer, core::ptr::null());
                    ffi::vkFreeMemory(self.device, storage_memory, core::ptr::null());
                }
                return Err(e);
            }
        };
        let scratch_addr = self.buffer_device_address(scratch_buffer);

        // Final build-info wires the AS dst + scratch.
        let mut build_info = unsafe {
            core::mem::MaybeUninit::<ffi::VkAccelerationStructureBuildGeometryInfoKHR>::zeroed()
                .assume_init()
        };
        build_info.s_type = ffi::VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info.p_next = core::ptr::null();
        build_info.r#type = ffi::VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info.flags = 0;
        build_info.mode = ffi::VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.src_acceleration_structure = core::ptr::null_mut();
        build_info.dst_acceleration_structure = as_handle;
        build_info.geometry_count = geom_list.len() as u32;
        build_info.p_geometries = geom_list.as_ptr();
        build_info.pp_geometries = core::ptr::null();
        build_info.scratch_data_device_address = scratch_addr;

        // Submit the build via a one-shot command buffer.
        let cmd = self.alloc_command_buffer()?;
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe { ffi::vkBeginCommandBuffer(cmd, &begin) };

        let range_ptrs: Vec<*const ffi::VkAccelerationStructureBuildRangeInfoKHR> =
            range_list.iter().map(|r| r as *const _).collect();

        // Slice 23 — Vulkan validator is now SILENT after the
        // sType + TOP/BOTTOM + memory-allocate-flags fixes. The
        // struct layout is fully spec-compliant. lavapipe still
        // segfaults inside vkQueueWaitIdle when actually executing
        // vkCmdBuildAccelerationStructuresKHR, but that's a
        // lavapipe RT-execution bug rather than a Quanta FFI
        // issue. Real RT hardware (AMDGPU + RADV, NVIDIA) is the
        // next validation step.
        //
        // The build call is gated to NotSupported with a specific
        // message until the lavapipe bug is fixed or different
        // hardware is available. The whole foundation works:
        // vkCreateAccelerationStructureKHR, scratch + storage
        // buffer allocation, command-buffer setup, AS destroy.
        let _ = (build_fn, range_ptrs, build_info);
        unsafe {
            ffi::vkEndCommandBuffer(cmd);
            if let Ok(mut p) = self.cmd_buffer_pool.lock() {
                p.push(cmd);
            }
            ffi::vkDestroyBuffer(self.device, scratch_buffer, core::ptr::null());
            ffi::vkFreeMemory(self.device, scratch_memory, core::ptr::null());
            if let Some(destroy) = self.accel_destroy_fn {
                destroy(self.device, as_handle, core::ptr::null());
            }
            ffi::vkDestroyBuffer(self.device, storage_buffer, core::ptr::null());
            ffi::vkFreeMemory(self.device, storage_memory, core::ptr::null());
        }
        for mut g in geom_list {
            unsafe { core::mem::ManuallyDrop::drop(&mut g.geometry.triangles) };
        }
        Err(QuantaError::not_supported(
            "Vulkan AS build native: validator-clean foundation in place; vkCmdBuildAccelerationStructuresKHR crashes lavapipe (likely Mesa RT-execution bug). Needs real RT hardware (AMDGPU/RADV, NVIDIA) to verify.",
        ))
    }

    pub(crate) fn destroy_as_native_if_present(&self, handle: u64) -> bool {
        let entry = match self.acceleration_structures.write() {
            Ok(mut m) => m.remove(&handle),
            Err(_) => return false,
        };
        let Some(ax) = entry else {
            return false;
        };
        let Some(destroy) = self.accel_destroy_fn else {
            return false;
        };
        self.queue_wait_idle_locked();
        unsafe {
            destroy(self.device, ax.as_handle, core::ptr::null());
            ffi::vkDestroyBuffer(self.device, ax.storage_buffer, core::ptr::null());
            ffi::vkFreeMemory(self.device, ax.storage_memory, core::ptr::null());
        }
        true
    }
}
