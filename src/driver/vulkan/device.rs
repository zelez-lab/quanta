//! VulkanDevice struct definition, discovery, and internal helpers.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{Caps, GpuDevice, Pulse, QuantaError, Vendor};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use super::ffi;

/// Vulkan-backed GPU device.
pub struct VulkanDevice {
    pub(super) instance: ffi::VkInstance,
    pub(super) physical_device: ffi::VkPhysicalDevice,
    pub(super) device: ffi::VkDevice,
    pub(super) queue: ffi::VkQueue,
    #[allow(dead_code)]
    pub(super) queue_family: u32,
    pub(super) command_pool: ffi::VkCommandPool,
    pub(super) caps: Caps,
    pub(super) max_push_constants_size: u32,
    // Resource storage — RwLock: dispatch/render paths take read locks; alloc/free take write locks.
    pub(super) buffers: RwLock<HashMap<u64, VkBuffer>>,
    pub(super) textures: RwLock<HashMap<u64, VkTexture>>,
    pub(super) compute_pipelines: RwLock<HashMap<u64, VkComputePipeline>>,
    pub(super) render_pipelines: RwLock<HashMap<u64, VkRenderPipeline>>,
    pub(super) samplers: RwLock<HashMap<u64, ffi::VkSampler>>,
    /// Standalone image views created via texture_view_create (not tied to a full VkTexture).
    pub(super) image_views: RwLock<HashMap<u64, ffi::VkImageView>>,
    pub(super) query_pools: RwLock<HashMap<u64, VkQueryPool>>,
    pub(super) queues: RwLock<HashMap<u64, ffi::VkQueue>>,
    pub(super) next_handle: AtomicU64,
    /// Pool of reusable command buffers — Arc<Mutex> for sharing with Pulse closures.
    pub(super) cmd_buffer_pool: std::sync::Arc<Mutex<Vec<ffi::VkCommandBuffer>>>,
}

pub(super) struct VkQueryPool {
    pub(super) pool: ffi::VkQueryPool,
    pub(super) count: u32,
}

#[allow(dead_code)]
pub(super) struct VkBuffer {
    pub(super) buffer: ffi::VkBuffer,
    pub(super) memory: ffi::VkDeviceMemory,
    pub(super) size: u64,
}

#[allow(dead_code)]
pub(super) struct VkTexture {
    pub(super) image: ffi::VkImage,
    pub(super) view: ffi::VkImageView,
    pub(super) memory: ffi::VkDeviceMemory,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: u32,
    pub(super) mip_levels: u32,
    pub(super) current_layout: std::sync::atomic::AtomicU32,
}

pub(super) struct VkComputePipeline {
    pub(super) pipeline: ffi::VkPipeline,
    pub(super) layout: ffi::VkPipelineLayout,
    pub(super) descriptor_set_layout: ffi::VkDescriptorSetLayout,
}

pub(super) struct VkRenderPipeline {
    pub(super) pipeline: ffi::VkPipeline,
    pub(super) layout: ffi::VkPipelineLayout,
    pub(super) render_pass: ffi::VkRenderPass,
    pub(super) descriptor_set_layout: ffi::VkDescriptorSetLayout,
}

impl VulkanDevice {
    pub(super) fn alloc_handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Check if a device extension is available on the physical device.
    pub(super) fn has_device_extension(&self, ext_name: &[u8]) -> bool {
        let mut count = 0u32;
        let result = unsafe {
            ffi::vkEnumerateDeviceExtensionProperties(
                self.physical_device,
                core::ptr::null(),
                &mut count,
                core::ptr::null_mut(),
            )
        };
        if result != ffi::VK_SUCCESS || count == 0 {
            return false;
        }
        let mut props = vec![ffi::VkExtensionProperties::default(); count as usize];
        let result = unsafe {
            ffi::vkEnumerateDeviceExtensionProperties(
                self.physical_device,
                core::ptr::null(),
                &mut count,
                props.as_mut_ptr(),
            )
        };
        if result != ffi::VK_SUCCESS {
            return false;
        }
        // ext_name is null-terminated; compare up to the null byte.
        let target = &ext_name[..ext_name.len() - 1]; // strip trailing \0
        props.iter().any(|p| {
            let name_bytes = &p.extension_name;
            let len = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            &name_bytes[..len] == target
        })
    }

    pub(super) fn alloc_command_buffer(&self) -> Result<ffi::VkCommandBuffer, QuantaError> {
        // Try to reuse a previously returned command buffer from the pool.
        if let Some(cmd) = self
            .cmd_buffer_pool
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .pop()
        {
            let result = unsafe { ffi::vkResetCommandBuffer(cmd, 0) };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            return Ok(cmd);
        }
        // Pool empty -- allocate a fresh one.
        let alloc_info = ffi::VkCommandBufferAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            command_pool: self.command_pool,
            level: ffi::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateCommandBuffers(self.device, &alloc_info, &mut cmd) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }
        Ok(cmd)
    }

    /// Submit a command buffer with a fence. Returns a Pulse that waits on the
    /// fence when wait() is called. The GPU executes asynchronously — the CPU
    /// can do other work before calling pulse.wait().
    pub(super) fn submit_and_wait(&self, cmd: ffi::VkCommandBuffer) -> Result<Pulse, QuantaError> {
        let fence_info = ffi::VkFenceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
        };
        let mut fence = ffi::null_handle();
        unsafe {
            let r = ffi::vkCreateFence(self.device, &fence_info, core::ptr::null(), &mut fence);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let submit = ffi::VkSubmitInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: core::ptr::null(),
            p_wait_dst_stage_mask: core::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            signal_semaphore_count: 0,
            p_signal_semaphores: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkQueueSubmit(self.queue, 1, &submit, fence);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let device = self.device;
        let pool = self.cmd_buffer_pool.clone();
        let handle = self.alloc_handle();
        Ok(Pulse {
            handle,
            completed: false,
            wait_fn: Some(Box::new(move || unsafe {
                ffi::vkWaitForFences(device, 1, &fence, 1, u64::MAX);
                ffi::vkDestroyFence(device, fence, core::ptr::null());
                if let Ok(mut p) = pool.lock() {
                    p.push(cmd);
                }
            })),
        })
    }
}

/// Discover Vulkan devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let app_info = ffi::VkApplicationInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: core::ptr::null(),
        p_application_name: core::ptr::null(),
        application_version: 0,
        p_engine_name: core::ptr::null(),
        engine_version: 0,
        api_version: ffi::make_api_version(0, 1, 3, 0),
    };

    let create_info = ffi::VkInstanceCreateInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: core::ptr::null(),
    };

    let mut instance = ffi::null_handle();
    let result = unsafe { ffi::vkCreateInstance(&create_info, core::ptr::null(), &mut instance) };
    if result != ffi::VK_SUCCESS {
        return Vec::new();
    }

    let mut count = 0u32;
    let result =
        unsafe { ffi::vkEnumeratePhysicalDevices(instance, &mut count, core::ptr::null_mut()) };
    if result != ffi::VK_SUCCESS || count == 0 {
        return Vec::new();
    }

    let mut physical_devices = vec![ffi::null_handle(); count as usize];
    let result = unsafe {
        ffi::vkEnumeratePhysicalDevices(instance, &mut count, physical_devices.as_mut_ptr())
    };
    if result != ffi::VK_SUCCESS {
        return Vec::new();
    }

    let mut devices: Vec<Box<dyn GpuDevice>> = Vec::new();

    for pd in physical_devices {
        let mut props = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceProperties>() };
        unsafe { ffi::vkGetPhysicalDeviceProperties(pd, &mut props) };

        let mut qf_count = 0u32;
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(pd, &mut qf_count, core::ptr::null_mut())
        };
        let mut queue_families = vec![ffi::VkQueueFamilyProperties::default(); qf_count as usize];
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(
                pd,
                &mut qf_count,
                queue_families.as_mut_ptr(),
            )
        };

        // Find a queue family that supports compute + graphics
        let queue_family = queue_families.iter().enumerate().find(|(_, qf)| {
            (qf.queue_flags & ffi::VK_QUEUE_GRAPHICS_BIT) != 0
                && (qf.queue_flags & ffi::VK_QUEUE_COMPUTE_BIT) != 0
        });

        let Some((qf_index, _)) = queue_family else {
            continue;
        };

        let queue_priorities = [1.0f32];
        let queue_create = ffi::VkDeviceQueueCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            queue_family_index: qf_index as u32,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
        };

        // Enable synchronization2 (Vulkan 1.3 core) for vkCmdPipelineBarrier2
        #[repr(C)]
        struct VkPhysicalDeviceSynchronization2Features {
            s_type: u32,
            p_next: *const core::ffi::c_void,
            synchronization2: u32,
        }
        let sync2_features = VkPhysicalDeviceSynchronization2Features {
            s_type: 1000314007, // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
            p_next: core::ptr::null(),
            synchronization2: 1, // VK_TRUE
        };

        let device_create = ffi::VkDeviceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: &sync2_features as *const _ as *const core::ffi::c_void,
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create,
            enabled_layer_count: 0,
            pp_enabled_layer_names: core::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: core::ptr::null(),
            p_enabled_features: core::ptr::null(),
        };

        let mut device = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateDevice(pd, &device_create, core::ptr::null(), &mut device) };
        if result != ffi::VK_SUCCESS {
            continue;
        }

        let mut queue = ffi::null_handle();
        unsafe { ffi::vkGetDeviceQueue(device, qf_index as u32, 0, &mut queue) };

        // Command pool
        let pool_info = ffi::VkCommandPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: qf_index as u32,
        };
        let mut command_pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateCommandPool(device, &pool_info, core::ptr::null(), &mut command_pool)
        };
        if result != ffi::VK_SUCCESS {
            continue;
        }

        let name = unsafe {
            let cstr =
                std::ffi::CStr::from_ptr(props.device_name.as_ptr() as *const core::ffi::c_char);
            cstr.to_string_lossy().to_string()
        };

        let vendor = match props.vendor_id {
            0x1002 => Vendor::Amd,
            0x10DE => Vendor::Nvidia,
            0x8086 => Vendor::Intel,
            0x13B5 | 0x14E4 => Vendor::Broadcom,
            _ => Vendor::Unknown,
        };

        // Query total device memory from the largest heap
        let mut mem_props = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceMemoryProperties>() };
        unsafe { ffi::vkGetPhysicalDeviceMemoryProperties(pd, &mut mem_props) };
        let total_memory = (0..mem_props.memory_heap_count as usize)
            .map(|i| mem_props.memory_heaps[i].size)
            .max()
            .unwrap_or(0);

        let caps = Caps {
            nuclei: props.limits.max_compute_work_group_count[0].min(1024),
            protons_per_nucleus: 1,
            quarks_per_proton: props.limits.max_compute_work_group_size[0],
            memory_bytes: total_memory,
            max_quarks_per_dispatch: props.limits.max_compute_work_group_invocations,
            max_groups: props.limits.max_compute_work_group_count,
            vendor,
            name,
        };

        devices.push(Box::new(VulkanDevice {
            instance,
            physical_device: pd,
            device,
            queue,
            queue_family: qf_index as u32,
            command_pool,
            caps,
            max_push_constants_size: props.limits.max_push_constants_size,
            buffers: RwLock::new(HashMap::new()),
            textures: RwLock::new(HashMap::new()),
            compute_pipelines: RwLock::new(HashMap::new()),
            render_pipelines: RwLock::new(HashMap::new()),
            samplers: RwLock::new(HashMap::new()),
            image_views: RwLock::new(HashMap::new()),
            query_pools: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            next_handle: AtomicU64::new(0),
            cmd_buffer_pool: std::sync::Arc::new(Mutex::new(Vec::new())),
        }));

        break; // Use first suitable device
    }

    devices
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            ffi::vkDeviceWaitIdle(self.device);

            // Clean up resources — write locks since we're draining.
            if let Ok(mut buffers) = self.buffers.write() {
                for (_, buf) in buffers.drain() {
                    ffi::vkDestroyBuffer(self.device, buf.buffer, core::ptr::null());
                    ffi::vkFreeMemory(self.device, buf.memory, core::ptr::null());
                }
            }
            if let Ok(mut textures) = self.textures.write() {
                for (_, tex) in textures.drain() {
                    ffi::vkDestroyImageView(self.device, tex.view, core::ptr::null());
                    ffi::vkDestroyImage(self.device, tex.image, core::ptr::null());
                    ffi::vkFreeMemory(self.device, tex.memory, core::ptr::null());
                }
            }
            if let Ok(mut pipelines) = self.compute_pipelines.write() {
                for (_, cp) in pipelines.drain() {
                    ffi::vkDestroyPipeline(self.device, cp.pipeline, core::ptr::null());
                    ffi::vkDestroyPipelineLayout(self.device, cp.layout, core::ptr::null());
                    ffi::vkDestroyDescriptorSetLayout(
                        self.device,
                        cp.descriptor_set_layout,
                        core::ptr::null(),
                    );
                }
            }
            if let Ok(mut pipelines) = self.render_pipelines.write() {
                for (_, rp) in pipelines.drain() {
                    ffi::vkDestroyPipeline(self.device, rp.pipeline, core::ptr::null());
                    ffi::vkDestroyPipelineLayout(self.device, rp.layout, core::ptr::null());
                    ffi::vkDestroyRenderPass(self.device, rp.render_pass, core::ptr::null());
                    ffi::vkDestroyDescriptorSetLayout(
                        self.device,
                        rp.descriptor_set_layout,
                        core::ptr::null(),
                    );
                }
            }
            if let Ok(mut samplers) = self.samplers.write() {
                for (_, sampler) in samplers.drain() {
                    ffi::vkDestroySampler(self.device, sampler, core::ptr::null());
                }
            }
            if let Ok(mut views) = self.image_views.write() {
                for (_, view) in views.drain() {
                    ffi::vkDestroyImageView(self.device, view, core::ptr::null());
                }
            }
            if let Ok(mut pools) = self.query_pools.write() {
                for (_, qp) in pools.drain() {
                    ffi::vkDestroyQueryPool(self.device, qp.pool, core::ptr::null());
                }
            }

            // Free pooled command buffers before destroying the pool.
            let pooled: Vec<_> = self
                .cmd_buffer_pool
                .lock()
                .map(|mut pool| pool.drain(..).collect())
                .unwrap_or_default();
            if !pooled.is_empty() {
                ffi::vkFreeCommandBuffers(
                    self.device,
                    self.command_pool,
                    pooled.len() as u32,
                    pooled.as_ptr(),
                );
            }

            ffi::vkDestroyCommandPool(self.device, self.command_pool, core::ptr::null());
            ffi::vkDestroyDevice(self.device, core::ptr::null());
            ffi::vkDestroyInstance(self.instance, core::ptr::null());
        }
    }
}
