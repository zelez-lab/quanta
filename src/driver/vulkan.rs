//! Vulkan driver for Linux, Android, and Windows.
//!
//! Uses the `ash` crate for raw Vulkan bindings.
//! Covers compute dispatch, render pass execution, texture management,
//! depth/stencil, instanced/indexed/indirect draw, MRT, and debug labels.

mod compute;
mod memory;
mod render;
mod sync;
mod texture;

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, ResourceState,
    Texture, TextureDesc, Vendor, Wave,
};
use ash::vk;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Mutex;

/// Vulkan-backed GPU device.
pub struct VulkanDevice {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family: u32,
    command_pool: vk::CommandPool,
    caps: Caps,
    // Resource storage
    buffers: Mutex<HashMap<u64, VkBuffer>>,
    textures: Mutex<HashMap<u64, VkTexture>>,
    compute_pipelines: Mutex<HashMap<u64, VkComputePipeline>>,
    render_pipelines: Mutex<HashMap<u64, VkRenderPipeline>>,
    samplers: Mutex<HashMap<u64, vk::Sampler>>,
    next_handle: Mutex<u64>,
    /// Pool of reusable command buffers. Instead of allocating and freeing
    /// command buffers per submission, completed buffers are reset and returned
    /// here. `alloc_command_buffer` draws from this pool first.
    cmd_buffer_pool: Mutex<Vec<vk::CommandBuffer>>,
}

#[allow(dead_code)]
struct VkBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

#[allow(dead_code)]
struct VkTexture {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    format: vk::Format,
}

struct VkComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

struct VkRenderPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

impl VulkanDevice {
    fn alloc_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        *h += 1;
        *h
    }

    fn alloc_command_buffer(&self) -> Result<vk::CommandBuffer, QuantaError> {
        // Try to reuse a previously returned command buffer from the pool.
        if let Some(cmd) = self.cmd_buffer_pool.lock().unwrap().pop() {
            unsafe {
                self.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|_| QuantaError::submit_failed())?;
            }
            return Ok(cmd);
        }
        // Pool empty -- allocate a fresh one.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let bufs = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|_| QuantaError::submit_failed())?
        };
        Ok(bufs[0])
    }

    fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), QuantaError> {
        let cmd_bufs = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())
                .map_err(|_| QuantaError::submit_failed())?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        // Return to pool for reuse instead of freeing.
        self.cmd_buffer_pool.lock().unwrap().push(cmd);
        Ok(())
    }
}

/// Discover Vulkan devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(_) => return Vec::new(), // Vulkan not available
    };

    let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));

    let layer_names: Vec<CString> = Vec::new();
    let layer_ptrs: Vec<*const i8> = layer_names.iter().map(|n| n.as_ptr()).collect();

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs);

    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };

    let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut devices: Vec<Box<dyn GpuDevice>> = Vec::new();

    for pd in physical_devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pd) };

        // Find a queue family that supports compute + graphics
        let queue_family = queue_families.iter().enumerate().find(|(_, qf)| {
            qf.queue_flags
                .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
        });

        let Some((qf_index, _)) = queue_family else {
            continue;
        };

        let queue_priorities = [1.0f32];
        let queue_create = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_index as u32)
            .queue_priorities(&queue_priorities);

        let device_create =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_create));

        let device = match unsafe { instance.create_device(pd, &device_create, None) } {
            Ok(d) => d,
            Err(_) => continue,
        };

        let queue = unsafe { device.get_device_queue(qf_index as u32, 0) };

        // Command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_index as u32)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(_) => continue,
        };

        let name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };

        let vendor = match props.vendor_id {
            0x1002 => Vendor::Amd,
            0x10DE => Vendor::Nvidia,
            0x8086 => Vendor::Intel,
            0x13B5 => Vendor::Broadcom,
            _ => Vendor::Unknown,
        };

        let caps = Caps {
            nuclei: props.limits.max_compute_work_group_count[0].min(1024),
            protons_per_nucleus: 1,
            quarks_per_proton: props.limits.max_compute_work_group_size[0],
            memory_bytes: 0, // Would need VK_EXT_memory_budget
            max_quarks_per_dispatch: props.limits.max_compute_work_group_invocations,
            max_groups: props.limits.max_compute_work_group_count,
            vendor,
            name,
        };

        devices.push(Box::new(VulkanDevice {
            _entry: entry.clone(),
            instance: instance.clone(),
            physical_device: pd,
            device,
            queue,
            queue_family: qf_index as u32,
            command_pool,
            caps,
            buffers: Mutex::new(HashMap::new()),
            textures: Mutex::new(HashMap::new()),
            compute_pipelines: Mutex::new(HashMap::new()),
            render_pipelines: Mutex::new(HashMap::new()),
            samplers: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(0),
            cmd_buffer_pool: Mutex::new(Vec::new()),
        }));

        break; // Use first suitable device
    }

    devices
}

impl GpuDevice for VulkanDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    // === Fields ===

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError> {
        self.field_alloc_impl(size, usage)
    }

    fn field_free(&self, handle: u64) {
        self.field_free_impl(handle)
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        self.field_write_bytes_impl(handle, data)
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        self.field_read_bytes_impl(handle, size)
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        self.field_copy_bytes_impl(dst, src, size)
    }

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        self.texture_create_impl(desc)
    }

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        self.texture_write_impl(texture, data)
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.texture_read_impl(texture)
    }

    fn sampler_create(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        self.sampler_create_impl(desc)
    }

    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        self.generate_mipmaps_impl(texture)
    }

    // === Compute ===

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_impl(kernel)
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_impl(wave, groups)
    }

    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_indirect_impl(wave, buffer, offset)
    }

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.pipeline_create_impl(desc)
    }

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        self.render_begin_impl(target)
    }

    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        self.render_end_impl(pass)
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        pulse.wait()
    }

    fn pulse_poll(&self, pulse: &Pulse) -> bool {
        pulse.is_done()
    }

    // === Barriers ===

    fn barrier(&self) -> Result<(), QuantaError> {
        self.barrier_impl()
    }

    fn barrier_buffer(
        &self,
        handle: u64,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.barrier_buffer_impl(handle, from, to)
    }

    fn barrier_texture(
        &self,
        texture: &Texture,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.barrier_texture_impl(texture, from, to)
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Clean up resources
            for (_, buf) in self.buffers.lock().unwrap().drain() {
                self.device.destroy_buffer(buf.buffer, None);
                self.device.free_memory(buf.memory, None);
            }
            for (_, tex) in self.textures.lock().unwrap().drain() {
                self.device.destroy_image_view(tex.view, None);
                self.device.destroy_image(tex.image, None);
                self.device.free_memory(tex.memory, None);
            }
            for (_, cp) in self.compute_pipelines.lock().unwrap().drain() {
                self.device.destroy_pipeline(cp.pipeline, None);
                self.device.destroy_pipeline_layout(cp.layout, None);
                self.device
                    .destroy_descriptor_set_layout(cp.descriptor_set_layout, None);
            }
            for (_, rp) in self.render_pipelines.lock().unwrap().drain() {
                self.device.destroy_pipeline(rp.pipeline, None);
                self.device.destroy_pipeline_layout(rp.layout, None);
                self.device.destroy_render_pass(rp.render_pass, None);
                self.device
                    .destroy_descriptor_set_layout(rp.descriptor_set_layout, None);
            }
            for (_, sampler) in self.samplers.lock().unwrap().drain() {
                self.device.destroy_sampler(sampler, None);
            }

            // Free pooled command buffers before destroying the pool.
            let pooled: Vec<_> = self.cmd_buffer_pool.lock().unwrap().drain(..).collect();
            if !pooled.is_empty() {
                self.device.free_command_buffers(self.command_pool, &pooled);
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ============================================================================
// Vulkan type conversions
// ============================================================================

fn format_to_vulkan(format: Format) -> vk::Format {
    match format {
        Format::RGBA8 => vk::Format::R8G8B8A8_UNORM,
        Format::BGRA8 => vk::Format::B8G8R8A8_UNORM,
        Format::R8 => vk::Format::R8_UNORM,
        Format::R16Float => vk::Format::R16_SFLOAT,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::RG32Float => vk::Format::R32G32_SFLOAT,
        Format::RGBA16Float => vk::Format::R16G16B16A16_SFLOAT,
        Format::RGBA32Float => vk::Format::R32G32B32A32_SFLOAT,
        Format::Depth32Float => vk::Format::D32_SFLOAT,
        // Compressed formats
        Format::Bc1Rgba => vk::Format::BC1_RGBA_UNORM_BLOCK,
        Format::Bc3Rgba => vk::Format::BC3_UNORM_BLOCK,
        Format::Bc5Rg => vk::Format::BC5_SNORM_BLOCK,
        Format::Bc7Rgba => vk::Format::BC7_UNORM_BLOCK,
        Format::Astc4x4 => vk::Format::ASTC_4X4_UNORM_BLOCK,
        Format::Astc6x6 => vk::Format::ASTC_6X6_UNORM_BLOCK,
        Format::Astc8x8 => vk::Format::ASTC_8X8_UNORM_BLOCK,
        Format::Etc2Rgb8 => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
        Format::Etc2Rgba8 => vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
    }
}

fn sample_count_to_vk(count: u32) -> vk::SampleCountFlags {
    match count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

fn format_bytes_per_pixel_vk(format: Format) -> usize {
    match format {
        Format::R8 => 1,
        Format::R16Float => 2,
        Format::R32Float | Format::RGBA8 | Format::BGRA8 => 4,
        Format::RG32Float | Format::RGBA16Float => 8,
        Format::RGBA32Float => 16,
        Format::Depth32Float => 4,
        // Compressed: block size in bytes
        Format::Bc1Rgba | Format::Etc2Rgb8 => 8,
        Format::Bc3Rgba | Format::Bc5Rg | Format::Bc7Rgba | Format::Etc2Rgba8 => 16,
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 16,
    }
}

fn blend_factor_to_vk(f: crate::BlendFactor) -> vk::BlendFactor {
    use crate::BlendFactor::*;
    match f {
        Zero => vk::BlendFactor::ZERO,
        One => vk::BlendFactor::ONE,
        SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        DstAlpha => vk::BlendFactor::DST_ALPHA,
        OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        SrcColor => vk::BlendFactor::SRC_COLOR,
        OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        DstColor => vk::BlendFactor::DST_COLOR,
        OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
    }
}

fn blend_op_to_vk(op: crate::BlendOp) -> vk::BlendOp {
    use crate::BlendOp::*;
    match op {
        Add => vk::BlendOp::ADD,
        Subtract => vk::BlendOp::SUBTRACT,
        ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        Min => vk::BlendOp::MIN,
        Max => vk::BlendOp::MAX,
    }
}

fn filter_to_vk(f: crate::render_pass::Filter) -> vk::Filter {
    match f {
        crate::render_pass::Filter::Nearest => vk::Filter::NEAREST,
        crate::render_pass::Filter::Linear => vk::Filter::LINEAR,
    }
}

fn address_to_vk(a: crate::render_pass::AddressMode) -> vk::SamplerAddressMode {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        crate::render_pass::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        crate::render_pass::AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}
