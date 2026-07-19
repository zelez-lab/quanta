//! Validation wrapper — catches common GPU API misuse at runtime.
//!
//! Enabled via `QUANTA_VALIDATE=1` environment variable.
//! Wraps the real driver and checks:
//! - field_write to a freed handle panics with a clear message
//! - dispatch with no bindings emits a warning
//! - texture_write with wrong data size panics

use alloc::boxed::Box;
use alloc::vec::Vec;
#[cfg(feature = "compute")]
use std::eprintln;

use crate::{
    Caps, FieldUsage, GpuDevice, Pulse, QuantaError, Texture, TextureDesc, TextureViewDesc,
    Timeline,
};
// `Wave` exists only on the compute face.
#[cfg(feature = "compute")]
use crate::Wave;
// Render types used only by the render-gated forwards (step 085).
#[cfg(feature = "render")]
use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
#[cfg(feature = "render")]
use crate::{Pipeline, RenderPass};
use std::collections::HashSet;
use std::sync::Mutex;

/// A validation layer that wraps any `GpuDevice`.
///
/// Tracks allocated handles and checks for misuse before forwarding
/// calls to the underlying driver.
pub struct ValidationDevice {
    inner: Box<dyn GpuDevice + Send + Sync>,
    live_fields: Mutex<HashSet<u64>>,
    live_textures: Mutex<HashSet<u64>>,
}

impl ValidationDevice {
    pub fn wrap(inner: Box<dyn GpuDevice>) -> Box<dyn GpuDevice> {
        Box::new(Self {
            inner,
            live_fields: Mutex::new(HashSet::new()),
            live_textures: Mutex::new(HashSet::new()),
        })
    }
}

impl crate::api::device::sealed::Sealed for ValidationDevice {}

impl GpuDevice for ValidationDevice {
    // The weak points at THIS wrapper's Arc; forwarding it lets the
    // inner driver's pulses keep the whole wrapper (and thus the inner
    // device it owns) alive.
    fn install_self_ref(&self, self_ref: alloc::sync::Weak<dyn GpuDevice>) {
        self.inner.install_self_ref(self_ref);
    }

    fn caps(&self) -> &Caps {
        self.inner.caps()
    }

    // === Fields ===

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError> {
        let handle = self.inner.field_alloc(size, usage)?;
        self.live_fields.lock().unwrap().insert(handle);
        Ok(handle)
    }

    fn field_free(&self, handle: u64) {
        self.live_fields.lock().unwrap().remove(&handle);
        self.inner.field_free(handle);
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        if !self.live_fields.lock().unwrap().contains(&handle) {
            panic!(
                "QUANTA_VALIDATE: field_write_bytes to freed handle {handle}. \
                 The field was already dropped or never allocated."
            );
        }
        self.inner.field_write_bytes(handle, data)
    }

    fn field_write_bytes_at(
        &self,
        handle: u64,
        byte_offset: usize,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        if !self.live_fields.lock().unwrap().contains(&handle) {
            panic!(
                "QUANTA_VALIDATE: field_write_bytes_at to freed handle {handle}. \
                 The field was already dropped or never allocated."
            );
        }
        self.inner.field_write_bytes_at(handle, byte_offset, data)
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        if !self.live_fields.lock().unwrap().contains(&handle) {
            panic!(
                "QUANTA_VALIDATE: field_read_bytes from freed handle {handle}. \
                 The field was already dropped or never allocated."
            );
        }
        self.inner.field_read_bytes(handle, size)
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let live = self.live_fields.lock().unwrap();
        if !live.contains(&src) {
            panic!("QUANTA_VALIDATE: field_copy_bytes from freed src handle {src}.");
        }
        if !live.contains(&dst) {
            panic!("QUANTA_VALIDATE: field_copy_bytes to freed dst handle {dst}.");
        }
        drop(live);
        self.inner.field_copy_bytes(dst, src, size)
    }

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let tex = self.inner.texture_create(desc)?;
        self.live_textures.lock().unwrap().insert(tex.handle());
        Ok(tex)
    }

    fn supports_texture_write_region(&self) -> bool {
        self.inner.supports_texture_write_region()
    }

    fn texture_write_region(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        size: (u32, u32),
        data: &[u8],
    ) -> Result<(), QuantaError> {
        if !self
            .live_textures
            .lock()
            .unwrap()
            .contains(&texture.handle())
        {
            panic!(
                "QUANTA_VALIDATE: texture_write_region to freed texture handle {}.",
                texture.handle()
            );
        }
        self.inner.texture_write_region(texture, origin, size, data)
    }

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        if !self
            .live_textures
            .lock()
            .unwrap()
            .contains(&texture.handle())
        {
            panic!(
                "QUANTA_VALIDATE: texture_write to freed texture handle {}.",
                texture.handle()
            );
        }
        let bpp = texture.format().bytes_per_pixel();
        let expected = (texture.width() * texture.height()) as usize * bpp;
        if data.len() != expected {
            panic!(
                "QUANTA_VALIDATE: texture_write data size mismatch. \
                 Expected {expected} bytes ({}x{}, {bpp} bpp) but got {} bytes.",
                texture.width(),
                texture.height(),
                data.len()
            );
        }
        self.inner.texture_write(texture, data)
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.inner.texture_read(texture)
    }

    fn sampler_create(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        self.inner.sampler_create(desc)
    }

    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        self.inner.generate_mipmaps(texture)
    }

    // === Render-resource lifecycle (destroy forwards) ===
    //
    // Must forward explicitly: the trait defaults are no-ops, so
    // relying on them here would leak the inner driver's entries.

    fn texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.live_textures.lock().unwrap().remove(&handle);
        self.inner.texture_destroy(handle)
    }

    fn sampler_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.sampler_destroy(handle)
    }

    #[cfg(feature = "render")]
    fn pipeline_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.pipeline_destroy(handle)
    }

    fn occlusion_query_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.occlusion_query_destroy(handle)
    }

    // Compute-resource lifecycle: same rule — forward explicitly so
    // the inner driver's wave registry entry is actually freed.
    #[cfg(feature = "compute")]
    fn wave_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.wave_destroy(handle)
    }

    fn debug_registry_counts(&self) -> crate::RegistryCounts {
        self.inner.debug_registry_counts()
    }

    // === Compute ===

    #[cfg(feature = "compute")]
    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.inner.wave(kernel)
    }

    #[cfg(feature = "compute")]
    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        if wave.binding_count == 0 {
            eprintln!(
                "QUANTA_VALIDATE warning: wave_dispatch with no bindings \
                 (handle {}). Did you forget to call wave.bind()?",
                wave.handle
            );
        }
        self.inner.wave_dispatch(wave, groups)
    }

    #[cfg(feature = "compute")]
    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        if wave.binding_count == 0 {
            eprintln!(
                "QUANTA_VALIDATE warning: wave_dispatch_indirect with no bindings \
                 (handle {}). Did you forget to call wave.bind()?",
                wave.handle
            );
        }
        self.inner.wave_dispatch_indirect(wave, buffer, offset)
    }

    // === Render === (render-gated, step 085)

    #[cfg(feature = "render")]
    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.inner.pipeline_create(desc)
    }

    #[cfg(feature = "render")]
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        self.inner.render_begin(target)
    }

    #[cfg(feature = "render")]
    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        self.inner.render_end(pass)
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        self.inner.pulse_wait(pulse)
    }

    fn pulse_poll(&self, pulse: &Pulse) -> bool {
        self.inner.pulse_poll(pulse)
    }

    // === Timestamps ===

    fn timestamp_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        self.inner.timestamp_query_create(count)
    }

    fn timestamp_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        self.inner.timestamp_query_read(handle)
    }

    // === Capability queries ===
    //
    // Must reflect the wrapped device, not the trait defaults —
    // otherwise QUANTA_VALIDATE=1 silently steers capability-gated
    // paths (64-bit kernels, subgroup reduce) onto their fallbacks.

    fn supports_f64(&self) -> bool {
        self.inner.supports_f64()
    }

    fn supports_i64(&self) -> bool {
        self.inner.supports_i64()
    }

    fn supports_subgroups(&self) -> bool {
        self.inner.supports_subgroups()
    }

    // === Async compute ===

    fn supports_async_compute(&self) -> bool {
        self.inner.supports_async_compute()
    }

    #[cfg(feature = "compute")]
    fn async_compute_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.inner.async_compute_dispatch(wave, groups)
    }

    // === Timeline semaphores ===

    fn timeline_create(&self) -> Result<Timeline, QuantaError> {
        self.inner.timeline_create()
    }

    fn timeline_signal(&self, timeline: &Timeline, value: u64) -> Result<(), QuantaError> {
        self.inner.timeline_signal(timeline, value)
    }

    fn timeline_wait(&self, timeline: &Timeline, value: u64) -> Result<(), QuantaError> {
        self.inner.timeline_wait(timeline, value)
    }

    // === Texture views ===

    fn texture_view_create(
        &self,
        texture: u64,
        desc: &TextureViewDesc,
    ) -> Result<u64, QuantaError> {
        self.inner.texture_view_create(texture, desc)
    }

    fn texture_view_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.texture_view_destroy(handle)
    }

    // === Barriers ===

    fn barrier(&self) -> Result<(), QuantaError> {
        self.inner.barrier()
    }

    fn wait_idle(&self) -> Result<(), QuantaError> {
        self.inner.wait_idle()
    }

    fn barrier_buffer(
        &self,
        handle: u64,
        from: crate::ResourceState,
        to: crate::ResourceState,
    ) -> Result<(), QuantaError> {
        self.inner.barrier_buffer(handle, from, to)
    }

    fn barrier_texture(
        &self,
        texture: &Texture,
        from: crate::ResourceState,
        to: crate::ResourceState,
    ) -> Result<(), QuantaError> {
        self.inner.barrier_texture(texture, from, to)
    }

    // === MSAA Resolve ===

    #[cfg(feature = "render")]
    fn resolve_texture(&self, src_handle: u64, dst_handle: u64) -> Result<(), QuantaError> {
        self.inner.resolve_texture(src_handle, dst_handle)
    }

    // === Multi-queue ===

    fn queue_families(&self) -> Vec<crate::QueueFamily> {
        self.inner.queue_families()
    }

    fn create_queue(&self, queue_type: crate::QueueType) -> Result<u64, QuantaError> {
        self.inner.create_queue(queue_type)
    }

    #[cfg(feature = "compute")]
    fn queue_dispatch(&self, queue: u64, wave: &Wave, groups: [u32; 3]) -> Result<(), QuantaError> {
        self.inner.queue_dispatch(queue, wave, groups)
    }

    fn queue_signal(&self, queue: u64, semaphore: u64) -> Result<(), QuantaError> {
        self.inner.queue_signal(queue, semaphore)
    }

    fn queue_wait(&self, queue: u64, semaphore: u64) -> Result<(), QuantaError> {
        self.inner.queue_wait(queue, semaphore)
    }

    // === Occlusion queries ===

    fn occlusion_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        self.inner.occlusion_query_create(count)
    }

    fn occlusion_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        self.inner.occlusion_query_read(handle)
    }

    // === Mesh shaders ===

    fn dispatch_mesh(&self, pipeline: u64, groups: [u32; 3]) -> Result<(), QuantaError> {
        self.inner.dispatch_mesh(pipeline, groups)
    }

    // === Ray tracing === (render-typed methods gated, step 085)

    #[cfg(feature = "render")]
    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        self.inner.build_acceleration_structure(geometry)
    }

    #[cfg(feature = "render")]
    fn create_ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        self.inner.create_ray_tracing_pipeline(desc)
    }

    fn dispatch_rays(&self, pipeline: u64, width: u32, height: u32) -> Result<(), QuantaError> {
        self.inner.dispatch_rays(pipeline, width, height)
    }

    fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.destroy_acceleration_structure(handle)
    }

    // === Sparse textures ===

    fn sparse_texture_create(&self, desc: &TextureDesc) -> Result<u64, QuantaError> {
        self.inner.sparse_texture_create(desc)
    }

    fn sparse_map_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
        backing: u64,
    ) -> Result<(), QuantaError> {
        self.inner.sparse_map_tile(texture, mip, x, y, backing)
    }

    fn sparse_unmap_tile(&self, texture: u64, mip: u32, x: u32, y: u32) -> Result<(), QuantaError> {
        self.inner.sparse_unmap_tile(texture, mip, x, y)
    }

    // === Indirect command buffers ===

    #[cfg(feature = "compute")]
    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        self.inner.indirect_buffer_create(max_commands)
    }

    #[cfg(feature = "compute")]
    fn icb_record_dispatch(
        &self,
        handle: u64,
        index: u32,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        self.inner.icb_record_dispatch(handle, index, wave, groups)
    }

    #[cfg(feature = "compute")]
    fn icb_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        self.inner
            .icb_record_draw(handle, index, pipeline, vertex_count, instance_count)
    }

    fn render_bundle_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        self.inner.render_bundle_create(max_commands)
    }

    fn render_bundle_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        self.inner
            .render_bundle_record_draw(handle, index, pipeline, vertex_count, instance_count)
    }

    fn render_bundle_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.render_bundle_destroy(handle)
    }

    #[cfg(feature = "compute")]
    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError> {
        self.inner.indirect_buffer_execute(handle, count)
    }

    #[cfg(feature = "compute")]
    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.indirect_buffer_destroy(handle)
    }

    // === Bindless resources ===

    fn bindless_texture_create(&self, cap: u32) -> Result<u64, QuantaError> {
        self.inner.bindless_texture_create(cap)
    }

    fn bindless_texture_set(
        &self,
        handle: u64,
        index: u32,
        texture: u64,
    ) -> Result<(), QuantaError> {
        self.inner.bindless_texture_set(handle, index, texture)
    }

    fn bindless_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.bindless_texture_destroy(handle)
    }

    fn bindless_buffer_create(&self, cap: u32) -> Result<u64, QuantaError> {
        self.inner.bindless_buffer_create(cap)
    }

    fn bindless_buffer_set(&self, handle: u64, index: u32, buffer: u64) -> Result<(), QuantaError> {
        self.inner.bindless_buffer_set(handle, index, buffer)
    }

    fn bindless_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.bindless_buffer_destroy(handle)
    }

    // === Debug ===

    fn debug_push(&self, label: &str) {
        self.inner.debug_push(label);
    }

    fn debug_pop(&self) {
        self.inner.debug_pop();
    }

    // === Presentation & interop (forwarded) ===

    fn supports_native_handle_export(&self) -> bool {
        self.inner.supports_native_handle_export()
    }

    fn texture_native_handle(
        &self,
        texture: &Texture,
    ) -> Result<crate::NativeTextureHandle, QuantaError> {
        self.inner.texture_native_handle(texture)
    }

    fn supports_surface_present(&self) -> bool {
        self.inner.supports_surface_present()
    }

    #[cfg(feature = "render")]
    fn surface_create(
        &self,
        target: &crate::surface::SurfaceTarget,
        config: &crate::surface::SurfaceConfig,
    ) -> Result<u64, QuantaError> {
        self.inner.surface_create(target, config)
    }

    #[cfg(feature = "render")]
    fn surface_configure(
        &self,
        surface: u64,
        config: &crate::surface::SurfaceConfig,
    ) -> Result<(), QuantaError> {
        self.inner.surface_configure(surface, config)
    }

    #[cfg(feature = "render")]
    fn surface_format(&self, surface: u64) -> Result<crate::Format, QuantaError> {
        self.inner.surface_format(surface)
    }

    #[cfg(feature = "render")]
    fn surface_current_extent(&self, surface: u64) -> Option<(u32, u32)> {
        self.inner.surface_current_extent(surface)
    }

    #[cfg(feature = "render")]
    fn surface_acquire(&self, surface: u64) -> Result<(u64, Texture), QuantaError> {
        self.inner.surface_acquire(surface)
    }

    #[cfg(feature = "render")]
    fn surface_present(&self, surface: u64, frame: u64) -> Result<(), QuantaError> {
        self.inner.surface_present(surface, frame)
    }

    #[cfg(feature = "render")]
    fn surface_discard(&self, surface: u64, frame: u64) -> Result<(), QuantaError> {
        self.inner.surface_discard(surface, frame)
    }

    #[cfg(feature = "render")]
    fn surface_destroy(&self, surface: u64) -> Result<(), QuantaError> {
        self.inner.surface_destroy(surface)
    }
}
