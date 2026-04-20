//! Validation wrapper — catches common GPU API misuse at runtime.
//!
//! Enabled via `QUANTA_VALIDATE=1` environment variable.
//! Wraps the real driver and checks:
//! - field_write to a freed handle panics with a clear message
//! - dispatch with no bindings emits a warning
//! - texture_write with wrong data size panics

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::eprintln;

use crate::{
    Caps, FieldUsage, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, Texture, TextureDesc,
    Timeline, Wave,
};
use hashbrown::HashSet;
use std::sync::Mutex;

/// A validation layer that wraps any `GpuDevice`.
///
/// Tracks allocated handles and checks for misuse before forwarding
/// calls to the underlying driver.
pub struct ValidationDevice {
    inner: Box<dyn GpuDevice>,
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

impl GpuDevice for ValidationDevice {
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
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        self.inner.sampler_create(desc)
    }

    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        self.inner.generate_mipmaps(texture)
    }

    // === Compute ===

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.inner.wave(kernel)
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        if wave.bindings.is_empty() {
            eprintln!(
                "QUANTA_VALIDATE warning: wave_dispatch with no bindings \
                 (handle {}). Did you forget to call wave.bind()?",
                wave.handle
            );
        }
        self.inner.wave_dispatch(wave, groups)
    }

    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        if wave.bindings.is_empty() {
            eprintln!(
                "QUANTA_VALIDATE warning: wave_dispatch_indirect with no bindings \
                 (handle {}). Did you forget to call wave.bind()?",
                wave.handle
            );
        }
        self.inner.wave_dispatch_indirect(wave, buffer, offset)
    }

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.inner.pipeline_create(desc)
    }

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        self.inner.render_begin(target)
    }

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

    // === Queries ===

    fn query_set_create(&self, count: u32) -> Result<u64, QuantaError> {
        self.inner.query_set_create(count)
    }

    fn query_set_read(&self, handle: u64, first: u32, count: u32) -> Result<Vec<u64>, QuantaError> {
        self.inner.query_set_read(handle, first, count)
    }

    // === Timestamps ===

    fn timestamp_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        self.inner.timestamp_query_create(count)
    }

    fn timestamp_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        self.inner.timestamp_query_read(handle)
    }

    // === Async compute ===

    fn supports_async_compute(&self) -> bool {
        self.inner.supports_async_compute()
    }

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

    // === Debug ===

    fn debug_push(&self, label: &str) {
        self.inner.debug_push(label);
    }

    fn debug_pop(&self) {
        self.inner.debug_pop();
    }
}
