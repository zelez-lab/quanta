//! Async-only public extension methods for the WebGPU driver.
//!
//! WebGPU's JS API is async for buffer read-back (`mapAsync`), query
//! resolution, and completion (`onSubmittedWorkDone`). The browser
//! event loop is non-blocking, so the sync `GpuDevice` trait methods
//! (`field_read_bytes`, `pulse_wait`, `texture_read`,
//! `occlusion_query_read`) return errors directing callers here.

use alloc::vec::Vec;

use crate::{Pulse, QuantaError, Texture};

use super::WebgpuDevice;
use super::executor::Promise;
use super::ffi::{self, buffer_usage};

// ── Async-only public extension methods ─────────────────────────────────────

#[allow(dead_code)] // public extension API used by browser-side callers.
impl WebgpuDevice {
    /// Async sibling of [`field_read_bytes`]: maps the buffer for read,
    /// awaits the GPU, copies into a `Vec<u8>`, and unmaps.
    ///
    /// `field_read_bytes` (sync trait) cannot be implemented on WebGPU
    /// because the browser event loop is non-blocking. Use this method
    /// from async Rust code.
    pub async fn field_read_bytes_async(
        &self,
        handle: u64,
        size: usize,
    ) -> Result<Vec<u8>, QuantaError> {
        let device = self.dev()?;
        let staging = unsafe {
            ffi::quanta_create_buffer(
                device,
                size as f64,
                buffer_usage::COPY_DST | buffer_usage::MAP_READ,
            )
        };

        {
            let buffers = self.state.buffers.0.borrow();
            let &src = buffers
                .get(&handle)
                .ok_or_else(|| Self::err("unknown buffer handle"))?;
            let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
            unsafe {
                ffi::quanta_encoder_copy_buffer_to_buffer(
                    encoder,
                    src,
                    0.0,
                    staging,
                    0.0,
                    size as f64,
                );
            }
            let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
            unsafe { ffi::quanta_queue_submit(device, cmd) };
        }

        Promise::register(|task| unsafe { ffi::quanta_map_async_read(staging, task) })
            .await
            .map_err(|_| Self::err("mapAsync rejected"))?;

        let mut out = alloc::vec![0u8; size];
        unsafe {
            ffi::quanta_get_mapped_range_copy(staging, out.as_mut_ptr(), size);
        }
        unsafe {
            ffi::quanta_unmap_buffer(staging);
            ffi::quanta_destroy_buffer(staging);
        }
        Ok(out)
    }

    /// Async sibling of [`occlusion_query_read`]: resolves the
    /// query set into a staging buffer, awaits `mapAsync`, and
    /// returns the per-slot u64 fragment counts.
    ///
    /// The sync trait method `occlusion_query_read` cannot be
    /// implemented on WebGPU because the browser event loop is
    /// non-blocking. Use this method from async Rust code (or
    /// drive the typed `quanta::Pulse` async waiter and read
    /// from a buffer you own via `field_read_bytes_async`).
    pub async fn occlusion_query_read_async(
        &self,
        handle: u64,
    ) -> Result<alloc::vec::Vec<u64>, QuantaError> {
        let device = self.dev()?;
        let (qs_js, count) = {
            let qs = self.state.query_sets.0.borrow();
            *qs.get(&handle)
                .ok_or_else(|| Self::err("unknown occlusion query handle"))?
        };
        let bytes = (count as f64) * 8.0;
        let staging = unsafe {
            ffi::quanta_create_buffer(
                device,
                bytes,
                buffer_usage::COPY_SRC
                    | buffer_usage::COPY_DST
                    | buffer_usage::QUERY_RESOLVE
                    | buffer_usage::MAP_READ,
            )
        };

        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        unsafe {
            ffi::quanta_encoder_resolve_query_set(encoder, qs_js, 0, count, staging, 0.0);
        }
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };

        Promise::register(|task| unsafe { ffi::quanta_map_async_read(staging, task) })
            .await
            .map_err(|_| Self::err("occlusion query mapAsync rejected"))?;

        let size = bytes as usize;
        let mut raw = alloc::vec![0u8; size];
        unsafe {
            ffi::quanta_get_mapped_range_copy(staging, raw.as_mut_ptr(), size);
            ffi::quanta_unmap_buffer(staging);
            ffi::quanta_destroy_buffer(staging);
        }

        // Each slot is a little-endian u64.
        let mut out = alloc::vec::Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let off = i * 8;
            let chunk: [u8; 8] = raw[off..off + 8].try_into().unwrap_or([0u8; 8]);
            out.push(u64::from_le_bytes(chunk));
        }
        Ok(out)
    }

    /// Async sibling of [`pulse_wait`]: awaits
    /// `device.queue.onSubmittedWorkDone()` for the pulse's submission.
    pub async fn pulse_wait_async(&self, _pulse: &mut Pulse) -> Result<(), QuantaError> {
        let device = self.dev()?;
        Promise::register(|task| unsafe { ffi::quanta_queue_on_submitted_work_done(device, task) })
            .await
            .map_err(|_| Self::err("onSubmittedWorkDone rejected"))?;
        Ok(())
    }

    /// Async sibling of [`texture_read`]: copies the texture to a
    /// staging buffer, awaits `mapAsync`, and returns the pixel bytes
    /// (tightly packed in the texture's native row stride).
    pub async fn texture_read_async(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let device = self.dev()?;
        let (tex_handle, view_dims, bytes_per_row, height, tight_row) = {
            let textures = self.state.textures.0.borrow();
            let entry = textures
                .get(&texture.handle)
                .ok_or_else(|| Self::err("unknown texture handle"))?;
            (
                entry.texture,
                (entry.width, entry.height),
                entry.bytes_per_row,
                entry.height,
                entry.width * entry.format.bytes_per_pixel() as u32,
            )
        };

        let total_bytes = (bytes_per_row as u64) * (height as u64);
        let staging = unsafe {
            ffi::quanta_create_buffer(
                device,
                total_bytes as f64,
                buffer_usage::COPY_DST | buffer_usage::MAP_READ,
            )
        };

        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        unsafe {
            ffi::quanta_encoder_copy_texture_to_buffer(
                encoder,
                tex_handle,
                staging,
                bytes_per_row,
                height,
                view_dims.0,
                view_dims.1,
                1,
            );
        }
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };

        Promise::register(|task| unsafe { ffi::quanta_map_async_read(staging, task) })
            .await
            .map_err(|_| Self::err("texture mapAsync rejected"))?;

        let total = (bytes_per_row as usize) * (height as usize);
        let mut padded = alloc::vec![0u8; total];
        unsafe {
            ffi::quanta_get_mapped_range_copy(staging, padded.as_mut_ptr(), total);
        }
        unsafe {
            ffi::quanta_unmap_buffer(staging);
            ffi::quanta_destroy_buffer(staging);
        }

        if bytes_per_row == tight_row {
            Ok(padded)
        } else {
            let mut out = Vec::with_capacity(tight_row as usize * height as usize);
            for row in 0..height as usize {
                let off = row * bytes_per_row as usize;
                out.extend_from_slice(&padded[off..off + tight_row as usize]);
            }
            Ok(out)
        }
    }
}
