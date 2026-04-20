//! Compute dispatch operations for Metal.

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::{Pulse, QuantaError, Wave};
use metal as mtl;

use super::MetalDevice;

impl MetalDevice {
    pub(crate) fn wave_impl(&self, kernel_source: &[u8]) -> Result<Wave, QuantaError> {
        // Try pre-compiled metallib binary first, fall back to MSL text compilation.
        let library = if kernel_source.len() >= 4 && &kernel_source[..4] == b"MTLB" {
            // Pre-compiled metallib binary — load directly, zero runtime compilation.
            self.device
                .new_library_with_data(kernel_source)
                .map_err(|e| QuantaError::compilation_failed(e.to_string()))?
        } else {
            // MSL text — compile at runtime (fallback when xcrun was not available).
            let source_str = std::str::from_utf8(kernel_source)
                .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in shader source"))?;
            let opts = mtl::CompileOptions::new();
            self.device
                .new_library_with_source(source_str, &opts)
                .map_err(|e| QuantaError::compilation_failed(e.to_string()))?
        };
        let func_names = library.function_names();
        let func_name = func_names
            .first()
            .ok_or_else(|| QuantaError::compilation_failed("no functions in kernel"))?;
        let func = library
            .get_function(func_name, None)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;

        let handle = self.alloc_handle();
        self.compute_pipelines
            .lock()
            .unwrap()
            .insert(handle, pipeline);
        Ok(Wave {
            handle,
            bindings: Vec::new(),
            push_constants: Vec::new(),
            texture_bindings: Vec::new(),
            drop_fn: None,
        })
    }

    pub(crate) fn wave_dispatch_impl(
        &self,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        // Note: Metal's CommandQueue.new_command_buffer() internally pools command
        // buffers. There is no need for an explicit pool — Metal manages reuse
        // automatically when command buffers complete.
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        let pipelines = self.compute_pipelines.lock().unwrap();
        let pipeline = pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch: handle {}", wave.handle))
        })?;
        encoder.set_compute_pipeline_state(pipeline);

        let buffers = self.buffers.lock().unwrap();
        for b in &wave.bindings {
            if let Some(buf) = buffers.get(&b.field_handle) {
                encoder.set_buffer(b.slot as u64, Some(buf), 0);
            }
        }
        for pc in &wave.push_constants {
            encoder.set_bytes(
                pc.slot as u64,
                pc.data.len() as u64,
                pc.data.as_ptr() as *const _,
            );
        }

        // Bind textures for compute access
        let textures = self.textures.lock().unwrap();
        for tb in &wave.texture_bindings {
            if let Some(tex) = textures.get(&tb.texture_handle) {
                encoder.set_texture(tb.slot as u64, Some(tex));
            }
        }
        drop(textures);

        let grid = mtl::MTLSize::new(groups[0] as u64, groups[1] as u64, groups[2] as u64);
        let group_size = mtl::MTLSize::new(64, 1, 1);
        encoder.dispatch_threads(grid, group_size);
        encoder.end_encoding();
        cmd.commit();

        let cmd_clone = cmd.to_owned();
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(move |_| {
                cmd_clone.wait_until_completed();
                Ok(())
            })),
            poll_fn: None,
            completed: false,
        })
    }

    pub(crate) fn wave_dispatch_indirect_impl(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        let pipelines = self.compute_pipelines.lock().unwrap();
        let pipeline = pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch_indirect: handle {}", wave.handle))
        })?;
        encoder.set_compute_pipeline_state(pipeline);

        let buffers = self.buffers.lock().unwrap();
        for b in &wave.bindings {
            if let Some(buf) = buffers.get(&b.field_handle) {
                encoder.set_buffer(b.slot as u64, Some(buf), 0);
            }
        }
        for pc in &wave.push_constants {
            encoder.set_bytes(
                pc.slot as u64,
                pc.data.len() as u64,
                pc.data.as_ptr() as *const _,
            );
        }

        let indirect_buf = buffers.get(&buffer).ok_or_else(|| {
            QuantaError::invalid_param("bad indirect buffer")
                .with_context(&format!("wave_dispatch_indirect: buffer handle {buffer}"))
        })?;
        let group_size = mtl::MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups_indirect(indirect_buf, offset, group_size);
        encoder.end_encoding();
        cmd.commit();

        let cmd_clone = cmd.to_owned();
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(move |_| {
                cmd_clone.wait_until_completed();
                Ok(())
            })),
            poll_fn: None,
            completed: false,
        })
    }
}
