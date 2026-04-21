//! Compute dispatch operations for Metal.

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use crate::{Pulse, QuantaError, Wave};

use super::MetalDevice;
use super::ffi;

impl MetalDevice {
    pub(crate) fn wave_impl(&self, kernel_source: &[u8]) -> Result<Wave, QuantaError> {
        let library = unsafe {
            if kernel_source.len() >= 4 && &kernel_source[..4] == b"MTLB" {
                // Pre-compiled metallib binary — load directly via dispatch_data.
                let (lib, error) = ffi::msg_new_library_with_data(
                    self.device,
                    kernel_source.as_ptr() as *const _,
                    kernel_source.len() as u64,
                );
                if lib.is_null() {
                    let msg = if !error.is_null() {
                        let desc = ffi::msg_id(error, b"localizedDescription\0");
                        let cstr = ffi::msg_utf8_string(desc);
                        std::ffi::CStr::from_ptr(cstr as *const _)
                            .to_string_lossy()
                            .into_owned()
                    } else {
                        "unknown metallib error".into()
                    };
                    return Err(QuantaError::compilation_failed(msg));
                }
                lib
            } else {
                // MSL text — compile at runtime.
                let source_str = std::str::from_utf8(kernel_source).map_err(|_| {
                    QuantaError::compilation_failed("invalid UTF-8 in shader source")
                })?;
                let mut src_bytes: Vec<u8> = source_str.bytes().collect();
                src_bytes.push(0);
                let ns_source = ffi::nsstring(&src_bytes);
                let (lib, error) =
                    ffi::msg_new_library_with_source(self.device, ns_source, ffi::NIL);
                if lib.is_null() {
                    let msg = if !error.is_null() {
                        let desc = ffi::msg_id(error, b"localizedDescription\0");
                        let cstr = ffi::msg_utf8_string(desc);
                        std::ffi::CStr::from_ptr(cstr as *const _)
                            .to_string_lossy()
                            .into_owned()
                    } else {
                        "unknown compilation error".into()
                    };
                    return Err(QuantaError::compilation_failed(msg));
                }
                lib
            }
        };

        // Get first function name from the library.
        let func_name = unsafe {
            let names = ffi::msg_function_names(library);
            let count = ffi::msg_array_count(names);
            if count == 0 {
                return Err(QuantaError::compilation_failed("no functions in kernel"));
            }
            ffi::msg_array_object_at(names, 0)
        };

        let func = unsafe { ffi::msg_get_function(library, func_name) };
        if func.is_null() {
            return Err(QuantaError::compilation_failed(
                "failed to get kernel function",
            ));
        }

        let (pipeline, error) = unsafe { ffi::msg_new_compute_pipeline(self.device, func) };
        if pipeline.is_null() {
            let msg = unsafe {
                if !error.is_null() {
                    let desc = ffi::msg_id(error, b"localizedDescription\0");
                    let cstr = ffi::msg_utf8_string(desc);
                    std::ffi::CStr::from_ptr(cstr as *const _)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "unknown pipeline error".into()
                }
            };
            return Err(QuantaError::compilation_failed(msg));
        }

        let handle = self.alloc_handle()?;
        self.compute_pipelines
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
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
        let cmd = unsafe { ffi::msg_id(self.queue, b"commandBuffer\0") };
        let encoder = unsafe { ffi::msg_id(cmd, b"computeCommandEncoder\0") };

        let pipelines = self
            .compute_pipelines
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipeline = pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch: handle {}", wave.handle))
        })?;
        unsafe {
            ffi::msg_void_id(encoder, b"setComputePipelineState:\0", *pipeline);
        }

        let buffers = self
            .buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for b in &wave.bindings {
            if let Some(buf) = buffers.get(&b.field_handle) {
                unsafe {
                    ffi::msg_set_buffer(
                        encoder,
                        b"setBuffer:offset:atIndex:\0",
                        *buf,
                        0,
                        b.slot as u64,
                    );
                }
            }
        }
        for pc in &wave.push_constants {
            unsafe {
                ffi::msg_set_bytes(
                    encoder,
                    b"setBytes:length:atIndex:\0",
                    pc.data.as_ptr() as *const _,
                    pc.data.len() as u64,
                    pc.slot as u64,
                );
            }
        }

        // Bind textures for compute access
        let textures = self
            .textures
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for tb in &wave.texture_bindings {
            if let Some(tex) = textures.get(&tb.texture_handle) {
                unsafe {
                    ffi::msg_set_texture(encoder, b"setTexture:atIndex:\0", *tex, tb.slot as u64);
                }
            }
        }
        drop(textures);

        let grid = ffi::MTLSize::new(groups[0] as u64, groups[1] as u64, groups[2] as u64);
        let group_size = ffi::MTLSize::new(64, 1, 1);
        unsafe {
            ffi::msg_dispatch_threads(encoder, grid, group_size);
            ffi::msg_void(encoder, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
        }

        Ok(Pulse {
            handle: self.alloc_handle()?,
            wait_fn: Some(Box::new(move |_| {
                unsafe { ffi::msg_void(cmd, b"waitUntilCompleted\0") };
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
        let cmd = unsafe { ffi::msg_id(self.queue, b"commandBuffer\0") };
        let encoder = unsafe { ffi::msg_id(cmd, b"computeCommandEncoder\0") };

        let pipelines = self
            .compute_pipelines
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipeline = pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch_indirect: handle {}", wave.handle))
        })?;
        unsafe {
            ffi::msg_void_id(encoder, b"setComputePipelineState:\0", *pipeline);
        }

        let buffers = self
            .buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for b in &wave.bindings {
            if let Some(buf) = buffers.get(&b.field_handle) {
                unsafe {
                    ffi::msg_set_buffer(
                        encoder,
                        b"setBuffer:offset:atIndex:\0",
                        *buf,
                        0,
                        b.slot as u64,
                    );
                }
            }
        }
        for pc in &wave.push_constants {
            unsafe {
                ffi::msg_set_bytes(
                    encoder,
                    b"setBytes:length:atIndex:\0",
                    pc.data.as_ptr() as *const _,
                    pc.data.len() as u64,
                    pc.slot as u64,
                );
            }
        }

        let indirect_buf = buffers.get(&buffer).ok_or_else(|| {
            QuantaError::invalid_param("bad indirect buffer")
                .with_context(&format!("wave_dispatch_indirect: buffer handle {buffer}"))
        })?;
        let group_size = ffi::MTLSize::new(64, 1, 1);
        unsafe {
            ffi::msg_dispatch_threadgroups_indirect(encoder, *indirect_buf, offset, group_size);
            ffi::msg_void(encoder, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
        }

        Ok(Pulse {
            handle: self.alloc_handle()?,
            wait_fn: Some(Box::new(move |_| {
                unsafe { ffi::msg_void(cmd, b"waitUntilCompleted\0") };
                Ok(())
            })),
            poll_fn: None,
            completed: false,
        })
    }
}
