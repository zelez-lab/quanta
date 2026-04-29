//! Compute dispatch operations for Metal.

use alloc::boxed::Box;
use alloc::format;

use crate::{Pulse, QuantaError, Wave};

use super::MetalDevice;
use super::ffi;

/// Create a Pulse backed by a dispatch_semaphore + addCompletedHandler.
/// The GPU signals the semaphore when the command buffer completes.
/// Pulse.wait() waits on the semaphore — no busy-polling, no thread parking.
pub(crate) fn make_async_pulse(device: &MetalDevice, cmd: ffi::Id) -> Pulse {
    unsafe {
        let sem = ffi::dispatch_semaphore_create(0);
        let block = ffi::make_completion_block(sem);
        ffi::msg_add_completed_handler(cmd, block);
        ffi::msg_void(cmd, b"commit\0");

        let handle = device.alloc_handle();
        Pulse {
            handle,
            completed: false,
            wait_fn: Some(Box::new(move || {
                ffi::dispatch_semaphore_wait(sem, ffi::DISPATCH_TIME_FOREVER);
                ffi::dispatch_release(sem);
                // Free the heap-allocated block
                drop(Box::from_raw(block));
            })),
        }
    }
}

impl MetalDevice {
    /// JIT-compile a kernel from serialized KernelDef IR.
    ///
    /// Deserializes the IR, emits MSL text, compiles via Metal runtime,
    /// and creates a compute pipeline.
    /// JIT-compile a kernel from serialized KernelDef IR.
    ///
    /// Deserializes the IR, emits MSL text, compiles via Metal runtime,
    /// and creates a compute pipeline.
    #[cfg(feature = "jit")]
    pub(crate) fn wave_jit_impl(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        use alloc::vec::Vec;

        let kernel = quanta_ir::deserialize_kernel(kernel_def_bytes)
            .map_err(|e| QuantaError::compilation_failed(format!("JIT deserialize: {}", e)))?;
        let msl = quanta_ir::emit_msl::emit(&kernel)
            .map_err(|e| QuantaError::compilation_failed(format!("JIT MSL emit: {}", e)))?;

        // Compile MSL text at runtime via Metal
        let mut src_bytes: Vec<u8> = msl.bytes().collect();
        src_bytes.push(0);
        let ns_src = ffi::nsstring(&src_bytes);
        let library = unsafe {
            let (lib, error) = ffi::msg_new_library_with_source(self.device, ns_src, ffi::NIL);
            if lib.is_null() {
                let msg = if !error.is_null() {
                    let desc = ffi::msg_id(error, b"localizedDescription\0");
                    let cstr = ffi::msg_utf8_string(desc);
                    std::ffi::CStr::from_ptr(cstr as *const _)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "unknown MSL compile error".into()
                };
                return Err(QuantaError::compilation_failed(format!(
                    "JIT Metal: {}",
                    msg
                )));
            }
            lib
        };

        // Get the kernel function and create pipeline
        let func_name = unsafe {
            let names = ffi::msg_function_names(library);
            let count = ffi::msg_array_count(names);
            if count == 0 {
                return Err(QuantaError::compilation_failed("JIT: no functions in MSL"));
            }
            ffi::msg_array_object_at(names, 0)
        };

        let func = unsafe { ffi::msg_get_function(library, func_name) };
        if func.is_null() {
            return Err(QuantaError::compilation_failed(
                "JIT: failed to get kernel function",
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
            return Err(QuantaError::compilation_failed(format!("JIT: {}", msg)));
        }

        let handle = self.alloc_handle();
        self.compute_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, pipeline);
        Ok(Wave {
            handle,
            bindings: [0u64; 16],
            binding_count: 0,
            texture_bindings: [0u64; 16],
            texture_count: 0,
            push_data: [0u8; 256],
            push_len: 0,
            push_mask: 0,
            // Honor the kernel's declared workgroup_size — previously
            // hardcoded to [64,1,1], which silently mismatched the
            // generated MSL's `[[max_total_threads_per_threadgroup(N)]]`
            // for kernels with smaller groups (e.g. the D-ext.3b.2 race
            // kernel uses [2,1,1]) and caused dispatchThreadgroups to
            // either clip silently or no-op.
            workgroup_size: kernel.workgroup_size,
            drop_fn: None,
        })
    }

    pub(crate) fn wave_impl(&self, kernel_source: &[u8]) -> Result<Wave, QuantaError> {
        // Require pre-compiled metallib binary (MTLB magic header).
        if kernel_source.len() < 4 || &kernel_source[..4] != b"MTLB" {
            return Err(QuantaError::compilation_failed(
                "Metal requires pre-compiled metallib binary",
            ));
        }

        let library = unsafe {
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

        let handle = self.alloc_handle();
        self.compute_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, pipeline);
        Ok(Wave {
            handle,
            bindings: [0u64; 16],
            binding_count: 0,
            texture_bindings: [0u64; 16],
            texture_count: 0,
            push_data: [0u8; 256],
            push_len: 0,
            push_mask: 0,
            workgroup_size: [64, 1, 1],
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
            .read()
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
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0
                && let Some(buf) = buffers.get(&handle)
            {
                unsafe {
                    ffi::msg_set_buffer(
                        encoder,
                        b"setBuffer:offset:atIndex:\0",
                        *buf,
                        0,
                        slot as u64,
                    );
                }
            }
        }

        // Push constants: send each occupied slot at its Metal buffer index.
        // Only send slots marked in the bitmask to avoid overwriting buffer bindings.
        {
            let mut mask = wave.push_mask;
            while mask != 0 {
                let slot = mask.trailing_zeros() as usize;
                let offset = slot * 16;
                let remaining = wave.push_len as usize - offset;
                let len = remaining.min(16);
                unsafe {
                    ffi::msg_set_bytes(
                        encoder,
                        b"setBytes:length:atIndex:\0",
                        wave.push_data[offset..].as_ptr() as *const _,
                        len as u64,
                        slot as u64,
                    );
                }
                mask &= mask - 1; // clear lowest set bit
            }
        }

        // Bind textures for compute access
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for slot in 0..wave.texture_count as usize {
            let handle = wave.texture_bindings[slot];
            if handle != 0
                && let Some(tex) = textures.get(&handle)
            {
                unsafe {
                    ffi::msg_set_texture(encoder, b"setTexture:atIndex:\0", *tex, slot as u64);
                }
            }
        }
        drop(textures);

        let grid = ffi::MTLSize::new(groups[0] as u64, groups[1] as u64, groups[2] as u64);
        let group_size = ffi::MTLSize::new(
            wave.workgroup_size[0] as u64,
            wave.workgroup_size[1] as u64,
            wave.workgroup_size[2] as u64,
        );
        unsafe {
            ffi::msg_dispatch_threadgroups(encoder, grid, group_size);
            ffi::msg_void(encoder, b"endEncoding\0");
        }
        Ok(make_async_pulse(self, cmd))
    }

    /// Dispatch by total thread count — Metal clips to exact grid size.
    pub(crate) fn wave_dispatch_threads_impl(
        &self,
        wave: &Wave,
        quarks: u32,
    ) -> Result<Pulse, QuantaError> {
        // Reuse the same binding/setup as wave_dispatch_impl, but use dispatchThreads
        let cmd = unsafe { ffi::msg_id(self.queue, b"commandBuffer\0") };
        let encoder = unsafe { ffi::msg_id(cmd, b"computeCommandEncoder\0") };

        let pipelines = self
            .compute_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipeline = pipelines
            .get(&wave.handle)
            .ok_or_else(|| QuantaError::invalid_param("bad wave handle"))?;
        unsafe {
            ffi::msg_void_id(encoder, b"setComputePipelineState:\0", *pipeline);
        }

        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0
                && let Some(buf) = buffers.get(&handle)
            {
                unsafe {
                    ffi::msg_set_buffer(
                        encoder,
                        b"setBuffer:offset:atIndex:\0",
                        *buf,
                        0,
                        slot as u64,
                    );
                }
            }
        }
        drop(buffers);

        // Push constants
        {
            let mut mask = wave.push_mask;
            while mask != 0 {
                let slot = mask.trailing_zeros() as usize;
                let offset = slot * 16;
                let remaining = wave.push_len as usize - offset;
                let len = remaining.min(16);
                unsafe {
                    ffi::msg_set_bytes(
                        encoder,
                        b"setBytes:length:atIndex:\0",
                        wave.push_data[offset..].as_ptr() as *const _,
                        len as u64,
                        slot as u64,
                    );
                }
                mask &= mask - 1;
            }
        }

        // Textures
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for slot in 0..wave.texture_count as usize {
            let handle = wave.texture_bindings[slot];
            if handle != 0
                && let Some(tex) = textures.get(&handle)
            {
                unsafe {
                    ffi::msg_set_texture(encoder, b"setTexture:atIndex:\0", *tex, slot as u64);
                }
            }
        }
        drop(textures);

        let grid = ffi::MTLSize::new(quarks as u64, 1, 1);
        let group_size = ffi::MTLSize::new(
            wave.workgroup_size[0] as u64,
            wave.workgroup_size[1] as u64,
            wave.workgroup_size[2] as u64,
        );
        unsafe {
            ffi::msg_dispatch_threads(encoder, grid, group_size);
            ffi::msg_void(encoder, b"endEncoding\0");
        }
        Ok(make_async_pulse(self, cmd))
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
            .read()
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
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0
                && let Some(buf) = buffers.get(&handle)
            {
                unsafe {
                    ffi::msg_set_buffer(
                        encoder,
                        b"setBuffer:offset:atIndex:\0",
                        *buf,
                        0,
                        slot as u64,
                    );
                }
            }
        }

        {
            let mut mask = wave.push_mask;
            while mask != 0 {
                let slot = mask.trailing_zeros() as usize;
                let offset = slot * 16;
                let remaining = wave.push_len as usize - offset;
                let len = remaining.min(16);
                unsafe {
                    ffi::msg_set_bytes(
                        encoder,
                        b"setBytes:length:atIndex:\0",
                        wave.push_data[offset..].as_ptr() as *const _,
                        len as u64,
                        slot as u64,
                    );
                }
                mask &= mask - 1;
            }
        }

        let indirect_buf = buffers.get(&buffer).ok_or_else(|| {
            QuantaError::invalid_param("bad indirect buffer")
                .with_context(&format!("wave_dispatch_indirect: buffer handle {buffer}"))
        })?;
        let group_size = ffi::MTLSize::new(
            wave.workgroup_size[0] as u64,
            wave.workgroup_size[1] as u64,
            wave.workgroup_size[2] as u64,
        );
        unsafe {
            ffi::msg_dispatch_threadgroups_indirect(encoder, *indirect_buf, offset, group_size);
            ffi::msg_void(encoder, b"endEncoding\0");
        }
        Ok(make_async_pulse(self, cmd))
    }
}
