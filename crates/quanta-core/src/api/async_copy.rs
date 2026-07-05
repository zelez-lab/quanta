//! Async memory copy queue (step 044).
//!
//! An async-copy queue runs buffer-to-buffer transfers concurrently
//! with compute / graphics queues. Backends:
//!
//! - Vulkan: queue family with `VK_QUEUE_TRANSFER_BIT` +
//!   `vkCmdCopyBuffer`.
//! - Metal: `MTLCommandQueue` + `MTLBlitCommandEncoder`.
//! - WebGPU: `GPUQueue.copyBufferToBuffer` (single global queue).
//! - CPU: serial memcpy via existing `field_copy_bytes`.
//!
//! The wrapper enforces the lifecycle proven in
//! `Quanta.AsyncCopy.Queue` (Lean) and
//! `quanta-api/async_copy_safety.rs` (Verus):
//!
//! - `copy_buffer(dst, src, size)` fails when the queue is destroyed.
//! - `Drop` releases the backend handle.

use alloc::sync::Arc;

use crate::{Field, GpuDevice, QuantaError};

/// A typed async-copy queue. Drop releases the backend handle.
///
/// Refines `Quanta.AsyncCopy.Queue`.
pub struct AsyncCopyQueue {
    pub(crate) handle: u64,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl AsyncCopyQueue {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Submit a buffer-to-buffer copy on this async queue. Returns
    /// when the copy has been recorded; cross-queue ordering must be
    /// established via `signal` / `wait` on the matching `Queue`
    /// (steps 018 + 019) if visibility to other queues is needed.
    ///
    /// Refines `Quanta.AsyncCopy.Queue.submitCopy` and the Verus
    /// theorem `t7851_submit_appends`.
    pub fn copy_buffer<T: Copy>(
        &self,
        dst: &Field<T>,
        src: &Field<T>,
        size: usize,
    ) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("async copy queue is not live"));
        }
        self.device
            .async_copy_submit(self.handle, dst.handle(), src.handle(), size)
    }

    /// Submit a raw-handle buffer copy (for users not holding typed
    /// `Field` wrappers).
    pub fn copy_buffer_raw(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("async copy queue is not live"));
        }
        self.device.async_copy_submit(self.handle, dst, src, size)
    }
}

impl Drop for AsyncCopyQueue {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.async_copy_destroy(self.handle);
            self.live = false;
        }
    }
}
