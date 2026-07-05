//! Multi-queue submission (steps 018 + 019).
//!
//! A `Queue` represents one GPU work queue with a specific
//! capability tier — graphics, compute, or transfer. Backends:
//!
//! - Vulkan: `VkQueue` from a queue family supporting the requested
//!   bitmask.
//! - Metal: `MTLCommandQueue` per family.
//! - WebGPU: single global queue (W3C exposes one per device).
//! - CPU: software FIFO model.
//!
//! The wrapper enforces the lifecycle proven in
//! `Quanta.MultiQueue.Queue` (Lean) and
//! `quanta-api/multi_queue_safety.rs` (Verus):
//!
//! - `submit(wave, groups)` and `signal(sem, value)` fail when the
//!   queue is destroyed.
//! - `Drop` releases the backend handle.

use alloc::sync::Arc;

use crate::{GpuDevice, QuantaError, QueueType, Wave};

/// A typed queue handle. Drop releases the backend handle.
///
/// Refines `Quanta.MultiQueue.Queue`.
pub struct Queue {
    pub(crate) handle: u64,
    pub(crate) kind: QueueType,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl Queue {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Queue capability tier.
    pub fn kind(&self) -> QueueType {
        self.kind
    }

    /// Submit a compute dispatch on this queue.
    ///
    /// Refines `Quanta.MultiQueue.Queue.submit` and the Verus
    /// theorem `t7751_submit_appends`.
    pub fn submit(&self, wave: &Wave, groups: [u32; 3]) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("queue is not live"));
        }
        self.device.queue_dispatch(self.handle, wave, groups)
    }

    /// Signal `(sem, value)` from this queue.
    pub fn signal(&self, semaphore: u64) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("queue is not live"));
        }
        self.device.queue_signal(self.handle, semaphore)
    }

    /// Wait on `semaphore` before executing more work on this queue.
    pub fn wait(&self, semaphore: u64) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("queue is not live"));
        }
        self.device.queue_wait(self.handle, semaphore)
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.queue_destroy(self.handle);
            self.live = false;
        }
    }
}
