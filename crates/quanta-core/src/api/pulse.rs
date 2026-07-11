use crate::{GpuDevice, QuantaError};
use alloc::boxed::Box;
use alloc::sync::Arc;

/// GPU completion signal. Returned by dispatch/render operations.
///
/// Named after quantum pulse — a discrete packet of energy.
/// A Pulse represents one submitted GPU operation.
///
/// Submission is asynchronous on the GPU backends: the command buffer is
/// committed without blocking, so the operation is generally still in
/// flight when the pulse is returned. Call [`Pulse::wait`] before any
/// CPU-side read of a target the operation writes (`Texture::read`,
/// `Field::read`), or drain the whole queue with `Gpu::wait_idle`.
/// Presenting an acquired surface frame needs no wait — same-queue
/// ordering covers it.
pub struct Pulse {
    pub(crate) handle: u64,
    pub(crate) completed: bool,
    /// Deferred GPU wait: called once by wait() to block until completion.
    pub(crate) wait_fn: Option<Box<dyn FnOnce()>>,
}

impl Pulse {
    /// Block until GPU completes this operation.
    pub fn wait(&mut self) -> Result<(), QuantaError> {
        if let Some(f) = self.wait_fn.take() {
            f();
        }
        self.completed = true;
        Ok(())
    }

    /// Check if `wait()` has already observed completion (non-blocking).
    /// This reflects the pulse's local state, not live GPU progress: an
    /// in-flight pulse reports `false` until `wait()` runs.
    pub fn is_done(&self) -> bool {
        self.completed
    }

    /// Reset the pulse so it can be reused for another operation.
    pub fn reset(&mut self) {
        self.completed = false;
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }
}

/// A monotonically increasing synchronization primitive.
///
/// Unlike a Pulse (binary: pending/done), a Timeline tracks a u64 counter.
/// GPU or CPU can signal a value; waiters block until the counter reaches
/// a target value. Enables multi-frame pipelining without per-frame fence objects.
pub struct Timeline {
    #[allow(dead_code)]
    pub(crate) handle: u64,
}

/// A set of GPU timestamp query slots.
///
/// Write timestamps at specific pipeline points, then read them
/// to measure GPU execution time.
pub struct TimestampQuery {
    pub(crate) handle: u64,
    pub(crate) count: u32,
}

impl TimestampQuery {
    /// The underlying query set handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Number of timestamp slots in this query set.
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// A set of occlusion query slots (M3.3).
///
/// Records how many fragments pass the depth/stencil test within a
/// begin/end bracket during a render pass. Used for visibility culling:
/// if zero fragments passed, the object is fully occluded.
///
/// Dropping the query set calls `GpuDevice::occlusion_query_destroy`
/// exactly once.
pub struct OcclusionQuery {
    pub(crate) handle: u64,
    pub(crate) count: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl OcclusionQuery {
    /// Construct a live query set over a driver handle. Internal hook
    /// for the `quanta-render` sibling crate
    /// (`RenderGpu::occlusion_query_create`); not part of the stable
    /// public surface.
    #[doc(hidden)]
    pub fn __new(handle: u64, count: u32, device: Arc<dyn GpuDevice>) -> Self {
        Self {
            handle,
            count,
            device,
            live: true,
        }
    }

    /// The underlying query set handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Number of occlusion query slots.
    pub fn count(&self) -> u32 {
        self.count
    }
}

impl Drop for OcclusionQuery {
    fn drop(&mut self) {
        if self.live {
            self.live = false;
            let _ = self.device.occlusion_query_destroy(self.handle);
        }
    }
}
