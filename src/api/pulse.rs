use crate::QuantaError;
use alloc::boxed::Box;

/// GPU completion signal. Returned by dispatch/render operations.
///
/// Named after quantum pulse — a discrete packet of energy.
/// A Pulse represents one completed GPU operation.
///
/// Both Metal and Vulkan drivers submit-and-wait synchronously, so the
/// pulse is already completed when returned. No boxed closure needed.
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

    /// Check if GPU has completed (non-blocking).
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
pub struct OcclusionQuery {
    pub(crate) handle: u64,
    pub(crate) count: u32,
}

impl OcclusionQuery {
    /// The underlying query set handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Number of occlusion query slots.
    pub fn count(&self) -> u32 {
        self.count
    }
}
