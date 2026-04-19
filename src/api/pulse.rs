use crate::QuantaError;

type WaitFn = Box<dyn FnOnce(u64) -> Result<(), QuantaError>>;

/// GPU completion signal. Returned by dispatch/render operations.
///
/// Named after quantum pulse — a discrete packet of energy.
/// A Pulse represents one completed GPU operation.
pub struct Pulse {
    pub(crate) handle: u64,
    pub(crate) wait_fn: Option<WaitFn>,
    pub(crate) poll_fn: Option<Box<dyn Fn(u64) -> bool>>,
    pub(crate) completed: bool,
}

impl Pulse {
    /// Block until GPU completes this operation.
    /// After waiting, the pulse is marked as completed.
    pub fn wait(&mut self) -> Result<(), QuantaError> {
        if let Some(f) = self.wait_fn.take() {
            f(self.handle)?;
        }
        self.completed = true;
        Ok(())
    }

    /// Check if GPU has completed (non-blocking).
    pub fn is_done(&self) -> bool {
        if self.completed {
            return true;
        }
        if let Some(f) = &self.poll_fn {
            f(self.handle)
        } else {
            true
        }
    }

    /// Reset the pulse so it can be reused for another operation.
    /// The pulse must have been waited on (completed) before resetting.
    pub fn reset(&mut self) {
        self.completed = false;
        self.wait_fn = None;
        self.poll_fn = None;
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
