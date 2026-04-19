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
}

impl Pulse {
    /// Block until GPU completes this operation.
    pub fn wait(mut self) -> Result<(), QuantaError> {
        if let Some(f) = self.wait_fn.take() {
            f(self.handle)
        } else {
            Ok(())
        }
    }

    /// Check if GPU has completed (non-blocking).
    pub fn is_done(&self) -> bool {
        if let Some(f) = &self.poll_fn {
            f(self.handle)
        } else {
            true
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }
}
