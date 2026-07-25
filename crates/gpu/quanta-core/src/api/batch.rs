use alloc::boxed::Box;

use crate::{Pulse, QuantaError, Wave};

/// A batch of GPU dispatches recorded into a single command buffer.
///
/// Multiple kernels are encoded without per-dispatch commit overhead.
/// Call `pulse()` to commit all dispatches at once with a single fence/semaphore.
///
/// ```ignore
/// let mut batch = gpu.batch()?;
/// batch.dispatch(&wave1, n);
/// batch.dispatch(&wave2, n);
/// let mut pulse = batch.pulse()?;
/// gpu.wait(&mut pulse)?;
/// ```
pub struct Batch {
    pub(crate) inner: Box<dyn BatchInner>,
}

impl Batch {
    /// Encode a dispatch into the batch.
    pub fn dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError> {
        self.inner.encode_dispatch(wave, quarks)
    }

    /// Submit all encoded dispatches as a single GPU submission.
    /// Returns a Pulse that completes when ALL dispatches finish.
    pub fn pulse(self) -> Result<Pulse, QuantaError> {
        self.inner.submit()
    }
}

/// `Send` so a `Batch` can live in the shared deferred-dispatch lane
/// (`Mutex`-guarded, one per device). Implementations over raw native
/// objects (command buffers, encoders) assert `Send` themselves: the
/// native APIs demand *external synchronization*, not thread affinity,
/// and both the lane's `Mutex` and `&mut self` on `Batch::dispatch`
/// guarantee exclusive access.
pub(crate) trait BatchInner: Send {
    fn encode_dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError>;
    fn submit(self: Box<Self>) -> Result<Pulse, QuantaError>;
}
