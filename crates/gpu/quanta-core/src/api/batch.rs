use alloc::boxed::Box;
use alloc::sync::Arc;

use crate::api::device::GpuDevice;
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
    // Declaration order is load-bearing: fields drop top-to-bottom, so
    // `inner` — whose Drop hands command buffers, descriptor pools and
    // pins back to the device through a raw pointer on every backend —
    // must be declared BEFORE `_device`, the keep-alive that guarantees
    // the device is still alive to receive them. This holds wherever
    // the batch ends up (parked in the deferred lane, held by a user
    // past the last `Gpu` clone): the batch owns its device.
    pub(crate) inner: Box<dyn BatchInner>,
    _device: Arc<dyn GpuDevice>,
}

impl Batch {
    /// The only way to build a `Batch`: drivers return the raw
    /// [`BatchInner`] and the api layer zips it with the device Arc it
    /// already holds — so a batch that outlives its device cannot be
    /// constructed by design.
    pub(crate) fn new(inner: Box<dyn BatchInner>, device: Arc<dyn GpuDevice>) -> Self {
        Batch {
            inner,
            _device: device,
        }
    }

    /// Encode a dispatch into the batch.
    pub fn dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError> {
        self.inner.encode_dispatch(wave, quarks)
    }

    /// Submit all encoded dispatches as a single GPU submission.
    /// Returns a Pulse that completes when ALL dispatches finish.
    pub fn pulse(self) -> Result<Pulse, QuantaError> {
        self.inner.submit()
    }

    /// Internal (deferred lane): a full ordering point between the
    /// dispatches encoded so far and those still to come.
    pub(crate) fn encode_barrier(&mut self) -> Result<(), QuantaError> {
        self.inner.encode_barrier()
    }
}

/// `Send` so a `Batch` can live in the shared deferred-dispatch lane
/// (`Mutex`-guarded, one per device). Implementations over raw native
/// objects (command buffers, encoders) assert `Send` themselves: the
/// native APIs demand *external synchronization*, not thread affinity,
/// and both the lane's `Mutex` and `&mut self` on `Batch::dispatch`
/// guarantee exclusive access.
///
/// `pub` only because [`GpuDevice`] (the public render-crate seam)
/// names it in `batch_begin`'s return type — same arrangement as
/// `Gpu::device_handle`. Not part of the stable surface.
#[doc(hidden)]
pub trait BatchInner: Send {
    fn encode_dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError>;
    /// Order every dispatch encoded after this call against every one
    /// encoded before it. A no-op on batches that are already fully
    /// ordered (the serial public batch, the synchronous CPU shim);
    /// on a CONCURRENT batch (the deferred lane's) this is the only
    /// ordering there is — the lane emits one at each hazard-run
    /// boundary.
    fn encode_barrier(&mut self) -> Result<(), QuantaError> {
        Ok(())
    }
    fn submit(self: Box<Self>) -> Result<Pulse, QuantaError>;
}
