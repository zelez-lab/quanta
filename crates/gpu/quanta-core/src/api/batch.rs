//! [`Batch`] — many commands encoded into one command buffer.
//!
//! A batch amortizes the per-submission commit: the work is recorded
//! as it arrives and the whole run reaches the queue once, behind a
//! single pulse. It owns a keep-alive on its device, so a batch parked
//! in the deferred lane can always hand its command buffers back.
//!
//! Two faces share the type: compute dispatches (the public
//! [`Batch::dispatch`], `compute` feature) and — through the deferred
//! lane only — render passes and MSAA resolves (`render` feature).
//! The lane is the one place both meet; a user-built batch records
//! dispatches only.

use alloc::boxed::Box;
use alloc::sync::Arc;

#[cfg(feature = "compute")]
use crate::Wave;
use crate::api::device::GpuDevice;
use crate::{Pulse, QuantaError};

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
    #[cfg(feature = "compute")]
    pub fn dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError> {
        self.inner.encode_dispatch(wave, quarks)
    }

    /// Submit all encoded work as a single GPU submission. Returns a
    /// Pulse that completes when ALL of it finishes.
    pub fn pulse(self) -> Result<Pulse, QuantaError> {
        self.inner.submit()
    }

    /// Internal (deferred lane): a full ordering point between the
    /// dispatches encoded so far and those still to come.
    #[cfg(feature = "compute")]
    pub(crate) fn encode_barrier(&mut self) -> Result<(), QuantaError> {
        self.inner.encode_barrier()
    }

    /// Internal (deferred lane): record a whole render pass into the
    /// batch's command buffer, after everything encoded so far.
    #[cfg(feature = "render")]
    pub(crate) fn encode_render(&mut self, pass: crate::RenderPass) -> Result<(), QuantaError> {
        self.inner.encode_render(pass)
    }

    /// Internal (deferred lane): record an MSAA resolve into the
    /// batch's command buffer, after everything encoded so far.
    #[cfg(feature = "render")]
    pub(crate) fn encode_resolve(&mut self, src: u64, dst: u64) -> Result<(), QuantaError> {
        self.inner.encode_resolve(src, dst)
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
    /// Record one compute dispatch.
    #[cfg(feature = "compute")]
    fn encode_dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError>;
    /// Order every dispatch encoded after this call against every one
    /// encoded before it. A no-op on batches that are already fully
    /// ordered (the serial public batch, the synchronous CPU shim);
    /// on a CONCURRENT batch (the deferred lane's) this is the only
    /// ordering there is — the lane emits one at each hazard-run
    /// boundary.
    #[cfg(feature = "compute")]
    fn encode_barrier(&mut self) -> Result<(), QuantaError> {
        Ok(())
    }
    /// Record a whole render pass after everything encoded so far,
    /// with the same visibility the pass-per-submission path gives:
    /// a later pass in the same batch that samples this pass's target
    /// sees the finished contents, and a dispatch on either side is
    /// ordered against it. Only batches whose device reports
    /// [`GpuDevice::supports_render_batching`] implement it — the lane
    /// never calls it elsewhere.
    #[cfg(feature = "render")]
    fn encode_render(&mut self, _pass: crate::RenderPass) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "render passes cannot be batched on this backend",
        ))
    }
    /// Record an MSAA resolve (`src` multisampled → `dst` single
    /// sample) after everything encoded so far. Same gate as
    /// `encode_render`.
    #[cfg(feature = "render")]
    fn encode_resolve(&mut self, _src: u64, _dst: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "resolves cannot be batched on this backend",
        ))
    }
    /// Commit everything recorded; the pulse completes when it has all
    /// executed.
    fn submit(self: Box<Self>) -> Result<Pulse, QuantaError>;
}
