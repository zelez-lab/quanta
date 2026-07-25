//! Deferred dispatch — the per-device pending lane, and the ONLY
//! dispatch model: every [`Gpu::dispatch`](crate::Gpu::dispatch)
//! encodes into a shared [`Batch`] instead of committing its own
//! command buffer, and the batch submits when something needs the
//! results: a [`Pulse::wait`](crate::Pulse::wait) on any returned
//! pulse, an explicit [`Gpu::flush`](crate::Gpu::flush),
//! [`Gpu::wait_idle`](crate::Gpu::wait_idle), or a `Field` byte op
//! touching a buffer the lane still owes work to. The sync contract
//! is the async one the API always had — reads require a wait — with
//! deferral only moving *when* work submits, never what a sync point
//! means.
//!
//! There is exactly ONE lane per device, shared by every `Gpu` clone —
//! the same anchoring as the MSAA pool. Two independent lanes on one
//! queue would commit in arbitrary order, and the queue executes
//! commit-order, so a dispatch in one lane could read a peer lane's
//! not-yet-committed write. One lane = one submission order = the
//! recorded program order.
//!
//! Backends without a [`Batch`] implementation stay eager: dispatch
//! commits and waits inline, returning a completed pulse. Semantics
//! are identical, only the batching win is absent.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::collections::HashSet;
use std::sync::Mutex;

use crate::{Batch, GpuDevice, Pulse, QuantaError, QuantaErrorKind, Wave};

/// Auto-submit threshold: at this many encoded dispatches the lane
/// submits the open batch (without waiting) and starts a fresh one.
/// Bounds command-buffer growth in read-free stretches (a training
/// loop that only reads its loss every K steps) and lets the GPU start
/// executing while the host keeps encoding. Cross-batch ordering holds
/// on the backends that reach this path: same queue, commit order,
/// hazard-tracked resources.
const AUTO_SUBMIT_ENCODES: u32 = 512;

struct LaneState {
    /// The open batch, created on first deferred dispatch. `None`
    /// between flushes.
    batch: Option<Batch>,
    /// Dispatches encoded into the open batch so far.
    encoded: u32,
    /// Submitted-but-unwaited batch pulses (threshold submits).
    outstanding: Vec<Pulse>,
    /// Every field handle bound by a wave encoded since the last
    /// completed flush — i.e. the buffers whose contents the lane may
    /// still owe work to. `Field` ops that touch buffer bytes outside
    /// the lane (`read`, `write`, `copy_from`, `native_handle`)
    /// consult this and flush only when their handle is in it, so
    /// fresh-buffer uploads mid-graph (scalar constants, input
    /// batches) never break an open batch.
    referenced: HashSet<u64>,
    /// A flush error that had no caller to surface to (it happened
    /// inside a pulse's deferred wait, which cannot return one). The
    /// next encode or flush takes and returns it.
    poisoned: Option<QuantaError>,
    /// Whether the device implements batching. `None` until the first
    /// deferred dispatch probes `batch_begin`; `Some(false)` routes
    /// every later dispatch down the eager path without re-probing.
    batch_capable: Option<bool>,
}

/// One device's deferred-dispatch state. Lives in [`crate::Gpu`]
/// beside the device Arc; every clone shares it.
pub(crate) struct PendingLane {
    state: Mutex<LaneState>,
}

impl Default for PendingLane {
    fn default() -> Self {
        PendingLane {
            state: Mutex::new(LaneState {
                batch: None,
                encoded: 0,
                outstanding: Vec::new(),
                referenced: HashSet::new(),
                poisoned: None,
                batch_capable: None,
            }),
        }
    }
}

impl PendingLane {
    /// Encode one dispatch into the lane. `Ok(true)` = encoded (the
    /// caller hands out a lazy pulse); `Ok(false)` = this device has
    /// no batch path (the caller dispatches eagerly). Surfaces any
    /// stored poison first, so an error from a deferred flush lands on
    /// the next op rather than vanishing.
    pub(crate) fn encode(
        &self,
        device: &Arc<dyn GpuDevice>,
        wave: &Wave,
        quarks: u32,
    ) -> Result<bool, QuantaError> {
        // Texture-binding waves take the eager path: completion
        // tracking covers field handles only (`referenced`), so a
        // deferred texture write could be observed stale through
        // `Texture::read`. Until textures get the same treatment,
        // correctness wins over batching for them. The caller
        // pre-submits the lane, so ordering against encoded buffer
        // work still holds.
        if wave.texture_count > 0 {
            return Ok(false);
        }
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        if state.batch_capable == Some(false) {
            return Ok(false);
        }
        if state.batch.is_none() {
            match device.batch_begin() {
                Ok(b) => {
                    state.batch_capable = Some(true);
                    state.batch = Some(b);
                }
                Err(QuantaError {
                    kind: QuantaErrorKind::NotSupported(_),
                    ..
                }) => {
                    state.batch_capable = Some(false);
                    return Ok(false);
                }
                Err(e) => return Err(e),
            }
        }
        state
            .batch
            .as_mut()
            .expect("open batch present after begin")
            .dispatch(wave, quarks)?;
        state.encoded += 1;
        for slot in 0..wave.binding_count as usize {
            if wave.bindings[slot] != 0 {
                state.referenced.insert(wave.bindings[slot]);
            }
        }
        if state.encoded >= AUTO_SUBMIT_ENCODES {
            Self::submit_open_batch(&mut state)?;
        }
        Ok(true)
    }

    /// Whether the lane may still owe work to the given field handle
    /// (bound by an encoded wave, not yet completed by a full flush).
    pub(crate) fn references(&self, handle: u64) -> bool {
        self.state
            .lock()
            .expect("deferred lane mutex poisoned")
            .referenced
            .contains(&handle)
    }

    /// Submit the open batch WITHOUT waiting — the ordering barrier
    /// for a submission that bypasses the lane (an explicit-groups or
    /// indirect dispatch, or an eager handle's dispatch): committing
    /// the pending batch first keeps queue order equal to program
    /// order, and the driver's hazard tracking does the rest. Handles
    /// stay `referenced` until a full flush actually waits.
    pub(crate) fn submit_pending(&self) -> Result<(), QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        Self::submit_open_batch(&mut state)
    }

    /// Submit the open batch (no wait) and stash its pulse.
    fn submit_open_batch(state: &mut LaneState) -> Result<(), QuantaError> {
        if let Some(batch) = state.batch.take() {
            state.encoded = 0;
            let pulse = batch.pulse()?;
            state.outstanding.push(pulse);
        }
        Ok(())
    }

    /// Submit the open batch and block until every outstanding
    /// submission completes. The lock is held across the waits on
    /// purpose: concurrent encoders queue behind a flush instead of
    /// racing the batch it is draining.
    pub(crate) fn flush_and_wait(&self) -> Result<(), QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        Self::submit_open_batch(&mut state)?;
        for mut pulse in state.outstanding.drain(..) {
            pulse.wait()?;
        }
        // Everything encoded has now completed: the lane owes nothing.
        state.referenced.clear();
        Ok(())
    }

    /// Store an error from a context that cannot return one (a lazy
    /// pulse's deferred wait). The next [`encode`](Self::encode) or
    /// [`flush_and_wait`](Self::flush_and_wait) surfaces it.
    pub(crate) fn poison(&self, e: QuantaError) {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        state.poisoned = Some(e);
    }
}

/// The pulse a deferred dispatch returns: waiting it flushes the whole
/// lane (this dispatch and everything encoded before or after it up to
/// the wait — over-waiting is conservative and correct), preserving
/// the documented wait-before-read contract verbatim.
pub(crate) fn lazy_pulse(lane: Arc<PendingLane>, device: Arc<dyn GpuDevice>) -> Pulse {
    Pulse {
        handle: 0,
        completed: false,
        wait_fn: Some(Box::new(move || {
            if let Err(e) = lane.flush_and_wait() {
                lane.poison(e);
            }
        })),
        keep_alive: Some(device),
    }
}
