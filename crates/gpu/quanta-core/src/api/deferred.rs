//! Deferred submission — the per-device pending lane, and the ONLY
//! submission model: every [`Gpu::dispatch`](crate::Gpu::dispatch),
//! every `RenderBuilder::pulse()` and every `resolve_texture` encodes
//! into a shared [`Batch`] instead of committing its own command
//! buffer, and the batch is submitted when something needs the
//! results: a [`Pulse::wait`](crate::Pulse::wait) on any returned
//! pulse, an explicit [`Gpu::flush`](crate::Gpu::flush) /
//! [`Gpu::submit`](crate::Gpu::submit),
//! [`Gpu::wait_idle`](crate::Gpu::wait_idle), a surface present, or a
//! `Field` / `Texture` byte op touching a resource the lane still owes
//! work to. The sync contract is the async one the API always had —
//! reads require a wait — with deferral only moving *when* work
//! submits, never what a sync point means.
//!
//! There is exactly ONE lane per device, shared by every `Gpu` clone —
//! the same anchoring as the MSAA pool. Two independent lanes on one
//! queue would commit in arbitrary order, and the queue executes
//! commit-order, so a dispatch in one lane could read a peer lane's
//! not-yet-committed write. One lane = one submission order = the
//! recorded program order.
//!
//! Every submitted batch carries a SERIAL, and every lazy pulse the
//! lane hands out remembers the serial of the batch its work went
//! into. Waiting a pulse completes the lane *through that serial* —
//! the batch (submitting it first if it is still open) and everything
//! submitted before it — and leaves later batches in flight, so a
//! frame loop that holds one pulse per frame and waits the one from N
//! frames back gets exactly depth-N pacing. Owed resources are tracked
//! per batch for the same reason: a `Texture::write` waits only the
//! submissions that drew with or sampled that texture, never the
//! frame being encoded.
//!
//! Backends without a [`Batch`] implementation stay eager: dispatch
//! commits and waits inline, returning a completed pulse. Semantics
//! are identical, only the batching win is absent. Backends whose
//! batches take no render work (`GpuDevice::supports_render_batching`
//! false — WebGPU today) submit each pass and resolve on its own.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::collections::HashSet;
use std::sync::Mutex;

#[cfg(feature = "compute")]
use crate::Wave;
use crate::{Batch, GpuDevice, Pulse, QuantaError, QuantaErrorKind};

/// Auto-submit threshold: at this many encodes the lane submits the
/// open batch (without waiting) and starts a fresh one. Bounds
/// command-buffer growth in read-free stretches (a training loop that
/// only reads its loss every K steps) and lets the GPU start executing
/// while the host keeps encoding. Cross-batch ordering holds on the
/// backends that reach this path: same queue, commit order,
/// hazard-tracked resources.
const AUTO_SUBMIT_ENCODES: u32 = 512;

/// A submitted batch the lane has not yet waited: its serial, the
/// driver pulse that completes it, and the resource handles its
/// recorded work references (the buffers bound by its waves and
/// passes, the textures its passes drew into or sampled, its resolve
/// sources and destinations).
struct Submitted {
    serial: u64,
    pulse: Pulse,
    referenced: HashSet<u64>,
}

struct LaneState {
    /// The open batch, created on first deferred encode. `None`
    /// between submissions.
    batch: Option<Batch>,
    /// Serial the open batch carries (the next submission takes it).
    /// Strictly increasing; a lazy pulse captures it at encode time.
    open_serial: u64,
    /// Encodes recorded into the open batch so far.
    encoded: u32,
    /// Handles the OPEN batch's recorded work references (see
    /// `Submitted::referenced`). Moves into the `Submitted` entry at
    /// submission.
    open_referenced: HashSet<u64>,
    /// Submitted-but-unwaited batches, in serial order.
    outstanding: Vec<Submitted>,
    /// Handles READ by the current hazard-free run (pure reads: bound
    /// slots whose `write_mask` bit is clear).
    run_reads: HashSet<u64>,
    /// Handles WRITTEN by the current run (`&mut` slots — read-write).
    run_writes: HashSet<u64>,
    /// A flush error that had no caller to surface to (it happened
    /// inside a pulse's deferred wait, which cannot return one). The
    /// next encode or flush takes and returns it.
    poisoned: Option<QuantaError>,
    /// Whether the device implements batching. `None` until the first
    /// deferred encode probes `batch_begin`; `Some(false)` routes
    /// every later encode down the eager path without re-probing.
    batch_capable: Option<bool>,
    /// Whether the device's batches take render passes and resolves
    /// (`GpuDevice::supports_render_batching`), probed once like
    /// `batch_capable`. `Some(false)` sends every pass down the
    /// per-submission path.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    render_capable: Option<bool>,
}

/// One device's deferred-submission state. Lives in [`crate::Gpu`]
/// beside the device Arc; every clone shares it.
pub(crate) struct PendingLane {
    state: Mutex<LaneState>,
}

impl Default for PendingLane {
    fn default() -> Self {
        PendingLane {
            state: Mutex::new(LaneState {
                batch: None,
                open_serial: 1,
                encoded: 0,
                open_referenced: HashSet::new(),
                outstanding: Vec::new(),
                run_reads: HashSet::new(),
                run_writes: HashSet::new(),
                poisoned: None,
                batch_capable: None,
                render_capable: None,
            }),
        }
    }
}

/// What the lane did with a render pass handed to
/// [`PendingLane::encode_render`].
#[cfg(feature = "render")]
pub(crate) enum RenderEncode {
    /// Encoded into the batch with this serial — hand out a lazy pulse.
    Deferred(u64),
    /// This device batches no render work; the pass comes back for the
    /// caller's per-submission path.
    Declined(crate::RenderPass),
}

impl PendingLane {
    /// Encode one dispatch into the lane. `Ok(Some(serial))` = encoded
    /// into the batch with that serial (the caller hands out a lazy
    /// pulse); `Ok(None)` = this device has no batch path (the caller
    /// dispatches eagerly). Surfaces any stored poison first, so an
    /// error from a deferred flush lands on the next op rather than
    /// vanishing.
    #[cfg(feature = "compute")]
    pub(crate) fn encode(
        &self,
        device: &Arc<dyn GpuDevice>,
        wave: &Wave,
        quarks: u32,
    ) -> Result<Option<u64>, QuantaError> {
        // Texture-binding waves take the eager path: the lane's
        // hazard-run analysis covers field handles only, so two
        // texture-touching dispatches in one batch would have no
        // ordering between them (render passes carry their own
        // barriers; dispatches rely on the run sets). The caller
        // pre-submits the lane, so ordering against encoded buffer
        // work still holds.
        if wave.texture_count > 0 {
            return Ok(None);
        }
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        if !Self::ensure_batch(&mut state, device)? {
            return Ok(None);
        }
        // Hazard-run grouping: this dispatch joins the current run
        // unless it conflicts with it — W∩(R'∪W') (its writes touch
        // anything the run used) or R∩W' (it reads something the run
        // wrote). Read-read sharing never orders (the common case:
        // many ops reading one input). On conflict, a batch barrier
        // closes the run BEFORE this encode. `&mut` slots count as
        // read-write, so they sit in the write set only and the
        // W-terms cover their reads.
        let mut reads: [u64; 16] = [0; 16];
        let mut writes: [u64; 16] = [0; 16];
        let (mut nr, mut nw) = (0usize, 0usize);
        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle == 0 {
                continue;
            }
            if wave.write_mask & (1 << slot) != 0 {
                writes[nw] = handle;
                nw += 1;
            } else {
                reads[nr] = handle;
                nr += 1;
            }
        }
        let hazard = writes[..nw]
            .iter()
            .any(|h| state.run_reads.contains(h) || state.run_writes.contains(h))
            || reads[..nr].iter().any(|h| state.run_writes.contains(h));
        if hazard {
            state
                .batch
                .as_mut()
                .expect("open batch present after begin")
                .encode_barrier()?;
            state.run_reads.clear();
            state.run_writes.clear();
        }
        state
            .batch
            .as_mut()
            .expect("open batch present after begin")
            .dispatch(wave, quarks)?;
        let serial = state.open_serial;
        state.encoded += 1;
        for &h in &reads[..nr] {
            state.run_reads.insert(h);
            state.open_referenced.insert(h);
        }
        for &h in &writes[..nw] {
            state.run_writes.insert(h);
            state.open_referenced.insert(h);
        }
        if state.encoded >= AUTO_SUBMIT_ENCODES {
            Self::submit_open_batch(&mut state)?;
        }
        Ok(Some(serial))
    }

    /// Open the lane's batch if none is open. `Ok(false)` = this
    /// device has no batch path (cached — never re-probed).
    fn ensure_batch(
        state: &mut LaneState,
        device: &Arc<dyn GpuDevice>,
    ) -> Result<bool, QuantaError> {
        if state.batch_capable == Some(false) {
            return Ok(false);
        }
        if state.batch.is_none() {
            match device.batch_begin_concurrent() {
                Ok(b) => {
                    state.batch_capable = Some(true);
                    // The wrapper takes the device Arc: a parked batch
                    // OWNS its device, so lane teardown can never hand
                    // resources back to a destroyed device — whatever
                    // order `Gpu`'s fields drop in.
                    state.batch = Some(Batch::new(b, device.clone()));
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
        Ok(true)
    }

    /// Whether this device's batches take render work — probed once
    /// through `supports_render_batching` and cached.
    #[cfg(feature = "render")]
    fn render_capable(state: &mut LaneState, device: &Arc<dyn GpuDevice>) -> bool {
        *state
            .render_capable
            .get_or_insert_with(|| device.supports_render_batching())
    }

    /// Encode a whole render pass into the lane, after everything
    /// encoded so far. A driver error leaves the batch exactly as it
    /// was when the driver validated before recording (dead handle,
    /// pass shape: only THIS pass fails); a failure mid-record marks
    /// the driver batch broken, and the next sync point surfaces it
    /// and discards the batch.
    #[cfg(feature = "render")]
    pub(crate) fn encode_render(
        &self,
        device: &Arc<dyn GpuDevice>,
        pass: crate::RenderPass,
    ) -> Result<RenderEncode, QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        if !Self::render_capable(&mut state, device) || !Self::ensure_batch(&mut state, device)? {
            return Ok(RenderEncode::Declined(pass));
        }
        // A render pass is a full ordering point (the driver batch
        // fences it against everything before and after), so the
        // hazard run the compute encodes were building ends here.
        state.run_reads.clear();
        state.run_writes.clear();
        let mut refs: Vec<u64> = Vec::new();
        pass.for_each_handle(|kind, h| {
            use crate::render_pass::HandleKind;
            if matches!(kind, HandleKind::Buffer | HandleKind::Texture) {
                refs.push(h);
            }
        });
        state
            .batch
            .as_mut()
            .expect("open batch present after begin")
            .encode_render(pass)?;
        let serial = state.open_serial;
        state.encoded += 1;
        state.open_referenced.extend(refs);
        if state.encoded >= AUTO_SUBMIT_ENCODES {
            Self::submit_open_batch(&mut state)?;
        }
        Ok(RenderEncode::Deferred(serial))
    }

    /// Encode an MSAA resolve into the lane, after everything encoded
    /// so far. `Ok(true)` = encoded; `Ok(false)` = declined (the
    /// caller resolves through the per-submission driver path).
    #[cfg(feature = "render")]
    pub(crate) fn encode_resolve(
        &self,
        device: &Arc<dyn GpuDevice>,
        src: u64,
        dst: u64,
    ) -> Result<bool, QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        if !Self::render_capable(&mut state, device) || !Self::ensure_batch(&mut state, device)? {
            return Ok(false);
        }
        state.run_reads.clear();
        state.run_writes.clear();
        state
            .batch
            .as_mut()
            .expect("open batch present after begin")
            .encode_resolve(src, dst)?;
        state.encoded += 1;
        state.open_referenced.insert(src);
        state.open_referenced.insert(dst);
        if state.encoded >= AUTO_SUBMIT_ENCODES {
            Self::submit_open_batch(&mut state)?;
        }
        Ok(true)
    }

    /// Test-support: how many encodes the OPEN batch holds (0 between
    /// submissions). Lets a test prove work stayed pending until a
    /// sync point without a driver-side probe.
    pub(crate) fn pending_encodes(&self) -> u32 {
        self.state
            .lock()
            .expect("deferred lane mutex poisoned")
            .encoded
    }

    /// Test-support: how many submitted batches nobody has waited yet.
    /// Lets a test prove that waiting one frame's pulse leaves later
    /// frames in flight.
    pub(crate) fn outstanding_batches(&self) -> usize {
        self.state
            .lock()
            .expect("deferred lane mutex poisoned")
            .outstanding
            .len()
    }

    /// Submit the open batch WITHOUT waiting — the ordering barrier
    /// for a submission that bypasses the lane (an explicit-groups or
    /// indirect dispatch, an eager handle's dispatch, a present):
    /// committing the pending batch first keeps queue order equal to
    /// program order, and the driver's hazard tracking does the rest.
    /// Owed handles stay owed until a wait actually completes them.
    pub(crate) fn submit_pending(&self) -> Result<(), QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        Self::submit_open_batch(&mut state)
    }

    /// Submit the open batch (no wait) only if its recorded work
    /// references `handle` — for a submission of the caller's own that
    /// must land behind it in queue order (`generate_mipmaps` after
    /// the pass that drew level 0). A no-op otherwise, so an unrelated
    /// resource never breaks an open batch.
    pub(crate) fn submit_if_referenced(&self, handle: u64) -> Result<(), QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        if state.open_referenced.contains(&handle) {
            Self::submit_open_batch(&mut state)?;
        }
        Ok(())
    }

    /// Submit the open batch (no wait) and stash its pulse under its
    /// serial. The next batch starts a fresh hazard run: cross-batch
    /// ordering is the backends' (Metal hazard tracking; the Vulkan
    /// batch's leading submission-order barrier).
    fn submit_open_batch(state: &mut LaneState) -> Result<(), QuantaError> {
        if let Some(batch) = state.batch.take() {
            state.encoded = 0;
            state.run_reads.clear();
            state.run_writes.clear();
            let serial = state.open_serial;
            state.open_serial += 1;
            let referenced = core::mem::take(&mut state.open_referenced);
            let pulse = batch.pulse()?;
            state.outstanding.push(Submitted {
                serial,
                pulse,
                referenced,
            });
        }
        Ok(())
    }

    /// Wait every outstanding batch with a serial `<= through`, in
    /// order, and forget them. Completion on one queue is in
    /// submission order, so the batches are waited oldest-first and
    /// each wait also runs that batch's deferred cleanup (descriptor
    /// pools back to the cache, per-pass objects destroyed).
    fn complete_outstanding_through(
        state: &mut LaneState,
        through: u64,
    ) -> Result<(), QuantaError> {
        let n = state
            .outstanding
            .iter()
            .take_while(|s| s.serial <= through)
            .count();
        for mut done in state.outstanding.drain(..n) {
            done.pulse.wait()?;
        }
        Ok(())
    }

    /// Complete the lane THROUGH the batch with `serial`: submit it if
    /// it is still open, then wait it and every batch submitted before
    /// it. Batches submitted after it stay in flight — this is what a
    /// lazy pulse's `wait` does, so waiting frame N's pulse never
    /// drains frame N+1. The lock is held across the waits on purpose:
    /// concurrent encoders queue behind a completion instead of racing
    /// the batch it is draining.
    pub(crate) fn complete_through(&self, serial: u64) -> Result<(), QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        if state.batch.is_some() && state.open_serial == serial {
            Self::submit_open_batch(&mut state)?;
        }
        Self::complete_outstanding_through(&mut state, serial)
    }

    /// Complete every batch whose recorded work references `handle` —
    /// the byte-op sync point (`Field::read`, `Texture::write`, …):
    /// submit the open batch if IT references the handle, then wait
    /// through the newest outstanding batch that does. A no-op (one
    /// lock + set probes) when the lane owes the handle nothing, so a
    /// fresh upload target never breaks an open batch, and later
    /// batches that never touched the resource stay in flight.
    pub(crate) fn complete_referencing(&self, handle: u64) -> Result<(), QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        if state.open_referenced.contains(&handle) {
            Self::submit_open_batch(&mut state)?;
        }
        let through = state
            .outstanding
            .iter()
            .filter(|s| s.referenced.contains(&handle))
            .map(|s| s.serial)
            .max();
        match through {
            Some(serial) => Self::complete_outstanding_through(&mut state, serial),
            None => Ok(()),
        }
    }

    /// Submit the open batch and block until every outstanding
    /// submission completes — `Gpu::flush`, `Gpu::wait_idle`, and the
    /// result reads that have no handle to narrow by.
    pub(crate) fn flush_and_wait(&self) -> Result<(), QuantaError> {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        if let Some(e) = state.poisoned.take() {
            return Err(e);
        }
        Self::submit_open_batch(&mut state)?;
        Self::complete_outstanding_through(&mut state, u64::MAX)
    }

    /// Store an error from a context that cannot return one (a lazy
    /// pulse's deferred wait). The next encode, completion or flush
    /// surfaces it.
    pub(crate) fn poison(&self, e: QuantaError) {
        let mut state = self.state.lock().expect("deferred lane mutex poisoned");
        state.poisoned = Some(e);
    }
}

/// The pulse a deferred encode returns: waiting it completes the lane
/// through the batch the work went into — that batch (submitted first
/// if still open) and everything submitted before it, never the
/// batches after — preserving the documented wait-before-read contract
/// verbatim while keeping later frames in flight.
pub(crate) fn lazy_pulse(lane: Arc<PendingLane>, device: Arc<dyn GpuDevice>, serial: u64) -> Pulse {
    Pulse {
        handle: 0,
        completed: false,
        wait_fn: Some(Box::new(move || {
            if let Err(e) = lane.complete_through(serial) {
                lane.poison(e);
            }
        })),
        keep_alive: Some(device),
    }
}
