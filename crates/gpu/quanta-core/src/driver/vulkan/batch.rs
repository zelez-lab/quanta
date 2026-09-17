//! The Vulkan [`crate::Batch`]: one command buffer, both faces.
//!
//! Compute dispatches are recorded with a global COMPUTE→COMPUTE
//! memory barrier at hazard-run boundaries (Vulkan gives no implicit
//! ordering between dispatches in a command buffer); render passes and
//! MSAA resolves are recorded through the shared `record_render_pass`
//! / `record_resolve` paths, bracketed by an ALL_COMMANDS memory
//! barrier on each side — the compute→render, render→render and
//! render→compute visibility the lane promises, without teaching the
//! compute run tracker about images. The whole buffer is one queue
//! submission with one fence.
//!
//! Lifetime bookkeeping an eager submission never needed:
//! - every handle the recorded commands reference — bound buffers, a
//!   wave's pipeline, a pass's targets / sampled textures / render
//!   pipeline / query pools / bundles — is PINNED
//!   (`VulkanDevice::batch_pins`) before the encode's registry
//!   lookups, so a destroy racing the open batch parks instead of
//!   freeing what the recorded commands reference;
//! - descriptor pools return to the cache, and per-pass objects
//!   (framebuffer, transient render pass, per-draw descriptor pool)
//!   are destroyed, only after the submission's fence completes;
//! - an abandoned batch (dropped un-submitted) resets and reclaims its
//!   command buffer, returns its pools, destroys its per-pass objects
//!   and unpins — parked destroys then retire behind the newest
//!   *submitted* serial, which is correct because nothing submitted
//!   references them.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;

#[cfg(feature = "compute")]
use crate::Wave;
use crate::{Pulse, QuantaError};

use super::VulkanDevice;
#[cfg(feature = "compute")]
use super::compute::{DispatchRecord, PreparedDispatch};
use super::ffi;
#[cfg(feature = "render")]
use super::render::RenderPassObjects;

pub(super) struct VulkanBatch {
    device: *const VulkanDevice,
    /// The exclusively owned command buffer — `None` once submitted
    /// (`submit_and_wait` consumes the lease into its fence waiter).
    /// An abandoned batch drops the lease back to the device's cache.
    lease: Option<super::device::CmdLease>,
    /// Copy of `lease.cmd` for the recording paths; dangling only
    /// after submit, which consumes the batch.
    cmd: ffi::VkCommandBuffer,
    /// Per-dispatch descriptor pools, returned to the cache after the
    /// fence.
    pools: Vec<ffi::VkDescriptorPool>,
    /// Per-pass objects, destroyed after the fence.
    #[cfg(feature = "render")]
    render_objects: Vec<RenderPassObjects>,
    /// One entry per `pin_for_batch` call (duplicates meaningful:
    /// unpin decrements per occurrence).
    pinned: Vec<u64>,
    any_encoded: bool,
    /// Serial mode (the public batch): a global barrier goes between
    /// EVERY pair of encodes. Concurrent mode (the deferred lane):
    /// barriers come only from `encode_barrier` at hazard-run
    /// boundaries.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    auto_barrier: bool,
    /// A recording failure mid-pass (`RecordFailure::Partial`): the
    /// command buffer holds a truncated pass, so later encodes refuse
    /// and `submit` returns this instead of submitting.
    broken: Option<QuantaError>,
}

// Safety: a `Batch` may be created on one thread and encoded/submitted
// on another (the deferred lane keeps one behind a `Mutex`). Vulkan
// command buffers require external synchronization, not thread
// affinity, and every access is exclusive (`&mut self` / by-value under
// that lock). The raw device pointer is valid for this batch's whole
// life, Drop included: the api `Batch` wrapper — the only way this type
// leaves the driver — owns a device `Arc` declared to drop AFTER the
// inner batch (see `api::batch::Batch`).
unsafe impl Send for VulkanBatch {}

impl VulkanBatch {
    pub(super) fn begin(device: &VulkanDevice, auto_barrier: bool) -> Result<Self, QuantaError> {
        let lease = device.alloc_command_buffer()?;
        let cmd = lease.cmd;
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        let r = unsafe { ffi::vkBeginCommandBuffer(cmd, &begin) };
        if r != ffi::VK_SUCCESS {
            // Dropping the lease returns the pair to the cache.
            return Err(QuantaError::submit_failed());
        }
        // Leading global barrier: Vulkan pipeline-barrier scopes cover
        // SUBMISSION order, not command-buffer extent — this one line
        // orders the whole batch after every previously submitted
        // work (the lane's threshold submits chain batches without
        // host waits, and nothing else provides that dependency).
        // ALL_COMMANDS, since the batch may open with a render pass
        // that samples what a previous submission drew.
        unsafe {
            emit_global_all_barrier(cmd);
        }
        Ok(VulkanBatch {
            device: device as *const VulkanDevice,
            lease: Some(lease),
            cmd,
            pools: Vec::new(),
            #[cfg(feature = "render")]
            render_objects: Vec::new(),
            pinned: Vec::new(),
            any_encoded: false,
            auto_barrier,
            broken: None,
        })
    }

    fn check_not_broken(&self) -> Result<(), QuantaError> {
        match &self.broken {
            Some(e) => Err(QuantaError::internal(
                "batch refused: an earlier pass failed mid-record and the command buffer \
                 holds a truncated pass — the next sync point surfaces that error and \
                 discards the batch",
            )
            .with_context(&alloc::format!("{e}"))),
            None => Ok(()),
        }
    }

    /// Pin every handle `pass` references, before any registry lookup
    /// the recording does, and remember them for the unpin.
    #[cfg(feature = "render")]
    fn pin_pass(&mut self, pass: &crate::RenderPass) -> usize {
        let device = unsafe { &*self.device };
        let base = self.pinned.len();
        pass.for_each_handle(|_, h| {
            device.pin_for_batch(h);
            self.pinned.push(h);
        });
        base
    }

    /// Drop the pins taken since `base` (a failed encode references
    /// nothing).
    fn unpin_from(&mut self, base: usize) {
        let device = unsafe { &*self.device };
        let fresh: Vec<u64> = self.pinned.split_off(base);
        device.unpin_for_batch(fresh.into_iter());
    }
}

/// The global COMPUTE→COMPUTE memory barrier both batch modes use
/// between dispatches: prior shader writes become visible to later
/// shader reads and writes, across the whole queue up to this point
/// in submission order.
///
/// # Safety
/// `cmd` must be in the recording state.
#[cfg(feature = "compute")]
unsafe fn emit_global_compute_barrier(cmd: ffi::VkCommandBuffer) {
    let barrier = ffi::VkMemoryBarrier {
        s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        p_next: core::ptr::null(),
        src_access_mask: ffi::VK_ACCESS_SHADER_WRITE_BIT,
        dst_access_mask: ffi::VK_ACCESS_SHADER_READ_BIT | ffi::VK_ACCESS_SHADER_WRITE_BIT,
    };
    unsafe {
        ffi::vkCmdPipelineBarrier(
            cmd,
            ffi::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            ffi::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier as *const ffi::VkMemoryBarrier as *const c_void,
            0,
            core::ptr::null(),
            0,
            core::ptr::null(),
        );
    }
}

/// The full-pipeline memory barrier around render work: every write by
/// anything before it is visible to everything after it. Used at the
/// batch's head and on both sides of a render pass or resolve — the
/// compute run tracker knows buffers only, so the render edges are
/// ordered wholesale (a pass dominates the cost of its two barriers).
///
/// # Safety
/// `cmd` must be in the recording state.
unsafe fn emit_global_all_barrier(cmd: ffi::VkCommandBuffer) {
    let barrier = ffi::VkMemoryBarrier {
        s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        p_next: core::ptr::null(),
        src_access_mask: ffi::VK_ACCESS_MEMORY_WRITE_BIT,
        dst_access_mask: ffi::VK_ACCESS_MEMORY_READ_BIT | ffi::VK_ACCESS_MEMORY_WRITE_BIT,
    };
    unsafe {
        ffi::vkCmdPipelineBarrier(
            cmd,
            ffi::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            ffi::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            1,
            &barrier as *const ffi::VkMemoryBarrier as *const c_void,
            0,
            core::ptr::null(),
            0,
            core::ptr::null(),
        );
    }
}

impl crate::batch::BatchInner for VulkanBatch {
    #[cfg(feature = "compute")]
    fn encode_dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError> {
        self.check_not_broken()?;
        let device = unsafe { &*self.device };

        // Pin FIRST — before any registry lookup — so a concurrent
        // destroy of a bound buffer or of the wave's pipeline can only
        // park, never free what this encode is about to reference.
        let pin_base = self.pinned.len();
        device.pin_for_batch(wave.handle);
        self.pinned.push(wave.handle);
        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0 {
                device.pin_for_batch(handle);
                self.pinned.push(handle);
            }
        }

        let records = match device.fold_dispatch_records(wave, quarks) {
            Ok(r) => r,
            Err(e) => {
                self.unpin_from(pin_base);
                return Err(e);
            }
        };
        let prep = match device.prepare_wave_dispatch(wave) {
            Ok(p) => p,
            Err(e) => {
                self.unpin_from(pin_base);
                return Err(e);
            }
        };

        // Vulkan orders nothing between dispatches on one command
        // buffer: a global memory barrier makes every prior encode's
        // shader writes visible to this one's reads and writes — the
        // chain shape the deferred lane records (op N+1 consumes op
        // N's output).
        if self.auto_barrier && self.any_encoded {
            unsafe {
                emit_global_compute_barrier(self.cmd);
            }
        }

        let recorded = unsafe { self_record(device, self.cmd, &prep, wave, &records) };
        match recorded {
            Ok(()) => {
                self.pools.push(prep.pool);
                self.any_encoded = true;
                Ok(())
            }
            Err(e) => {
                device.return_descriptor_pool(prep.pool);
                self.unpin_from(pin_base);
                Err(e)
            }
        }
    }

    #[cfg(feature = "compute")]
    fn encode_barrier(&mut self) -> Result<(), QuantaError> {
        // Meaningful in concurrent mode (the lane's hazard-run
        // boundary); harmless over-ordering in serial mode.
        unsafe {
            emit_global_compute_barrier(self.cmd);
        }
        Ok(())
    }

    #[cfg(feature = "render")]
    fn encode_render(&mut self, pass: crate::RenderPass) -> Result<(), QuantaError> {
        use crate::driver::RecordFailure;
        self.check_not_broken()?;
        let device = unsafe { &*self.device };
        let pin_base = self.pin_pass(&pass);
        // Everything before the pass → visible to it; the pass → visible
        // to everything after. Emitted around the record, so a Clean
        // refusal (nothing recorded) still leaves a matched pair.
        unsafe {
            emit_global_all_barrier(self.cmd);
        }
        let result = device.record_render_pass(self.cmd, pass);
        unsafe {
            emit_global_all_barrier(self.cmd);
        }
        match result {
            Ok(objects) => {
                self.render_objects.push(objects);
                self.any_encoded = true;
                Ok(())
            }
            Err(RecordFailure::Clean(e)) => {
                self.unpin_from(pin_base);
                Err(e)
            }
            Err(RecordFailure::Partial(e)) => {
                // The truncated commands still name the pinned handles
                // until the buffer is discarded — pins stay until Drop.
                self.broken = Some(e.clone());
                Err(e)
            }
        }
    }

    #[cfg(feature = "render")]
    fn encode_resolve(&mut self, src: u64, dst: u64) -> Result<(), QuantaError> {
        use crate::driver::RecordFailure;
        self.check_not_broken()?;
        let device = unsafe { &*self.device };
        let pin_base = self.pinned.len();
        for h in [src, dst] {
            device.pin_for_batch(h);
            self.pinned.push(h);
        }
        unsafe {
            emit_global_all_barrier(self.cmd);
        }
        let result = device.record_resolve(self.cmd, src, dst);
        unsafe {
            emit_global_all_barrier(self.cmd);
        }
        match result {
            Ok(()) => {
                self.any_encoded = true;
                Ok(())
            }
            Err(RecordFailure::Clean(e)) => {
                self.unpin_from(pin_base);
                Err(e)
            }
            Err(RecordFailure::Partial(e)) => {
                self.broken = Some(e.clone());
                Err(e)
            }
        }
    }

    fn submit(self: Box<Self>) -> Result<Pulse, QuantaError> {
        let mut this = self;
        if let Some(e) = this.broken.take() {
            // Drop reclaims the command buffer, pools, objects, pins.
            return Err(e);
        }
        let device = unsafe { &*this.device };
        let r = unsafe { ffi::vkEndCommandBuffer(this.cmd) };
        if r != ffi::VK_SUCCESS {
            // Drop reclaims the (still-owned) command buffer, pools, pins.
            return Err(QuantaError::submit_failed());
        }
        // Consumed: `submit_and_wait` owns the lease from here (its
        // fence waiter returns it after the GPU is done; on submit
        // failure it drops straight back to the cache) — Drop must not
        // return it a second time, hence the take().
        let Some(lease) = this.lease.take() else {
            return Err(QuantaError::internal("batch submitted twice"));
        };
        let mut inner = device.submit_and_wait(lease)?;

        // The submission has its serial: unpin now — anything parked
        // for these handles retires behind the newest serial (ours).
        let pins = core::mem::take(&mut this.pinned);
        device.unpin_for_batch(pins.into_iter());

        // Descriptor pools return, and per-pass objects die, only
        // after the fence: compose the submission pulse with both.
        let pools = core::mem::take(&mut this.pools);
        #[cfg(feature = "render")]
        let render_objects = core::mem::take(&mut this.render_objects);
        let keep_alive = inner.keep_alive.take();
        let handle = inner.handle;
        struct AfterFence {
            device: *const VulkanDevice,
            pools: Vec<ffi::VkDescriptorPool>,
            #[cfg(feature = "render")]
            render_objects: Vec<RenderPassObjects>,
        }
        impl AfterFence {
            fn run(self) {
                let device = unsafe { &*self.device };
                for pool in self.pools {
                    device.return_descriptor_pool(pool);
                }
                #[cfg(feature = "render")]
                for objects in self.render_objects {
                    objects.destroy(device.device);
                }
            }
        }
        /// The submission's fence wait plus the after-fence work,
        /// run EXACTLY ONCE whether the pulse is waited or dropped
        /// unwaited (the lane drops its outstanding pulses at device
        /// teardown; a public batch's caller may drop the pulse
        /// anytime). Dropping the wait closure unrun would leak the
        /// fence, the framebuffers and the transient render passes
        /// past `vkDestroyDevice` — the same shape the per-pass path
        /// guards with `RenderPassCleanup`.
        struct BatchCleanup {
            inner: Pulse,
            after: Option<AfterFence>,
        }
        // Safety: same argument as `FenceWaiter` in submit_and_wait —
        // the fence wait is legal from any thread, the pool cache sits
        // behind its mutex, the per-pass objects are exclusively ours,
        // and the outer pulse's keep-alive holds the device across the
        // deferred wait (its `wait_fn` drops before its `keep_alive`).
        unsafe impl Send for BatchCleanup {}
        impl Drop for BatchCleanup {
            fn drop(&mut self) {
                // Waits the fence (which destroys it, returns the command
                // buffer lease and completes the retire serial), then
                // hands the pools back and destroys the per-pass objects.
                let _ = self.inner.wait();
                if let Some(after) = self.after.take() {
                    after.run();
                }
            }
        }
        let cleanup = BatchCleanup {
            inner,
            after: Some(AfterFence {
                device: this.device,
                pools,
                #[cfg(feature = "render")]
                render_objects,
            }),
        };
        Ok(Pulse {
            handle,
            completed: false,
            keep_alive,
            wait_fn: Some(Box::new(move || drop(cleanup))),
        })
    }
}

impl Drop for VulkanBatch {
    fn drop(&mut self) {
        let device = unsafe { &*self.device };
        // A submitted batch reaches here with the lease consumed and
        // pools/objects/pins drained — every arm below no-ops. An
        // ABANDONED batch was never submitted: dropping its lease
        // returns the (still-recording) buffer to the cache, where
        // reacquisition's pool reset clears it; its descriptor sets
        // and per-pass objects are unreferenced by the GPU, and its
        // parked destroys retire behind the newest submitted serial
        // (nothing submitted references them).
        drop(self.lease.take());
        for pool in self.pools.drain(..) {
            device.return_descriptor_pool(pool);
        }
        #[cfg(feature = "render")]
        for objects in self.render_objects.drain(..) {
            objects.destroy(device.device);
        }
        let pins = core::mem::take(&mut self.pinned);
        device.unpin_for_batch(pins.into_iter());
    }
}

/// Free-fn shim: `record_wave_commands` is a device method, but the
/// borrow checker cannot see through `&mut self` on the batch plus
/// `&*self.device` — record through the device reference directly.
#[cfg(feature = "compute")]
unsafe fn self_record(
    device: &VulkanDevice,
    cmd: ffi::VkCommandBuffer,
    prep: &PreparedDispatch,
    wave: &Wave,
    records: &[DispatchRecord],
) -> Result<(), QuantaError> {
    unsafe { device.record_wave_commands(cmd, prep, wave, records) }
}
