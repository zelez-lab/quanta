//! Typed `IndirectCommandBuffer` (steps 032 + 033).
//!
//! GPU-driven dispatch primitive: pre-record N dispatch commands into
//! a buffer, then have the GPU execute them without re-issuing each
//! command from the host. The user-facing wrapper enforces the
//! lifetime model proven in `Quanta.Icb` (Lean) and
//! `quanta-api/icb_safety.rs` (Verus):
//!
//! - `record_dispatch` fails if the buffer is full or destroyed.
//! - `execute(count)` requires `count ≤ recorded`.
//! - `Drop` calls `indirect_buffer_destroy` exactly once.
//!
//! Backends (Metal `MTLIndirectCommandBuffer`, Vulkan secondary
//! command buffers, WebGPU `GPURenderBundle`) are responsible for
//! refining this model.

use alloc::sync::Arc;

use crate::{GpuDevice, QuantaError};
// `Wave` is a compute type; only the compute-gated `IndirectCommandBuffer`
// half of this module references it.
#[cfg(feature = "compute")]
use crate::Wave;
// `Pipeline` is a render type; only the render-gated `record_draw` paths
// (compute ICB draw + the render bundle) reference it.
#[cfg(feature = "render")]
use crate::Pipeline;

/// A pre-recorded sequence of GPU dispatch commands. Created via
/// [`Gpu::indirect_command_buffer`](crate::Gpu::indirect_command_buffer).
///
/// The buffer has a fixed capacity supplied at create time. Records
/// past `capacity()` return an error. `execute(count)` runs the
/// first `count` recorded dispatches in order; passing `count >
/// len()` returns an error.
///
/// Destruction is automatic on `Drop` — the underlying handle is
/// released once.
///
/// Compute-only (records `Wave` dispatches); gated with the `compute`
/// feature. The render-path sibling is [`IndirectRenderBundle`].
#[cfg(feature = "compute")]
pub struct IndirectCommandBuffer {
    pub(crate) handle: u64,
    pub(crate) cap: u32,
    pub(crate) recorded: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

#[cfg(feature = "compute")]
impl IndirectCommandBuffer {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Maximum number of commands this buffer can hold.
    pub fn capacity(&self) -> u32 {
        self.cap
    }

    /// Number of commands recorded so far.
    pub fn len(&self) -> u32 {
        self.recorded
    }

    /// Whether no commands have been recorded.
    pub fn is_empty(&self) -> bool {
        self.recorded == 0
    }

    /// Append a compute dispatch command to the buffer.
    ///
    /// Records the wave's pipeline, current bindings, and the dispatch
    /// group counts. Backends snapshot the binding state at record
    /// time — later mutating the wave does not affect recorded
    /// commands.
    ///
    /// Refines the `Quanta.Icb.Command.dispatch` constructor from the
    /// Lean equivalence theorem.
    ///
    /// Returns `Err(InvalidParam)` when the buffer is full or has
    /// been consumed.
    pub fn record_dispatch(&mut self, wave: &Wave, groups: [u32; 3]) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("ICB is not live"));
        }
        if self.recorded >= self.cap {
            return Err(QuantaError::invalid_param("ICB is full"));
        }
        self.device
            .icb_record_dispatch(self.handle, self.recorded, wave, groups)?;
        self.recorded += 1;
        Ok(())
    }

    /// Append a render-path draw command to the buffer.
    ///
    /// Records the render pipeline, vertex / instance counts, and
    /// the current resource bindings carried by the pipeline at
    /// record time. The recorded draw is replayed inside an active
    /// render pass when `execute` runs (Metal
    /// `executeCommandsInBuffer:withRange:` on a render encoder,
    /// Vulkan `vkCmdExecuteCommands` inside a render pass, WebGPU
    /// `executeBundles` on a render pass).
    ///
    /// Refines the `Quanta.Icb.Command.draw` constructor.
    ///
    /// Returns `Err(InvalidParam)` when the buffer is full, has been
    /// consumed, or the backend has not yet wired its render-path
    /// ICB lowering (the proof contract is in place; per-backend
    /// lowering is staged in follow-up commits).
    ///
    /// Render-only (step 085): takes a render `Pipeline`.
    #[cfg(feature = "render")]
    pub fn record_draw(
        &mut self,
        pipeline: &Pipeline,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("ICB is not live"));
        }
        if self.recorded >= self.cap {
            return Err(QuantaError::invalid_param("ICB is full"));
        }
        self.device.icb_record_draw(
            self.handle,
            self.recorded,
            pipeline.handle(),
            vertex_count,
            instance_count,
        )?;
        self.recorded += 1;
        Ok(())
    }

    /// Execute the first `count` recorded commands.
    ///
    /// Backends translate this to `executeCommandsInBuffer:withRange:`
    /// (Metal) or `vkCmdExecuteCommands` (Vulkan). Returns
    /// `Err(InvalidParam)` if `count > len()` or the buffer has been
    /// destroyed.
    pub fn execute(&self, count: u32) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("ICB is not live"));
        }
        if count > self.recorded {
            return Err(QuantaError::invalid_param(
                "ICB execute count exceeds recorded length",
            ));
        }
        self.device.indirect_buffer_execute(self.handle, count)
    }

    /// Execute every recorded command. Equivalent to
    /// `execute(self.len())`.
    pub fn execute_all(&self) -> Result<(), QuantaError> {
        self.execute(self.recorded)
    }
}

#[cfg(feature = "compute")]
impl Drop for IndirectCommandBuffer {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.indirect_buffer_destroy(self.handle);
            self.live = false;
        }
    }
}

/// A render-path Indirect Command Buffer.
///
/// Holds recorded draw commands that the GPU replays inside an
/// active render pass via
/// [`RenderPass::execute_bundle`](crate::RenderPass::execute_bundle).
/// Backends lower this to Metal `MTLIndirectCommandBuffer` with
/// DRAW command types, Vulkan secondary command buffers recorded
/// in `RENDER_PASS_CONTINUE` mode, or WebGPU `GPURenderBundle`.
///
/// Refines the `Quanta.Icb.Command.draw` constructor from the Lean
/// equivalence theorem (T7000 / T7006). The buffer has a fixed
/// capacity supplied at create time; records past `capacity()`
/// return an error; `Drop` releases the underlying handle.
///
/// Render-only (step 085): gated with the `render` feature.
#[cfg(feature = "render")]
pub struct IndirectRenderBundle {
    pub(crate) handle: u64,
    pub(crate) cap: u32,
    pub(crate) recorded: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

#[cfg(feature = "render")]
impl IndirectRenderBundle {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Maximum number of draws this bundle can hold.
    pub fn capacity(&self) -> u32 {
        self.cap
    }

    /// Number of draws recorded so far.
    pub fn len(&self) -> u32 {
        self.recorded
    }

    /// Whether no draws have been recorded.
    pub fn is_empty(&self) -> bool {
        self.recorded == 0
    }

    /// Append a draw command to the bundle.
    ///
    /// Records the render pipeline, vertex / instance counts. The
    /// recorded draw is replayed when a `RenderPass` calls
    /// `execute_bundle(self, count)`.
    ///
    /// Refines `Quanta.Icb.Command.draw`.
    pub fn record_draw(
        &mut self,
        pipeline: &Pipeline,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("render bundle is not live"));
        }
        if self.recorded >= self.cap {
            return Err(QuantaError::invalid_param("render bundle is full"));
        }
        self.device.render_bundle_record_draw(
            self.handle,
            self.recorded,
            pipeline.handle(),
            vertex_count,
            instance_count,
        )?;
        self.recorded += 1;
        Ok(())
    }
}

#[cfg(feature = "render")]
impl Drop for IndirectRenderBundle {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.render_bundle_destroy(self.handle);
            self.live = false;
        }
    }
}
