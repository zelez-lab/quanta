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

use crate::{GpuDevice, QuantaError, Wave};

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
pub struct IndirectCommandBuffer {
    pub(crate) handle: u64,
    pub(crate) cap: u32,
    pub(crate) recorded: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

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

    /// Append a dispatch command to the buffer.
    ///
    /// Records the wave's pipeline, current bindings, and the dispatch
    /// group counts. Backends snapshot the binding state at record
    /// time — later mutating the wave does not affect recorded
    /// commands.
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

impl Drop for IndirectCommandBuffer {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.indirect_buffer_destroy(self.handle);
            self.live = false;
        }
    }
}
