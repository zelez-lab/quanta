//! GPU printf debugging (step 049).
//!
//! Lets a kernel emit debug messages from inside a workgroup; the
//! host drains them after dispatch. Backends:
//!
//! - Vulkan: `VK_EXT_debug_printf` (validation-layer feature) +
//!   `debug_printfEXT(...)` SPIR-V intrinsic.
//! - Metal: `os_log` from MSL via the Metal Debugger.
//! - WebGPU: software shim through a storage-buffer ring.
//! - CPU: software ring buffer.
//!
//! The wrapper enforces the lifecycle proven in
//! `Quanta.Printf.Buffer` (Lean) and
//! `quanta-api/printf_safety.rs` (Verus):
//!
//! - `record` fails when the buffer is full or destroyed.
//! - `drain` returns recorded messages and empties the buffer.
//! - `Drop` releases the backend handle.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::{GpuDevice, QuantaError};

/// A typed printf buffer. Drop releases the backend handle.
///
/// Refines `Quanta.Printf.Buffer`.
pub struct PrintfBuffer {
    pub(crate) handle: u64,
    pub(crate) cap: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl PrintfBuffer {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Maximum number of recorded messages.
    pub fn capacity(&self) -> u32 {
        self.cap
    }

    /// Record a single message id from the host (e.g. for testing or
    /// for backends that do not have a native printf shader path).
    /// Refines `Quanta.Printf.Buffer.record`.
    pub fn record(&self, msg_id: u64) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("printf buffer is not live"));
        }
        self.device.printf_record(self.handle, msg_id)
    }

    /// Drain the recorded messages, leaving the buffer empty.
    /// Refines `Quanta.Printf.Buffer.drain`.
    pub fn drain(&self) -> Result<Vec<u64>, QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("printf buffer is not live"));
        }
        self.device.printf_drain(self.handle)
    }
}

impl Drop for PrintfBuffer {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.printf_destroy(self.handle);
            self.live = false;
        }
    }
}
