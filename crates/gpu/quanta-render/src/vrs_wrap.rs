//! Variable rate shading typed API (steps 028 + 029).
//!
//! The [`ShadingRate`] data model lives in `quanta-core` (the render
//! op stream the drivers execute carries it); this module holds the
//! typed [`VrsState`] wrapper.
//!
//! The wrapper enforces the lifecycle proven in `Quanta.Vrs.State`
//! (Lean) and `quanta-api/vrs_safety.rs` (Verus):
//!
//! - `set_rate(rate)` fails if the state is destroyed.
//! - `Drop` calls `vrs_destroy` exactly once.

use alloc::sync::Arc;

use quanta_core::{GpuDevice, QuantaError, ShadingRate};

/// A typed VRS state — Drop releases the backend handle.
///
/// Refines `Quanta.Vrs.State`.
pub struct VrsState {
    pub(crate) handle: u64,
    pub(crate) current: ShadingRate,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl VrsState {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// The currently bound shading rate.
    pub fn current(&self) -> ShadingRate {
        self.current
    }

    /// Set the shading rate the next draw will use.
    ///
    /// Returns `Err(InvalidParam)` if the state has been destroyed.
    /// Refines `Quanta.Vrs.State.setRate` and the Verus theorem
    /// `t7551_set_rate_writes`.
    pub fn set_rate(&mut self, rate: ShadingRate) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("VRS state is not live"));
        }
        self.device.vrs_set_rate(self.handle, rate.code())?;
        self.current = rate;
        Ok(())
    }
}

impl Drop for VrsState {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.vrs_destroy(self.handle);
            self.live = false;
        }
    }
}
