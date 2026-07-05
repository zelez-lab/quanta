//! Variable rate shading typed API (steps 028 + 029).
//!
//! VRS lets the renderer reduce shading rate per region. A
//! `ShadingRate` of `2x2` means one fragment-shader invocation
//! covers a 2×2 block of pixels. Backends:
//!
//! - Metal: `MTLRasterizationRateMap` per-tile rates (Apple Silicon).
//! - Vulkan: `VK_KHR_fragment_shading_rate` +
//!   `vkCmdSetFragmentShadingRateKHR(rate, combiner_op)`.
//! - WebGPU: not in W3C — `NotSupported`.
//! - CPU: software lifecycle only.
//!
//! The wrapper enforces the lifecycle proven in `Quanta.Vrs.State`
//! (Lean) and `quanta-api/vrs_safety.rs` (Verus):
//!
//! - `set_rate(rate)` fails if the state is destroyed.
//! - `Drop` calls `vrs_destroy` exactly once.

use alloc::sync::Arc;

use crate::{GpuDevice, QuantaError};

/// Cross-vendor shading rate. The 7 entries match Vulkan
/// `VK_KHR_fragment_shading_rate` Tier 1 + Metal Apple Silicon
/// rate maps.
///
/// Marked `#[non_exhaustive]`: rates can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShadingRate {
    R1x1,
    R1x2,
    R2x1,
    R2x2,
    R2x4,
    R4x2,
    R4x4,
}

impl ShadingRate {
    /// Horizontal axis (pixels per fragment).
    pub fn x_axis(self) -> u32 {
        match self {
            Self::R1x1 | Self::R1x2 => 1,
            Self::R2x1 | Self::R2x2 | Self::R2x4 => 2,
            Self::R4x2 | Self::R4x4 => 4,
        }
    }

    /// Vertical axis (pixels per fragment).
    pub fn y_axis(self) -> u32 {
        match self {
            Self::R1x1 | Self::R2x1 => 1,
            Self::R1x2 | Self::R2x2 | Self::R4x2 => 2,
            Self::R2x4 | Self::R4x4 => 4,
        }
    }

    /// Encode as the 8-bit code passed across the device boundary
    /// (matches the Verus `ShadingRate` u8 type).
    pub fn code(self) -> u8 {
        match self {
            Self::R1x1 => 0,
            Self::R1x2 => 1,
            Self::R2x1 => 2,
            Self::R2x2 => 3,
            Self::R2x4 => 4,
            Self::R4x2 => 5,
            Self::R4x4 => 6,
        }
    }
}

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
