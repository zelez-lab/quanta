//! Device capabilities that change *how* a kernel is emitted.
//!
//! The emitters are device-agnostic by default (the portable path uses only
//! core features). A driver that has queried its device can pass an
//! `EmitCaps` so the emitter can take a faster path where the hardware
//! supports it — e.g. storing bf16 in native 16-bit buffers instead of the
//! portable one-bf16-per-u32-word fallback.

/// Optional device capabilities consulted during emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmitCaps {
    /// The device supports 16-bit storage-buffer access (Vulkan
    /// `storageBuffer16BitAccess` + `Int16`, Metal native `bfloat`/`ushort`
    /// buffers). When true, bf16 fields use true 2-byte storage; when false,
    /// bf16 is stored one value per 32-bit word (portable, +2 bytes/elem).
    ///
    /// WGSL has no 16-bit storage type at all, so the WGSL emitter ignores
    /// this and always uses the u32-word fallback.
    pub bf16_native_storage: bool,
}

impl EmitCaps {
    /// The portable capability set: assume no optional features. Every
    /// backend can run what this produces.
    pub const fn portable() -> Self {
        EmitCaps {
            bf16_native_storage: false,
        }
    }
}

impl Default for EmitCaps {
    fn default() -> Self {
        Self::portable()
    }
}
