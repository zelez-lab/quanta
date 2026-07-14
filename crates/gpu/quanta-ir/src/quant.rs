//! Quantization scheme — the orthogonal-axis abstraction.
//!
//! Quantization is not a self-describing dtype (a value `q` means
//! `real = scale * (q - zero_point)`, with the params external). Following
//! the design that survived a decade in CUTLASS / burn / cubecl, the scheme
//! keeps four axes ORTHOGONAL so new formats slot in additively:
//!
//!   - `value` — the numeric semantics (Q8S, Q4S, …)
//!   - `store` — the physical layout (one byte, or packed nibbles)
//!   - `level` — the granularity of the scale (per-tensor, block, …)
//!   - `mode`  — symmetric (zero_point = 0) vs affine (full zero_point)
//!
//! First increment: per-tensor symmetric int8 + int4. The reserved variants
//! are present in the enums (so adding them later is a value change, not a
//! shape change) but the lowering returns `NotSupported` for them.

/// The numeric value type of a quantized element. `*S` = symmetric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantValue {
    /// Symmetric signed int8, range [-128, 127].
    Q8S,
    /// Symmetric signed int4, range [-8, 7].
    Q4S,
    // Reserved (not yet lowered): full-range/asymmetric int8/int4, int2…
}

impl QuantValue {
    /// Storage bit width of one element.
    pub fn bits(self) -> u32 {
        match self {
            QuantValue::Q8S => 8,
            QuantValue::Q4S => 4,
        }
    }

    /// The signed integer range `(lo, hi)` of representable codes.
    pub fn range(self) -> (i32, i32) {
        crate::dtype::quant_range(self.bits())
    }

    /// The storage ScalarType carrying the int payload. Q8 rides the
    /// portable i32 slot (one code per word — the int8-ness is the range
    /// clamp, not the storage width); Q4 packs nibbles via the dedicated
    /// I4 type. Native sub-byte int8 storage is a later `store`-axis fork.
    pub fn storage_scalar(self) -> crate::ScalarType {
        match self {
            QuantValue::Q8S => crate::ScalarType::I32,
            QuantValue::Q4S => crate::ScalarType::I4,
        }
    }
}

/// How quantized values are physically packed into buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantStore {
    /// One element per storage slot (int8 → one byte / u32-slot).
    Native,
    /// Sub-byte values packed into 32-bit words (int4 → 8 nibbles/word).
    PackedU32,
}

/// Granularity of the scale / zero-point. First increment: `PerTensor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantLevel {
    /// One (scale, zero_point) for the whole tensor.
    PerTensor,
    /// Reserved: one scale per fixed-size block of `0` = group size.
    Block(u32),
    /// Reserved: one scale per channel along a dimension.
    PerChannel,
}

/// Symmetric (zero_point = 0) vs affine (full zero_point). First increment:
/// `Symmetric`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantMode {
    /// zero_point is fixed at 0; `real = scale * q`.
    Symmetric,
    /// Reserved: `real = scale * (q - zero_point)`.
    Affine,
}

/// The full quantization scheme threaded through Quantize/Dequantize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantScheme {
    pub value: QuantValue,
    pub store: QuantStore,
    pub level: QuantLevel,
    pub mode: QuantMode,
}

impl QuantScheme {
    /// The per-tensor symmetric scheme for a value type, with the natural
    /// storage (int8 native, int4 packed). The only schemes lowered in the
    /// first increment.
    pub fn per_tensor_symmetric(value: QuantValue) -> Self {
        let store = match value {
            QuantValue::Q8S => QuantStore::Native,
            QuantValue::Q4S => QuantStore::PackedU32,
        };
        QuantScheme {
            value,
            store,
            level: QuantLevel::PerTensor,
            mode: QuantMode::Symmetric,
        }
    }

    /// Whether this scheme is in the lowered subset (per-tensor symmetric).
    /// Reserved levels/modes return `false` → backends emit `NotSupported`.
    pub fn is_supported(self) -> bool {
        matches!(self.level, QuantLevel::PerTensor) && matches!(self.mode, QuantMode::Symmetric)
    }
}
