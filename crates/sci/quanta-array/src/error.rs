//! Error type for quanta-array.

use core::fmt;

/// Errors from array construction, manipulation, and dispatch.
#[derive(Debug)]
pub enum ArrayError {
    /// A shape/layout operation failed (bad rank, non-broadcastable, etc.).
    Layout(quanta_tensor::LayoutError),
    /// A shape was invalid (zero extent, etc.).
    Shape(quanta_tensor::ShapeError),
    /// The data length didn't match the shape's element count.
    LengthMismatch { expected: usize, got: usize },
    /// A GPU operation (alloc / dispatch / read) failed.
    Gpu(quanta_core::QuantaError),
    /// An operation requires a contiguous array; call `.contiguous()` first.
    NotContiguous,
    /// An npy / npz interop fault (header, container, or dtype).
    Npy(crate::npy::NpyError),
    /// A quantization fault (rank, granularity, grid, or checkpoint).
    Quant(crate::quant::QuantError),
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayError::Layout(e) => write!(f, "layout error: {e:?}"),
            ArrayError::Shape(e) => write!(f, "shape error: {e:?}"),
            ArrayError::LengthMismatch { expected, got } => {
                write!(f, "data length {got} does not match shape size {expected}")
            }
            ArrayError::Gpu(e) => write!(f, "gpu error: {e}"),
            ArrayError::NotContiguous => {
                write!(
                    f,
                    "operation requires a contiguous array (call .contiguous())"
                )
            }
            // NpyError / QuantError messages are self-contained per the
            // interop message contract (offender / workaround always
            // present) — no wrapper prefix on top.
            ArrayError::Npy(e) => write!(f, "{e}"),
            ArrayError::Quant(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ArrayError {}

impl From<quanta_tensor::LayoutError> for ArrayError {
    fn from(e: quanta_tensor::LayoutError) -> Self {
        ArrayError::Layout(e)
    }
}
impl From<quanta_tensor::ShapeError> for ArrayError {
    fn from(e: quanta_tensor::ShapeError) -> Self {
        ArrayError::Shape(e)
    }
}
impl From<quanta_core::QuantaError> for ArrayError {
    fn from(e: quanta_core::QuantaError) -> Self {
        ArrayError::Gpu(e)
    }
}
impl From<crate::npy::NpyError> for ArrayError {
    fn from(e: crate::npy::NpyError) -> Self {
        ArrayError::Npy(e)
    }
}
impl From<crate::quant::QuantError> for ArrayError {
    fn from(e: crate::quant::QuantError) -> Self {
        ArrayError::Quant(e)
    }
}
