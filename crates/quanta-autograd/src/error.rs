//! Error type for `quanta-autograd`.

use quanta_array::ArrayError;

/// An autograd failure — almost always a wrapped array/GPU error from running
/// a forward op or accumulating a gradient.
#[derive(Debug)]
pub enum AutogradError {
    /// An underlying `quanta-array` op failed (GPU error, shape mismatch, …).
    Array(ArrayError),
    /// `backward` was called on a `Var` from a different tape.
    ForeignVar,
}

impl From<ArrayError> for AutogradError {
    fn from(e: ArrayError) -> Self {
        AutogradError::Array(e)
    }
}

impl core::fmt::Display for AutogradError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AutogradError::Array(e) => write!(f, "autograd: array op failed: {e:?}"),
            AutogradError::ForeignVar => write!(f, "autograd: Var belongs to a different tape"),
        }
    }
}

impl std::error::Error for AutogradError {}
