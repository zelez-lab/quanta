/// Errors returned by Quanta operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantaError {
    /// No GPU device found at the given index.
    NoDevice,
    /// GPU memory allocation failed.
    OutOfMemory,
    /// Shader/kernel compilation failed.
    CompilationFailed(String),
    /// Command submission failed.
    SubmitFailed,
    /// GPU operation timed out.
    Timeout,
    /// The device was lost (hardware removed, driver crash).
    DeviceLost,
    /// Invalid parameter.
    InvalidParam(&'static str),
}

impl core::fmt::Display for QuantaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no GPU device found"),
            Self::OutOfMemory => write!(f, "GPU out of memory"),
            Self::CompilationFailed(msg) => write!(f, "compilation failed: {msg}"),
            Self::SubmitFailed => write!(f, "command submission failed"),
            Self::Timeout => write!(f, "GPU operation timed out"),
            Self::DeviceLost => write!(f, "GPU device lost"),
            Self::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}
