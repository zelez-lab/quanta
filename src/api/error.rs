use alloc::format;
use alloc::string::String;

/// Errors returned by Quanta operations.
#[derive(Debug, Clone)]
pub struct QuantaError {
    pub kind: QuantaErrorKind,
    pub context: Option<String>,
}

/// The category of error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantaErrorKind {
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

impl QuantaError {
    /// Attach context to this error (e.g. which operation produced it).
    pub fn with_context(mut self, ctx: &str) -> Self {
        self.context = Some(String::from(ctx));
        self
    }

    // --- Convenience constructors (keep call-sites concise) ---

    pub fn no_device() -> Self {
        Self {
            kind: QuantaErrorKind::NoDevice,
            context: None,
        }
    }

    pub fn out_of_memory() -> Self {
        Self {
            kind: QuantaErrorKind::OutOfMemory,
            context: None,
        }
    }

    pub fn compilation_failed(msg: impl Into<String>) -> Self {
        Self {
            kind: QuantaErrorKind::CompilationFailed(msg.into()),
            context: None,
        }
    }

    pub fn submit_failed() -> Self {
        Self {
            kind: QuantaErrorKind::SubmitFailed,
            context: None,
        }
    }

    pub fn timeout() -> Self {
        Self {
            kind: QuantaErrorKind::Timeout,
            context: None,
        }
    }

    pub fn device_lost() -> Self {
        Self {
            kind: QuantaErrorKind::DeviceLost,
            context: None,
        }
    }

    pub fn invalid_param(msg: &'static str) -> Self {
        Self {
            kind: QuantaErrorKind::InvalidParam(msg),
            context: None,
        }
    }
}

impl PartialEq for QuantaError {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Eq for QuantaError {}

impl core::fmt::Display for QuantaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let base = match &self.kind {
            QuantaErrorKind::NoDevice => String::from("no GPU device found"),
            QuantaErrorKind::OutOfMemory => String::from("GPU out of memory"),
            QuantaErrorKind::CompilationFailed(msg) => format!("compilation failed: {msg}"),
            QuantaErrorKind::SubmitFailed => String::from("command submission failed"),
            QuantaErrorKind::Timeout => String::from("GPU operation timed out"),
            QuantaErrorKind::DeviceLost => String::from("GPU device lost"),
            QuantaErrorKind::InvalidParam(msg) => format!("invalid parameter: {msg}"),
        };
        if let Some(ctx) = &self.context {
            write!(f, "{base} [{ctx}]")
        } else {
            write!(f, "{base}")
        }
    }
}
