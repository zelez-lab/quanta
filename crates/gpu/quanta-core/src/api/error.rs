use alloc::format;
use alloc::string::String;

/// Errors returned by Quanta operations.
///
/// Implements [`core::error::Error`], so it composes with `?` and
/// error-reporting frameworks.
///
/// Marked `#[non_exhaustive]`: fields can be added without a breaking
/// change. Construct through the convenience constructors
/// ([`QuantaError::not_supported`], [`QuantaError::invalid_param`], …).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct QuantaError {
    pub kind: QuantaErrorKind,
    pub context: Option<String>,
}

/// The category of error.
///
/// Every message-carrying variant holds a `String`, so drivers can
/// report both static category messages and dynamic detail (compiler
/// logs, handle values) uniformly.
///
/// Marked `#[non_exhaustive]`: categories can be added — match with a
/// wildcard arm.
#[non_exhaustive]
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
    /// Invalid parameter — caller passed a value outside the
    /// documented range. Distinct from `NotSupported` (feature is
    /// genuinely unavailable) and `NotFound` (handle does not exist).
    InvalidParam(String),
    /// The requested feature is not implemented on this backend
    /// (e.g. mesh shaders on a software CPU device, ray tracing on
    /// pre-Apple-family-6 Metal). Callers should branch on this to
    /// fall back to a non-accelerated path.
    NotSupported(String),
    /// The given handle does not refer to a live resource. Usually
    /// means the wrapping typed handle was double-freed or never
    /// allocated by this device.
    NotFound(String),
    /// The presentation surface no longer matches its target (the
    /// window / layer was resized or its properties changed since the
    /// last `configure`). Recoverable: call `Surface::configure` with
    /// the new extent, then acquire again.
    SurfaceOutdated(String),
    /// Internal error (e.g. poisoned mutex).
    Internal(String),
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

    pub fn invalid_param(msg: impl Into<String>) -> Self {
        Self {
            kind: QuantaErrorKind::InvalidParam(msg.into()),
            context: None,
        }
    }

    /// Construct a `NotSupported` error — the feature is genuinely
    /// unavailable on this backend / device.
    pub fn not_supported(msg: impl Into<String>) -> Self {
        Self {
            kind: QuantaErrorKind::NotSupported(msg.into()),
            context: None,
        }
    }

    /// Construct a `NotFound` error — the given handle does not
    /// refer to a live resource.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            kind: QuantaErrorKind::NotFound(msg.into()),
            context: None,
        }
    }

    /// Construct a `SurfaceOutdated` error — the presentation surface
    /// no longer matches its target and must be reconfigured before
    /// the next acquire.
    pub fn surface_outdated(msg: impl Into<String>) -> Self {
        Self {
            kind: QuantaErrorKind::SurfaceOutdated(msg.into()),
            context: None,
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            kind: QuantaErrorKind::Internal(msg.into()),
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
            QuantaErrorKind::NotSupported(msg) => format!("not supported on this backend: {msg}"),
            QuantaErrorKind::NotFound(msg) => format!("not found: {msg}"),
            QuantaErrorKind::SurfaceOutdated(msg) => format!("surface outdated: {msg}"),
            QuantaErrorKind::Internal(msg) => format!("internal error: {msg}"),
        };
        if let Some(ctx) = &self.context {
            write!(f, "{base} [{ctx}]")
        } else {
            write!(f, "{base}")
        }
    }
}

impl core::error::Error for QuantaError {}
