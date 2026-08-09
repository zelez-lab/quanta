//! Test seam for the npy/npz format layer — `#[doc(hidden)]`, not API.
//!
//! The header codec, ZIP container, and inflate live in private modules
//! (the crate-internal seam the typed npy/npz layer builds on). The
//! format-layer integration tests (`tests/npy_format.rs`) need to drive
//! them directly, so they are re-exported here. Nothing in this module
//! is part of the public surface and any of it may change without
//! notice.

pub use crate::npy_codec::{DATA_ALIGN, Descr, NpyDtype, parse_descr, write_header};
pub use crate::zip::inflate::inflate;
pub use crate::zip::{Entry, crc32, read_entries, write_stored};
