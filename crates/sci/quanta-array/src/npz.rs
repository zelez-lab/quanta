//! npz interop — the multi-array numpy container (`np.savez` /
//! `np.savez_compressed`).
//!
//! An npz file is an ordinary ZIP archive with one `<name>.npy` entry
//! per array. `np.savez` writes stored (method 0) entries;
//! `np.savez_compressed` writes deflate (method 8). The container layer
//! lives in the private `zip` module: a deterministic stored-entry
//! writer (fixed 1980 timestamps, caller entry order, real sizes +
//! CRC-32 — byte-for-byte the container class `np.savez` produces) and a
//! central-directory-driven reader with hand-rolled RFC 1951 inflate,
//! per-entry CRC verification, and loud ZIP64 refusal.
//!
//! The typed surface:
//!
//! - [`save_named`] — appends `.npy` to each name (numpy's convention),
//!   encodes each array through the shared npy codec, and writes stored
//!   entries in caller order. Duplicate names are a loud error.
//! - [`load_named`] — reads stored and deflate entries (so
//!   `np.savez_compressed` files load too), strips the `.npy` suffix,
//!   and returns archive order. A non-`.npy` entry means the ZIP wasn't
//!   written as an npz — a loud error naming the entry, never a silent
//!   skip.
//!
//! Bytes-level like the rest of the interop surface:
//! `std::fs::write(path, npz::save_named(&entries)?)`.

use quanta_core::Gpu;

use crate::error::ArrayError;
use crate::npy::{self, NpyArray, NpyError};

/// Serialize named arrays as an npz archive — ZIP **stored** (method 0),
/// the container `np.savez` writes. Entry names get `.npy` appended
/// (numpy's convention), caller order is preserved, and the bytes are
/// deterministic (fixed 1980-01-01 timestamps): identical input yields
/// identical archives. A duplicate name is a loud error.
pub fn save_named(entries: &[(String, NpyArray)]) -> Result<Vec<u8>, ArrayError> {
    let mut blobs = Vec::with_capacity(entries.len());
    for (name, array) in entries {
        blobs.push((format!("{name}.npy"), npy::save_dyn(array)?));
    }
    let refs: Vec<(&str, &[u8])> = blobs
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    Ok(crate::zip::write_stored(&refs)?)
}

/// Read every array of an npz archive, in archive order, with the
/// `.npy` suffix stripped from each name. Stored and deflate entries are
/// both accepted (`np.savez` and `np.savez_compressed` files alike),
/// CRC-verified, central-directory-driven. Every fault names the entry:
/// container damage through the ZIP taxonomy, and an entry whose npy
/// payload fails to decode is reported as [`NpyError::Zip`] carrying the
/// entry name with the inner fault's message.
pub fn load_named(gpu: &Gpu, bytes: &[u8]) -> Result<Vec<(String, NpyArray)>, ArrayError> {
    let entries = crate::zip::read_entries(bytes)?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let name = entry
            .name
            .strip_suffix(".npy")
            .expect("read_entries validates the suffix");
        let array = npy::load_dyn(gpu, &entry.data).map_err(|e| match e {
            // The message contract: the entry name is always present
            // where one exists. GPU/upload faults pass through untouched.
            ArrayError::Npy(inner) => ArrayError::Npy(NpyError::Zip {
                entry: Some(entry.name.clone()),
                what: format!("entry is not a loadable npy: {inner}"),
            }),
            other => other,
        })?;
        out.push((name.to_string(), array));
    }
    Ok(out)
}
