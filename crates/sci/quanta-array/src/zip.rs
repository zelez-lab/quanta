//! The npz container — a hand-rolled ZIP layer scoped to exactly what
//! numpy writes. Private substrate under [`crate::npz`].
//!
//! Write side ([`write_stored`]): stored entries only (method 0 — what
//! `np.savez` produces), real sizes and CRC-32 in the local headers (no
//! data descriptors), a central directory, and the EOCD record. Bytes
//! are deterministic: fixed 1980-01-01 DOS timestamps, zeroed attrs, and
//! caller entry order — identical input yields identical archives.
//!
//! Read side ([`read_entries`]): the EOCD → central directory walk is
//! authoritative for names, offsets, sizes and CRCs; local headers are
//! validated against it (which also sidesteps streaming-mode bit-3
//! entries — the CD always has real sizes). Stored and deflate (method
//! 8, [`inflate`]) entries are accepted, CRC-verified after extraction.
//! Extra fields and archive comments are tolerated per the ZIP spec;
//! ZIP64 markers are detected and refused loudly (the claim boundary:
//! numpy emits ZIP64 only above 4 GiB, and safetensors is the declared
//! big-weights lane). Names must end in `.npy` — anything else means the
//! ZIP was not written as an npz.
//!
//! Seam for the typed layer: `npz::save_named` appends `.npy` to its
//! names and feeds per-array npy bytes to [`write_stored`];
//! `npz::load_named` takes [`read_entries`] output, strips the suffix,
//! and decodes each entry through the shared npy codec.

pub mod inflate;

use std::collections::HashSet;

use crate::npy::NpyError;

const LOCAL_SIG: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];
const CD_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
const ZIP64_LOCATOR_SIG: [u8; 4] = [0x50, 0x4b, 0x06, 0x07];

/// 1980-01-01, the DOS epoch — the fixed timestamp every entry carries.
const DOS_DATE_1980: u16 = 0x0021;
/// "version made by" / "version needed": 2.0, plain stored/deflate.
const ZIP_VERSION: u16 = 20;
/// The ZIP64 extra-field header id.
const ZIP64_EXTRA_ID: u16 = 0x0001;

fn zip_err(entry: Option<&str>, what: impl Into<String>) -> NpyError {
    NpyError::Zip {
        entry: entry.map(str::to_string),
        what: what.into(),
    }
}

fn zip64_err(entry: Option<&str>) -> NpyError {
    zip_err(
        entry,
        "ZIP64 markers present — archives or entries at/above 4 GiB are not supported \
         (safetensors is the big-weights lane)",
    )
}

// ── CRC-32 (IEEE, reflected) ────────────────────────────────────────────

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// Table-based CRC-32 (IEEE) — the `"123456789"` → `0xCBF43926` one.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

// ── Write ───────────────────────────────────────────────────────────────

fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// A value that must fit u32 without touching the 0xFFFFFFFF ZIP64
/// sentinel.
fn fits_u32(v: usize) -> bool {
    v < u32::MAX as usize
}

/// Write a stored-only (method 0) archive: one local header + data run
/// per entry in caller order, then the central directory and EOCD.
/// Deterministic bytes; duplicate names and ZIP64-scale inputs are loud
/// errors.
pub fn write_stored(entries: &[(&str, &[u8])]) -> Result<Vec<u8>, NpyError> {
    if entries.len() >= 0xFFFF {
        return Err(zip_err(
            None,
            format!(
                "{} entries — 65535 or more requires ZIP64, which is not supported",
                entries.len()
            ),
        ));
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut cd = Vec::new();
    for (name, data) in entries {
        if !seen.insert(*name) {
            return Err(zip_err(Some(name), "duplicate entry name"));
        }
        let name_bytes = name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(zip_err(Some(name), "entry name longer than 65535 bytes"));
        }
        if !fits_u32(data.len()) {
            return Err(zip_err(
                Some(name),
                "entry at/above 4 GiB requires ZIP64, which is not supported \
                 (safetensors is the big-weights lane)",
            ));
        }
        let offset = out.len();
        if !fits_u32(offset) {
            return Err(zip_err(
                Some(name),
                "archive at/above 4 GiB requires ZIP64, which is not supported",
            ));
        }
        let crc = crc32(data);
        // General-purpose bit 11: the name is UTF-8 (only flagged when
        // it matters — ASCII names coincide with cp437).
        let flags: u16 = if name.is_ascii() { 0 } else { 0x0800 };
        let size = data.len() as u32;

        out.extend_from_slice(&LOCAL_SIG);
        push_u16(&mut out, ZIP_VERSION); // version needed
        push_u16(&mut out, flags);
        push_u16(&mut out, 0); // method 0 = stored
        push_u16(&mut out, 0); // mod time 00:00:00
        push_u16(&mut out, DOS_DATE_1980);
        push_u32(&mut out, crc);
        push_u32(&mut out, size); // compressed
        push_u32(&mut out, size); // uncompressed
        push_u16(&mut out, name_bytes.len() as u16);
        push_u16(&mut out, 0); // extra length
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(data);

        cd.extend_from_slice(&CD_SIG);
        push_u16(&mut cd, ZIP_VERSION); // version made by
        push_u16(&mut cd, ZIP_VERSION); // version needed
        push_u16(&mut cd, flags);
        push_u16(&mut cd, 0); // method
        push_u16(&mut cd, 0); // mod time
        push_u16(&mut cd, DOS_DATE_1980);
        push_u32(&mut cd, crc);
        push_u32(&mut cd, size);
        push_u32(&mut cd, size);
        push_u16(&mut cd, name_bytes.len() as u16);
        push_u16(&mut cd, 0); // extra length
        push_u16(&mut cd, 0); // comment length
        push_u16(&mut cd, 0); // disk number start
        push_u16(&mut cd, 0); // internal attrs
        push_u32(&mut cd, 0); // external attrs
        push_u32(&mut cd, offset as u32);
        cd.extend_from_slice(name_bytes);
    }
    let cd_offset = out.len();
    if !fits_u32(cd_offset) || !fits_u32(cd.len()) {
        return Err(zip_err(
            None,
            "archive at/above 4 GiB requires ZIP64, which is not supported",
        ));
    }
    out.extend_from_slice(&cd);
    out.extend_from_slice(&EOCD_SIG);
    push_u16(&mut out, 0); // this disk
    push_u16(&mut out, 0); // CD start disk
    push_u16(&mut out, entries.len() as u16); // entries on this disk
    push_u16(&mut out, entries.len() as u16); // entries total
    push_u32(&mut out, cd.len() as u32);
    push_u32(&mut out, cd_offset as u32);
    push_u16(&mut out, 0); // comment length
    Ok(out)
}

// ── Read ────────────────────────────────────────────────────────────────

/// One extracted archive entry: the full name (with its `.npy` suffix)
/// and the decompressed, CRC-verified bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub data: Vec<u8>,
}

/// Little-endian field reads inside an already bounds-checked record.
fn rd_u16(rec: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([rec[off], rec[off + 1]])
}
fn rd_u32(rec: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([rec[off], rec[off + 1], rec[off + 2], rec[off + 3]])
}

/// Locate the EOCD record: scan back from the end (the archive comment
/// may follow it), requiring the comment-length field to land exactly on
/// the end of the input.
fn find_eocd(bytes: &[u8]) -> Result<usize, NpyError> {
    if bytes.len() < 22 {
        return Err(zip_err(
            None,
            "too short to hold an end-of-central-directory record (not a ZIP archive)",
        ));
    }
    let lo = bytes.len().saturating_sub(22 + u16::MAX as usize);
    for pos in (lo..=bytes.len() - 22).rev() {
        if bytes[pos..pos + 4] == EOCD_SIG {
            let comment_len = rd_u16(&bytes[pos..pos + 22], 20) as usize;
            if pos + 22 + comment_len == bytes.len() {
                return Ok(pos);
            }
        }
    }
    Err(zip_err(
        None,
        "no end-of-central-directory record found (not a ZIP archive, or truncated)",
    ))
}

struct CdEntry {
    name: String,
    method: u16,
    crc: u32,
    compressed: usize,
    uncompressed: usize,
    local_offset: usize,
}

/// Walk one central-directory entry at `bytes[pos..]`; returns the entry
/// and the position just past it.
fn read_cd_entry(bytes: &[u8], pos: usize) -> Result<(CdEntry, usize), NpyError> {
    let rec = bytes
        .get(pos..pos + 46)
        .ok_or_else(|| zip_err(None, "central directory truncated"))?;
    if rec[..4] != CD_SIG {
        return Err(zip_err(None, "bad central-directory entry signature"));
    }
    let method = rd_u16(rec, 10);
    let crc = rd_u32(rec, 16);
    let compressed = rd_u32(rec, 20);
    let uncompressed = rd_u32(rec, 24);
    let name_len = rd_u16(rec, 28) as usize;
    let extra_len = rd_u16(rec, 30) as usize;
    let comment_len = rd_u16(rec, 32) as usize;
    let disk_start = rd_u16(rec, 34);
    let local_offset = rd_u32(rec, 42);

    let name_bytes = bytes
        .get(pos + 46..pos + 46 + name_len)
        .ok_or_else(|| zip_err(None, "central-directory entry name truncated"))?;
    let name = core::str::from_utf8(name_bytes)
        .map_err(|_| zip_err(None, "central-directory entry name is not valid UTF-8"))?
        .to_string();

    if compressed == u32::MAX || uncompressed == u32::MAX || local_offset == u32::MAX {
        return Err(zip64_err(Some(&name)));
    }
    if disk_start != 0 {
        return Err(zip_err(
            Some(&name),
            "multi-disk archives are not supported",
        ));
    }
    if method != 0 && method != 8 {
        return Err(zip_err(
            Some(&name),
            format!("compression method {method} is not supported (stored and deflate only)"),
        ));
    }
    // Extra fields are skipped per the ZIP spec, but a ZIP64 field means
    // the sizes above are lies — walk the (id, size) pairs to detect it.
    let extra_start = pos + 46 + name_len;
    let mut extra = bytes
        .get(extra_start..extra_start + extra_len)
        .ok_or_else(|| zip_err(Some(&name), "central-directory extra field truncated"))?;
    while !extra.is_empty() {
        if extra.len() < 4 {
            return Err(zip_err(Some(&name), "malformed extra field"));
        }
        let id = rd_u16(extra, 0);
        let size = rd_u16(extra, 2) as usize;
        if id == ZIP64_EXTRA_ID {
            return Err(zip64_err(Some(&name)));
        }
        extra = extra
            .get(4 + size..)
            .ok_or_else(|| zip_err(Some(&name), "malformed extra field"))?;
    }
    // The entry comment is skipped, bounds-checked.
    let end = extra_start + extra_len + comment_len;
    if end > bytes.len() {
        return Err(zip_err(
            Some(&name),
            "central-directory entry comment truncated",
        ));
    }
    Ok((
        CdEntry {
            name,
            method,
            crc,
            compressed: compressed as usize,
            uncompressed: uncompressed as usize,
            local_offset: local_offset as usize,
        },
        end,
    ))
}

/// Extract one entry's bytes, validating the local header against the
/// authoritative central-directory record.
fn extract(bytes: &[u8], cd: &CdEntry) -> Result<Vec<u8>, NpyError> {
    let name = cd.name.as_str();
    let rec = bytes
        .get(cd.local_offset..cd.local_offset + 30)
        .ok_or_else(|| {
            zip_err(
                name.into(),
                "central-directory offset points outside the archive",
            )
        })?;
    if rec[..4] != LOCAL_SIG {
        return Err(zip_err(
            name.into(),
            "no local header at the central directory's offset",
        ));
    }
    let flags = rd_u16(rec, 6);
    if rd_u16(rec, 8) != cd.method {
        return Err(zip_err(
            name.into(),
            "local header method disagrees with the central directory",
        ));
    }
    let name_len = rd_u16(rec, 26) as usize;
    let extra_len = rd_u16(rec, 28) as usize;
    let local_name = bytes
        .get(cd.local_offset + 30..cd.local_offset + 30 + name_len)
        .ok_or_else(|| zip_err(name.into(), "local header name truncated"))?;
    if local_name != name.as_bytes() {
        return Err(zip_err(
            name.into(),
            "local header name disagrees with the central directory",
        ));
    }
    // Bit 3 (streaming mode) moves sizes/CRC to a data descriptor and
    // zeroes them here; the CD stays authoritative either way. Without
    // bit 3 the two records must agree.
    if flags & 0x0008 == 0
        && (rd_u32(rec, 14) != cd.crc
            || rd_u32(rec, 18) as usize != cd.compressed
            || rd_u32(rec, 22) as usize != cd.uncompressed)
    {
        return Err(zip_err(
            name.into(),
            "local header sizes/CRC disagree with the central directory",
        ));
    }
    let data_start = cd.local_offset + 30 + name_len + extra_len;
    let raw = data_start
        .checked_add(cd.compressed)
        .and_then(|end| bytes.get(data_start..end))
        .ok_or_else(|| zip_err(name.into(), "entry data overruns the archive"))?;

    let data = match cd.method {
        0 => {
            if cd.compressed != cd.uncompressed {
                return Err(zip_err(
                    name.into(),
                    format!(
                        "stored entry sizes disagree: compressed {}, uncompressed {}",
                        cd.compressed, cd.uncompressed
                    ),
                ));
            }
            raw.to_vec()
        }
        8 => {
            let out = inflate::inflate(raw, cd.uncompressed)
                .map_err(|e| zip_err(name.into(), format!("deflate: {e}")))?;
            if out.len() != cd.uncompressed {
                return Err(zip_err(
                    name.into(),
                    format!(
                        "deflate stream inflated to {} bytes, central directory declares {}",
                        out.len(),
                        cd.uncompressed
                    ),
                ));
            }
            out
        }
        _ => unreachable!("method validated during the CD walk"),
    };
    let got = crc32(&data);
    if got != cd.crc {
        return Err(zip_err(
            name.into(),
            format!(
                "CRC-32 mismatch: computed {got:08x}, central directory says {:08x}",
                cd.crc
            ),
        ));
    }
    Ok(data)
}

/// Read every entry of an npz archive, central-directory-driven, in CD
/// order. Names keep their `.npy` suffix (the npz layer strips it); a
/// non-`.npy` name, a duplicate, a CRC mismatch, a local/CD
/// disagreement, or any ZIP64 marker is a loud error.
pub fn read_entries(bytes: &[u8]) -> Result<Vec<Entry>, NpyError> {
    let eocd_pos = find_eocd(bytes)?;
    // A ZIP64 EOCD locator immediately precedes the EOCD when present.
    if eocd_pos >= 20 && bytes[eocd_pos - 20..eocd_pos - 16] == ZIP64_LOCATOR_SIG {
        return Err(zip64_err(None));
    }
    let eocd = &bytes[eocd_pos..eocd_pos + 22];
    let this_disk = rd_u16(eocd, 4);
    let cd_disk = rd_u16(eocd, 6);
    let n_this = rd_u16(eocd, 8);
    let n_total = rd_u16(eocd, 10);
    let cd_size = rd_u32(eocd, 12);
    let cd_offset = rd_u32(eocd, 16);
    if this_disk != 0 || cd_disk != 0 || n_this != n_total {
        return Err(zip_err(None, "multi-disk archives are not supported"));
    }
    if n_total == u16::MAX || cd_size == u32::MAX || cd_offset == u32::MAX {
        return Err(zip64_err(None));
    }
    let cd_offset = cd_offset as usize;
    let cd_end = cd_offset
        .checked_add(cd_size as usize)
        .filter(|&e| e <= eocd_pos)
        .ok_or_else(|| zip_err(None, "central directory overruns the archive"))?;

    let mut entries = Vec::with_capacity(n_total as usize);
    let mut names = HashSet::new();
    let mut pos = cd_offset;
    for _ in 0..n_total {
        let (cd, next) = read_cd_entry(bytes, pos)?;
        if next > cd_end {
            return Err(zip_err(
                None,
                "central-directory entry overruns the directory's declared size",
            ));
        }
        if !cd.name.ends_with(".npy") {
            return Err(zip_err(
                Some(&cd.name),
                "not a .npy entry — this ZIP was not written as an npz archive",
            ));
        }
        if !names.insert(cd.name.clone()) {
            return Err(zip_err(Some(&cd.name), "duplicate entry name"));
        }
        let data = extract(bytes, &cd)?;
        entries.push(Entry {
            name: cd.name,
            data,
        });
        pos = next;
    }
    if pos != cd_end {
        return Err(zip_err(
            None,
            "central directory size disagrees with its entry count",
        ));
    }
    Ok(entries)
}
