//! Format-layer tests for npy/npz interop: the header codec, the ZIP
//! container, RFC 1951 inflate, and the error taxonomy. Pure host
//! parsing — no `Array`-typed load/save here; that suite arrives with
//! the dtype matrix, alongside the numpy-generated fixtures under
//! `tests/fixtures/npy/` (generation is a pending maintainer step —
//! see `gen_fixtures.py` there; numpy was absent on the authoring
//! machine, so the deflate vectors below are hand-built instead).

use quanta_array::ArrayError;
use quanta_array::format_internals as fi;
use quanta_array::npy::{self, NpyError, NpyHeader};

/// Unwrap the `ArrayError::Npy` layer the public API returns.
fn npy_err(e: ArrayError) -> NpyError {
    match e {
        ArrayError::Npy(e) => e,
        other => panic!("expected ArrayError::Npy, got: {other}"),
    }
}

// ── npy header: spec-byte-exact reference ───────────────────────────────

/// The reference preamble for `<f4`, C-order, shape (2, 3): built by
/// hand from the spec — magic, v1.0, u16 length 118, the dict,
/// space-padding, `\n` — 128 bytes total, so the data section starts
/// 64-byte aligned exactly as modern numpy writes it.
fn reference_header_bytes() -> Vec<u8> {
    let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    let mut b = Vec::new();
    b.extend_from_slice(b"\x93NUMPY");
    b.push(1);
    b.push(0);
    b.extend_from_slice(&118u16.to_le_bytes());
    b.extend_from_slice(dict.as_bytes());
    b.resize(127, b' ');
    b.push(b'\n');
    assert_eq!(b.len(), 128);
    b
}

#[test]
fn write_header_matches_reference_bytes() {
    assert_eq!(
        fi::write_header("<f4", false, &[2, 3]),
        reference_header_bytes()
    );
}

#[test]
fn header_parses_reference_bytes() {
    // `header` needs only the preamble — introspection never touches
    // the data section.
    let h = npy::header(&reference_header_bytes()).unwrap();
    assert_eq!(
        h,
        NpyHeader {
            descr: "<f4".to_string(),
            fortran_order: false,
            shape: vec![2, 3],
            version: (1, 0),
            data_offset: 128,
        }
    );
}

#[test]
fn header_roundtrip_and_alignment() {
    let shapes: &[&[usize]] = &[&[], &[3], &[1, 2, 3], &[65535, 7], &[9; 24]];
    for descr in ["<f4", "<f8", "|u1", "<i2", "|b1", "<u8"] {
        for &shape in shapes {
            for fortran in [false, true] {
                let bytes = fi::write_header(descr, fortran, shape);
                assert_eq!(bytes.len() % fi::DATA_ALIGN, 0);
                let h = npy::header(&bytes).unwrap();
                assert_eq!(h.descr, descr);
                assert_eq!(h.fortran_order, fortran);
                assert_eq!(h.shape, shape);
                assert_eq!(h.version, (1, 0));
                assert_eq!(h.data_offset, bytes.len());
            }
        }
    }
}

#[test]
fn shape_tuple_forms() {
    // Rank 0 writes numpy's `()`, rank 1 the trailing-comma `(3,)`.
    // (The header text starts after the 10-byte v1 preamble.)
    let b0 = fi::write_header("<f8", false, &[]);
    assert!(
        std::str::from_utf8(&b0[10..])
            .unwrap()
            .contains("'shape': (), }")
    );
    let b1 = fi::write_header("<f8", false, &[3]);
    assert!(
        std::str::from_utf8(&b1[10..])
            .unwrap()
            .contains("'shape': (3,), }")
    );
}

#[test]
fn v2_auto_upgrade_on_u16_overflow() {
    // ~30k axes of extent 1 pushes the dict past 64 KiB — numpy's rule
    // upgrades the version rather than truncating the length field.
    let shape = vec![1usize; 30_000];
    let bytes = fi::write_header("<f4", false, &shape);
    assert_eq!(bytes.len() % fi::DATA_ALIGN, 0);
    let h = npy::header(&bytes).unwrap();
    assert_eq!(h.version, (2, 0));
    assert_eq!(h.shape, shape);
    assert_eq!(h.data_offset, bytes.len());
}

#[test]
fn v3_header_accepted() {
    // v3.0 differs from v2.0 only in header text encoding, invisible in
    // the grammar we accept — same dict, version bytes 3.0, u32 length.
    let dict = "{'descr': '<i4', 'fortran_order': True, 'shape': (5,), }";
    let mut b = Vec::new();
    b.extend_from_slice(b"\x93NUMPY");
    b.push(3);
    b.push(0);
    let text_len = (dict.len() + 1) as u32;
    b.extend_from_slice(&text_len.to_le_bytes());
    b.extend_from_slice(dict.as_bytes());
    b.push(b'\n');
    let h = npy::header(&b).unwrap();
    assert_eq!(h.version, (3, 0));
    assert_eq!(h.descr, "<i4");
    assert!(h.fortran_order);
    assert_eq!(h.shape, vec![5]);
    assert_eq!(h.data_offset, b.len());
}

// ── npy header: hostile input ───────────────────────────────────────────

/// Wrap an arbitrary dict text in a valid v1.0 preamble (exact length,
/// no padding) — the grammar-fuzzing harness.
fn npy_with_dict(dict: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"\x93NUMPY");
    b.push(1);
    b.push(0);
    b.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    b.extend_from_slice(dict.as_bytes());
    b
}

#[test]
fn magic_errors_show_first_bytes() {
    let e = npy_err(npy::header(b"").unwrap_err());
    assert!(matches!(e, NpyError::Magic { .. }));
    assert!(e.to_string().contains("empty input"));

    let e = npy_err(npy::header(b"PK\x03\x04rest").unwrap_err());
    assert!(matches!(e, NpyError::Magic { .. }));
    let msg = e.to_string();
    assert!(msg.contains("50 4b 03 04"), "first bytes shown: {msg}");

    // Truncated before the version bytes is a Magic error too.
    let e = npy_err(npy::header(b"\x93NUM").unwrap_err());
    assert!(matches!(e, NpyError::Magic { .. }));
}

#[test]
fn version_error_names_accepted_versions() {
    let mut b = b"\x93NUMPY".to_vec();
    b.push(9);
    b.push(0);
    b.extend_from_slice(&[0, 0]);
    let e = npy_err(npy::header(&b).unwrap_err());
    assert!(matches!(e, NpyError::Version { major: 9, minor: 0 }));
    let msg = e.to_string();
    assert!(msg.contains("1.0") && msg.contains("2.0") && msg.contains("3.0"));
}

#[test]
fn header_length_bounds_checked() {
    // Length field truncated.
    let mut b = b"\x93NUMPY".to_vec();
    b.push(1);
    b.push(0);
    b.push(7); // one of two u16 bytes
    let e = npy_err(npy::header(&b).unwrap_err());
    assert!(matches!(e, NpyError::Header { at: 8, .. }), "{e}");

    // Length overrunning the buffer — trusted nowhere.
    let mut b = b"\x93NUMPY".to_vec();
    b.push(1);
    b.push(0);
    b.extend_from_slice(&500u16.to_le_bytes());
    b.extend_from_slice(b"{'descr': '<f4'");
    let e = npy_err(npy::header(&b).unwrap_err());
    assert!(matches!(e, NpyError::Header { at: 8, .. }));
    assert!(e.to_string().contains("overruns"));
}

#[test]
fn grammar_violations_are_loud_with_offsets() {
    let cases: &[(&str, &str)] = &[
        ("['descr']", "expected '{'"),
        ("{'descrr': '<f4', }", "unknown key"),
        (
            "{'descr': '<f4', 'descr': '<f4', }",
            "duplicate key 'descr'",
        ),
        (
            "{'descr': '<f4', 'fortran_order': False, }",
            "missing key 'shape'",
        ),
        (
            "{'fortran_order': False, 'shape': (2,), }",
            "missing key 'descr'",
        ),
        (
            "{'descr': '<f4', 'fortran_order': Maybe, 'shape': (2,), }",
            "expected True or False",
        ),
        (
            "{'descr': '<f4', 'fortran_order': False, 'shape': [2, 3], }",
            "expected '('",
        ),
        (
            "{'descr': '<f4', 'fortran_order': False, 'shape': (3), }",
            "'(n,)'",
        ),
        (
            "{'descr': '<f4', 'fortran_order': False, 'shape': (x), }",
            "expected a shape extent",
        ),
        (
            "{'descr': '<f4', 'fortran_order': False, 'shape': \
             (99999999999999999999999999,), }",
            "overflows",
        ),
        (
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }garbage",
            "after the closing '}'",
        ),
        (
            "{'descr': '<f4' 'fortran_order': False}",
            "expected ',' or '}'",
        ),
        (
            "{'descr': 'a\\\\b', 'fortran_order': False, 'shape': (), }",
            "escape",
        ),
        ("{'descr': '<f4", "unterminated string"),
    ];
    for (dict, expect) in cases {
        let e = npy_err(npy::header(&npy_with_dict(dict)).unwrap_err());
        let msg = e.to_string();
        assert!(
            msg.contains(expect),
            "dict {dict:?}: expected {expect:?} in {msg:?}"
        );
        // The Header contract: a byte offset is always present.
        if matches!(e, NpyError::Header { .. }) {
            assert!(msg.contains("at byte"), "offset missing: {msg}");
        }
    }
}

#[test]
fn header_offsets_are_file_absolute() {
    // The dict starts at byte 10 in a v1 file; an error at its first
    // byte must report an offset >= 10, not a dict-relative 0.
    let e = npy_err(npy::header(&npy_with_dict("x")).unwrap_err());
    match e {
        NpyError::Header { at, .. } => assert_eq!(at, 10),
        other => panic!("expected Header, got {other:?}"),
    }
}

#[test]
fn structured_descr_is_a_dtype_error() {
    let dict = "{'descr': [('a', '<f4'), ('b', '<i4')], 'fortran_order': False, 'shape': (2,), }";
    let e = npy_err(npy::header(&npy_with_dict(dict)).unwrap_err());
    match &e {
        NpyError::Dtype { descr } => assert!(descr.starts_with('['), "{descr:?}"),
        other => panic!("expected Dtype, got {other:?}"),
    }
    assert!(e.to_string().contains("structured"));
}

#[test]
fn non_ascii_header_byte_rejected() {
    let mut b = npy_with_dict("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }");
    b[12] = 0xE9; // corrupt a byte inside the 'descr' key string
    let e = npy_err(npy::header(&b).unwrap_err());
    assert!(matches!(e, NpyError::Header { .. }), "{e}");
}

// ── The descr table ─────────────────────────────────────────────────────

#[test]
fn descr_table_covers_all_supported_types() {
    use fi::NpyDtype::*;
    let cases: &[(&str, fi::NpyDtype, bool, usize)] = &[
        ("<f4", F32, false, 4),
        ("<f8", F64, false, 8),
        ("<i4", I32, false, 4),
        ("<u4", U32, false, 4),
        ("<i8", I64, false, 8),
        ("<u8", U64, false, 8),
        ("<u2", U16, false, 2),
        ("<i2", I16, false, 2),
        ("|u1", U8, false, 1),
        ("|i1", I8, false, 1),
        ("<f2", F16, false, 2),
        ("|b1", Bool, false, 1),
        // Big-endian variants: same dtype, byteswap-on-load flag set.
        (">f4", F32, true, 4),
        (">f8", F64, true, 8),
        (">i4", I32, true, 4),
        (">u4", U32, true, 4),
        (">i8", I64, true, 8),
        (">u8", U64, true, 8),
        (">u2", U16, true, 2),
        (">i2", I16, true, 2),
        (">f2", F16, true, 2),
        // Byte order is meaningless at width 1 — numpy writes `|`, but
        // `<`/`>` from other writers are unambiguous.
        ("<u1", U8, false, 1),
        (">i1", I8, false, 1),
        ("|b1", Bool, false, 1),
    ];
    for &(s, dtype, big_endian, width) in cases {
        let d = fi::parse_descr(s).unwrap();
        assert_eq!(d, fi::Descr { dtype, big_endian }, "descr {s}");
        assert_eq!(d.dtype.width(), width, "descr {s}");
    }
}

#[test]
fn canonical_descrs_roundtrip() {
    use fi::NpyDtype::*;
    for dtype in [F32, F64, I32, U32, I64, U64, U8, I8, U16, I16, F16, Bool] {
        let d = fi::parse_descr(dtype.canonical_descr()).unwrap();
        assert_eq!(d.dtype, dtype);
        assert!(!d.big_endian);
    }
}

#[test]
fn descr_rejections_carry_reasons() {
    // `=` is its own taxonomy row: numpy never writes it.
    let e = fi::parse_descr("=f4").unwrap_err();
    assert!(matches!(e, NpyError::ByteOrder { .. }));
    assert!(e.to_string().contains("'='"));

    let cases: &[(&str, &str)] = &[
        ("|O", "pickle"),
        ("<c8", "complex"),
        ("<c16", "complex"),
        ("|S8", "string"),
        ("<U4", "string"), // unicode STRING — case distinguishes it from <u4
        ("|V10", "structured"),
        ("|f4", "not an element dtype"), // malformed order for a 4-byte type
        ("f4", "not an element dtype"),  // numpy always writes an order char
        ("<f3", "not an element dtype"),
        ("<x4", "not an element dtype"),
        ("", "not an element dtype"),
    ];
    for (s, reason) in cases {
        let e = fi::parse_descr(s).unwrap_err();
        assert!(matches!(e, NpyError::Dtype { .. }), "descr {s:?}: {e:?}");
        let msg = e.to_string();
        assert!(
            msg.contains(reason),
            "descr {s:?}: {reason:?} not in {msg:?}"
        );
        // The Dtype contract: the descr and the supported list are named.
        assert!(msg.contains(s) || s.is_empty(), "descr missing: {msg}");
        assert!(msg.contains("<f4"), "supported list missing: {msg}");
    }
}

// ── CRC-32 ──────────────────────────────────────────────────────────────

#[test]
fn crc32_reference_vectors() {
    assert_eq!(fi::crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(fi::crc32(b""), 0);
}

// ── ZIP container: write ────────────────────────────────────────────────

/// The reference stored archive for one entry `a.npy` holding `DATA`,
/// hand-built field by field from the ZIP spec.
fn reference_zip_bytes() -> Vec<u8> {
    let crc = fi::crc32(b"DATA");
    let mut b = Vec::new();
    // Local header.
    b.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
    b.extend_from_slice(&20u16.to_le_bytes()); // version needed
    b.extend_from_slice(&0u16.to_le_bytes()); // flags (ascii name)
    b.extend_from_slice(&0u16.to_le_bytes()); // method 0 = stored
    b.extend_from_slice(&0u16.to_le_bytes()); // time 00:00:00
    b.extend_from_slice(&0x0021u16.to_le_bytes()); // date 1980-01-01
    b.extend_from_slice(&crc.to_le_bytes());
    b.extend_from_slice(&4u32.to_le_bytes()); // compressed size
    b.extend_from_slice(&4u32.to_le_bytes()); // uncompressed size
    b.extend_from_slice(&5u16.to_le_bytes()); // name length
    b.extend_from_slice(&0u16.to_le_bytes()); // extra length
    b.extend_from_slice(b"a.npy");
    b.extend_from_slice(b"DATA");
    assert_eq!(b.len(), 39);
    // Central directory (one entry, at offset 39).
    b.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
    b.extend_from_slice(&20u16.to_le_bytes()); // version made by
    b.extend_from_slice(&20u16.to_le_bytes()); // version needed
    b.extend_from_slice(&0u16.to_le_bytes()); // flags
    b.extend_from_slice(&0u16.to_le_bytes()); // method
    b.extend_from_slice(&0u16.to_le_bytes()); // time
    b.extend_from_slice(&0x0021u16.to_le_bytes()); // date
    b.extend_from_slice(&crc.to_le_bytes());
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&5u16.to_le_bytes()); // name length
    b.extend_from_slice(&0u16.to_le_bytes()); // extra
    b.extend_from_slice(&0u16.to_le_bytes()); // comment
    b.extend_from_slice(&0u16.to_le_bytes()); // disk start
    b.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
    b.extend_from_slice(&0u32.to_le_bytes()); // external attrs
    b.extend_from_slice(&0u32.to_le_bytes()); // local offset
    b.extend_from_slice(b"a.npy");
    assert_eq!(b.len(), 39 + 51);
    // EOCD.
    b.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
    b.extend_from_slice(&0u16.to_le_bytes()); // this disk
    b.extend_from_slice(&0u16.to_le_bytes()); // CD disk
    b.extend_from_slice(&1u16.to_le_bytes()); // entries this disk
    b.extend_from_slice(&1u16.to_le_bytes()); // entries total
    b.extend_from_slice(&51u32.to_le_bytes()); // CD size
    b.extend_from_slice(&39u32.to_le_bytes()); // CD offset
    b.extend_from_slice(&0u16.to_le_bytes()); // comment length
    b
}

#[test]
fn write_stored_matches_reference_bytes() {
    assert_eq!(
        fi::write_stored(&[("a.npy", b"DATA")]).unwrap(),
        reference_zip_bytes()
    );
}

#[test]
fn zip_roundtrip_preserves_order_names_and_bytes() {
    let entries: &[(&str, &[u8])] = &[
        ("weights.npy", &[1, 2, 3, 4, 5]),
        ("π_labels.npy", &[0xFF, 0x00, 0x80]), // unicode name
        ("empty.npy", &[]),
    ];
    let bytes = fi::write_stored(entries).unwrap();
    let read = fi::read_entries(&bytes).unwrap();
    assert_eq!(read.len(), 3);
    for ((name, data), entry) in entries.iter().zip(&read) {
        assert_eq!(&entry.name, name);
        assert_eq!(&entry.data, data);
    }
}

#[test]
fn zip_bytes_are_deterministic() {
    let entries: &[(&str, &[u8])] = &[("a.npy", b"one"), ("b.npy", b"two")];
    assert_eq!(
        fi::write_stored(entries).unwrap(),
        fi::write_stored(entries).unwrap()
    );
}

#[test]
fn zip_empty_archive_roundtrips() {
    let bytes = fi::write_stored(&[]).unwrap();
    assert_eq!(bytes.len(), 22); // bare EOCD
    assert_eq!(fi::read_entries(&bytes).unwrap(), Vec::<fi::Entry>::new());
}

#[test]
fn zip_write_duplicate_name_is_loud() {
    let e = fi::write_stored(&[("a.npy", b"x"), ("a.npy", b"y")]).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("a.npy") && msg.contains("duplicate"), "{msg}");
}

#[test]
fn zip_write_oversized_name_is_loud() {
    let name = format!("{}.npy", "a".repeat(70_000));
    let e = fi::write_stored(&[(name.as_str(), b"x")]).unwrap_err();
    assert!(e.to_string().contains("65535"), "{e}");
}

// ── ZIP container: hostile input ────────────────────────────────────────

/// Hand-build an archive with full control over method / sizes / crc —
/// the harness for deflate entries and for lying metadata.
fn craft_zip(entries: &[(&str, u16, &[u8], u32, u32)], cd_extra: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    let mut cd = Vec::new();
    for (name, method, comp, uncomp_size, crc) in entries {
        let offset = b.len() as u32;
        b.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
        b.extend_from_slice(&20u16.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&method.to_le_bytes());
        b.extend_from_slice(&[0, 0, 0x21, 0]);
        b.extend_from_slice(&crc.to_le_bytes());
        b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        b.extend_from_slice(&uncomp_size.to_le_bytes());
        b.extend_from_slice(&(name.len() as u16).to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(comp);

        cd.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&method.to_le_bytes());
        cd.extend_from_slice(&[0, 0, 0x21, 0]);
        cd.extend_from_slice(&crc.to_le_bytes());
        cd.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        cd.extend_from_slice(&uncomp_size.to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&(cd_extra.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u32.to_le_bytes());
        cd.extend_from_slice(&offset.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
        cd.extend_from_slice(cd_extra);
    }
    let cd_offset = b.len() as u32;
    b.extend_from_slice(&cd);
    b.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    b.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    b.extend_from_slice(&(cd.len() as u32).to_le_bytes());
    b.extend_from_slice(&cd_offset.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b
}

#[test]
fn zip_not_an_archive() {
    let e = fi::read_entries(b"hello, not a zip").unwrap_err();
    assert!(e.to_string().contains("end-of-central-directory"), "{e}");
}

#[test]
fn zip_crc_mismatch_names_the_entry() {
    let mut bytes = fi::write_stored(&[("a.npy", b"DATA")]).unwrap();
    bytes[35] ^= 0xFF; // first data byte (local header is 30 + 5 name)
    let e = fi::read_entries(&bytes).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("a.npy") && msg.contains("CRC-32"), "{msg}");
}

#[test]
fn zip_local_cd_name_disagreement_is_loud() {
    let mut bytes = fi::write_stored(&[("a.npy", b"DATA")]).unwrap();
    bytes[30] = b'b'; // local header name, CD untouched
    let e = fi::read_entries(&bytes).unwrap_err();
    assert!(e.to_string().contains("disagrees"), "{e}");
}

#[test]
fn zip_non_npy_entry_is_loud() {
    // The writer is name-agnostic; the npz READER refuses foreign ZIPs.
    let bytes = fi::write_stored(&[("notes.txt", b"hi")]).unwrap();
    let e = fi::read_entries(&bytes).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("notes.txt") && msg.contains("npz"), "{msg}");
}

#[test]
fn zip_duplicate_cd_names_are_loud() {
    let bytes = craft_zip(
        &[
            ("a.npy", 0, b"x", 1, fi::crc32(b"x")),
            ("a.npy", 0, b"y", 1, fi::crc32(b"y")),
        ],
        &[],
    );
    let e = fi::read_entries(&bytes).unwrap_err();
    assert!(e.to_string().contains("duplicate"), "{e}");
}

#[test]
fn zip_unsupported_method_is_loud() {
    let bytes = craft_zip(&[("a.npy", 9, b"x", 1, fi::crc32(b"x"))], &[]);
    let e = fi::read_entries(&bytes).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("method 9") && msg.contains("a.npy"), "{msg}");
}

#[test]
fn zip_stored_size_lie_is_loud() {
    let bytes = craft_zip(&[("a.npy", 0, b"xx", 7, fi::crc32(b"xx"))], &[]);
    let e = fi::read_entries(&bytes).unwrap_err();
    assert!(e.to_string().contains("sizes disagree"), "{e}");
}

#[test]
fn zip64_markers_are_refused_loudly() {
    // Entry count sentinel in the EOCD.
    let good = fi::write_stored(&[("a.npy", b"DATA")]).unwrap();
    let eocd = good.len() - 22;
    let mut bad = good.clone();
    bad[eocd + 8..eocd + 12].fill(0xFF); // both entry-count fields
    let e = fi::read_entries(&bad).unwrap_err();
    assert!(e.to_string().contains("ZIP64"), "{e}");

    // CD offset sentinel.
    let mut bad = good.clone();
    bad[eocd + 16..eocd + 20].fill(0xFF);
    let e = fi::read_entries(&bad).unwrap_err();
    assert!(e.to_string().contains("ZIP64"), "{e}");

    // A ZIP64 extra field (id 0x0001) on a CD entry.
    let extra = [0x01, 0x00, 0x04, 0x00, 0xAA, 0xBB, 0xCC, 0xDD];
    let bytes = craft_zip(&[("a.npy", 0, b"x", 1, fi::crc32(b"x"))], &extra);
    let e = fi::read_entries(&bytes).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("ZIP64") && msg.contains("a.npy"), "{msg}");
}

#[test]
fn zip_multi_disk_is_refused() {
    let good = fi::write_stored(&[("a.npy", b"DATA")]).unwrap();
    let eocd = good.len() - 22;
    let mut bad = good;
    bad[eocd + 4] = 1; // "this disk" = 1
    let e = fi::read_entries(&bad).unwrap_err();
    assert!(e.to_string().contains("multi-disk"), "{e}");
}

#[test]
fn zip_cd_count_lie_is_loud() {
    let good = fi::write_stored(&[("a.npy", b"DATA")]).unwrap();
    let eocd = good.len() - 22;
    let mut bad = good;
    bad[eocd + 8] = 2; // claims two entries, CD holds one
    bad[eocd + 10] = 2;
    let e = fi::read_entries(&bad).unwrap_err();
    assert!(e.to_string().contains("central"), "{e}");
}

#[test]
fn zip_truncation_fuzz_never_panics() {
    let bytes = fi::write_stored(&[("a.npy", b"DATA"), ("b.npy", b"MORE-DATA")]).unwrap();
    for cut in 0..bytes.len() {
        assert!(
            fi::read_entries(&bytes[..cut]).is_err(),
            "prefix of {cut} bytes parsed as a whole archive"
        );
    }
}

// ── Inflate: RFC 1951 vectors ───────────────────────────────────────────

/// An LSB-first bit writer for hand-assembling deflate streams in the
/// RFC's own vocabulary.
struct Bw {
    bytes: Vec<u8>,
    bit: u32,
}

impl Bw {
    fn new() -> Self {
        Bw {
            bytes: Vec::new(),
            bit: 0,
        }
    }
    /// Push `n` bits LSB-first (header fields, extra bits).
    fn lsb(&mut self, v: u32, n: u32) {
        for i in 0..n {
            if self.bit == 0 {
                self.bytes.push(0);
            }
            let byte = self.bytes.last_mut().unwrap();
            *byte |= (((v >> i) & 1) as u8) << self.bit;
            self.bit = (self.bit + 1) % 8;
        }
    }
    /// Push a Huffman code MSB-first (how RFC 1951 packs codes).
    fn code(&mut self, code: u32, n: u32) {
        for i in (0..n).rev() {
            self.lsb((code >> i) & 1, 1);
        }
    }
    /// The fixed-Huffman literal/length code for a symbol.
    fn fixed_lit(&mut self, sym: u32) {
        match sym {
            0..=143 => self.code(0x30 + sym, 8),
            144..=255 => self.code(0x190 + (sym - 144), 9),
            256..=279 => self.code(sym - 256, 7),
            _ => self.code(0xC0 + (sym - 280), 8),
        }
    }
}

#[test]
fn inflate_stored_blocks() {
    // Two stored blocks (BFINAL=0 then BFINAL=1) exercise the block loop.
    let mut v = vec![0x00, 3, 0, !3u8, 0xFF];
    v.extend_from_slice(b"abc");
    v.extend_from_slice(&[0x01, 2, 0, !2u8, 0xFF]);
    v.extend_from_slice(b"de");
    assert_eq!(fi::inflate(&v, 5).unwrap(), b"abcde");
}

#[test]
fn inflate_fixed_empty() {
    // The classic 2-byte empty stream: BFINAL=1, BTYPE=01, EOB.
    assert_eq!(fi::inflate(&[0x03, 0x00], 0).unwrap(), b"");
}

#[test]
fn inflate_fixed_abc() {
    // zlib's own raw-deflate encoding of b"abc" (fixed Huffman, no
    // matches) — the interchange ground truth vector.
    assert_eq!(
        fi::inflate(&[0x4b, 0x4c, 0x4a, 0x06, 0x00], 3).unwrap(),
        b"abc"
    );
}

#[test]
fn inflate_fixed_window_copy() {
    // 'a', then <len 6, dist 1> — an overlapping copy: "aaaaaaa".
    let mut w = Bw::new();
    w.lsb(1, 1); // BFINAL
    w.lsb(1, 2); // BTYPE fixed
    w.fixed_lit(97); // 'a'
    w.fixed_lit(260); // length code: base 6, 0 extra
    w.code(0, 5); // distance code 0 → dist 1
    w.fixed_lit(256); // EOB
    assert_eq!(fi::inflate(&w.bytes, 7).unwrap(), b"aaaaaaa");
}

#[test]
fn inflate_fixed_nine_bit_literal_and_length_extra() {
    // Byte 200 takes the 9-bit literal row; <len 11, dist 2> takes a
    // length code with an extra bit. Output: [200, 97] then 11 bytes
    // alternating from distance 2.
    let mut w = Bw::new();
    w.lsb(1, 1);
    w.lsb(1, 2);
    w.fixed_lit(200);
    w.fixed_lit(97);
    w.fixed_lit(265); // length base 11, 1 extra bit
    w.lsb(0, 1); // extra → len 11
    w.code(1, 5); // distance code 1 → dist 2
    w.fixed_lit(256);
    let mut expect = vec![200u8, 97];
    for i in 0..11 {
        expect.push(expect[i]);
    }
    assert_eq!(fi::inflate(&w.bytes, 13).unwrap(), expect);
}

/// A minimal dynamic-Huffman block for the byte `'X'`: two literal
/// codes ('X' and EOB, both length 1), an all-zero distance alphabet,
/// and 18-coded zero runs in the code-length stream.
fn dynamic_minimal_x() -> Vec<u8> {
    let mut w = Bw::new();
    w.lsb(1, 1); // BFINAL
    w.lsb(2, 2); // BTYPE dynamic
    w.lsb(0, 5); // HLIT → 257
    w.lsb(0, 5); // HDIST → 1
    w.lsb(14, 4); // HCLEN → 18 entries
    // Code-length-code lengths in CL_ORDER
    // [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]:
    // 18→1, 0→2, 1→2, rest 0. Canonical: 18=0, 0=10, 1=11.
    let cl = [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2];
    for l in cl {
        w.lsb(l, 3);
    }
    // 257 literal lengths: 88 zeros, len 1 at 'X'(88), 167 zeros,
    // len 1 at 256. Then 1 distance length: zero.
    w.code(0, 1); // 18: run of zeros
    w.lsb(77, 7); // 11 + 77 = 88 zeros
    w.code(3, 2); // symbol 1 → lengths['X'] = 1
    w.code(0, 1);
    w.lsb(127, 7); // 138 zeros (89..=226)
    w.code(0, 1);
    w.lsb(18, 7); // 29 zeros (227..=255)
    w.code(3, 2); // symbol 1 → lengths[256] = 1
    w.code(2, 2); // symbol 0 → the single distance length is 0
    // Literal code: 'X'=0, EOB=1 (canonical over two length-1 codes).
    w.code(0, 1); // 'X'
    w.code(1, 1); // EOB
    w.bytes
}

#[test]
fn inflate_dynamic_minimal() {
    assert_eq!(fi::inflate(&dynamic_minimal_x(), 1).unwrap(), b"X");
}

#[test]
fn inflate_dynamic_full_with_repeats_and_degenerate_distance() {
    // "XXXXXX" = 'X' + <len 5, dist 1>. Exercises every code-length
    // meta-symbol (16: repeat, 17: short zero run, 18: long zero run),
    // and a single-code (incomplete but legal) distance table.
    //
    // Literal lengths: 'X'(88)→1; 256..=259→3 (256=EOB, 259=len-5).
    // Canonical: 'X'=0; 256=100, 257=101, 258=110, 259=111.
    // CL lengths: 1→2, 3→2, 18→2, 16→3, 17→3.
    // Canonical CL: 1=00, 3=01, 18=10, 16=110, 17=111.
    let mut w = Bw::new();
    w.lsb(1, 1); // BFINAL
    w.lsb(2, 2); // BTYPE dynamic
    w.lsb(3, 5); // HLIT → 260
    w.lsb(0, 5); // HDIST → 1
    w.lsb(14, 4); // HCLEN → 18 entries
    // CL_ORDER: [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]
    let cl = [3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2];
    for l in cl {
        w.lsb(l, 3);
    }
    // 260 literal lengths:
    w.code(2, 2); // 18
    w.lsb(77, 7); // 88 zeros (0..=87)
    w.code(0, 2); // symbol 1 → lengths[88] = 1
    w.code(2, 2); // 18
    w.lsb(127, 7); // 138 zeros (89..=226)
    w.code(7, 3); // 17
    w.lsb(7, 3); // 10 zeros (227..=236)
    w.code(2, 2); // 18
    w.lsb(8, 7); // 19 zeros (237..=255)
    w.code(1, 2); // symbol 3 → lengths[256] = 3
    w.code(6, 3); // 16: repeat previous length…
    w.lsb(0, 2); // …3 times → 257, 258, 259 all length 3
    // 1 distance length: symbol 1 → the lone distance code (dist 1).
    w.code(0, 2);
    // Data: 'X', <len 5 (259), dist 1>, EOB.
    w.code(0, 1); // 'X'
    w.code(7, 3); // 259 → length 5
    w.code(0, 1); // the single distance code → dist 1
    w.code(4, 3); // 256 EOB
    assert_eq!(fi::inflate(&w.bytes, 6).unwrap(), b"XXXXXX");
}

#[test]
fn inflate_ignores_trailing_bytes() {
    // The container's compressed-size field delimits the stream; bytes
    // after the final block are ignored (CRC + size checks upstream
    // catch real corruption).
    assert_eq!(
        fi::inflate(&[0x4b, 0x4c, 0x4a, 0x06, 0x00, 0xFF], 3).unwrap(),
        b"abc"
    );
}

// ── Inflate: hostile input ──────────────────────────────────────────────

#[test]
fn inflate_reserved_block_type() {
    let e = fi::inflate(&[0x07], 8).unwrap_err();
    assert!(e.contains("reserved"), "{e}");
}

#[test]
fn inflate_stored_len_nlen_mismatch() {
    let e = fi::inflate(&[0x01, 3, 0, 0, 0], 8).unwrap_err();
    assert!(e.contains("LEN/NLEN"), "{e}");
}

#[test]
fn inflate_distance_before_start() {
    // 'a', then <len 3, dist 2> with only 1 byte of output so far.
    let mut w = Bw::new();
    w.lsb(1, 1);
    w.lsb(1, 2);
    w.fixed_lit(97);
    w.fixed_lit(257); // len 3
    w.code(1, 5); // dist 2
    w.fixed_lit(256);
    let e = fi::inflate(&w.bytes, 8).unwrap_err();
    assert!(e.contains("before the start"), "{e}");
}

#[test]
fn inflate_bomb_guard_stored() {
    // A stored block holding more than the declared uncompressed size.
    let mut v = vec![0x01, 10, 0, !10u8, 0xFF];
    v.extend_from_slice(&[0u8; 10]);
    let e = fi::inflate(&v, 5).unwrap_err();
    assert!(e.contains("zip-bomb"), "{e}");
}

#[test]
fn inflate_bomb_guard_window_copy() {
    // The "aaaaaaa" vector declared as 3 bytes: inflates past the claim.
    let mut w = Bw::new();
    w.lsb(1, 1);
    w.lsb(1, 2);
    w.fixed_lit(97);
    w.fixed_lit(260);
    w.code(0, 5);
    w.fixed_lit(256);
    let e = fi::inflate(&w.bytes, 3).unwrap_err();
    assert!(e.contains("zip-bomb"), "{e}");
}

#[test]
fn inflate_oversubscribed_code_lengths() {
    // Four length-1 code-length codes: Kraft sum 2.0 — impossible.
    let mut w = Bw::new();
    w.lsb(1, 1);
    w.lsb(2, 2);
    w.lsb(0, 5);
    w.lsb(0, 5);
    w.lsb(0, 4); // HCLEN → 4 entries: 16, 17, 18, 0
    for _ in 0..4 {
        w.lsb(1, 3);
    }
    let e = fi::inflate(&w.bytes, 8).unwrap_err();
    assert!(e.contains("oversubscribed"), "{e}");
}

#[test]
fn inflate_hole_in_incomplete_code() {
    // One length-1 CL code (for symbol 18): reading bit 1 walks into
    // the unassigned half of the code space.
    let mut w = Bw::new();
    w.lsb(1, 1);
    w.lsb(2, 2);
    w.lsb(0, 5);
    w.lsb(0, 5);
    w.lsb(0, 4); // 4 entries: 16, 17, 18, 0
    w.lsb(0, 3); // 16 → 0
    w.lsb(0, 3); // 17 → 0
    w.lsb(1, 3); // 18 → 1
    w.lsb(0, 3); // 0 → 0
    w.lsb(0x7FFF, 15); // decode walks 1-bits into the hole, full depth
    let e = fi::inflate(&w.bytes, 8).unwrap_err();
    assert!(e.contains("invalid Huffman code"), "{e}");
}

#[test]
fn inflate_repeat_overrun_and_orphan_repeat() {
    // 18 with max extra overruns a 258-length table near its end.
    let mut base = Bw::new();
    base.lsb(1, 1);
    base.lsb(2, 2);
    base.lsb(0, 5);
    base.lsb(0, 5);
    base.lsb(14, 4);
    let cl = [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2];
    for l in cl {
        base.lsb(l, 3);
    }
    let mut w = Bw::new();
    w.bytes = base.bytes.clone();
    w.bit = base.bit;
    w.code(0, 1); // 18
    w.lsb(127, 7); // 138 zeros
    w.code(0, 1); // 18
    w.lsb(127, 7); // 138 more → 276 > 258
    let e = fi::inflate(&w.bytes, 8).unwrap_err();
    assert!(e.contains("overruns"), "{e}");

    // Symbol 16 as the very first code length has nothing to repeat.
    // CL table: 16→1, 0→2, 1→2 (canonical: 16=0, 0=10, 1=11).
    let mut w = Bw::new();
    w.lsb(1, 1);
    w.lsb(2, 2);
    w.lsb(0, 5);
    w.lsb(0, 5);
    w.lsb(14, 4);
    let cl = [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2];
    for l in cl {
        w.lsb(l, 3);
    }
    w.code(0, 1); // 16 first
    w.lsb(0, 2);
    let e = fi::inflate(&w.bytes, 8).unwrap_err();
    assert!(e.contains("no previous"), "{e}");
}

#[test]
fn inflate_missing_end_of_block_code() {
    // Two literal codes, but length 0 at symbol 256: the stream could
    // never terminate — rejected at table time.
    let mut w = Bw::new();
    w.lsb(1, 1);
    w.lsb(2, 2);
    w.lsb(0, 5);
    w.lsb(0, 5);
    w.lsb(14, 4);
    let cl = [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2];
    for l in cl {
        w.lsb(l, 3);
    }
    w.code(3, 2); // lengths[0] = 1
    w.code(3, 2); // lengths[1] = 1
    w.code(0, 1); // 18
    w.lsb(127, 7); // 138 zeros
    w.code(0, 1); // 18
    w.lsb(106, 7); // 117 zeros → 257 lengths, 256 still zero
    w.code(2, 2); // distance length: 0
    let e = fi::inflate(&w.bytes, 8).unwrap_err();
    assert!(e.contains("end-of-block"), "{e}");
}

#[test]
fn inflate_truncation_fuzz_never_panics() {
    let vectors: Vec<(Vec<u8>, usize)> = vec![
        (vec![0x4b, 0x4c, 0x4a, 0x06, 0x00], 3),
        (
            {
                let mut v = vec![0x01, 3, 0, !3u8, 0xFF];
                v.extend_from_slice(b"abc");
                v
            },
            3,
        ),
        (dynamic_minimal_x(), 1),
    ];
    for (v, declared) in vectors {
        for cut in 0..v.len() {
            assert!(
                fi::inflate(&v[..cut], declared).is_err(),
                "prefix of {cut}/{} bytes inflated cleanly",
                v.len()
            );
        }
    }
}

// ── Deflate entries inside the container ────────────────────────────────

#[test]
fn zip_deflate_entry_roundtrips() {
    let comp = [0x4b, 0x4c, 0x4a, 0x06, 0x00]; // deflate of b"abc"
    let bytes = craft_zip(&[("a.npy", 8, &comp, 3, fi::crc32(b"abc"))], &[]);
    let read = fi::read_entries(&bytes).unwrap();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].name, "a.npy");
    assert_eq!(read[0].data, b"abc");
}

#[test]
fn zip_deflate_bomb_declaration_is_loud() {
    // CD declares 2 bytes; the stream holds 3 — the guard fires.
    let comp = [0x4b, 0x4c, 0x4a, 0x06, 0x00];
    let bytes = craft_zip(&[("a.npy", 8, &comp, 2, fi::crc32(b"ab"))], &[]);
    let e = fi::read_entries(&bytes).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("a.npy") && msg.contains("zip-bomb"), "{msg}");
}

#[test]
fn zip_deflate_short_stream_is_loud() {
    // CD declares 10 bytes; the stream inflates to 3.
    let comp = [0x4b, 0x4c, 0x4a, 0x06, 0x00];
    let bytes = craft_zip(&[("a.npy", 8, &comp, 10, fi::crc32(b"abc"))], &[]);
    let e = fi::read_entries(&bytes).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("declares 10"), "{msg}");
}

#[test]
fn zip_deflate_garbage_stream_is_loud() {
    let comp = [0xFF, 0xFF, 0xFF, 0xFF];
    let bytes = craft_zip(&[("a.npy", 8, &comp, 4, 0)], &[]);
    let e = fi::read_entries(&bytes).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("a.npy") && msg.contains("deflate"), "{msg}");
}

// ── Error taxonomy: the message contract ────────────────────────────────

#[test]
fn error_messages_carry_the_promised_context() {
    // Variants constructed by the typed layer land now with their
    // Display contracts — entry / offset / workaround always present.
    let cases: Vec<(NpyError, Vec<&str>)> = vec![
        (NpyError::BoolValue { at: 42 }, vec!["42", "corrupt"]),
        (
            NpyError::DtypeMismatch {
                file: "<f8".into(),
                requested: "<f4".into(),
            },
            vec!["<f8", "<f4", "load_dyn"],
        ),
        (
            NpyError::EmptyShape { shape: vec![0, 4] },
            vec!["[0, 4]", "zero"],
        ),
        (
            NpyError::DataLength {
                expected: 24,
                got: 20,
            },
            vec!["24", "20"],
        ),
        (
            NpyError::Zip {
                entry: Some("w.npy".into()),
                what: "CRC-32 mismatch".into(),
            },
            vec!["w.npy", "CRC-32"],
        ),
        (
            NpyError::Zip {
                entry: None,
                what: "no end-of-central-directory record found".into(),
            },
            vec!["archive"],
        ),
    ];
    for (e, pieces) in cases {
        let msg = e.to_string();
        for piece in pieces {
            assert!(msg.contains(piece), "{piece:?} missing from {msg:?}");
        }
    }
}

#[test]
fn npy_error_wires_into_array_error() {
    let e: ArrayError = NpyError::Version { major: 4, minor: 2 }.into();
    assert!(matches!(e, ArrayError::Npy(NpyError::Version { .. })));
    // The Display passthrough: NpyError messages are self-contained.
    assert!(e.to_string().contains("npy version 4.2"));
}
