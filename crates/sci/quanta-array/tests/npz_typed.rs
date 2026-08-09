//! Typed-layer tests for npz interop: mixed-dtype archives through
//! `npz::save_named` / `npz::load_named` — entry order, unicode names,
//! the `.npy` suffix convention, deflate reads, and the per-entry error
//! contract. The raw container (ZIP grammar, CRC, inflate vectors) is
//! covered by `npy_format.rs`.
//!
//! numpy-generated npz fixture tests at the bottom SKIP LOUDLY while the
//! fixtures are absent (see `tests/fixtures/npy/gen_fixtures.py`).

use quanta_array::format_internals as fi;
use quanta_array::npy::{self, NpyArray};
use quanta_array::npz;
use quanta_array::{Array, ArrayError};

/// The device these tests run on: the real GPU under a hardware backend
/// feature (metal / vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

fn err_msg(e: ArrayError) -> String {
    e.to_string()
}

// ── Round-trip: mixed dtypes, order, unicode names ──────────────────────

#[test]
fn mixed_dtype_archive_roundtrips_in_order() {
    let g = gpu();
    let entries = vec![
        (
            "poids".to_string(),
            NpyArray::from(Array::from_slice(&g, &[0.5f32, -1.5, 2.0, 3.5], &[2, 2]).unwrap()),
        ),
        (
            "étiquettes π".to_string(), // unicode name (UTF-8 flagged in the ZIP)
            NpyArray::from(Array::from_slice(&g, &[7u64, 8, u64::MAX], &[3]).unwrap()),
        ),
        (
            "mask".to_string(),
            NpyArray::from(Array::from_slice(&g, &[1u8, 0, 1, 1], &[4]).unwrap()),
        ),
        (
            "δ".to_string(),
            NpyArray::from(Array::from_slice(&g, &[-300i16, 300], &[2]).unwrap()),
        ),
    ];
    let bytes = npz::save_named(&entries).unwrap();
    // Deterministic bytes: the fixed-timestamp guarantee.
    assert_eq!(bytes, npz::save_named(&entries).unwrap());

    let loaded = npz::load_named(&g, &bytes).unwrap();
    let names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["poids", "étiquettes π", "mask", "δ"]);

    let dtypes: Vec<&str> = loaded.iter().map(|(_, a)| a.dtype()).collect();
    assert_eq!(dtypes, vec!["<f4", "<u8", "|u1", "<i2"]);

    match &loaded[0].1 {
        NpyArray::F32(a) => {
            assert_eq!(a.shape(), &[2, 2]);
            assert_eq!(a.to_vec().unwrap(), vec![0.5, -1.5, 2.0, 3.5]);
        }
        other => panic!("expected F32, got {other:?}"),
    }
    match &loaded[1].1 {
        NpyArray::U64(a) => assert_eq!(a.to_vec().unwrap(), vec![7, 8, u64::MAX]),
        other => panic!("expected U64, got {other:?}"),
    }
    match &loaded[2].1 {
        NpyArray::U8(a) => assert_eq!(a.to_vec().unwrap(), vec![1, 0, 1, 1]),
        other => panic!("expected U8, got {other:?}"),
    }
    match &loaded[3].1 {
        NpyArray::I16(a) => assert_eq!(a.to_vec().unwrap(), vec![-300, 300]),
        other => panic!("expected I16, got {other:?}"),
    }
}

#[test]
fn rank0_arrays_travel_through_npz() {
    let g = gpu();
    let entries = vec![(
        "scalaire".to_string(),
        NpyArray::from(Array::from_slice(&g, &[42i32], &[]).unwrap()),
    )];
    let bytes = npz::save_named(&entries).unwrap();
    let loaded = npz::load_named(&g, &bytes).unwrap();
    assert_eq!(loaded[0].0, "scalaire");
    assert_eq!(loaded[0].1.shape(), &[] as &[usize]);
    match &loaded[0].1 {
        NpyArray::I32(a) => assert_eq!(a.to_vec().unwrap(), vec![42]),
        other => panic!("expected I32, got {other:?}"),
    }
}

#[test]
fn empty_archive_roundtrips() {
    let g = gpu();
    let bytes = npz::save_named(&[]).unwrap();
    assert!(npz::load_named(&g, &bytes).unwrap().is_empty());
}

// ── The .npy suffix convention ──────────────────────────────────────────

#[test]
fn entry_names_carry_the_npy_suffix_in_the_archive() {
    let g = gpu();
    let entries = vec![(
        "weights".to_string(),
        NpyArray::from(Array::from_slice(&g, &[1.0f32], &[1]).unwrap()),
    )];
    let bytes = npz::save_named(&entries).unwrap();
    // The raw container holds `<name>.npy` (numpy's convention), and
    // each entry's payload is a well-formed npy file.
    let raw = fi::read_entries(&bytes).unwrap();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0].name, "weights.npy");
    assert_eq!(npy::header(&raw[0].data).unwrap().descr, "<f4");
}

// ── Error contract ──────────────────────────────────────────────────────

#[test]
fn duplicate_names_are_loud() {
    let g = gpu();
    let a = || NpyArray::from(Array::from_slice(&g, &[1.0f32], &[1]).unwrap());
    let entries = vec![("x".to_string(), a()), ("x".to_string(), a())];
    let msg = err_msg(npz::save_named(&entries).unwrap_err());
    assert!(msg.contains("duplicate") && msg.contains("x.npy"), "{msg}");
}

#[test]
fn foreign_zip_entry_is_loud_through_load_named() {
    let g = gpu();
    let bytes = fi::write_stored(&[("notes.txt", b"hello")]).unwrap();
    let msg = err_msg(npz::load_named(&g, &bytes).unwrap_err());
    assert!(msg.contains("notes.txt") && msg.contains("npz"), "{msg}");
}

#[test]
fn entry_with_undecodable_npy_payload_names_the_entry() {
    let g = gpu();
    let bytes = fi::write_stored(&[("bad.npy", b"junk, not an npy")]).unwrap();
    let msg = err_msg(npz::load_named(&g, &bytes).unwrap_err());
    assert!(
        msg.contains("bad.npy") && msg.contains("not an npy file"),
        "{msg}"
    );
}

#[test]
fn entry_with_unsupported_dtype_names_the_entry_and_reason() {
    let g = gpu();
    // A structurally valid npy payload with a descr the matrix excludes.
    let mut payload = Vec::new();
    let dict = "{'descr': '<c8', 'fortran_order': False, 'shape': (1,), }";
    payload.extend_from_slice(b"\x93NUMPY\x01\x00");
    payload.extend_from_slice(&((dict.len() + 1) as u16).to_le_bytes());
    payload.extend_from_slice(dict.as_bytes());
    payload.push(b'\n');
    payload.extend_from_slice(&[0; 8]);
    let bytes = fi::write_stored(&[("complexe.npy", &payload)]).unwrap();
    let msg = err_msg(npz::load_named(&g, &bytes).unwrap_err());
    assert!(
        msg.contains("complexe.npy") && msg.contains("complex"),
        "{msg}"
    );
}

// ── Deflate reads (the savez_compressed container class) ────────────────

/// Wrap bytes as a single stored-mode RFC 1951 deflate block — a valid
/// method-8 stream any inflater must accept (what "stored → deflate"
/// means: the payload is uncompressed, the framing is deflate).
fn deflate_stored(data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= u16::MAX as usize);
    let len = data.len() as u16;
    let mut v = vec![0x01]; // BFINAL=1, BTYPE=00 (stored)
    v.extend_from_slice(&len.to_le_bytes());
    v.extend_from_slice(&(!len).to_le_bytes());
    v.extend_from_slice(data);
    v
}

/// Hand-build an archive whose entries use the given method — the
/// deflate-npz fixture builder (mirrors the format suite's `craft_zip`).
fn craft_zip(entries: &[(&str, u16, &[u8], &[u8])]) -> Vec<u8> {
    let mut b = Vec::new();
    let mut cd = Vec::new();
    for (name, method, comp, plain) in entries {
        let crc = fi::crc32(plain);
        let offset = b.len() as u32;
        b.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
        b.extend_from_slice(&20u16.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&method.to_le_bytes());
        b.extend_from_slice(&[0, 0, 0x21, 0]);
        b.extend_from_slice(&crc.to_le_bytes());
        b.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        b.extend_from_slice(&(plain.len() as u32).to_le_bytes());
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
        cd.extend_from_slice(&(plain.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&[0; 12]);
        cd.extend_from_slice(&offset.to_le_bytes());
        cd.extend_from_slice(name.as_bytes());
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
fn deflate_entries_load_through_load_named() {
    let g = gpu();
    let w = Array::from_slice(&g, &[1.5f32, -2.5, 0.25], &[3]).unwrap();
    let m = Array::from_slice(&g, &[1u8, 2, 3, 4], &[2, 2]).unwrap();
    let w_npy = npy::save(&w).unwrap();
    let m_npy = npy::save(&m).unwrap();
    let (w_comp, m_comp) = (deflate_stored(&w_npy), deflate_stored(&m_npy));
    let bytes = craft_zip(&[("w.npy", 8, &w_comp, &w_npy), ("m.npy", 8, &m_comp, &m_npy)]);
    let loaded = npz::load_named(&g, &bytes).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].0, "w");
    match &loaded[0].1 {
        NpyArray::F32(a) => assert_eq!(a.to_vec().unwrap(), vec![1.5, -2.5, 0.25]),
        other => panic!("expected F32, got {other:?}"),
    }
    assert_eq!(loaded[1].0, "m");
    match &loaded[1].1 {
        NpyArray::U8(a) => {
            assert_eq!(a.shape(), &[2, 2]);
            assert_eq!(a.to_vec().unwrap(), vec![1, 2, 3, 4]);
        }
        other => panic!("expected U8, got {other:?}"),
    }
}

// ── numpy-generated fixtures (arm once gen_fixtures.py has run) ─────────

/// Read a checked-in numpy fixture, or skip the test LOUDLY while the
/// maintainer step (running `gen_fixtures.py` with the pinned numpy) is
/// pending.
fn fixture(name: &str) -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npy")
        .join(name);
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!(
                "SKIPPED: numpy fixture {name} is absent — run \
                 tests/fixtures/npy/gen_fixtures.py with the pinned numpy and commit \
                 the outputs to arm this test"
            );
            None
        }
    }
}

/// The three arrays `gen_fixtures.py` puts in both containers, checked
/// by name against a loaded archive.
fn check_fixture_arrays(loaded: &[(String, NpyArray)]) {
    let weights: Vec<f32> = (0..12).map(|i| i as f32 / 8.0).collect();
    match loaded.iter().find(|(n, _)| n == "weights").map(|(_, a)| a) {
        Some(NpyArray::F32(a)) => {
            assert_eq!(a.shape(), &[3, 4]);
            assert_eq!(a.to_vec().unwrap(), weights);
        }
        other => panic!("weights: expected F32, got {other:?}"),
    }
    match loaded.iter().find(|(n, _)| n == "labels").map(|(_, a)| a) {
        Some(NpyArray::U64(a)) => assert_eq!(a.to_vec().unwrap(), vec![0, 1, 2]),
        other => panic!("labels: expected U64, got {other:?}"),
    }
    match loaded.iter().find(|(n, _)| n == "mask").map(|(_, a)| a) {
        Some(NpyArray::U8(a)) => assert_eq!(a.to_vec().unwrap(), vec![1, 0, 1]),
        other => panic!("mask: expected U8, got {other:?}"),
    }
}

#[test]
fn numpy_fixture_stored_npz_loads() {
    let Some(bytes) = fixture("stored.npz") else {
        return;
    };
    let g = gpu();
    let loaded = npz::load_named(&g, &bytes).unwrap();
    let names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["weights", "labels", "mask"]);
    check_fixture_arrays(&loaded);
}

#[test]
fn numpy_fixture_compressed_npz_loads() {
    // The positive inflate case: real zlib output from numpy, with a
    // repetitive entry sized to exercise window copies and dynamic
    // Huffman, next to small fixed-Huffman entries.
    let Some(bytes) = fixture("compressed.npz") else {
        return;
    };
    let g = gpu();
    let loaded = npz::load_named(&g, &bytes).unwrap();
    assert_eq!(loaded.len(), 4);
    check_fixture_arrays(&loaded);
    match loaded
        .iter()
        .find(|(n, _)| n == "repetitive")
        .map(|(_, a)| a)
    {
        Some(NpyArray::F32(a)) => {
            let v = a.to_vec().unwrap();
            assert_eq!(v.len(), 64 * 256);
            assert!(v.iter().enumerate().all(|(i, &x)| x == (i % 64) as f32));
        }
        other => panic!("repetitive: expected F32, got {other:?}"),
    }
}
