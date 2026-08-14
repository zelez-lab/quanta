//! Batch kNN with no host round-trip — the Thiaba shape.
//!
//! `cdist_sq` (quanta::sci) builds a `[B, N]` squared-distance matrix on
//! the device, and `block_segmented_top_k_f32_buffer` (quanta::prims)
//! selects the k nearest of every query row straight out of that same
//! buffer: the array's backing field is bound into the primitive's
//! dispatch, so nothing crosses back to the host between the stages.
//! Only the finished `[B, K]` result is read.
//!
//! Top-k is a *largest*-first selection and nearest means *smallest*
//! distance, so the distances are negated before the selection — the k
//! largest of −d are the k smallest of d.
//!
//! Run: cargo run --features "sci prims" --example knn_batch_top_k

use quanta::prims::{block_segmented_top_k_f32_buffer, reference};
use quanta::sci::Array;

/// Queries per batch. Thiaba's validated GPU regime starts at 16.
const B: usize = 16;
/// Database rows — one segment of candidate distances per query.
const N: usize = 64;
/// Feature dimensions.
const D: usize = 8;
/// Neighbours kept per query.
const K: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    // A small synthetic corpus: row j is the point (j, j+1, …) scaled
    // down, so row j's nearest neighbour is itself and then j±1.
    let db_host: Vec<f32> = (0..N * D).map(|i| (i % D + i / D) as f32 * 0.5).collect();
    // The queries ARE the first B rows, which pins the expected answer:
    // query q's nearest row is q, at distance exactly 0.
    let q_host: Vec<f32> = db_host[..B * D].to_vec();

    let db = Array::<f32>::from_slice(&gpu, &db_host, &[N, D])?;
    let queries = Array::<f32>::from_slice(&gpu, &q_host, &[B, D])?;

    // ── device-resident from here to the readback ────────────────────
    let dist = queries.cdist_sq(&db)?; // [B, N]
    let neg = dist.neg()?; // nearest = largest under totalOrder
    let keys = neg
        .backing_field()
        .expect("cdist_sq output is contiguous, so it has a backing field");

    let out = gpu.field::<f32>(B * K)?;
    let mut wave = block_segmented_top_k_f32_buffer(&gpu)?;
    wave.bind(0, keys);
    wave.bind(1, &out);
    wave.set_value(2, N as u32); // one segment per query
    wave.set_value(3, K as u32);
    gpu.dispatch(&wave, (B * N) as u32)?.wait()?;

    let got = out.read()?;
    // ── readback done ────────────────────────────────────────────────

    // Oracle: the same selection on the host, over the distances the GPU
    // itself produced — so this pins the SELECTION stage bit-for-bit and
    // can't drift on the summation order of cdist_sq.
    let neg_host = neg.to_vec()?;
    let mut want = vec![0.0f32; B * K];
    reference::segmented_top_k_f32_blocks(&neg_host, &mut want, N, K);
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "segmented top-k disagrees with the host oracle"
    );

    // Semantic check: query q sits on row q, so its nearest distance is 0
    // and the k distances come back ascending once un-negated.
    for q in 0..B {
        let nearest: Vec<f32> = got[q * K..(q + 1) * K].iter().map(|v| -v).collect();
        assert_eq!(nearest[0], 0.0, "query {q} must find its own row first");
        assert!(
            nearest.windows(2).all(|w| w[0] <= w[1]),
            "query {q}: neighbours not ordered nearest-first"
        );
    }

    println!(
        "batch knn ok: {B} queries × {N} candidates → top-{K}, one dispatch, \
         no host round-trip"
    );
    let first: Vec<f32> = got[..K].iter().map(|v| -v).collect();
    println!("query 0 nearest squared distances: {first:?}");
    Ok(())
}
