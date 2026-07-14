//! `gather_last` — gather along the last axis by a runtime index array.

use quanta_array::Array;

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

/// Host oracle: input [R, D], idx [R, K] → out[r, j] = input[r, idx[r, j]].
fn host_gather(input: &[f32], idx: &[u32], r: usize, d: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * k];
    for row in 0..r {
        for j in 0..k {
            let col = idx[row * k + j] as usize;
            out[row * k + j] = input[row * d + col];
        }
    }
    out
}

#[test]
fn gather_last_picks_columns() {
    let g = gpu();
    // [2, 3] input, pick 2 columns per row.
    let input: Vec<f32> = (0..6).map(|i| i as f32).collect(); // [[0,1,2],[3,4,5]]
    let idx = vec![2u32, 0, 1, 2]; // row0: cols 2,0 ; row1: cols 1,2
    let a = Array::from_slice(&g, &input, &[2, 3]).unwrap();
    let ix = Array::from_slice(&g, &idx, &[2, 2]).unwrap();
    let out = a.gather_last(&ix).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.to_vec().unwrap(), host_gather(&input, &idx, 2, 3, 2));
    // spot check: row0 → [input[2], input[0]] = [2, 0]; row1 → [4, 5].
    assert_eq!(out.to_vec().unwrap(), vec![2.0, 0.0, 4.0, 5.0]);
}

#[test]
fn gather_last_allows_repeats_and_oversampling() {
    let g = gpu();
    // K > D: repeat picks (e.g. sampling with replacement).
    let input: Vec<f32> = vec![10.0, 20.0]; // [1, 2]
    let idx = vec![0u32, 1, 1, 0, 0]; // [1, 5]
    let a = Array::from_slice(&g, &input, &[1, 2]).unwrap();
    let ix = Array::from_slice(&g, &idx, &[1, 5]).unwrap();
    let out = a.gather_last(&ix).unwrap();
    assert_eq!(out.shape(), &[1, 5]);
    assert_eq!(out.to_vec().unwrap(), vec![10.0, 20.0, 20.0, 10.0, 10.0]);
}

#[test]
fn gather_last_3d_leading_dims() {
    let g = gpu();
    // [2, 2, 4] input, pick 1 column per (a, b) row → [2, 2, 1].
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let idx = vec![3u32, 0, 2, 1]; // [2, 2, 1]
    let a = Array::from_slice(&g, &input, &[2, 2, 4]).unwrap();
    let ix = Array::from_slice(&g, &idx, &[2, 2, 1]).unwrap();
    let out = a.gather_last(&ix).unwrap();
    assert_eq!(out.shape(), &[2, 2, 1]);
    // rows are [0..4],[4..8],[8..12],[12..16]; pick cols 3,0,2,1.
    assert_eq!(out.to_vec().unwrap(), vec![3.0, 4.0, 10.0, 13.0]);
}

#[test]
fn gather_last_matches_gather_rows_when_k_is_1() {
    let g = gpu();
    // gather_last with K=1 (then squeezed) == gather_rows.
    let input: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect(); // [4, 3]
    let picks = vec![1u32, 2, 0, 2];
    let a = Array::from_slice(&g, &input, &[4, 3]).unwrap();
    let ix1 = Array::from_slice(&g, &picks, &[4, 1]).unwrap();
    let ixr = Array::from_slice(&g, &picks, &[4]).unwrap();
    let via_last = a.gather_last(&ix1).unwrap().reshape(&[4]).unwrap();
    let via_rows = a.gather_rows(&ixr).unwrap();
    assert_eq!(via_last.to_vec().unwrap(), via_rows.to_vec().unwrap());
}

#[test]
fn gather_last_leading_dim_mismatch_errors() {
    let g = gpu();
    let a = Array::<f32>::zeros(&g, &[2, 3]).unwrap();
    let ix = Array::from_slice(&g, &[0u32, 1, 2], &[3, 1]).unwrap(); // leading 3 != 2
    assert!(a.gather_last(&ix).is_err());
}

/// Host oracle for the gather_last adjoint: out[row, col] = Σ_j grad[row,j]·[idx[row,j]==col].
fn host_scatter(grad: &[f32], idx: &[u32], r: usize, k: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * d];
    for row in 0..r {
        for j in 0..k {
            let col = idx[row * k + j] as usize;
            out[row * d + col] += grad[row * k + j];
        }
    }
    out
}

#[test]
fn scatter_last_add_routes_grad() {
    let g = gpu();
    // grad [2,2] into [2,3], idx picks columns.
    let grad = vec![1.0f32, 2.0, 3.0, 4.0];
    let idx = vec![2u32, 0, 1, 2];
    let gr = Array::from_slice(&g, &grad, &[2, 2]).unwrap();
    let ix = Array::from_slice(&g, &idx, &[2, 2]).unwrap();
    let out = gr.scatter_last_add(&ix, 3).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec().unwrap(), host_scatter(&grad, &idx, 2, 2, 3));
    // row0: col2←1, col0←2 → [2,0,1]; row1: col1←3, col2←4 → [0,3,4].
    assert_eq!(out.to_vec().unwrap(), vec![2.0, 0.0, 1.0, 0.0, 3.0, 4.0]);
}

#[test]
fn scatter_last_add_accumulates_repeats() {
    let g = gpu();
    // Repeated picks of the same column must SUM (the key adjoint property).
    let grad = vec![1.0f32, 10.0, 100.0]; // [1, 3]
    let idx = vec![0u32, 0, 1]; // cols 0,0,1
    let gr = Array::from_slice(&g, &grad, &[1, 3]).unwrap();
    let ix = Array::from_slice(&g, &idx, &[1, 3]).unwrap();
    let out = gr.scatter_last_add(&ix, 2).unwrap();
    // col0 gets 1+10=11, col1 gets 100.
    assert_eq!(out.to_vec().unwrap(), vec![11.0, 100.0]);
}

#[test]
fn gather_scatter_are_adjoint() {
    // <gather(x), g> == <x, scatter(g)> — the defining adjoint identity, so the
    // VJP is correct. Use random-ish x and g.
    let g = gpu();
    let (r, d, k) = (3usize, 4usize, 5usize);
    let x: Vec<f32> = (0..r * d).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let grad: Vec<f32> = (0..r * k).map(|i| (i as f32) * 0.1 + 0.2).collect();
    let idx: Vec<u32> = (0..r * k).map(|i| ((i * 7 + 1) % d) as u32).collect();
    let xa = Array::from_slice(&g, &x, &[r, d]).unwrap();
    let ga = Array::from_slice(&g, &grad, &[r, k]).unwrap();
    let ix = Array::from_slice(&g, &idx, &[r, k]).unwrap();

    let gathered = xa.gather_last(&ix).unwrap().to_vec().unwrap();
    let scattered = ga.scatter_last_add(&ix, d).unwrap().to_vec().unwrap();
    let lhs: f32 = gathered.iter().zip(&grad).map(|(a, b)| a * b).sum();
    let rhs: f32 = scattered.iter().zip(&x).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-3 * (1.0 + lhs.abs()),
        "adjoint: {lhs} vs {rhs}"
    );
}
