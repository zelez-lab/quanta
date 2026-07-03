//! `select_rows` (embedding lookup) and `scatter_rows_into` (sparse adjoint).

use quanta_array::Array;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

#[test]
fn select_rows_looks_up_embeddings() {
    let g = gpu();
    // table [4, 3]: 4 vocab, embedding dim 3.
    let table: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = Array::from_slice(&g, &table, &[4, 3]).unwrap();
    let ids = Array::from_slice(&g, &[2u32, 0, 3], &[3]).unwrap();
    let out = t.select_rows(&ids).unwrap();
    assert_eq!(out.shape(), &[3, 3]);
    // row 2 = [6,7,8], row 0 = [0,1,2], row 3 = [9,10,11]
    assert_eq!(
        out.to_vec().unwrap(),
        vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn scatter_rows_into_routes_and_accumulates() {
    let g = gpu();
    // grad [3, 2] scattered into table [3, 2] by ids with a REPEAT.
    let grad = vec![1.0f32, 2.0, 10.0, 20.0, 100.0, 200.0];
    let gr = Array::from_slice(&g, &grad, &[3, 2]).unwrap();
    let ids = Array::from_slice(&g, &[0u32, 2, 0], &[3]).unwrap(); // rows 0,2,0
    let out = gr.scatter_rows_into(&ids, 3).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    // row0 = grad[0]+grad[2] = [1+100, 2+200] = [101, 202]
    // row1 = 0 ; row2 = grad[1] = [10, 20]
    assert_eq!(
        out.to_vec().unwrap(),
        vec![101.0, 202.0, 0.0, 0.0, 10.0, 20.0]
    );
}

#[test]
fn embedding_lookup_scatter_are_adjoint() {
    // <select(table), g> == <table, scatter(g)> — the VJP identity.
    let g = gpu();
    let (v, e, b) = (5usize, 4usize, 6usize);
    let table: Vec<f32> = (0..v * e).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let grad: Vec<f32> = (0..b * e).map(|i| (i as f32) * 0.1 + 0.2).collect();
    let ids: Vec<u32> = (0..b).map(|i| ((i * 3 + 1) % v) as u32).collect();
    let ta = Array::from_slice(&g, &table, &[v, e]).unwrap();
    let ga = Array::from_slice(&g, &grad, &[b, e]).unwrap();
    let ix = Array::from_slice(&g, &ids, &[b]).unwrap();

    let selected = ta.select_rows(&ix).unwrap().to_vec().unwrap();
    let scattered = ga.scatter_rows_into(&ix, v).unwrap().to_vec().unwrap();
    let lhs: f32 = selected.iter().zip(&grad).map(|(a, b)| a * b).sum();
    let rhs: f32 = scattered.iter().zip(&table).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-3 * (1.0 + lhs.abs()),
        "adjoint {lhs} vs {rhs}"
    );
}
