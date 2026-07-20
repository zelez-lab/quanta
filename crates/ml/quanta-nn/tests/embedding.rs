//! Embedding — gather values against a host reference, the scatter-add
//! gradient with repeated ids (the sparse update), and the unit-std init.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::embedding::Embedding;
use quanta_nn::layer::Key;

fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device (metal/vulkan feature is on)")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

#[test]
fn gather_matches_the_table_rows() {
    let gpu = gpu();
    let emb = Embedding { vocab: 5, dim: 3 };
    // table[v][e] = 10·v + e — every row distinct and readable by eye.
    let host: Vec<f32> = (0..5)
        .flat_map(|v| (0..3).map(move |e| (10 * v + e) as f32))
        .collect();
    let table = Array::from_slice(&gpu, &host, &[5, 3]).unwrap();
    let ids = Array::from_slice(&gpu, &[3u32, 0, 3, 4], &[4]).unwrap();

    let tape = Tape::<f32>::new();
    let tv = tape.var(table);
    let out = emb.apply(&tv, &ids).unwrap();
    assert_eq!(out.value().shape(), [4, 3]);
    assert_eq!(
        out.value().to_vec().unwrap(),
        vec![
            30.0, 31.0, 32.0, // id 3
            0.0, 1.0, 2.0, // id 0
            30.0, 31.0, 32.0, // id 3 again
            40.0, 41.0, 42.0, // id 4
        ]
    );
}

#[test]
fn gradient_scatter_adds_repeated_ids() {
    let gpu = gpu();
    let emb = Embedding { vocab: 4, dim: 2 };
    let table = Array::from_slice(&gpu, &[0.0f32; 8], &[4, 2]).unwrap();
    // id 2 appears three times, id 0 once, ids 1 and 3 never.
    let ids = Array::from_slice(&gpu, &[2u32, 2, 0, 2], &[4]).unwrap();

    let tape = Tape::<f32>::new();
    let tv = tape.var(table);
    let out = emb.apply(&tv, &ids).unwrap();
    let loss = out.sum().unwrap();
    let dt = loss.grad(&tv).unwrap().to_vec().unwrap();

    // d(sum)/dtable row v = (occurrences of v in ids) per column.
    assert_eq!(
        dt,
        vec![
            1.0, 1.0, // row 0: once
            0.0, 0.0, // row 1: never
            3.0, 3.0, // row 2: three times — the accumulation case
            0.0, 0.0, // row 3: never
        ]
    );
}

#[test]
fn shape_contract_is_loud() {
    let gpu = gpu();
    let emb = Embedding { vocab: 5, dim: 3 };
    let wrong = Array::from_slice(&gpu, &[0.0f32; 6], &[2, 3]).unwrap();
    let ids = Array::from_slice(&gpu, &[0u32], &[1]).unwrap();
    let tape = Tape::<f32>::new();
    let tv = tape.var(wrong);
    assert!(emb.apply(&tv, &ids).is_err());
}

#[test]
fn init_has_unit_std_and_zero_mean() {
    let gpu = gpu();
    let emb = Embedding {
        vocab: 200,
        dim: 50,
    };
    let table: Array<f32> = emb.init(&gpu, Key::new(8)).unwrap();
    assert_eq!(table.shape(), [200, 50]);
    let host = table.to_vec().unwrap();
    let n = host.len() as f64;
    let mean = host.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = host.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    assert!(mean.abs() < 0.05, "init mean {mean}");
    assert!(
        (var.sqrt() - 1.0).abs() < 0.05,
        "init std {} (uniform(−√3, √3) has unit std)",
        var.sqrt()
    );
}
