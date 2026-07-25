//! The named init family — determinism, formula bounds, fan
//! convention, distribution sanity, and the layer-delegation
//! invariant (Linear/Conv2d defaults ARE the named schemes).

use quanta_nn::conv::Conv2d;
use quanta_nn::init::{Init, fans};
use quanta_nn::layer::{Key, Layer, Linear};

fn gpu() -> quanta::Gpu {
    quanta::init().expect("a device")
}

#[test]
fn fans_follow_the_shape_convention() {
    assert_eq!(fans(&[64]), (64, 64));
    assert_eq!(fans(&[128, 32]), (128, 32)); // [in, out]
    assert_eq!(fans(&[16, 8, 3, 3]), (8 * 9, 16 * 9)); // [Cout, Cin, kh, kw]
}

#[test]
fn same_key_same_tensor_different_key_different() {
    let g = gpu();
    let a: Vec<f32> = Init::XavierNormal
        .sample::<f32>(&g, Key::new(7), &[32, 16])
        .unwrap()
        .to_vec()
        .unwrap();
    let b: Vec<f32> = Init::XavierNormal
        .sample::<f32>(&g, Key::new(7), &[32, 16])
        .unwrap()
        .to_vec()
        .unwrap();
    let c: Vec<f32> = Init::XavierNormal
        .sample::<f32>(&g, Key::new(8), &[32, 16])
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(a, b, "same key must reproduce exactly");
    assert_ne!(a, c, "different keys must diverge");
}

#[test]
fn uniform_variants_respect_their_bounds() {
    let g = gpu();
    let (fan_in, fan_out) = (48usize, 24usize);
    let xav = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    let kai = (6.0f32 / fan_in as f32).sqrt();
    let xs: Vec<f32> = Init::XavierUniform
        .sample::<f32>(&g, Key::new(1), &[fan_in, fan_out])
        .unwrap()
        .to_vec()
        .unwrap();
    let ks: Vec<f32> = Init::KaimingUniform
        .sample::<f32>(&g, Key::new(1), &[fan_in, fan_out])
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(xs.iter().all(|v| v.abs() <= xav));
    assert!(ks.iter().all(|v| v.abs() <= kai));
    // Kaiming's bound is wider than Xavier's; the same key stream must
    // actually reach beyond Xavier's bound somewhere.
    assert!(
        ks.iter().any(|v| v.abs() > xav),
        "kaiming should exceed the xavier bound for fan_out < fan_in"
    );
}

#[test]
fn normal_variants_match_their_moments() {
    let g = gpu();
    let n = 100_000usize;
    let std_want = (2.0f32 / 400.0).sqrt(); // kaiming-normal, fan_in = 400
    let v: Vec<f32> = Init::KaimingNormal
        .sample::<f32>(&g, Key::new(3), &[400, n / 400])
        .unwrap()
        .to_vec()
        .unwrap();
    let mean = v.iter().sum::<f32>() / n as f32;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    assert!(mean.abs() < 3e-3, "mean {mean} not ~0");
    let std = var.sqrt();
    assert!(
        (std - std_want).abs() / std_want < 0.02,
        "std {std} vs wanted {std_want}"
    );
}

#[test]
fn zeros_ones_and_plain_forms() {
    let g = gpu();
    let z: Vec<f32> = Init::Zeros
        .sample::<f32>(&g, Key::new(0), &[8])
        .unwrap()
        .to_vec()
        .unwrap();
    let o: Vec<f32> = Init::Ones
        .sample::<f32>(&g, Key::new(0), &[8])
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(z.iter().all(|&v| v == 0.0));
    assert!(o.iter().all(|&v| v == 1.0));
    let u: Vec<f32> = Init::Uniform { lo: 2.0, hi: 3.0 }
        .sample::<f32>(&g, Key::new(5), &[64])
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(u.iter().all(|&v| (2.0..3.0).contains(&v)));
}

#[test]
fn layer_defaults_are_the_named_schemes() {
    let g = gpu();
    // Linear's init must equal KaimingUniform.sample with the key's
    // weight half — the delegation invariant that keeps checkpoints
    // reproducible across the refactor.
    let lin = Linear {
        in_dim: 24,
        out_dim: 8,
        bias: false,
    };
    let params = Layer::<f32>::init(&lin, &g, Key::new(42)).unwrap();
    let (kw, _) = Key::new(42).split();
    let want: Vec<f32> = Init::KaimingUniform
        .sample::<f32>(&g, kw, &[24, 8])
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(params.w.to_vec().unwrap(), want);

    let conv = Conv2d {
        cin: 3,
        cout: 4,
        kh: 3,
        kw: 3,
        stride: 1,
        pad: 0,
        bias: true,
    };
    let cp = Layer::<f32>::init(&conv, &g, Key::new(9)).unwrap();
    let (ckw, ckb) = Key::new(9).split();
    let cwant: Vec<f32> = Init::KaimingUniform
        .sample::<f32>(&g, ckw, &[4, 3, 3, 3])
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(cp.w.to_vec().unwrap(), cwant);
    let _ = ckb;
    assert!(cp.b.unwrap().to_vec().unwrap().iter().all(|&v| v == 0.0));
}
