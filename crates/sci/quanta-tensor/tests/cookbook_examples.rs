//! Validates every snippet in `COOKBOOK.md` and `GETTING_STARTED.md`
//! compiles and produces the asserted output. If a recipe drifts
//! from the actual API, this test fails before users hit the doc.
//!
//! Each test name mirrors the cookbook section so a failure points
//! straight at the recipe to fix.

use quanta_tensor::Layout;

// ── GETTING_STARTED.md ──────────────────────────────────────────

#[test]
fn getting_started_first_layout() {
    let m = Layout::row_major(&[2, 3]).unwrap();
    assert_eq!(m.shape().dims(), &[2, 3]);
    assert_eq!(m.strides(), &[3, 1]);
    assert_eq!(m.linear_size(), 6);
    assert_eq!(m.at(&[1, 2]).unwrap(), 5);
}

#[test]
fn getting_started_local_ops_compose() {
    let src = Layout::row_major(&[2, 3, 4]).unwrap();

    let t = src.transpose(1, 2).unwrap();
    assert_eq!(t.shape().dims(), &[2, 4, 3]);

    let p = src.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape().dims(), &[4, 2, 3]);

    let s = src.slice(0, 1, 2).unwrap();
    assert_eq!(s.shape().dims(), &[1, 3, 4]);

    let b = Layout::row_major(&[3]).unwrap().broadcast(&[2, 3]).unwrap();
    assert_eq!(b.shape().dims(), &[2, 3]);
    assert_eq!(b.strides(), &[0, 1]);
}

#[test]
fn getting_started_chained_view() {
    let view = Layout::row_major(&[2, 3, 4])
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .permute(&[2, 0, 1])
        .unwrap()
        .slice(0, 1, 3)
        .unwrap();
    assert_eq!(view.shape().dims(), &[2, 2, 4]);
}

#[test]
fn getting_started_row_vs_column_major() {
    let r = Layout::row_major(&[2, 3]).unwrap();
    let c = Layout::column_major(&[2, 3]).unwrap();
    assert_eq!(r.at(&[1, 2]).unwrap(), 5); // 1*3 + 2*1
    assert_eq!(c.at(&[1, 2]).unwrap(), 5); // 1*1 + 2*2
}

#[test]
fn getting_started_block_tiling() {
    let buffer = Layout::row_major(&[72]).unwrap();
    let block_tile = Layout::row_major(&[36]).unwrap();
    let blocked = buffer.logical_divide(&block_tile).unwrap();
    assert_eq!(blocked.shape().dims(), &[36, 2]);
    assert_eq!(blocked.at(&[0, 0]).unwrap(), 0);
    assert_eq!(blocked.at(&[35, 1]).unwrap(), 71);
}

#[test]
fn getting_started_rank2_tiler() {
    let buffer = Layout::row_major(&[24]).unwrap();
    let tile = Layout::row_major(&[2, 3]).unwrap();
    let tiled = buffer.logical_divide(&tile).unwrap();
    assert_eq!(tiled.shape().dims(), &[2, 3, 4]);
    assert_eq!(tiled.strides(), &[3, 1, 6]);
}

#[test]
fn getting_started_compose_identity() {
    let mat = Layout::row_major(&[2, 3]).unwrap();
    let flat = Layout::row_major(&[6]).unwrap();
    let view = mat.compose(&flat).unwrap();
    assert_eq!(view.shape().dims(), &[2, 3]);
}

// ── COOKBOOK.md ─────────────────────────────────────────────────

#[test]
fn cookbook_row_major_dense() {
    let m = Layout::row_major(&[4, 8]).unwrap();
    assert_eq!(m.strides(), &[8, 1]);
    assert_eq!(m.at(&[2, 5]).unwrap(), 21);
}

#[test]
fn cookbook_column_major_dense() {
    let m = Layout::column_major(&[4, 8]).unwrap();
    assert_eq!(m.strides(), &[1, 4]);
    assert_eq!(m.at(&[2, 5]).unwrap(), 22);
}

#[test]
fn cookbook_transposed_view() {
    let a = Layout::row_major(&[4, 8]).unwrap();
    let at = a.transpose(0, 1).unwrap();
    assert_eq!(at.shape().dims(), &[8, 4]);
    assert_eq!(at.strides(), &[1, 8]);
    assert_eq!(at.at(&[5, 2]).unwrap(), a.at(&[2, 5]).unwrap());
}

#[test]
fn cookbook_broadcast_bias() {
    let bias = Layout::row_major(&[8]).unwrap();
    let view = bias.broadcast(&[4, 8]).unwrap();
    assert_eq!(view.shape().dims(), &[4, 8]);
    assert_eq!(view.strides(), &[0, 1]);
    for row in 0..4 {
        for col in 0..8 {
            assert_eq!(view.at(&[row, col]).unwrap(), col);
        }
    }
}

#[test]
fn cookbook_gemm_block_tile_rank1() {
    let out = Layout::row_major(&[4096]).unwrap();
    let tile = Layout::row_major(&[64]).unwrap();
    let tiled = out.logical_divide(&tile).unwrap();
    assert_eq!(tiled.shape().dims(), &[64, 64]);
    assert_eq!(tiled.at(&[0, 0]).unwrap(), 0);
    assert_eq!(tiled.at(&[63, 0]).unwrap(), 63);
    assert_eq!(tiled.at(&[0, 1]).unwrap(), 64);
    assert_eq!(tiled.at(&[63, 63]).unwrap(), 4095);
}

#[test]
fn cookbook_gemm_block_tile_rank2() {
    let buffer = Layout::row_major(&[24]).unwrap();
    let tile = Layout::row_major(&[2, 3]).unwrap();
    let tiled = buffer.logical_divide(&tile).unwrap();
    assert_eq!(tiled.shape().dims(), &[2, 3, 4]);
    assert_eq!(tiled.strides(), &[3, 1, 6]);
}

#[test]
fn cookbook_iterated_block_then_warp() {
    const ELEMS_PER_WARP: usize = 12;
    const WARPS_PER_BLOCK: usize = 3;
    const BLOCKS: usize = 2;
    const BLOCK_SIZE: usize = ELEMS_PER_WARP * WARPS_PER_BLOCK;
    const TOTAL: usize = BLOCK_SIZE * BLOCKS;

    let buffer = Layout::row_major(&[TOTAL]).unwrap();
    let blocked = buffer
        .logical_divide(&Layout::row_major(&[BLOCK_SIZE]).unwrap())
        .unwrap();
    assert_eq!(blocked.shape().dims(), &[BLOCK_SIZE, BLOCKS]);

    let inner = Layout::row_major(&[BLOCK_SIZE]).unwrap();
    let warped = inner
        .logical_divide(&Layout::row_major(&[ELEMS_PER_WARP]).unwrap())
        .unwrap();
    assert_eq!(warped.shape().dims(), &[ELEMS_PER_WARP, WARPS_PER_BLOCK]);

    for block in 0..BLOCKS {
        for warp in 0..WARPS_PER_BLOCK {
            for elem in 0..ELEMS_PER_WARP {
                let inner_elem = warp * ELEMS_PER_WARP + elem;
                let got = blocked.at(&[inner_elem, block]).unwrap();
                let expected = block * BLOCK_SIZE + warp * ELEMS_PER_WARP + elem;
                assert_eq!(got, expected);
            }
        }
    }
}

#[test]
fn cookbook_compose_identity_roundtrip() {
    let m = Layout::row_major(&[2, 3, 4]).unwrap();
    let flat = Layout::row_major(&[m.linear_size()]).unwrap();
    let v = m.compose(&flat).unwrap();
    assert_eq!(v.shape().dims(), m.shape().dims());
    assert_eq!(v.strides(), m.strides());
    assert_eq!(v.base_offset(), m.base_offset());
}

#[test]
fn cookbook_subtensor_slice() {
    let m = Layout::row_major(&[8, 16]).unwrap();
    let sub = m.slice(0, 2, 5).unwrap();
    assert_eq!(sub.shape().dims(), &[3, 16]);
    assert_eq!(sub.strides(), &[16, 1]);
    assert_eq!(sub.base_offset(), 32);
    assert_eq!(sub.at(&[0, 0]).unwrap(), 32);
    assert_eq!(sub.at(&[2, 15]).unwrap(), 79);
}

#[test]
fn cookbook_axis_permutation_nhwc_to_nchw() {
    let nhwc = Layout::row_major(&[2, 32, 32, 3]).unwrap();
    let nchw = nhwc.permute(&[0, 3, 1, 2]).unwrap();
    assert_eq!(nchw.shape().dims(), &[2, 3, 32, 32]);
}
