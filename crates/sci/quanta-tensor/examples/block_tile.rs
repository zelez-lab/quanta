//! GEMM-style block tiling: split a contiguous buffer into blocks
//! and walk one block at a time.
//!
//! This is the load-bearing recipe for downstream math kernels —
//! every tiled matmul, every block radix sort, every Stockham FFT
//! pass turns into one or more `logical_divide` calls.
//!
//! Run: cargo run -p quanta-tensor --example block_tile

use quanta_tensor::Layout;

fn main() {
    // A flat 4096-element output buffer. Each block has 64 elements,
    // so there are 64 blocks total.
    let buffer = Layout::row_major(&[4096]).unwrap();
    let block_tile = Layout::row_major(&[64]).unwrap();

    let tiled = buffer.logical_divide(&block_tile).unwrap();

    println!("== block tiling ==");
    println!(
        "buffer       : shape {:?}, strides {:?}",
        buffer.shape().dims(),
        buffer.strides()
    );
    println!(
        "block tile   : shape {:?}, strides {:?}",
        block_tile.shape().dims(),
        block_tile.strides()
    );
    println!(
        "tiled output : shape {:?}, strides {:?}",
        tiled.shape().dims(),
        tiled.strides()
    );
    println!("  (first mode walks within a block; second mode walks across blocks)");
    println!();

    // Show the first 4 elements of the first 3 blocks.
    println!("== first elements of the first 3 blocks ==");
    for block in 0..3 {
        let first = tiled.at(&[0, block]).unwrap();
        let last = tiled.at(&[63, block]).unwrap();
        println!("  block {block}: elements 0..63 -> offsets {first}..{last}");
    }
    println!();

    // Rank-2 tiler: a 2x3 block, footprint 6, applied to a 24-element
    // buffer gives 4 tiles laid out as a rank-3 layout.
    let buffer = Layout::row_major(&[24]).unwrap();
    let tile = Layout::row_major(&[2, 3]).unwrap();
    let tiled = buffer.logical_divide(&tile).unwrap();

    println!("== rank-2 tiler (2x3 over 24-element buffer) ==");
    println!(
        "tiled output : shape {:?}, strides {:?}",
        tiled.shape().dims(),
        tiled.strides()
    );
    println!("  (within-tile coordinates first, tile index last)");
    println!();
    for tile_idx in 0..4 {
        let corner = tiled.at(&[0, 0, tile_idx]).unwrap();
        let opposite = tiled.at(&[1, 2, tile_idx]).unwrap();
        println!("  tile {tile_idx}: corner -> offset {corner}, opposite -> offset {opposite}");
    }
}
