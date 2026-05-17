//! Broadcasting: read a length-N bias vector as if it were an MxN
//! matrix, without copying it M times. The trick is stride 0 on the
//! broadcast axis — every row of the view reads the same N source
//! elements.
//!
//! Run: cargo run -p quanta-tensor --example broadcast_bias

use quanta_tensor::Layout;

fn main() {
    let bias = Layout::row_major(&[8]).unwrap();
    let view = bias.broadcast(&[4, 8]).unwrap();

    println!("== broadcast bias ==");
    println!(
        "bias  : shape {:?}, strides {:?}",
        bias.shape().dims(),
        bias.strides()
    );
    println!(
        "view  : shape {:?}, strides {:?}",
        view.shape().dims(),
        view.strides()
    );
    println!("  (stride 0 on the row axis -> every row reads the same 8 bias elements)");
    println!();

    println!("== offsets for the 4x8 broadcast view ==");
    for row in 0..4 {
        let row_offsets: Vec<usize> = (0..8).map(|col| view.at(&[row, col]).unwrap()).collect();
        println!("  row {row}: {row_offsets:?}");
    }
}
