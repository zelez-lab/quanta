//! Axis permutation: convert between channel-last (TensorFlow NHWC)
//! and channel-first (PyTorch NCHW) layouts without copying. The
//! buffer is unchanged; only the index function changes.
//!
//! Run: cargo run -p quanta-tensor --example nhwc_to_nchw

use quanta_tensor::Layout;

fn main() {
    // NHWC: batch, height, width, channels. Default for TF.
    let nhwc = Layout::row_major(&[2, 32, 32, 3]).unwrap();
    // NCHW: batch, channels, height, width. Default for PyTorch.
    let nchw = nhwc.permute(&[0, 3, 1, 2]).unwrap();

    println!("== axis permutation (NHWC <-> NCHW) ==");
    println!(
        "NHWC : shape {:?}, strides {:?}",
        nhwc.shape().dims(),
        nhwc.strides()
    );
    println!(
        "NCHW : shape {:?}, strides {:?}",
        nchw.shape().dims(),
        nchw.strides()
    );
    println!();

    // Show that the same buffer offset is reached by the matching
    // coordinates in each layout: NHWC[b,h,w,c] == NCHW[b,c,h,w].
    let (b, h, w, c) = (1usize, 5, 7, 2);
    let off_nhwc = nhwc.at(&[b, h, w, c]).unwrap();
    let off_nchw = nchw.at(&[b, c, h, w]).unwrap();
    println!("== same source pixel via either layout ==");
    println!("  NHWC[b={b},h={h},w={w},c={c}] -> offset {off_nhwc}");
    println!("  NCHW[b={b},c={c},h={h},w={w}] -> offset {off_nchw}");
    println!(
        "  {}",
        if off_nhwc == off_nchw {
            "matches — same buffer, different walk order"
        } else {
            "MISMATCH"
        }
    );

    // Round-trip: NHWC -> NCHW -> NHWC should be the identity.
    let round_trip = nchw.permute(&[0, 2, 3, 1]).unwrap();
    let ok =
        round_trip.shape().dims() == nhwc.shape().dims() && round_trip.strides() == nhwc.strides();
    println!();
    println!(
        "== round-trip NHWC -> NCHW -> NHWC : {}",
        if ok { "identity" } else { "DRIFT" }
    );
}
