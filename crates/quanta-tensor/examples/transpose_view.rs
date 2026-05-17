//! Free transpose: produce A^T from the same buffer as A without
//! copying. Every BLAS-style kernel relies on this.
//!
//! Run: cargo run -p quanta-tensor --example transpose_view

use quanta_tensor::Layout;

fn main() {
    let a = Layout::row_major(&[4, 8]).unwrap();
    let at = a.transpose(0, 1).unwrap();

    println!("== free transpose ==");
    println!(
        "A   : shape {:?}, strides {:?}",
        a.shape().dims(),
        a.strides()
    );
    println!(
        "A^T : shape {:?}, strides {:?}",
        at.shape().dims(),
        at.strides()
    );
    println!();

    // Confirm at(j, i) == a(i, j) — the transpose identity.
    println!("== element-wise check (transposed coordinate -> same offset) ==");
    for (i, j) in [(0usize, 0usize), (1, 3), (2, 5), (3, 7)] {
        let a_off = a.at(&[i, j]).unwrap();
        let at_off = at.at(&[j, i]).unwrap();
        let ok = if a_off == at_off { "OK" } else { "MISMATCH" };
        println!("  A[{i},{j}] = {a_off}, A^T[{j},{i}] = {at_off}  [{ok}]");
    }
    println!();

    // Bonus: compose transpose with a slice to take the first 3
    // columns of A^T (= first 3 rows of A) without allocating.
    let sub = at.slice(0, 0, 3).unwrap();
    println!("== A^T sliced to its first 3 rows ==");
    println!(
        "  shape {:?}, strides {:?}, base_offset {}",
        sub.shape().dims(),
        sub.strides(),
        sub.base_offset()
    );
}
