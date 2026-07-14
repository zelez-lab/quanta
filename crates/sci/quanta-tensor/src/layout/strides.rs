//! Stride computation for row-major / column-major layouts.
//!
//! Free functions, no Layout knowledge. The constructors in
//! `super::Layout` use these to build their strides.

pub(super) fn compute_row_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    let mut running: isize = 1;
    for i in (0..dims.len()).rev() {
        strides[i] = running;
        running *= dims[i] as isize;
    }
    strides
}

pub(super) fn compute_column_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    let mut running: isize = 1;
    for i in 0..dims.len() {
        strides[i] = running;
        running *= dims[i] as isize;
    }
    strides
}
