//! `Array<T>` — a GPU-backed N-dimensional array.

use std::sync::Arc;

use quanta::{Field, Gpu, GpuType};
use quanta_tensor::Layout;

use crate::error::ArrayError;

/// An N-dimensional array stored in GPU memory.
///
/// Owns a [`Field<T>`] (the flat buffer, behind an `Arc` so zero-copy
/// views can share it) and a [`Layout`] (shape + strides over that
/// buffer). Shape manipulations (`reshape`, `permute`, `transpose`, …)
/// produce zero-copy views that share the same `Field` with a different
/// `Layout`; only materializing operations (`contiguous`, ufuncs,
/// reductions, `to_vec`) touch the device. The buffer is freed once, when
/// the last view drops.
pub struct Array<T: GpuType> {
    field: Arc<Field<T>>,
    layout: Layout,
    gpu: Gpu,
}

impl<T: GpuType> Array<T> {
    /// Wrap an existing `Field` + `Layout`. The layout's element count must
    /// fit within the field. Internal constructor used by the builders.
    pub(crate) fn from_parts(gpu: Gpu, field: Field<T>, layout: Layout) -> Self {
        debug_assert!(
            layout.base_offset() as usize + layout.linear_size() <= field.len(),
            "layout overruns the backing field"
        );
        Array {
            field: Arc::new(field),
            layout,
            gpu,
        }
    }

    // ── Shape / metadata ────────────────────────────────────────────────

    /// The shape (extent of each dimension).
    pub fn shape(&self) -> &[usize] {
        self.layout.shape().dims()
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    /// Total number of logical elements.
    pub fn len(&self) -> usize {
        self.layout.linear_size()
    }

    /// Whether the array has zero elements. (Always false today — shapes
    /// with a zero extent are rejected at construction — but kept for the
    /// idiomatic `is_empty` pairing with `len`.)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The strides (in elements) of each dimension.
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// The backing layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The `Gpu` handle this array lives on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Borrow the backing field (for binding into a dispatch).
    pub(crate) fn field_ref(&self) -> &Field<T> {
        &self.field
    }

    /// A new `Array` view sharing the same backing field + layout (cheap
    /// `Arc` share — used when an op's input is already in the form it
    /// needs and no copy is required).
    pub(crate) fn shallow_clone(&self) -> Array<T> {
        Array {
            field: Arc::clone(&self.field),
            layout: self.layout.clone(),
            gpu: self.gpu.clone(),
        }
    }

    /// Whether the layout is row-major contiguous from offset 0 (the fast
    /// path: a plain linear walk matches logical order).
    pub fn is_contiguous(&self) -> bool {
        if self.layout.base_offset() != 0 {
            return false;
        }
        match Layout::row_major(self.layout.shape().dims()) {
            Ok(rm) => rm.strides() == self.layout.strides(),
            Err(_) => false,
        }
    }

    // ── Zero-copy views (pure Layout ops on the same Field) ─────────────

    /// Permute the axes (a transposing view, zero-copy).
    pub fn permute(&self, perm: &[usize]) -> Result<Array<T>, ArrayError> {
        let layout = self.layout.permute(perm)?;
        Ok(Array {
            field: self.field.clone(),
            layout,
            gpu: self.gpu.clone(),
        })
    }

    /// Swap two axes (zero-copy transposing view).
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Array<T>, ArrayError> {
        let layout = self.layout.transpose(d0, d1)?;
        Ok(Array {
            field: self.field.clone(),
            layout,
            gpu: self.gpu.clone(),
        })
    }

    /// Reshape to a new shape with the same element count. Requires a
    /// contiguous array (a strided view must be `.contiguous()`-ified
    /// first); the result is a zero-copy row-major view.
    pub fn reshape(&self, shape: &[usize]) -> Result<Array<T>, ArrayError> {
        if !self.is_contiguous() {
            return Err(ArrayError::NotContiguous);
        }
        let new = Layout::row_major(shape)?;
        if new.linear_size() != self.len() {
            return Err(ArrayError::LengthMismatch {
                expected: self.len(),
                got: new.linear_size(),
            });
        }
        Ok(Array {
            field: self.field.clone(),
            layout: new,
            gpu: self.gpu.clone(),
        })
    }

    /// Broadcast to a target shape (zero-copy view with zero-strides on
    /// the broadcast axes).
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Array<T>, ArrayError> {
        let layout = self.layout.broadcast(shape)?;
        Ok(Array {
            field: self.field.clone(),
            layout,
            gpu: self.gpu.clone(),
        })
    }

    // ── Materialization ─────────────────────────────────────────────────

    /// Download the array to host memory in **logical row-major order**.
    /// Contiguous arrays read the field directly; strided views are
    /// gathered on the host from the raw buffer via the layout.
    pub fn to_vec(&self) -> Result<Vec<T>, ArrayError> {
        let raw = self.field.read()?;
        if self.is_contiguous() {
            return Ok(raw[..self.len()].to_vec());
        }
        // Strided / offset view: walk logical coordinates and gather.
        let dims = self.layout.shape().dims().to_vec();
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        let mut coord = vec![0usize; dims.len()];
        for _ in 0..n {
            let idx = self.layout.at(&coord)?;
            out.push(raw[idx]);
            // increment the mixed-radix coordinate (row-major, last axis fastest)
            for axis in (0..dims.len()).rev() {
                coord[axis] += 1;
                if coord[axis] < dims[axis] {
                    break;
                }
                coord[axis] = 0;
            }
        }
        Ok(out)
    }
}
