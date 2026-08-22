//! `SPV_KHR_cooperative_matrix` lowering of the three cooperative-matrix
//! ops — the SPIR-V side of what Metal lowers to `simdgroup_matrix`.
//!
//! A fragment register holds an `OpTypeCooperativeMatrixKHR` value, not a
//! scalar. Two places need to know that before any op is emitted:
//! register demotion (an accumulator reassigned inside the K loop becomes
//! a `Function`-storage variable, whose type must be the matrix type), and
//! the capability section (`CooperativeMatrixKHR` + the extension string
//! must precede the type declarations). Both are served by
//! [`SpvEmitter::scan_coop_frags`], which runs in the kernel prologue.
//!
//! Shapes are the device's, not ours: the ops carry `m`, `n`, `k` and the
//! element type, and the driver only reaches pipeline creation with a
//! module like this on a device that enumerated the shape (see
//! `Gpu::cooperative_matrix_shapes`). The emitter therefore lowers any
//! shape faithfully and leaves shape policy to the consumer.
//!
//! Row-major only: Quanta's tiles are dense row-major with an explicit
//! element `stride`, the same contract `simdgroup_load` takes, so every
//! load/store passes `RowMajorKHR` and the stride register through.

use super::constants::*;
use super::emitter::SpvEmitter;
use crate::{KernelOp, MatrixFrag, Reg, ScalarType};

/// `OpTypeCooperativeMatrixKHR`.
pub(crate) const OP_TYPE_COOPERATIVE_MATRIX_KHR: u16 = 4456;
/// `OpCooperativeMatrixLoadKHR`.
pub(crate) const OP_COOPERATIVE_MATRIX_LOAD_KHR: u16 = 4457;
/// `OpCooperativeMatrixStoreKHR`.
pub(crate) const OP_COOPERATIVE_MATRIX_STORE_KHR: u16 = 4458;
/// `OpCooperativeMatrixMulAddKHR`.
pub(crate) const OP_COOPERATIVE_MATRIX_MUL_ADD_KHR: u16 = 4459;
/// `Capability CooperativeMatrixKHR`.
pub(crate) const CAPABILITY_COOPERATIVE_MATRIX_KHR: u32 = 6022;
/// `CooperativeMatrixUse` operands.
const USE_MATRIX_A: u32 = 0;
const USE_MATRIX_B: u32 = 1;
const USE_MATRIX_ACCUMULATOR: u32 = 2;
/// `CooperativeMatrixLayout::RowMajorKHR`.
const LAYOUT_ROW_MAJOR: u32 = 0;

/// The shape of the matrix value a fragment register holds: `(rows,
/// cols, use, element type)`. Derived from the op that writes the register
/// — `A` is `m×k`, `B` is `k×n`, the accumulator (load or MMA result) is
/// `m×n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoopFrag {
    pub rows: u8,
    pub cols: u8,
    pub use_: u32,
    pub ty: ScalarType,
}

fn frag_shape(frag: MatrixFrag, m: u8, n: u8, k: u8, ty: ScalarType) -> CoopFrag {
    let (rows, cols, use_) = match frag {
        MatrixFrag::A => (m, k, USE_MATRIX_A),
        MatrixFrag::B => (k, n, USE_MATRIX_B),
        MatrixFrag::Accumulator => (m, n, USE_MATRIX_ACCUMULATOR),
    };
    CoopFrag {
        rows,
        cols,
        use_,
        ty,
    }
}

impl SpvEmitter {
    /// Record the fragment shape of every register written by a
    /// cooperative-matrix op (body, nested arms, device functions are the
    /// caller's job). Runs before demotion so a demoted fragment register
    /// gets a matrix-typed variable, and before the capability section so
    /// the extension is declared exactly when a fragment exists.
    pub(crate) fn scan_coop_frags(&mut self, ops: &[KernelOp]) {
        for op in ops {
            match op {
                KernelOp::CooperativeMatrixLoad {
                    dst,
                    frag,
                    m,
                    n,
                    k,
                    ty,
                    ..
                } => {
                    self.coop_frag_regs
                        .insert(dst.0, frag_shape(*frag, *m, *n, *k, *ty));
                }
                KernelOp::CooperativeMMA {
                    dst, m, n, k, ty, ..
                } => {
                    self.coop_frag_regs
                        .insert(dst.0, frag_shape(MatrixFrag::Accumulator, *m, *n, *k, *ty));
                }
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    self.scan_coop_frags(then_ops);
                    self.scan_coop_frags(else_ops);
                }
                KernelOp::Loop { body, .. } => self.scan_coop_frags(body),
                _ => {}
            }
        }
    }

    /// Declare the capability + extension once per module.
    fn ensure_coopmat_ext(&mut self) {
        if self.coopmat_declared {
            return;
        }
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_COOPERATIVE_MATRIX_KHR],
        );
        let name_words = Self::string_words("SPV_KHR_cooperative_matrix");
        Self::emit_op(&mut self.sec_extension, OP_EXTENSION, &name_words);
        // The Vulkan memory model is mandatory alongside CooperativeMatrixKHR;
        // the module's OpMemoryModel already selected it (kernel prologue).
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_VULKAN_MEMORY_MODEL],
        );
        let vmm_words = Self::string_words("SPV_KHR_vulkan_memory_model");
        Self::emit_op(&mut self.sec_extension, OP_EXTENSION, &vmm_words);
        self.coopmat_declared = true;
    }

    /// `OpTypeCooperativeMatrixKHR %elem %scope %rows %cols %use`, cached
    /// per shape. Scope is always `Subgroup`; the operands are `OpConstant`
    /// ids, emitted ahead of the type in the same section.
    pub(crate) fn coopmat_type(&mut self, frag: CoopFrag) -> u32 {
        let key = format!(
            "coopmat:{:?}:{}:{}:{}",
            frag.ty, frag.rows, frag.cols, frag.use_
        );
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        self.ensure_coopmat_ext();
        let elem = self.scalar_type_id(frag.ty);
        let scope = self.emit_constant_u32(SCOPE_SUBGROUP);
        let rows = self.emit_constant_u32(u32::from(frag.rows));
        let cols = self.emit_constant_u32(u32::from(frag.cols));
        let use_ = self.emit_constant_u32(frag.use_);
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_COOPERATIVE_MATRIX_KHR,
            &[id, elem, scope, rows, cols, use_],
        );
        self.type_cache.insert(key, id);
        id
    }

    /// Pointer to the tile's top-left element: a `StorageBuffer` field
    /// (struct member 0, runtime array, `index`) or a `Workgroup` shared
    /// array (`index`).
    fn coop_tile_pointer(
        &mut self,
        field: u32,
        index: Reg,
        from_shared: bool,
    ) -> Result<u32, String> {
        let idx = self.reg_value_id(index)?;
        if from_shared {
            let (var_id, elem_ty) = *self
                .shared_vars
                .get(&field)
                .ok_or_else(|| format!("shared memory {} not declared", field))?;
            let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, elem_ty);
            let chain = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ACCESS_CHAIN,
                &[ptr_elem, chain, var_id, idx],
            );
            Ok(chain)
        } else {
            let (var_id, elem_ty, _) = *self
                .field_vars
                .get(&field)
                .ok_or_else(|| format!("field {} not declared", field))?;
            let zero = self.emit_constant_u32(0);
            let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
            let chain = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_ACCESS_CHAIN,
                &[ptr_elem, chain, var_id, zero, idx],
            );
            Ok(chain)
        }
    }

    /// `OpCooperativeMatrixLoadKHR %ty %dst %ptr %RowMajor %stride`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_coop_load(
        &mut self,
        dst: Reg,
        field: u32,
        index: Reg,
        stride: Reg,
        frag: MatrixFrag,
        from_shared: bool,
        m: u8,
        n: u8,
        k: u8,
        ty: ScalarType,
    ) -> Result<(), String> {
        let shape = frag_shape(frag, m, n, k, ty);
        let mat_ty = self.coopmat_type(shape);
        let ptr = self.coop_tile_pointer(field, index, from_shared)?;
        let layout = self.emit_constant_u32(LAYOUT_ROW_MAJOR);
        let stride_id = self.reg_value_id(stride)?;
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COOPERATIVE_MATRIX_LOAD_KHR,
            &[mat_ty, result, ptr, layout, stride_id],
        );
        self.set_reg(dst, result, mat_ty);
        Ok(())
    }

    /// `OpCooperativeMatrixStoreKHR %ptr %src %RowMajor %stride`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_coop_store(
        &mut self,
        field: u32,
        index: Reg,
        stride: Reg,
        src: Reg,
    ) -> Result<(), String> {
        let ptr = self.coop_tile_pointer(field, index, false)?;
        let src_id = self.reg_value_id(src)?;
        let layout = self.emit_constant_u32(LAYOUT_ROW_MAJOR);
        let stride_id = self.reg_value_id(stride)?;
        Self::emit_op(
            &mut self.sec_function,
            OP_COOPERATIVE_MATRIX_STORE_KHR,
            &[ptr, src_id, layout, stride_id],
        );
        Ok(())
    }

    /// `OpCooperativeMatrixMulAddKHR %acc_ty %dst %a %b %c`. Float element
    /// types need no `CooperativeMatrixOperands`; the validator keeps
    /// integer fragments off this path until the signedness operands are
    /// wired.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_coop_mma(
        &mut self,
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
        m: u8,
        n: u8,
        k: u8,
        ty: ScalarType,
    ) -> Result<(), String> {
        let acc_ty = self.coopmat_type(frag_shape(MatrixFrag::Accumulator, m, n, k, ty));
        let a_id = self.reg_value_id(a)?;
        let b_id = self.reg_value_id(b)?;
        let c_id = self.reg_value_id(c)?;
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COOPERATIVE_MATRIX_MUL_ADD_KHR,
            &[acc_ty, result, a_id, b_id, c_id],
        );
        self.set_reg(dst, result, acc_ty);
        Ok(())
    }
}
