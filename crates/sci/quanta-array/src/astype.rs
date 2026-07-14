//! `astype` — elementwise dtype conversion between array element types.
//!
//! Reads each element as `T`, emits a `Cast` to `U`, writes `U`. One thread per
//! element. The common need is `u32 → f32` (turning an index/mask array into
//! something that can feed a matmul), but it works between any two
//! `ArrayScalar` types the `Cast` IR op supports.

use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, serialize_kernel};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

impl<T: ArrayScalar> Array<T> {
    /// Convert this array's elements to type `U` (`out[i] = self[i] as U`),
    /// row-major and elementwise. Float↔int follow the usual truncation /
    /// rounding of the `Cast` op.
    pub fn astype<U: ArrayScalar>(&self) -> Result<Array<U>, ArrayError> {
        let from = T::scalar_type();
        let to = U::scalar_type();
        let src = self.contiguous_or_self()?;
        let n = src.len();

        let def = KernelDef {
            name: "qa_astype".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "src".into(),
                    slot: 0,
                    scalar_type: from,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 1,
                    scalar_type: to,
                },
            ],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 0,
                    index: Reg(0),
                    ty: from,
                },
                KernelOp::Cast {
                    dst: Reg(2),
                    src: Reg(1),
                    from,
                    to,
                },
                KernelOp::Store {
                    field: 1,
                    index: Reg(0),
                    src: Reg(2),
                    ty: to,
                },
            ],
            body_source: None,
            next_reg: 3,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let out = self.gpu().field::<U>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(self.shape())?,
        ))
    }
}
