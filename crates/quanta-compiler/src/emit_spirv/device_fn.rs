//! Device function emission — SPIR-V OpFunction definitions for `#[quanta::device]`.
//!
//! Device functions are emitted before the main kernel function so their
//! IDs are available for `OpFunctionCall` during body emission.

use quanta_ir::{KernelDef, KernelOp, ScalarType};

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    /// Emit device functions as SPIR-V OpFunction definitions.
    ///
    /// Must be called before emit_kernel so that function IDs are available
    /// for OpFunctionCall during body emission. The function bodies are
    /// emitted into sec_device_fns which is placed before sec_function
    /// in the final module.
    pub(crate) fn emit_device_functions(
        &mut self,
        kernel: &KernelDef,
        gid_var: u32,
        local_id_var: u32,
        group_id_var: u32,
        num_wg_var: u32,
    ) -> Result<(), String> {
        for device_fn in &kernel.device_functions {
            let ret_ty = self.scalar_type_id(device_fn.return_type);

            // Build parameter type IDs
            let mut param_type_ids = Vec::new();
            for (_name, ty) in &device_fn.params {
                param_type_ids.push(self.scalar_type_id(*ty));
            }

            // Create function type: OpTypeFunction ret_ty param_types...
            let func_type = self.ensure_type_function(ret_ty, &param_type_ids);

            // Allocate function ID
            let fn_id = self.alloc_id();
            self.emit_name(fn_id, &device_fn.name);

            // Store mapping for OpFunctionCall
            self.device_fn_ids.insert(
                device_fn.name.clone(),
                (fn_id, ret_ty, param_type_ids.clone()),
            );

            // OpFunction
            Self::emit_op(
                &mut self.sec_device_fns,
                OP_FUNCTION,
                &[ret_ty, fn_id, FUNCTION_CONTROL_NONE, func_type],
            );

            // OpFunctionParameter for each param — save the register mapping
            let old_reg_ids = self.reg_ids.clone();
            let old_reg_types = self.reg_types.clone();
            self.reg_ids.clear();
            self.reg_types.clear();

            for (i, (pname, ty)) in device_fn.params.iter().enumerate() {
                let param_id = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_device_fns,
                    OP_FUNCTION_PARAMETER,
                    &[param_type_ids[i], param_id],
                );
                self.emit_name(param_id, pname);
                // Map the register that the parser assigned to this param
                // The parser allocates registers 0..N for N params
                let type_id = self.scalar_type_id(*ty);
                self.reg_ids.insert(i as u32, param_id);
                self.reg_types.insert(i as u32, type_id);
            }

            // OpLabel for the function body
            let body_label = self.alloc_id();
            Self::emit_op(&mut self.sec_device_fns, OP_LABEL, &[body_label]);

            // Emit the function body into a temporary buffer, then move to sec_device_fns
            let saved_fn = std::mem::take(&mut self.sec_function);
            self.emit_ops(
                &device_fn.body,
                gid_var,
                local_id_var,
                group_id_var,
                num_wg_var,
            )?;

            // Find the last value produced — that's the return value.
            let return_val = self.find_return_value(&device_fn.body, device_fn.return_type);

            let body_words = std::mem::replace(&mut self.sec_function, saved_fn);
            self.sec_device_fns.extend_from_slice(&body_words);

            // OpReturnValue with the return value
            if let Some(ret_id) = return_val {
                Self::emit_op(
                    &mut self.sec_device_fns,
                    252, // OpReturnValue
                    &[ret_id],
                );
            } else {
                Self::emit_op(&mut self.sec_device_fns, OP_RETURN, &[]);
            }

            Self::emit_op(&mut self.sec_device_fns, OP_FUNCTION_END, &[]);

            // Restore main function's register context
            self.reg_ids = old_reg_ids;
            self.reg_types = old_reg_types;
        }
        Ok(())
    }

    /// Find the SPIR-V ID of the return value for a device function body.
    /// The last expression in the body determines the return value.
    pub(crate) fn find_return_value(&self, ops: &[KernelOp], _ret_ty: ScalarType) -> Option<u32> {
        // Walk backwards to find the last op that writes to a dst register
        for op in ops.iter().rev() {
            let dst_reg = match op {
                KernelOp::BinOp { dst, .. }
                | KernelOp::UnaryOp { dst, .. }
                | KernelOp::Cmp { dst, .. }
                | KernelOp::Cast { dst, .. }
                | KernelOp::Const { dst, .. }
                | KernelOp::Load { dst, .. }
                | KernelOp::SharedLoad { dst, .. }
                | KernelOp::MathCall { dst, .. }
                | KernelOp::Copy { dst, .. }
                | KernelOp::DeviceCall { dst, .. }
                | KernelOp::Bitcast { dst, .. }
                | KernelOp::CountTrailingZeros { dst, .. }
                | KernelOp::CountLeadingZeros { dst, .. }
                | KernelOp::PopCount { dst, .. }
                | KernelOp::Dot { dst, .. }
                | KernelOp::SubgroupReduceAdd { dst, .. }
                | KernelOp::SubgroupReduceMin { dst, .. }
                | KernelOp::SubgroupReduceMax { dst, .. }
                | KernelOp::SubgroupExclusiveAdd { dst, .. }
                | KernelOp::SubgroupInclusiveAdd { dst, .. }
                | KernelOp::TextureLoad2D { dst, .. }
                | KernelOp::SubgroupSize { dst, .. } => Some(dst.0),
                _ => None,
            };
            if let Some(reg_num) = dst_reg
                && let Some(&id) = self.reg_ids.get(&reg_num)
            {
                return Some(id);
            }
        }
        None
    }
}
